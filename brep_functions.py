import os.path
import pickle
import itertools
import numpy as np
import torch
import pulp
import random
from tqdm import tqdm
from multiprocessing.pool import Pool
from chamferdist import ChamferDistance
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface, GeomAPI_PointsToBSpline
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.gp import gp_Pnt
from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Wire, ShapeFix_Edge
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
from collections import defaultdict


def create_vertex_edge_adjacency(edgeCorner):
    vertex_edge_dict = defaultdict(list)
    for edge_id, (v1, v2) in enumerate(edgeCorner):
        vertex_edge_dict[v1].append(edge_id)
        vertex_edge_dict[v2].append(edge_id)
    return vertex_edge_dict


def check_loop_edge(path):

    with open(os.path.join('data_process/furniture_parsed', path), 'rb') as tf:
        data = pickle.load(tf)

    # ne*2, [(edge1, edge2, ...), ...], ne*2
    edgeFace, faceEdge, edgeCorner = data['edgeFace_adj'], data['faceEdge_adj'], data['edgeCorner_adj']

    vertex_edge_dict = create_vertex_edge_adjacency(edgeCorner)

    """Check Loops"""
    face_loop = []
    for edges in faceEdge:
        visited_edges = set()
        loops = []

        for start_edge in edges:
            if start_edge in visited_edges:
                continue

            loop = []
            current_edge = start_edge
            current_vertex = edgeCorner[current_edge][0]
            while current_edge not in visited_edges:
                visited_edges.add(current_edge)
                loop.append(current_edge)

                # find next vertex
                if edgeCorner[current_edge][0] == current_vertex:
                    current_vertex = edgeCorner[current_edge][1]
                else:
                    current_vertex = edgeCorner[current_edge][0]

                # find next edge
                for e in vertex_edge_dict[current_vertex]:
                    if e != current_edge and e in edges:
                        current_edge = e
                        break

            loops.append(loop)

        if len(loops) > 1:
            print(loops)
        face_loop.append(loops)



    """Check Common Edges"""
    for face1_edges, face2_edges in itertools.combinations(faceEdge, 2):
        connected = are_faces_connected(face1_edges, face2_edges, edgeCorner)
        if not connected:
            print(path)
            return 0
    return 1


class BrepTopology:
    def __init__(self, edgeFace_adj, faceEdge_adj):
        self.edgeFace_adj = edgeFace_adj
        self.faceEdge_adj = faceEdge_adj

    def opposite_face(self, edge_id, face_id):
        faces = self.edgeFace_adj[edge_id]
        return faces[0] if faces[1] == face_id else faces[1]


def edge_choice(edge_id, condition, edges):

    return random.sample(edges, len(edges))


def construct_topology(brepTopo: BrepTopology):   # ne*2, [[e1, e2, ...], ...]

    edgeFace_adj = brepTopo.edgeFace_adj.tolist()
    faceEdge_adj = brepTopo.faceEdge_adj
    handled_faces = {}

    # for face_id, edges in enumerate(faceEdge_adj):
    #     start_edge = None
    #     edge_order = []
    #     edge_loop = []
    #
    #     edge = edges[0]
    #     while True:
    #         if start_edge is None:
    #             start_edge = edge
    #             edge_loop.append(edge)
    #
    #         if brepTopo.opposite_face(edge, face_id) not in handled_faces:
    #             edge_sort = edges[i+1:]
    #             edge_sort = edge_choice(edge, edge_sort.append(-1) if len(edge_loop) > 1 else edge_order, edges)
    #             for next_edge in edge_sort:


def are_faces_connected(face1_edges, face2_edges, edgeCorner):
    # Find common edges between two faces
    edges = list(set(face1_edges) & set(face2_edges))

    # Check if the common edges are connected
    if len(edges) < 2:
        return True

    # Create a map of edge to its vertices
    edge_vertices = {edge: set(edgeCorner[edge]) for edge in edges}

    # Start from the first edge
    current_edge = edges[0]
    current_vertices = edge_vertices[current_edge]
    connected_edges = set()
    connected_edges.add(current_edge)

    while len(connected_edges) < len(edges):
        found_next_edge = False

        for edge in edges:
            if edge in connected_edges:
                continue
            if not current_vertices.isdisjoint(edge_vertices[edge]):
                connected_edges.add(edge)
                current_vertices = current_vertices.union(edge_vertices[edge])
                found_next_edge = True
                break

        if not found_next_edge:
            return False

    return True


def scale_surf(bbox, surf, node_mask):   # b*n*6, b*n*p*3, b*n

    face_wcs = []

    for box, face, mask in zip(bbox, surf, node_mask):  #
        box = box[mask]       # n1*6
        face = face[mask]     # n1*p*3
        min_xyz, max_xyz = box.reshape(-1, 2, 3).min(1).values, box.reshape(-1, 2, 3).max(1).values   # n1*3, n1*3
        center_point = (min_xyz + max_xyz) * 0.5  # n1*3
        face = face * ((max_xyz - min_xyz) * 0.5).unsqueeze(1) + center_point.unsqueeze(1)  # n1*p*3
        face_wcs.append(face)

    return face_wcs


def manual_edgeVert_topology(edge_pnts, edgeFace_adj):   # ne*32*3, ne*2
    assert (edgeFace_adj[:, 0] - edgeFace_adj[:, 1]).abs().max() > 0

    edge_start_end = edge_pnts[:, [0, -1], :]   # ne*2*3

    # Initialize the faceEdge_adj list with empty lists for each face
    faceEdge_adj = [[] for _ in range(edgeFace_adj.max()+1)]     # [[edge_id1, edge_id2, ...], ...]

    total_merge_vertex = []

    # Iterate over each edge in edgeFace_adj
    for i, edge in enumerate(edgeFace_adj):
        face1, face2 = edge.tolist()
        faceEdge_adj[face1].append(i)
        faceEdge_adj[face2].append(i)

    for edge_ids in faceEdge_adj:   # list[edge_id1, edge_id2, ...]
        face_edges_flatten = edge_start_end[edge_ids].reshape(-1, 3)   # (m*2)*3
        vertex_id = torch.cat((2*torch.tensor(edge_ids).unsqueeze(-1), 2*torch.tensor(edge_ids).unsqueeze(-1)+1), dim=-1).reshape(-1)  # (m*2,)

        # connect end points by closest distance
        merged_vertex_id = []
        for edge_idx in edge_ids:
            start_end = edge_start_end[edge_idx]  # 2*3
            self_id = [2 * edge_idx, 2 * edge_idx + 1]

            # left endpoint
            distance = torch.norm(face_edges_flatten - start_end[0], dim=1)  # (m*2, )
            min_id = vertex_id[torch.argsort(distance)].tolist()
            id_no_self = [x for x in min_id if x not in self_id]
            merged_vertex_id.append(sorted([2 * edge_idx, id_no_self[0]]))

            # right endpoint
            distance = torch.norm(face_edges_flatten - start_end[1], dim=1)
            min_id = vertex_id[torch.argsort(distance)].tolist()
            id_no_self = [x for x in min_id if x not in self_id]
            merged_vertex_id.append(sorted([2 * edge_idx + 1, id_no_self[0]]))

        merged_vertex_id = torch.unique(torch.tensor(merged_vertex_id), dim=0)   # ?*2
        if len(merged_vertex_id) != len(edge_ids):
            print("Failed!")
            return False, None, None

        total_merge_vertex.append(merged_vertex_id)

    total_merge_vertex = torch.unique(torch.cat(total_merge_vertex, dim=0), dim=0)  # ?*2
    assert (total_merge_vertex[:, 0] - total_merge_vertex[:, 1]).abs().max() > 0
    vertex_pnt = edge_start_end.reshape(-1, 3)   # (2*ne, 3)
    edgeVertex_adj = torch.tensor([[2*i, 2*i+1] for i in range(len(edge_pnts))])    # ne*2

    merge_cluster = [set(total_merge_vertex[1].tolist())]
    for merge_pair in total_merge_vertex[1:]:
        merge_pair = set(merge_pair.tolist())
        intersection_idx = []
        for i, existing_cluster in enumerate(merge_cluster):
            if merge_pair & existing_cluster:
                intersection_idx.append(i)
        if intersection_idx:
            # Combine the sets at intersection_idx with merge_pair
            combined_set = set.union(*{merge_cluster[idx] for idx in intersection_idx}, merge_pair)

            # Convert intersection_idx to a set for faster lookup
            intersection_set = set(intersection_idx)

            # Remove the sets at intersection_idx from merge_cluster
            merge_cluster = [s for idx, s in enumerate(merge_cluster) if idx not in intersection_set]

            # Add the combined set to merge_cluster
            merge_cluster.append(combined_set)
        else:
            merge_cluster.append(merge_pair)
    unique_vertex_pnt = []
    new_vertex_idx = torch.arange(vertex_pnt.shape[0])  # (2*ne, )
    for idx, cluster in enumerate(merge_cluster):
        unique_vertex_pnt.append(vertex_pnt[list(cluster)].mean(0))
        new_vertex_idx[list(cluster)] = idx

    edgeVertex_adj = new_vertex_idx[edgeVertex_adj]   # ne*2

    return True, torch.tensor(unique_vertex_pnt), edgeVertex_adj


def optimize_edgeVert_topology(edge_pnts, edgeFace_adj):   # ne*32*3, ne*2

    assert (edgeFace_adj[:, 0] - edgeFace_adj[:, 1]).abs().min() > 0

    edge_start_end = edge_pnts[:, [0, -1], :]   # ne*2*3

    # Initialize the faceEdge_adj list with empty lists for each face
    faceEdge_adj = [[] for _ in range(edgeFace_adj.max()+1)]     # [[edge_id1, edge_id2, ...], ...]

    total_merge_vertex = []

    # Iterate over each edge in edgeFace_adj
    for i, edge in enumerate(edgeFace_adj):
        face1, face2 = edge.tolist()
        faceEdge_adj[face1].append(i)
        faceEdge_adj[face2].append(i)

    for edge_ids in faceEdge_adj:   # list[edge_id1, edge_id2, ...]
        face_edges_flatten = edge_start_end[edge_ids].reshape(-1, 3)   # (m*2)*3
        n = face_edges_flatten.shape[0]   # n=2m
        vertex_id = torch.cat((2*torch.tensor(edge_ids).unsqueeze(-1), 2*torch.tensor(edge_ids).unsqueeze(-1)+1), dim=-1).reshape(-1)  # (m*2,)
        distance = torch.sqrt(torch.sum((face_edges_flatten.unsqueeze(1) - face_edges_flatten.unsqueeze(0)) ** 2, dim=-1))  # (m*2)*(m*2)
        self_id = torch.tensor([[2*i, 2*i+1] for i in range(n//2)])    # m*2
        max_value = 10000
        distance[self_id[:, 0], self_id[:, 1]] = max_value
        triu_indices = torch.triu_indices(n, n, offset=1)   # 2*(n^2-n)/2, idx tensor
        cof = distance[triu_indices[0], triu_indices[1]]  # (n^2-n)/2

        prob = pulp.LpProblem("0-1_Minimization_Problem", pulp.LpMinimize)

        variable_num = len(cof)
        x = [pulp.LpVariable(f'x{i}', cat='Binary') for i in range(variable_num)]

        prob += pulp.lpSum(cof[i] * x[i] for i in range(variable_num)), "Objective_Function"

        for j in range(n):
            temp = torch.zeros((n, n))
            temp[j, :] = 1
            temp[:, j] = 1
            A = temp[triu_indices[0], triu_indices[1]]  # (n^2-n)/2
            prob += pulp.lpSum(A[i] * x[i] for i in range(variable_num)) == 1, f"Constraint_{j}"

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        solution_val = torch.tensor([v.varValue for v in prob.variables()])   # (n^2-n)/2
        solution_name = torch.tensor([int(v.name[1:]) for v in prob.variables()])  # (n^2-n)/2
        solution = torch.zeros(variable_num)
        solution[solution_name] = solution_val
        assert (solution[cof == max_value] == 0).all()
        # print(pulp.value(prob.objective))

        # connect end points by optimization method
        merged_vertex_id, _ = torch.sort(vertex_id[triu_indices[:, solution > 0].transpose(0, 1)])   # ?*2
        merged_vertex_id = torch.unique(merged_vertex_id, dim=0)   # ?*2

        assert len(merged_vertex_id) == len(edge_ids)

        total_merge_vertex.append(merged_vertex_id)

    total_merge_vertex = torch.unique(torch.cat(total_merge_vertex, dim=0), dim=0)  # ?*2
    assert (total_merge_vertex[:, 0] - total_merge_vertex[:, 1]).abs().max() > 0
    vertex_pnt = edge_start_end.reshape(-1, 3)   # (2*ne, 3)
    edgeVertex_adj = torch.tensor([[2*i, 2*i+1] for i in range(len(edge_pnts))])    # ne*2

    merge_cluster = [set(total_merge_vertex[0].tolist())]
    for merge_pair in total_merge_vertex[1:]:
        merge_pair = set(merge_pair.tolist())
        intersection_idx = []
        for i, existing_cluster in enumerate(merge_cluster):
            if merge_pair & existing_cluster:
                intersection_idx.append(i)
        if intersection_idx:
            # Combine the sets at intersection_idx with merge_pair
            combined_set = merge_pair.union(*[merge_cluster[idx] for idx in intersection_idx])

            # Convert intersection_idx to a set for faster lookup
            intersection_set = set(intersection_idx)

            # Remove the sets at intersection_idx from merge_cluster
            merge_cluster = [s for idx, s in enumerate(merge_cluster) if idx not in intersection_set]

            # Add the combined set to merge_cluster
            merge_cluster.append(combined_set)
        else:
            merge_cluster.append(merge_pair)
    unique_vertex_pnt = []
    new_vertex_idx = torch.arange(vertex_pnt.shape[0])  # (2*ne, )
    for idx, cluster in enumerate(merge_cluster):
        unique_vertex_pnt.append(vertex_pnt[list(cluster)].mean(0))
        new_vertex_idx[list(cluster)] = idx

    edgeVertex_adj = new_vertex_idx[edgeVertex_adj]   # ne*2

    assert (edgeVertex_adj[:, 0] - edgeVertex_adj[:, 1]).abs().min() > 0

    return torch.stack(unique_vertex_pnt), edgeVertex_adj, faceEdge_adj


def compute_bbox_center_and_size(min_corner, max_corner):
    # Calculate the center
    center_x = (min_corner[0] + max_corner[0]) / 2
    center_y = (min_corner[1] + max_corner[1]) / 2
    center_z = (min_corner[2] + max_corner[2]) / 2
    center = np.array([center_x, center_y, center_z])
    # Calculate the size
    size_x = max_corner[0] - min_corner[0]
    size_y = max_corner[1] - min_corner[1]
    size_z = max_corner[2] - min_corner[2]
    size = max(size_x, size_y, size_z)
    return center, size


def get_bbox_minmax(point_cloud):
    # Find the minimum and maximum coordinates along each axis
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])

    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])

    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])

    # Create the 3D bounding box using the min and max values
    min_point = np.array([min_x, min_y, min_z])
    max_point = np.array([max_x, max_y, max_z])
    return (min_point, max_point)


def get_bbox_norm(point_cloud):
    # Find the minimum and maximum coordinates along each axis
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])

    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])

    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])

    # Create the 3D bounding box using the min and max values
    min_point = np.array([min_x, min_y, min_z])
    max_point = np.array([max_x, max_y, max_z])
    return np.linalg.norm(max_point - min_point)


class STModel(torch.nn.Module):
    def __init__(self, num_edge, num_surf):
        super().__init__()
        self.edge_t = torch.nn.Parameter(torch.zeros((num_edge, 3)))
        self.surf_st = torch.nn.Parameter(torch.FloatTensor([1,0,0,0]).unsqueeze(0).repeat(num_surf,1))


def joint_optimize(surf_ncs, edge_ncs, surfPos, unique_vertices, EdgeVertexAdj, FaceEdgeAdj, num_edge, num_surf):
    """
    Jointly optimize the face/edge/vertex based on topology
    """
    loss_func = ChamferDistance()

    model = STModel(num_edge, num_surf)
    model = model.cuda().train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
        eps=1e-08,
    )

    # Optimize edges (directly compute)
    edge_ncs_se = edge_ncs[:, [0, -1]]
    edge_vertex_se = unique_vertices[EdgeVertexAdj]

    edge_wcs = []
    print('Joint Optimization...')
    for wcs, ncs_se, vertex_se in zip(edge_ncs, edge_ncs_se, edge_vertex_se):
        # scale
        scale_target = np.linalg.norm(vertex_se[0] - vertex_se[1])
        scale_ncs = np.linalg.norm(ncs_se[0] - ncs_se[1])
        edge_scale = scale_target / scale_ncs

        edge_updated = wcs * edge_scale
        edge_se = ncs_se * edge_scale

        # offset
        offset = (vertex_se - edge_se)
        offset_rev = (vertex_se - edge_se[::-1])

        # swap start / end if necessary
        offset_error = np.abs(offset[0] - offset[1]).mean()
        offset_rev_error = np.abs(offset_rev[0] - offset_rev[1]).mean()
        if offset_rev_error < offset_error:
            edge_updated = edge_updated[::-1]
            offset = offset_rev

        edge_updated = edge_updated + offset.mean(0)[np.newaxis, np.newaxis, :]
        edge_wcs.append(edge_updated)

    edge_wcs = np.vstack(edge_wcs)

    # Replace start/end points with corner, and backprop change along curve
    for index in range(len(edge_wcs)):
        start_vec = edge_vertex_se[index, 0] - edge_wcs[index, 0]
        end_vec = edge_vertex_se[index, 1] - edge_wcs[index, -1]
        weight = np.tile((np.arange(32) / 31)[:, np.newaxis], (1, 3))
        weighted_vec = np.tile(start_vec[np.newaxis, :], (32, 1)) * (1 - weight) + np.tile(end_vec, (32, 1)) * weight
        edge_wcs[index] += weighted_vec

        # Optimize surfaces
    face_edges = []
    for adj in FaceEdgeAdj:
        all_pnts = edge_wcs[adj]
        face_edges.append(torch.FloatTensor(all_pnts).cuda())

    # Initialize surface in wcs based on surface pos
    surf_wcs_init = []
    bbox_threshold_min = []
    bbox_threshold_max = []
    for edges_perface, ncs, bbox in zip(face_edges, surf_ncs, surfPos):
        surf_center, surf_scale = compute_bbox_center_and_size(bbox[0:3], bbox[3:])
        edges_perface_flat = edges_perface.reshape(-1, 3).detach().cpu().numpy()
        min_point, max_point = get_bbox_minmax(edges_perface_flat)
        edge_center, edge_scale = compute_bbox_center_and_size(min_point, max_point)
        bbox_threshold_min.append(min_point)
        bbox_threshold_max.append(max_point)

        # increase surface size if does not fully cover the wire bbox
        if surf_scale < edge_scale:
            surf_scale = 1.05 * edge_scale

        wcs = ncs * (surf_scale / 2) + surf_center
        surf_wcs_init.append(wcs)

    surf_wcs_init = np.stack(surf_wcs_init)

    # optimize the surface offset
    surf = torch.FloatTensor(surf_wcs_init).cuda()
    for iters in range(200):
        surf_scale = model.surf_st[:, 0].reshape(-1, 1, 1, 1)
        surf_offset = model.surf_st[:, 1:].reshape(-1, 1, 1, 3)
        surf_updated = surf + surf_offset

        surf_loss = 0
        for surf_pnt, edge_pnts in zip(surf_updated, face_edges):
            surf_pnt = surf_pnt.reshape(-1, 3)
            edge_pnts = edge_pnts.reshape(-1, 3).detach()
            surf_loss += loss_func(surf_pnt.unsqueeze(0), edge_pnts.unsqueeze(0), bidirectional=False, reverse=True)
        surf_loss /= len(surf_updated)

        optimizer.zero_grad()
        (surf_loss).backward()
        optimizer.step()

        # print(f'Iter {iters} surf:{surf_loss:.5f}')

    surf_wcs = surf_updated.detach().cpu().numpy()

    return (surf_wcs, edge_wcs)


def fix_wires(face, debug=False):
    top_exp = TopologyExplorer(face)
    for wire in top_exp.wires():
        if debug:
            wire_checker = ShapeAnalysis_Wire(wire, face, 0.01)
            print(f"Check order 3d {wire_checker.CheckOrder()}")
            print(f"Check 3d gaps {wire_checker.CheckGaps3d()}")
            print(f"Check closed {wire_checker.CheckClosed()}")
            print(f"Check connected {wire_checker.CheckConnected()}")
        wire_fixer = ShapeFix_Wire(wire, face, 0.01)

        # wire_fixer.SetClosedWireMode(True)
        # wire_fixer.SetFixConnectedMode(True)
        # wire_fixer.SetFixSeamMode(True)

        assert wire_fixer.IsReady()
        ok = wire_fixer.Perform()
        # assert ok


def fix_face(face):
    fixer = ShapeFix_Face(face)
    fixer.SetPrecision(0.01)
    fixer.SetMaxTolerance(0.1)
    ok = fixer.Perform()
    # assert ok
    fixer.FixOrientation()
    face = fixer.Face()
    return face


def add_pcurves_to_edges(face):
    edge_fixer = ShapeFix_Edge()
    top_exp = TopologyExplorer(face)
    for wire in top_exp.wires():
        wire_exp = WireExplorer(wire)
        for edge in wire_exp.ordered_edges():
            edge_fixer.FixAddPCurve(edge, face, False, 0.001)


def construct_brep(surf_wcs, edge_wcs, FaceEdgeAdj, EdgeVertexAdj):
    """
    Fit parametric surfaces / curves and trim into B-rep
    """
    print('Building the B-rep...')
    # Fit surface bspline
    recon_faces = []
    for points in surf_wcs:
        num_u_points, num_v_points = 32, 32
        uv_points_array = TColgp_Array2OfPnt(1, num_u_points, 1, num_v_points)
        for u_index in range(1, num_u_points + 1):
            for v_index in range(1, num_v_points + 1):
                pt = points[u_index - 1, v_index - 1]
                point_3d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
                uv_points_array.SetValue(u_index, v_index, point_3d)
        approx_face = GeomAPI_PointsToBSplineSurface(uv_points_array, 3, 8, GeomAbs_C2, 5e-2).Surface()
        recon_faces.append(approx_face)

    recon_edges = []
    for points in edge_wcs:
        num_u_points = 32
        u_points_array = TColgp_Array1OfPnt(1, num_u_points)
        for u_index in range(1, num_u_points + 1):
            pt = points[u_index - 1]
            point_2d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
            u_points_array.SetValue(u_index, point_2d)
        try:
            approx_edge = GeomAPI_PointsToBSpline(u_points_array, 0, 8, GeomAbs_C2, 5e-3).Curve()
        except Exception as e:
            print('high precision failed, trying mid precision...')
            try:
                approx_edge = GeomAPI_PointsToBSpline(u_points_array, 0, 8, GeomAbs_C2, 8e-3).Curve()
            except Exception as e:
                print('mid precision failed, trying low precision...')
                approx_edge = GeomAPI_PointsToBSpline(u_points_array, 0, 8, GeomAbs_C2, 5e-2).Curve()
        recon_edges.append(approx_edge)

    # Create edges from the curve list
    edge_list = []
    for curve in recon_edges:
        edge = BRepBuilderAPI_MakeEdge(curve).Edge()
        edge_list.append(edge)

    # Cut surface by wire
    post_faces = []
    post_edges = []
    for idx, (surface, edge_incides) in enumerate(zip(recon_faces, FaceEdgeAdj)):

        """Jing Test 2024/06/16"""
        # print("number of edges:", len(edge_incides))
        # from OCC.Display.SimpleGui import init_display
        # display, start_display, add_menu, add_function_to_menu = init_display()
        # display.DisplayShape(surface, update=True, color='BLUE')  # 以蓝色显示面
        # for i in edge_incides:
        #     display.DisplayShape(edge_list[i], update=True, color='RED')  # 以红色显示边
        # start_display()
        """*******************"""

        corner_indices = EdgeVertexAdj[edge_incides]

        # ordered loop
        loops = []
        ordered = [0]
        seen_corners = [corner_indices[0, 0], corner_indices[0, 1]]
        next_index = corner_indices[0, 1]

        while len(ordered) < len(corner_indices):
            while True:
                next_row = [idx for idx, edge in enumerate(corner_indices) if next_index in edge and idx not in ordered]
                if len(next_row) == 0:
                    break
                ordered += next_row
                next_index = list(set(corner_indices[next_row][0]) - set(seen_corners))
                if len(next_index) == 0:
                    break
                else:
                    next_index = next_index[0]
                seen_corners += [corner_indices[next_row][0][0], corner_indices[next_row][0][1]]

            cur_len = int(np.array([len(x) for x in loops]).sum())  # add to inner / outer loops
            loops.append(ordered[cur_len:])

            # Swith to next loop
            next_corner = list(set(np.arange(len(corner_indices))) - set(ordered))
            if len(next_corner) == 0:
                break
            else:
                next_corner = next_corner[0]
            next_index = corner_indices[next_corner][0]
            ordered += [next_corner]
            seen_corners += [corner_indices[next_corner][0], corner_indices[next_corner][1]]
            next_index = corner_indices[next_corner][1]

        # Determine the outer loop by bounding box length (?)
        bbox_spans = [get_bbox_norm(edge_wcs[x].reshape(-1, 3)) for x in loops]

        # Create wire from ordered edges
        _edge_incides_ = [edge_incides[x] for x in ordered]
        edge_post = [edge_list[x] for x in _edge_incides_]
        post_edges += edge_post

        out_idx = np.argmax(np.array(bbox_spans))
        inner_idx = list(set(np.arange(len(loops))) - set([out_idx]))

        # Outer wire
        wire_builder = BRepBuilderAPI_MakeWire()
        for edge_idx in loops[out_idx]:
            wire_builder.Add(edge_list[edge_incides[edge_idx]])
        outer_wire = wire_builder.Wire()

        # Inner wires
        inner_wires = []
        for idx in inner_idx:
            wire_builder = BRepBuilderAPI_MakeWire()
            for edge_idx in loops[idx]:
                wire_builder.Add(edge_list[edge_incides[edge_idx]])
            inner_wires.append(wire_builder.Wire())

        # Cut by wires
        face_builder = BRepBuilderAPI_MakeFace(surface, outer_wire)
        for wire in inner_wires:
            face_builder.Add(wire)
        face_occ = face_builder.Shape()
        fix_wires(face_occ)
        add_pcurves_to_edges(face_occ)
        fix_wires(face_occ)
        face_occ = fix_face(face_occ)
        post_faces.append(face_occ)

        # """Jing Test 2024/06/16"""
        # from OCC.Core.TopExp import TopExp_Explorer
        # from OCC.Core.TopAbs import TopAbs_EDGE
        # display, start_display, add_menu, add_function_to_menu = init_display()
        # display.DisplayShape(face_occ, color='BLUE', update=True)
        # # 创建一个explorer来遍历面上的边缘(edges)
        # edge_explorer = TopExp_Explorer(face_occ, TopAbs_EDGE)
        # # 遍历并显示每一边
        # ii = 0
        # while edge_explorer.More():
        #     edge = edge_explorer.Current()
        #     display.DisplayShape(edge, color='RED', update=True)
        #     edge_explorer.Next()
        #     ii += 1
        # # 开始交互式事件循环
        # print("number of edges:", ii)
        # start_display()

    # Sew faces into solid
    sewing = BRepBuilderAPI_Sewing()
    for face in post_faces:
        sewing.Add(face)

    # Perform the sewing operation
    sewing.Perform()
    sewn_shell = sewing.SewedShape()

    # Make a solid from the shell
    maker = BRepBuilderAPI_MakeSolid()
    maker.Add(sewn_shell)
    maker.Build()
    solid = maker.Solid()

    return solid


def main():
    with open('data_process/furniture_data_split_6bit.pkl', 'rb') as tf:
        files = pickle.load(tf)['train']

    # convert_iter = Pool(os.cpu_count()).imap(check_loop_edge, files)
    # valid = 0
    # for status in tqdm(convert_iter, total=len(files)):
    #     valid += status
    #
    # print(valid)

    for file in files:
        check_loop_edge(file)


if __name__ == '__main__':
    main()
