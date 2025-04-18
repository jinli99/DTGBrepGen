import numpy as np
import torch
import time
import sys
import os
import gmsh
import multiprocessing
from multiprocessing.pool import Pool
from tqdm import tqdm
from chamferdist import ChamferDistance
from contextlib import contextmanager
from typing import Optional
from concurrent.futures import ProcessPoolExecutor
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface, GeomAPI_PointsToBSpline, GeomAPI_ProjectPointOnSurf
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Pln, gp_Ax3, gp_Vec
from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Wire, ShapeFix_Edge
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.Geom import (Geom_BSplineSurface, Geom_BSplineCurve, Geom_Plane,
                           Geom_SphericalSurface, Geom_CylindricalSurface, Geom_ConicalSurface)
from OCC.Core.GeomConvert import GeomConvert_ApproxSurface
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_UniformAbscissa
from utils import load_data_with_prefix
from inference.primitive_fitting import process_one_surface
from topology.transfer import face_edge_trans
from OCC.Core.TopoDS import topods_Shell, topods_Wire, topods_Face, topods_Edge, TopoDS_Solid
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire, ShapeAnalysis_Shell, ShapeAnalysis_FreeBounds
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_SHELL, TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from occwl.io import load_step


def check_brep_validity(step_file_path):

    if isinstance(step_file_path, str):
        # Read the STEP file
        step_reader = STEPControl_Reader()
        status = step_reader.ReadFile(step_file_path)

        if status != IFSelect_RetDone:
            print("Error: Unable to read STEP file")
            return False

        step_reader.TransferRoot()
        shape = step_reader.Shape()

    elif isinstance(step_file_path, TopoDS_Solid):
        shape = step_file_path

    else:
        return False

    # Initialize check results
    wire_order_ok = True
    wire_self_intersection_ok = True
    shell_bad_edges_ok = True
    brep_closed_ok = True  # Initialize closed BRep check
    solid_one_ok = True

    # 1. Check if BRep has more than one solid
    if isinstance(step_file_path, str):
        try:
            cad_solid = load_step(step_file_path)
            if len(cad_solid) != 1:
                solid_one_ok = False
        except Exception as e:
            return False

    # 2. Check all wires
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = topods_Face(face_explorer.Current())
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_explorer.More():
            wire = topods_Wire(wire_explorer.Current())

            # Create a ShapeFix_Wire object
            wire_fixer = ShapeFix_Wire(wire, face, 0.01)
            wire_fixer.Load(wire)
            wire_fixer.SetFace(face)
            wire_fixer.SetPrecision(0.01)
            wire_fixer.SetMaxTolerance(1)
            wire_fixer.SetMinTolerance(0.0001)

            # Fix the wire
            wire_fixer.Perform()
            fixed_wire = wire_fixer.Wire()

            # Analyze the fixed wire
            wire_analysis = ShapeAnalysis_Wire(fixed_wire, face, 0.01)
            wire_analysis.Load(fixed_wire)
            wire_analysis.SetPrecision(0.01)
            wire_analysis.SetSurface(BRep_Tool.Surface(face))

            # 1. Check wire edge order
            order_status = wire_analysis.CheckOrder()
            if order_status != 0:  # 0 means no error
                # print(f"Wire order issue detected: {order_status}")
                wire_order_ok = False

            # 2. Check wire self-intersection
            if wire_analysis.CheckSelfIntersection():
                wire_self_intersection_ok = False

            wire_explorer.Next()
        face_explorer.Next()

    # 3. Check for bad edges in shells
    shell_explorer = TopExp_Explorer(shape, TopAbs_SHELL)
    while shell_explorer.More():
        shell = topods_Shell(shell_explorer.Current())
        shell_analysis = ShapeAnalysis_Shell()
        shell_analysis.LoadShells(shell)

        if shell_analysis.HasBadEdges():
            shell_bad_edges_ok = False

        shell_explorer.Next()

    # 4. Check if BRep is closed (no free edges)
    free_bounds = ShapeAnalysis_FreeBounds(shape)
    free_edges = free_bounds.GetOpenWires()
    edge_explorer = TopExp_Explorer(free_edges, TopAbs_EDGE)
    num_free_edges = 0
    while edge_explorer.More():
        edge = topods_Edge(edge_explorer.Current())
        num_free_edges += 1
        # print(f"Free edge: {edge}")
        edge_explorer.Next()
    if num_free_edges > 0:
        brep_closed_ok = False

    return int(wire_order_ok and wire_self_intersection_ok and shell_bad_edges_ok and brep_closed_ok and solid_one_ok)


class Brep2Mesh:

    METHOD_GMSH = 'gmsh'
    METHOD_OCC = 'occ'

    def __init__(self, input_path: str, save_path: Optional[str] = None,
                 method: str = METHOD_OCC, timeout: int = 4, deflection: float = 0.001):

        """Initialize the Brep2Mesh converter.

        Args:
            input_path (str): Path to the input step files.
            save_path (str, optional): Path to save the output mesh files. Defaults to input_path.
            method (str): Mesh generation method ('gmsh' or 'occ'). Defaults to 'gmsh'.
            timeout (int): Timeout for mesh generation using gmsh method. Defaults to 2.
            deflection (float): Deflection parameter for mesh generation. Defaults to 0.001.
        """

        self.files = load_data_with_prefix(input_path, '.step')
        self.save_path = save_path or input_path
        os.makedirs(self.save_path, exist_ok=True)

        if method not in [self.METHOD_GMSH, self.METHOD_OCC]:
            raise ValueError(f"Invalid method: {method}. Choose either 'gmsh' or 'occ'.")
        self.method: str = method

        self.timeout: int = timeout
        self.deflection: float = deflection

    def occMesh(self, file):
        step_reader = STEPControl_Reader()
        status = step_reader.ReadFile(file)

        if status != IFSelect_RetDone:
            raise ValueError("Error: Cannot read the STEP file")

        step_reader.TransferRoot()
        shape = step_reader.Shape()

        mesh = BRepMesh_IncrementalMesh(shape, self.deflection)
        mesh.Perform()

        stl_writer = StlAPI_Writer()
        stl_writer.Write(shape, os.path.join(self.save_path, os.path.splitext(os.path.basename(file))[0] + '.stl'))

    @contextmanager
    def suppress_stdout_stderr(self):
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    def gMesh_single(self, file_path):
        try:
            with self.suppress_stdout_stderr():
                gmsh.initialize()
                gmsh.option.setNumber("General.Terminal", 0)
                gmsh.open(file_path)

                gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay for 2D meshes
                gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay for 3D meshes
                gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
                gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
                gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
                gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 20)
                gmsh.option.setNumber("Mesh.MeshSizeFactor", 1)

                # gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1)
                # gmsh.option.setNumber("Mesh.MeshSizeMax", 10)

                gmsh.model.mesh.generate(3)
                output_file = os.path.join(self.save_path, os.path.splitext(os.path.basename(file_path))[0] + '.stl')
                gmsh.write(output_file)
        except Exception as e:
            return str(e)
        finally:
            gmsh.finalize()

    def gMesh(self, file_path):
        def target_func(q, path):
            q.put(self.gMesh_single(path))
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=target_func, args=(result_queue, file_path))
        process.start()

        start_time = time.time()
        while process.is_alive():
            if time.time() - start_time > self.timeout:
                # print(f"Mesh generation for {file_path} timed out after {timeout} seconds")
                process.terminate()
                process.join()
                return False
            time.sleep(0.1)

        if not result_queue.empty():
            result = result_queue.get()
            if result is None:
                # print(f"Successfully generated mesh for {file_path}")
                return True
            else:
                # print(f"Mesh generation for {file_path} failed: {result}")
                return False
        else:
            # print(f"Mesh generation for {file_path} failed with unknown error")
            return False

    def generate(self, parallel=True):
        if self.method == self.METHOD_GMSH:
            for file in tqdm(self.files):
                if not self.gMesh(file):
                    self.occMesh(file)
        elif self.method == self.METHOD_OCC:
            if parallel:
                convert_iter = Pool(os.cpu_count()).imap(self.occMesh, self.files)
                for _ in tqdm(convert_iter, total=len(self.files)):
                    pass
            else:
                for file in tqdm(self.files):
                    self.occMesh(file)
        else:
            raise ValueError(f"Unsupported method: {self.method}")


def create_bspline_surface(ctrs):

    assert ctrs.shape[0] == 16

    poles = TColgp_Array2OfPnt(1, 4, 1, 4)
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            poles.SetValue(i + 1, j + 1, gp_Pnt(*ctrs[idx]))

    u_knots = TColStd_Array1OfReal(1, 2)
    v_knots = TColStd_Array1OfReal(1, 2)

    u_knots.SetValue(1, 0.0)
    u_knots.SetValue(2, 1.0)
    v_knots.SetValue(1, 0.0)
    v_knots.SetValue(2, 1.0)

    u_mults = TColStd_Array1OfInteger(1, 2)
    v_mults = TColStd_Array1OfInteger(1, 2)

    u_mults.SetValue(1, 4)
    u_mults.SetValue(2, 4)
    v_mults.SetValue(1, 4)
    v_mults.SetValue(2, 4)

    bspline_surface = Geom_BSplineSurface(poles, u_knots, v_knots, u_mults, v_mults, 3, 3)

    return bspline_surface


def create_bspline_curve(ctrs):

    assert ctrs.shape[0] == 4

    poles = TColgp_Array1OfPnt(1, 4)
    for i, ctr in enumerate(ctrs, 1):
        poles.SetValue(i, gp_Pnt(*ctr))

    n_knots = 2
    knots = TColStd_Array1OfReal(1, n_knots)
    knots.SetValue(1, 0.0)
    knots.SetValue(2, 1.0)

    mults = TColStd_Array1OfInteger(1, n_knots)
    mults.SetValue(1, 4)
    mults.SetValue(2, 4)

    bspline_curve = Geom_BSplineCurve(poles, knots, mults, 3)

    return bspline_curve


def sample_bspline_surface(bspline_surface, num_u=32, num_v=32):
    u_start, u_end, v_start, v_end = bspline_surface.Bounds()
    u_range = np.linspace(u_start, u_end, num_u)
    v_range = np.linspace(v_start, v_end, num_v)

    points = np.zeros((num_u, num_v, 3))

    for i, u in enumerate(u_range):
        for j, v in enumerate(v_range):
            pnt = bspline_surface.Value(u, v)
            points[i, j] = [pnt.X(), pnt.Y(), pnt.Z()]

    return points      # 32*32*3


def sample_bspline_curve(bspline_curve, num_points=32):
    u_start, u_end = bspline_curve.FirstParameter(), bspline_curve.LastParameter()
    u_range = np.linspace(u_start, u_end, num_points)

    points = np.zeros((num_points, 3))

    for i, u in enumerate(u_range):
        pnt = bspline_curve.Value(u)
        points[i] = [pnt.X(), pnt.Y(), pnt.Z()]

    return points    # 32*3


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


def joint_optimize(edge_ncs, face_ncs, surfPos, unique_vertices, EdgeVertexAdj, FaceEdgeAdj, num_edge, num_surf):
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
    for edges_perface, ncs, bbox in zip(face_edges, face_ncs, surfPos):
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

    return edge_wcs, surf_wcs


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


def project_point_on_surface(point, surface):
    """
    Project a point onto a surface.

    Args:
        point: A 3D point (x, y, z).
        surface: OCC.Core.Geom surface object.

    Returns:
        A projected 3D point (x, y, z) on the surface.
    """
    # Convert the point to OCC gp_Pnt
    pnt = gp_Pnt(point[0], point[1], point[2])

    # Project the point onto the surface
    projector = GeomAPI_ProjectPointOnSurf(pnt, surface)
    projector.Perform(pnt)

    # Get the projected point
    if projector.IsDone():
        projected_pnt = projector.NearestPoint()
        return np.array([projected_pnt.X(), projected_pnt.Y(), projected_pnt.Z()])
    else:
        return point


def fit_basic_surface(outer_points, inner_points):
    """
    Args:
        outer_points: ne*32*3
        inner_points: 32*32*3
    """
    # outer_points, inner_points = args

    indices = np.linspace(0, outer_points.shape[1] - 1, 16, dtype=int)
    outer_points = np.array([outer_points[i][indices] for i in range(outer_points.shape[0])])
    outer_points = outer_points.reshape(-1, 3)
    inner_points = inner_points[::8, ::8, :].reshape(-1, 3)

    out = process_one_surface(np.concatenate([inner_points, outer_points]),
                              device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                              weights=np.array([0.5] * inner_points.shape[0] + [1.0] * outer_points.shape[0]).reshape(
                                  -1, 1))

    return out


def fit_surface(face_id, edge_wcs, face_wcs, FaceEdgeAdj):
    out = fit_basic_surface(outer_points=edge_wcs[FaceEdgeAdj[face_id]], inner_points=face_wcs[face_id])
    return out


def construct_brep(face_wcs, edge_wcs, FaceEdgeAdj, EdgeVertexAdj):
    """
    Fit parametric surfaces / curves and trim into B-rep
    """
    print('Building the B-rep...')

    # Fit surface
    recon_faces = []

    with ProcessPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(fit_surface, range(len(FaceEdgeAdj)), [edge_wcs]*len(FaceEdgeAdj), [face_wcs]*len(FaceEdgeAdj), [FaceEdgeAdj]*len(FaceEdgeAdj)))

    for idx, out in enumerate(results):

        if out['err'] < 0.01:

            if out['type'] == 'plane':
                direction, distance = out['params']
                dire = gp_Dir(direction[0], direction[1], direction[2])
                point = gp_Pnt(direction[0] * distance, direction[1] * distance, direction[2] * distance)
                plane = gp_Pln(point, dire)
                approx_face = Geom_Plane(plane)
            elif out['type'] == 'sphere':
                center, radius = out['params']
                center_point = gp_Pnt(center[0], center[1], center[2])
                axis = gp_Ax3(center_point, gp_Dir(0, 0, 1))
                approx_face = Geom_SphericalSurface(axis, radius)
            elif out['type'] == 'cylinder':
                direction, pnt, radius = out['params']
                dire = gp_Dir(direction[0], direction[1], direction[2])
                origin = gp_Pnt(pnt[0], pnt[1], pnt[2])
                axis = gp_Ax3(origin, dire)
                approx_face = Geom_CylindricalSurface(axis, radius)
            else:
                assert out['type'] == 'cone'
                apex, axis, theta = out['params']
                apex_point = gp_Pnt(apex[0], apex[1], apex[2])
                axis_dir = gp_Dir(axis[0], axis[1], axis[2])
                ax3 = gp_Ax3(apex_point, axis_dir)
                approx_face = Geom_ConicalSurface(ax3, theta, 1)

        else:
            points = face_wcs[idx]
            num_u_points, num_v_points = points.shape[0], points.shape[1]
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
        num_u_points = points.shape[0]
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

        """Test 2024/06/16"""
        # print("number of edges:", len(edge_incides))
        # from OCC.Display.SimpleGui import init_display
        # display, start_display, add_menu, add_function_to_menu = init_display()
        # display.DisplayShape(surface, update=True, color='BLUE')  # 以蓝色显示面
        # for i in edge_incides:
        #     display.DisplayShape(edge_list[i], update=True, color='RED')  # 以红色显示边
        # start_display()
        # """*******************"""

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
        for idx_ in inner_idx:
            wire_builder = BRepBuilderAPI_MakeWire()
            for edge_idx in loops[idx_]:
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

        # """Test 2024/09/21"""
        # analyzer = BRepCheck_Analyzer(face_occ)
        # if not analyzer.IsValid():
        #     fixer = ShapeFix_Shape(face_occ)
        #     fixer.Perform()
        #     print(idx, type(fixer.Shape()))
        #     if isinstance(face_occ, TopoDS_Face):
        #         face_occ = fixer.Shape()
        #         explorer = TopExp_Explorer(face_occ, TopAbs_FACE)
        #         while explorer.More():
        #             post_faces.append(topods_Face(explorer.Current()))
        #             explorer.Next()
        #         continue

        post_faces.append(face_occ)

        # """Test 2024/06/16"""
        # from OCC.Core.TopExp import TopExp_Explorer
        # from OCC.Core.TopAbs import TopAbs_EDGE
        # from OCC.Display.SimpleGui import init_display
        # display, start_display, add_menu, add_function_to_menu = init_display()
        # display.DisplayShape(face_occ, color='BLUE', update=True)
        # edge_explorer = TopExp_Explorer(face_occ, TopAbs_EDGE)
        # ii = 0
        # while edge_explorer.More():
        #     edge = edge_explorer.Current()
        #     display.DisplayShape(edge, color='RED', update=True)
        #     edge_explorer.Next()
        #     ii += 1
        # print("number of edges:", ii)
        # start_display()
        # print(111)

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

    if not check_brep_validity(solid):
        return None

    return solid


def main():
    pass


if __name__ == '__main__':
    main()
