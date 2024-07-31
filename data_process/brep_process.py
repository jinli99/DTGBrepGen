import os
import pickle
import argparse
from tqdm import tqdm
from multiprocessing.pool import Pool
from occwl.io import load_step
import json
import numpy as np
from occwl.uvgrid import ugrid, uvgrid
from occwl.compound import Compound
from occwl.solid import Solid
from occwl.shell import Shell
from occwl.entity_mapper import EntityMapper

# To speed up processing, define maximum threshold
MAX_FACE = 70


def normalize(face_pnts, edge_pnts, corner_pnts):
    """
    Various levels of normalization
    """
    # Global normalization to -1~1
    total_points = np.array(face_pnts).reshape(-1, 3)
    min_vals = np.min(total_points, axis=0)
    max_vals = np.max(total_points, axis=0)
    global_offset = min_vals + (max_vals - min_vals) / 2
    global_scale = max(max_vals - min_vals)
    assert global_scale != 0, 'scale is zero'

    faces_wcs, edges_wcs, faces_ncs, edges_ncs = [], [], [], []

    # Normalize corner
    corner_wcs = (corner_pnts - global_offset[np.newaxis, :]) / (global_scale * 0.5)

    # Normalize surface
    for face_pnt in face_pnts:
        # Normalize CAD to WCS
        face_pnt_wcs = (face_pnt - global_offset[np.newaxis, np.newaxis, :]) / (global_scale * 0.5)
        faces_wcs.append(face_pnt_wcs)
        # Normalize Surface to NCS
        min_vals = np.min(face_pnt_wcs.reshape(-1, 3), axis=0)
        max_vals = np.max(face_pnt_wcs.reshape(-1, 3), axis=0)
        local_offset = min_vals + (max_vals - min_vals) / 2
        local_scale = max(max_vals - min_vals)
        pnt_ncs = (face_pnt_wcs - local_offset[np.newaxis, np.newaxis, :]) / (local_scale * 0.5)
        faces_ncs.append(pnt_ncs)

    # Normalize edge
    for edge_pnt in edge_pnts:
        # Normalize CAD to WCS
        edge_pnt_wcs = (edge_pnt - global_offset[np.newaxis, :]) / (global_scale * 0.5)
        edges_wcs.append(edge_pnt_wcs)
        # Normalize Edge to NCS
        min_vals = np.min(edge_pnt_wcs.reshape(-1, 3), axis=0)
        max_vals = np.max(edge_pnt_wcs.reshape(-1, 3), axis=0)
        local_offset = min_vals + (max_vals - min_vals) / 2
        local_scale = max(max_vals - min_vals)
        pnt_ncs = (edge_pnt_wcs - local_offset) / (local_scale * 0.5)
        edges_ncs.append(pnt_ncs)
        assert local_scale != 0, 'scale is zero'

    faces_wcs = np.stack(faces_wcs)
    faces_ncs = np.stack(faces_ncs)
    edges_wcs = np.stack(edges_wcs)
    edges_ncs = np.stack(edges_ncs)

    return faces_wcs, edges_wcs, faces_ncs, edges_ncs, corner_wcs


def get_bbox(point_cloud):
    """
    Get the tighest fitting 3D bounding box giving a set of points (axis-aligned)
    """
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
    return min_point, max_point


def load_abc_step(root_dir, use_deepcad):
    """
    Recursively searches through a given parent directory and its subdirectories
    to find the paths of all ABC STEP files.

    Args:
    - root_dir (str): Path to the root directory where the search begins.
    - use_deepcad (bool): Process deepcad or not

    Returns:
    - step_dirs [str]: A list containing the paths to all STEP parent directory
    """
    # Load DeepCAD UID
    if use_deepcad:
        with open('train_val_test_split.json', 'r') as json_file:
            deepcad_data = json.load(json_file)
        deepcad_data = deepcad_data['train'] + deepcad_data['validation'] + deepcad_data['test']
        deepcad_uid = set([uid.split('/')[1] for uid in deepcad_data])

    # Create STEP file folder path (based on the default ABC STEP format)
    dirs_nested = [[f'{root_dir}/abc_{str(i).zfill(4)}_step_v00'] * 10000 for i in range(100)]
    dirs = [item for sublist in dirs_nested for item in sublist]
    subdirs = [f'{str(i).zfill(8)}' for i in range(1000000)]

    if use_deepcad:
        step_dirs_ = [root + '/' + sub for root, sub in zip(dirs, subdirs) if sub in deepcad_uid]
    else:
        step_dirs_ = [root + '/' + sub for root, sub in zip(dirs, subdirs)]

    return step_dirs_


def load_furniture_step(root_dir):
    """
    Recursively searches through a given parent directory and its subdirectories
    to find the paths of all Furniture STEP files.

    Args:
    - root_dir (str): Path to the root directory where the search begins.

    Returns:
    - data_files [str]: A list containing the paths to all STEP parent directory
    """
    data_files = []
    # Walk through the directory tree starting from the root folder
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            # Check if the file ends with the specified prefix
            if filename.endswith('.step'):
                file_path = os.path.join(root, filename)
                data_files.append(file_path)
    return data_files


def extract_primitive(solid):
    """
    Extract all primitive information from splitted solid

    Args:
    - solid (occwl.Solid): A single b-rep solid in occwl format

    Returns:
    - face_pnts (N x 32 x 32 x 3): Sampled uv-grid points on the bounded surface region (face)
    - edge_pnts (M x 32 x 3): Sampled u-grid points on the boundged curve region (edge)
    - edge_corner_pnts (M x 2 x 3): Start & end vertices per edge
    - edgeFace_IncM (M x 2): Edge-Face incident matrix, every edge is connect to two face IDs
    - faceEdge_IncM: A list of N sublist, where each sublist represents the adjacent edge IDs to a face
    """
    assert isinstance(solid, Solid)

    # Retrieve face, edge geometry and face-edge adjacency
    face_dict, edge_dict, edgeFace_IncM = face_edge_adj(solid)

    # Skip unused index key, and update the adj
    face_dict, face_map = update_mapping(face_dict)
    edge_dict, edge_map = update_mapping(edge_dict)
    edgeFace_IncM_update = {}
    for key, value in edgeFace_IncM.items():
        new_face_indices = [face_map[x] for x in value]
        edgeFace_IncM_update[edge_map[key]] = new_face_indices
    edgeFace_IncM = edgeFace_IncM_update

    # Face-edge adj
    num_faces = len(face_dict)
    edgeFace_IncM = np.stack([x for x in edgeFace_IncM.values()])
    faceEdge_IncM = []
    for face_idx in range(num_faces):
        face_edges, _ = np.where(edgeFace_IncM == face_idx)
        faceEdge_IncM.append(face_edges)

    # Sample uv-grid from surface (32x32)
    graph_face_feat = {}
    for face_idx, face_feature in face_dict.items():
        _, face = face_feature
        points = uvgrid(
            face, method="point", num_u=32, num_v=32
        )
        visibility_status = uvgrid(
            face, method="visibility_status", num_u=32, num_v=32
        )
        mask = np.logical_or(visibility_status == 0, visibility_status == 2)  # 0: Inside, 1: Outside, 2: On boundary
        # Concatenate channel-wise to form face feature tensor
        face_feat = np.concatenate((points, mask), axis=-1)
        graph_face_feat[face_idx] = face_feat
    face_pnts = np.stack([x for x in graph_face_feat.values()])[:, :, :, :3]

    # sample u-grid from curve (1x32)
    graph_edge_feat = {}
    graph_corner_feat = {}
    for edge_idx, edge in edge_dict.items():
        points = ugrid(edge, method="point", num_u=32)
        graph_edge_feat[edge_idx] = points
        #### edge corners as start/end vertex ###
        v_start = points[0]
        v_end = points[-1]
        graph_corner_feat[edge_idx] = (v_start, v_end)
    edge_pnts = np.stack([x for x in graph_edge_feat.values()])
    edge_corner_pnts = np.stack([x for x in graph_corner_feat.values()])

    return [face_pnts, edge_pnts, edge_corner_pnts, edgeFace_IncM, faceEdge_IncM]


def update_mapping(data_dict):
    """
    Remove unused key index from data dictionary.
    """
    dict_new = {}
    mapping = {}
    max_idx = max(data_dict.keys())
    skipped_indices = np.array(sorted(list(set(np.arange(max_idx)) - set(data_dict.keys()))))
    for idx, value in data_dict.items():
        skips = (skipped_indices < idx).sum()
        idx_new = idx - skips
        dict_new[idx_new] = value
        mapping[idx] = idx_new
    return dict_new, mapping


def face_edge_adj(shape):
    """
    *** COPY AND MODIFIED FROM THE ORIGINAL OCCWL SOURCE CODE ***
    Extract face/edge geometry and create a face-edge adjacency
    graph from the given shape (Solid or Compound)

    Args:
    - shape (Shell, Solid, or Compound): Shape

    Returns:
    - face_dict: Dictionary of occwl faces, with face ID as the key
    - edge_dict: Dictionary of occwl edges, with edge ID as the key
    - edgeFace_IncM: Edge ID as the key, Adjacent faces ID as the value
    """
    assert isinstance(shape, (Shell, Solid, Compound))
    mapper = EntityMapper(shape)

    ### Faces ###
    face_dict = {}
    for face in shape.faces():
        face_idx = mapper.face_index(face)
        face_dict[face_idx] = (face.surface_type(), face)

    ### Edges and IncidenceMat ###
    edgeFace_IncM = {}
    edge_dict = {}
    for edge in shape.edges():
        if not edge.has_curve():
            continue

        connected_faces = list(shape.faces_from_edge(edge))
        if len(connected_faces) == 2 and not edge.seam(connected_faces[0]) and not edge.seam(connected_faces[1]):
            left_face, right_face = edge.find_left_and_right_faces(connected_faces)
            if left_face is None or right_face is None:
                continue
            edge_idx = mapper.edge_index(edge)
            edge_dict[edge_idx] = edge
            left_index = mapper.face_index(left_face)
            right_index = mapper.face_index(right_face)

            if edge_idx in edgeFace_IncM:
                edgeFace_IncM[edge_idx] += [left_index, right_index]
            else:
                edgeFace_IncM[edge_idx] = [left_index, right_index]
        else:
            pass  # ignore seam

    return face_dict, edge_dict, edgeFace_IncM


def parse_solid(solid):
    """
    Parse the surface, curve, face, edge, vertex in a CAD solid.

    Args:
    - solid (occwl.solid): A single brep solid in occwl data format.

    Returns:
    - data: A dictionary containing all parsed data
    """
    assert isinstance(solid, Solid)

    # Split closed surface and closed curve to halve
    solid = solid.split_all_closed_faces(num_splits=0)
    solid = solid.split_all_closed_edges(num_splits=0)

    if len(list(solid.faces())) > MAX_FACE:
        return None

    # Extract all B-rep primitives and their adjacency information
    face_pnts, edge_pnts, edge_corner_pnts, edgeFace_IncM, faceEdge_IncM = extract_primitive(solid)

    # Normalize the CAD model
    faces_wcs, edges_wcs, faces_ncs, edges_ncs, corner_wcs = normalize(face_pnts, edge_pnts, edge_corner_pnts)

    # Remove duplicate and merge corners
    corner_wcs = np.round(corner_wcs, 4)
    corner_unique = []
    for corner_pnt in corner_wcs.reshape(-1, 3):
        if len(corner_unique) == 0:
            corner_unique = corner_pnt.reshape(1, 3)
        else:
            # Check if it exists or not
            exists = np.any(np.all(corner_unique == corner_pnt, axis=1))
            if exists:
                continue
            else:
                corner_unique = np.concatenate([corner_unique, corner_pnt.reshape(1, 3)], 0)

    # Edge-corner adjacency
    edgeCorner_IncM = []
    for edge_corner in corner_wcs:
        start_corner_idx = np.where((corner_unique == edge_corner[0]).all(axis=1))[0].item()
        end_corner_idx = np.where((corner_unique == edge_corner[1]).all(axis=1))[0].item()
        edgeCorner_IncM.append([start_corner_idx, end_corner_idx])
    edgeCorner_IncM = np.array(edgeCorner_IncM)

    # Surface global bbox
    face_bboxes = []
    for pnts in faces_wcs:
        min_point, max_point = get_bbox(pnts.reshape(-1, 3))
        face_bboxes.append(np.concatenate([min_point, max_point]))
    face_bboxes = np.vstack(face_bboxes)

    # Edge global bbox
    edge_bboxes = []
    for pnts in edges_wcs:
        min_point, max_point = get_bbox(pnts.reshape(-1, 3))
        edge_bboxes.append(np.concatenate([min_point, max_point]))
    edge_bboxes = np.vstack(edge_bboxes)

    # Convert to float32 to save space
    data = {
        'face_wcs': faces_wcs.astype(np.float32),
        'edge_wcs': edges_wcs.astype(np.float32),
        'face_ncs': faces_ncs.astype(np.float32),
        'edge_ncs': edges_ncs.astype(np.float32),
        'corner_wcs': corner_wcs.astype(np.float32),
        'edgeFace_adj': edgeFace_IncM,
        'edgeCorner_adj': edgeCorner_IncM,
        'faceEdge_adj': faceEdge_IncM,
        'face_bbox_wcs': face_bboxes.astype(np.float32),
        'edge_bbox_wcs': edge_bboxes.astype(np.float32),
        'corner_unique': corner_unique.astype(np.float32),
    }

    return data


def count_fe_topo(face_edge):
    """
    Calculate the number of common edges between two adjacent faces

    Args:
    - face_edge (list): Face-Edge List
    
    Returns:
    - fe_topo (numpy.ndarray): Number of common edges between any paired faces
    """
    num_faces = len(face_edge)
    fe_topo = np.zeros((num_faces, num_faces), dtype=int)
    face_edge_sets = [set(fe) for fe in face_edge]
    for i in range(num_faces):
        for j in range(i+1, num_faces):
            common_elements = face_edge_sets[i].intersection(face_edge_sets[j])
            common_count = len(common_elements)
            fe_topo[i, j] = common_count
            fe_topo[j, i] = common_count

    return fe_topo


def process(step_folder, print_error=False):
    """
    Helper function to load step files and process in parallel

    Args:
    - step_folder (str): Path to the STEP parent-folder
    Returns:
    - Complete status: Valid (1) / Non-valid (0).
    """
    try:
        step_path = ""
        # Load cad data
        if step_folder.endswith('.step'):
            step_path = step_folder
            process_furniture = True
        else:
            for _, _, files in os.walk(step_folder):
                assert len(files) == 1
                step_path = os.path.join(step_folder, files[0])
            process_furniture = False

        # Check single solid
        cad_solid = load_step(step_path)
        if len(cad_solid) != 1:
            if print_error:
                print('Skipping multi solids...')
            return 0

        # Start data parsing
        data = parse_solid(cad_solid[0])
        if data is None:
            if print_error:
                print('Exceeding threshold...')
            return 0  # number of faces or edges exceed pre-determined threshold

        # Save the parsed result 
        if process_furniture:
            data_uid = step_path.split('/')[-2] + '_' + step_path.split('/')[-1]
            sub_folder = step_path.split('/')[-3]
        else:
            data_uid = step_path.split('/')[-2]
            sub_folder = data_uid[:4]

        if data_uid.endswith('.step'):
            data_uid = data_uid[:-5]  # furniture avoid .step

        data['uid'] = data_uid
        save_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT, sub_folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        fe_topo = count_fe_topo(data['faceEdge_adj'])
        data['fe_topo'] = fe_topo

        save_path = os.path.join(save_folder, data['uid'] + '.pkl')
        with open(save_path, "wb") as tf:
            pickle.dump(data, tf)

        return 1

    except Exception as e:
        print('not saving due to error...')
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Data folder path", default='/home/jing/Datasets/furniture_dataset')
    parser.add_argument("--option", type=str, choices=['abc', 'deepcad', 'furniture'], default='furniture',
                        help="Choose between dataset option [abc/deepcad/furniture] (default: abc)")
    parser.add_argument("--interval", type=int, help="Data range index, only required for abc/deepcad")
    args = parser.parse_args()

    if args.option == 'deepcad':
        OUTPUT = 'deepcad_parsed'
    elif args.option == 'abc':
        OUTPUT = 'abc_parsed'
    else:
        OUTPUT = 'furniture_parsed'

    # Load all STEP files
    if args.option == 'furniture':
        step_dirs = load_furniture_step(args.input)
    else:
        step_dirs = load_abc_step(args.input, args.option == 'deepcad')
        step_dirs = step_dirs[args.interval * 10000: (args.interval + 1) * 10000]

    # Process B-reps in parallel
    # process(step_dirs[0])
    valid = 0
    convert_iter = Pool(os.cpu_count()).imap(process, step_dirs)
    for status in tqdm(convert_iter, total=len(step_dirs)):
        valid += status
    print(f'Done... Data Converted Ratio {100.0 * valid / len(step_dirs)}%', valid, len(step_dirs))
