import os
import torch
import math
import pickle
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
from multiprocessing.pool import Pool
from utils import pad_and_stack, pad_zero, sort_bbox_multi, check_step_ok


# furniture class labels
text2int = {'bathtub': 0, 'bed': 1, 'bench': 2, 'bookshelf': 3, 'cabinet': 4, 'chair': 5, 'couch': 6, 'lamp': 7,
            'sofa': 8, 'table': 9}


def compute_dataset_info(name='furniture'):

    with open(os.path.join('data_process', name+'_data_split_6bit.pkl'), "rb") as file:
        datas = pickle.load(file)['train']

    max_num_edge = 0
    max_vert = 0
    max_vertFace = 0
    for path in tqdm(datas):
        with open(os.path.join('data_process/GeomDatasets', name+'_parsed', path), 'rb') as file:
            data = pickle.load(file)
            if not check_step_ok(data):
                continue
            fef_adj = data['fef_adj']
            max_num_edge = max(max_num_edge, len(data['edgeFace_adj']))
            max_vert = max(max_vert, len(data['vert_wcs']))
            max_vertFace = max(max_vertFace, max([len(i) for i in data['vertFace_adj']]))
            assert np.array_equal(fef_adj, fef_adj.T) and np.all(np.diag(fef_adj) == 0)

    print(f'{name} dataset has max_num_edge:{max_num_edge}, max_vert:{max_vert}, max_vertFace:{max_vertFace}')


def rotate_point_cloud(point_cloud, angle_degrees, axis):
    """
    Rotate a point cloud around it's center by a specified angle in degrees along a specified axis.

    Args:
    - point_cloud: Numpy array of shape (N, 3) representing the point cloud.
    - angle_degrees: Angle of rotation in degrees.
    - axis: Axis of rotation. Can be 'x', 'y', or 'z'.

    Returns:
    - rotated_point_cloud: Numpy array of shape (N, 3) representing the rotated point cloud.
    """

    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Compute rotation matrix based on the specified axis
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(angle_radians), -np.sin(angle_radians)],
                                    [0, np.sin(angle_radians), np.cos(angle_radians)]])
    elif axis == 'y':
        rotation_matrix = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)],
                                    [0, 1, 0],
                                    [-np.sin(angle_radians), 0, np.cos(angle_radians)]])
    elif axis == 'z':
        rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                                    [np.sin(angle_radians), np.cos(angle_radians), 0],
                                    [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")

    # Center the point cloud
    center = np.mean(point_cloud, axis=0)
    centered_point_cloud = point_cloud - center

    # Apply rotation
    rotated_point_cloud = np.dot(centered_point_cloud, rotation_matrix.T)

    # Translate back to original position
    rotated_point_cloud += center

    # Find the maximum absolute coordinate value
    max_abs_coord = np.max(np.abs(rotated_point_cloud))

    # Scale the point cloud to fit within the -1 to 1 cube
    normalized_point_cloud = rotated_point_cloud / max_abs_coord

    return normalized_point_cloud


def filter_data(data):
    """
    Helper function to check if a brep needs to be included
        in the training data or not
    """
    data_path, max_face, max_edge, scaled_value, threshold_value, edge_classes, data_class = data
    # Load data
    with open(data_path, "rb") as tf:
        data = pickle.load(tf)
    if not check_step_ok(data, max_face=max_face, max_edge=max_edge, edge_classes=edge_classes):
        return None, None
    return data_path, data_class


def load_data(input_data, input_list, validate, args):
    # Filter data list
    with open(input_list, "rb") as tf:
        if validate:
            data_list = pickle.load(tf)['test']
        else:
            data_list = pickle.load(tf)['train']

    data_paths = []
    data_classes = []

    for uid in data_list:
        path = os.path.join(input_data, uid)
        if args.use_cf:
            class_label = text2int[uid.split('/')[0]]      # conditional generation (furniture)
        else:
            class_label = -1                               # unconditional generation (abc/deepcad)

        data_paths.append(path)
        data_classes.append(class_label)

    # Filter data in parallel
    loaded_data = []
    params = zip(data_paths, [args.max_face] * len(data_list), [args.max_edge] * len(data_list),
                 [args.bbox_scaled] * len(data_list), [args.threshold] * len(data_list),
                 [args.edge_classes] * len(data_list), data_classes)
    convert_iter = Pool(os.cpu_count()).imap(filter_data, params)
    for data_path, data_class in tqdm(convert_iter, total=len(data_list)):
        if data_path is not None:
            if data_class < 0:  # abc or deepcad
                loaded_data.append(data_path)
            else:  # furniture
                loaded_data.append((data_path, data_class))

    print(f'Processed {len(loaded_data)}/{len(data_list)}')
    return loaded_data


class EdgeGeomData(torch.utils.data.Dataset):
    """ Edge Feature Dataloader """

    def __init__(self, args, validate=False):
        self.max_face = args.max_face
        self.max_edge = args.max_edge
        self.max_num_edge = args.max_num_edge
        self.bbox_scaled = args.bbox_scaled
        self.aug = args.data_aug
        self.use_cf = args.use_cf
        self.use_pc = args.use_pc
        self.data = []
        # Load data
        self.data = load_data(args.data, args.train_list, validate, args)
        if len(self.data) < 2000 and not validate:
            self.data = self.data*50

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load data
        data_class = None
        if isinstance(self.data[index], tuple):
            data_path, data_class = self.data[index]
        else:
            data_path = self.data[index]

        with open(data_path, "rb") as tf:
            data = pickle.load(tf)

        face_bbox, edge_ncs, edgeFace_adj, vert_geom, edgeVert_adj = (
            data['face_bbox_wcs'],                     # nf*6
            data['edge_ncs'],                          # ne*32*3
            data['edgeFace_adj'],                      # ne*2
            data['vert_wcs'],                          # nv*3
            data['edgeVert_adj']                       # ne*2
        )

        # Increase value range
        face_bbox = sort_bbox_multi(face_bbox)
        face_bbox = face_bbox * self.bbox_scaled      # nf*6
        vert_geom *= self.bbox_scaled                 # nv*3

        edgeFace_bbox = face_bbox[edgeFace_adj]       # ne*2*6
        edgeVert_geom = vert_geom[edgeVert_adj]       # ne*2*3

        random_indices = np.random.permutation(edge_ncs.shape[0])
        edgeFace_bbox = edgeFace_bbox[random_indices]
        edge_geom = edge_ncs[random_indices]               # ne*32*3
        edgeVert_geom = edgeVert_geom[random_indices]      # ne*2*3

        edgeFace_bbox, mask = pad_zero(edgeFace_bbox, max_len=self.max_num_edge, dim=0)
        edge_geom, _ = pad_zero(edge_geom, max_len=self.max_num_edge, dim=0)
        edgeVert_geom, _ = pad_zero(edgeVert_geom, max_len=self.max_num_edge, dim=0)
        mask = mask.sum(keepdims=True)

        if data_class is not None:
            return (
                torch.FloatTensor(edge_geom),              # max_num_edge*32*3
                torch.FloatTensor(edgeFace_bbox),          # max_num_edge*2*6
                torch.FloatTensor(edgeVert_geom),          # max_num_edge*2*3
                torch.from_numpy(mask),                    # 1
                torch.LongTensor([data_class + 1])         # add 1, class 0 = uncond (furniture)
            )
        else:
            return (
                torch.FloatTensor(edge_geom),              # max_num_edge*32*3
                torch.FloatTensor(edgeFace_bbox),          # max_num_edge*2*6
                torch.FloatTensor(edgeVert_geom),          # max_num_edge*2*3
                torch.from_numpy(mask),                    # 1
            )


class FaceGeomData(torch.utils.data.Dataset):
    def __init__(self, args, validate=False):
        self.max_face = args.max_face
        self.max_edge = args.max_edge
        self.bbox_scaled = args.bbox_scaled
        self.aug = args.data_aug
        self.use_cf = args.use_cf
        self.use_pc = args.use_pc

        self.data = []
        # Load data
        self.data = load_data(args.data, args.train_list, validate, args)
        if len(self.data) < 2000 and not validate:
            self.data = self.data*50

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load data
        data_class = None
        if isinstance(self.data[index], tuple):
            data_path, data_class = self.data[index]
        else:
            data_path = self.data[index]

        with open(data_path, "rb") as tf:
            data = pickle.load(tf)

        face_ctrs, face_bbox, edge_ctrs, vert_geom, faceEdge_adj, edgeVert_adj = (
            data['face_ctrs'].reshape(-1, 48),         # nf*16*3
            data['face_bbox_wcs'],                     # nf*6
            data['edge_ctrs'].reshape(-1, 12),         # ne*12
            data['vert_wcs'],                          # nv*3
            data['faceEdge_adj'],                      # [[e1, e2, ...], ...]
            data['edgeVert_adj']                       # ne*2
        )

        # Increase value range
        face_bbox = sort_bbox_multi(face_bbox)
        face_bbox *= self.bbox_scaled                                                                # nf*6
        face_ctrs *= self.bbox_scaled                                                                # nf*48
        edge_ctrs *= self.bbox_scaled                                                                # ne*12
        vert_geom *= self.bbox_scaled                                                                # nv*3
        faceEdge_geom = [edge_ctrs[i] for i in faceEdge_adj]                                         # [fe*12, ...]
        faceVert_geom = [vert_geom[np.unique(edgeVert_adj[i].flatten())] for i in faceEdge_adj]      # [fv*3, ...]

        # Randomly shuffle the sequence
        random_indices = np.random.permutation(face_bbox.shape[0])
        face_geom = face_ctrs[random_indices]
        face_bbox = face_bbox[random_indices]
        faceVert_geom = [faceVert_geom[i] for i in random_indices]
        faceEdge_geom = [faceEdge_geom[i] for i in random_indices]

        face_bbox, mask = pad_zero(face_bbox, max_len=self.max_face, dim=0)                     # max_faces*6, max_faces
        mask = mask.sum(keepdims=True)                                                          # 1
        faceVert_geom, faceVert_mask = pad_and_stack(faceVert_geom, max_n=self.max_edge)        # nf*fv*3, nf*fv
        faceVert_mask = faceVert_mask.sum(-1, keepdims=True)                                    # nf*1
        faceEdge_geom, faceEdge_mask = pad_and_stack(faceEdge_geom, max_n=self.max_edge)        # nf*fe*12, nf*fe
        faceEdge_mask = faceEdge_mask.sum(-1, keepdims=True)                                    # nf*1
        assert mask == data['face_ctrs'].shape[0]
        face_geom, _ = pad_zero(face_geom, max_len=self.max_face, dim=0)                        # max_faces*48
        faceVert_geom, _ = pad_zero(faceVert_geom, max_len=self.max_face, dim=0)                # max_faces*fv*3
        faceVert_mask, _ = pad_zero(faceVert_mask, max_len=self.max_face, dim=0)                # max_faces*1
        faceEdge_geom, _ = pad_zero(faceEdge_geom, max_len=self.max_face, dim=0)                # max_faces*fe*12
        faceEdge_mask, _ = pad_zero(faceEdge_mask, max_len=self.max_face, dim=0)                # max_faces*1
        if data_class is not None:
            return (
                torch.FloatTensor(face_geom),          # max_faces*48
                torch.FloatTensor(face_bbox),          # max_faces*6
                torch.FloatTensor(faceVert_geom),      # max_faces*fv*3
                torch.FloatTensor(faceEdge_geom),      # max_faces*fe*12
                torch.from_numpy(mask),                # 1
                torch.from_numpy(faceVert_mask),       # max_faces*1
                torch.from_numpy(faceEdge_mask),       # max_faces*1
                torch.LongTensor([data_class + 1])     # add 1, class 0 = un-cond (furniture)
            )
        else:
            return (
                torch.FloatTensor(face_geom),          # max_faces*48
                torch.FloatTensor(face_bbox),          # max_faces*6
                torch.FloatTensor(faceVert_geom),      # max_faces*fv*3
                torch.FloatTensor(faceEdge_geom),      # max_faces*fe*12
                torch.from_numpy(mask),                # 1
                torch.from_numpy(faceVert_mask),       # max_faces*1
                torch.from_numpy(faceEdge_mask),       # max_faces*1 un-cond deepcad/abc
            )


if __name__ == '__main__':
    compute_dataset_info(name='furniture')
