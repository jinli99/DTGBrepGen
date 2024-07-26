import os
import torch
import math
import pickle
import numpy as np
import random
from tqdm import tqdm
from multiprocessing.pool import Pool


# furniture class labels
text2int = {'bathtub': 0, 'bed': 1, 'bench': 2, 'bookshelf': 3, 'cabinet': 4, 'chair': 5, 'couch': 6, 'lamp': 7,
            'sofa': 8, 'table': 9}


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
    faceEdge_adj, surf_bbox, edge_bbox, ff_edges = (data['faceEdge_adj'], data['surf_bbox_wcs'],
                                                    data['edge_bbox_wcs'], data['ff_edges'])

    # Skip over max edge-classes
    if ff_edges.max() >= edge_classes:
        return None, None

    # Skip over max size data
    if len(surf_bbox) > max_face:
        return None, None

    for surf_edges in faceEdge_adj:
        if len(surf_edges) > max_edge:
            return None, None

    # Skip surfaces too close to each other
    surf_bbox = surf_bbox * scaled_value  # make bbox difference larger

    _surf_bbox_ = surf_bbox.reshape(len(surf_bbox), 2, 3)
    non_repeat = _surf_bbox_[:1]
    for bbox in _surf_bbox_:
        diff = np.max(np.max(np.abs(non_repeat - bbox), -1), -1)
        same = diff < threshold_value
        if same.sum() >= 1:
            continue  # repeat value
        else:
            non_repeat = np.concatenate([non_repeat, bbox[np.newaxis, :, :]], 0)
    if len(non_repeat) != len(_surf_bbox_):
        return None, None

    # Skip edges too close to each other
    se_bbox = []
    for adj in faceEdge_adj:
        if len(edge_bbox[adj]) == 0:
            return None, None
        se_bbox.append(edge_bbox[adj] * scaled_value)

    for bbb in se_bbox:
        _edge_bbox_ = bbb.reshape(len(bbb), 2, 3)
        non_repeat = _edge_bbox_[:1]
        for bbox in _edge_bbox_:
            diff = np.max(np.max(np.abs(non_repeat - bbox), -1), -1)
            same = diff < threshold_value
            if same.sum() >= 1:
                continue  # repeat value
            else:
                non_repeat = np.concatenate([non_repeat, bbox[np.newaxis, :, :]], 0)
        if len(non_repeat) != len(_edge_bbox_):
            return None, None
    return data_path, data_class


def load_data(input_data, input_list, validate, args):
    # Filter data list
    with open(input_list, "rb") as tf:
        if validate:
            data_list = pickle.load(tf)['val']
        else:
            data_list = pickle.load(tf)['train']

    data_paths = []
    data_classes = []
    for uid in data_list:
        try:
            path = os.path.join(input_data, str(math.floor(int(uid.split('.')[0]) / 10000)).zfill(4), uid)
            class_label = -1   # unconditional generation (abc/deepcad)
        except Exception:
            path = os.path.join(input_data, uid)
            class_label = text2int[uid.split('/')[0]]   # conditional generation (furniture)
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


def pad_zero(x, max_len, return_mask=False):
    keys = np.ones(len(x))
    padding = np.zeros((max_len-len(x))).astype(int)
    mask = 1-np.concatenate([keys, padding]) == 1
    padding = np.zeros((max_len-len(x), *x.shape[1:]))
    x_padded = np.concatenate([x, padding], axis=0)
    if return_mask:
        return x_padded, mask
    else:
        return x_padded


class SurfData(torch.utils.data.Dataset):
    """ Surface VAE Dataloader """

    def __init__(self, input_data, input_list, validate=False, aug=False):
        self.validate = validate
        self.aug = aug

        # Load validation data
        if self.validate:
            print('Loading validation data...')
            with open(input_list, "rb") as tf:
                data_list = pickle.load(tf)['val']

            datas = []
            for uid in data_list:
                try:
                    path = os.path.join(input_data, str(math.floor(int(uid.split('.')[0]) / 10000)).zfill(4), uid)
                except Exception:
                    path = os.path.join(input_data, uid)

                with open(path, "rb") as tf:
                    data = pickle.load(tf)
                surf_uv = data['surf_ncs']
                datas.append(surf_uv)
            self.data = np.vstack(datas)

        # Load training data (deduplicated)
        else:
            print('Loading training data...')
            with open(input_list, "rb") as tf:
                self.data = pickle.load(tf)

        print(len(self.data))
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        surf_uv = self.data[index]
        if np.random.rand() > 0.5 and self.aug:
            for axis in ['x', 'y', 'z']:
                angle = random.choice([90, 180, 270])
                surf_uv = rotate_point_cloud(surf_uv.reshape(-1, 3), angle, axis).reshape(32, 32, 3)
        return torch.FloatTensor(surf_uv)


class EdgeData(torch.utils.data.Dataset):
    """ Edge VAE Dataloader """

    def __init__(self, input_data, input_list, validate=False, aug=False):
        self.validate = validate
        self.aug = aug

        # Load validation data
        if self.validate:
            print('Loading validation data...')
            with open(input_list, "rb") as tf:
                data_list = pickle.load(tf)['val']

            datas = []
            for uid in tqdm(data_list):
                try:
                    path = os.path.join(input_data, str(math.floor(int(uid.split('.')[0]) / 10000)).zfill(4), uid)
                except Exception:
                    path = os.path.join(input_data, uid)

                with open(path, "rb") as tf:
                    data = pickle.load(tf)

                edge_u = data['edge_ncs']
                datas.append(edge_u)
            self.data = np.vstack(datas)

        # Load training data (deduplicated)
        else:
            print('Loading training data...')
            with open(input_list, "rb") as tf:
                self.data = pickle.load(tf)

        print(len(self.data))
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        edge_u = self.data[index]
        # Data augmentation, randomly rotate 50% of the times
        if np.random.rand() > 0.5 and self.aug:
            for axis in ['x', 'y', 'z']:
                angle = random.choice([90, 180, 270])
                edge_u = rotate_point_cloud(edge_u, angle, axis)
        return torch.FloatTensor(edge_u)


class SurfGraphData(torch.utils.data.Dataset):
    """ Surface Feature and Edge Topology Dataloader """

    def __init__(self, input_data, input_list, validate=False, aug=False, args=None):
        self.max_face = args.max_face
        self.max_edge = args.max_edge
        self.bbox_scaled = args.bbox_scaled
        self.aug = aug
        # Load data
        self.data = load_data(input_data, input_list, validate, args)
        # Inflate furniture x50 times for training
        if len(self.data) < 2000 and not validate:
            self.data = self.data * 12
        # if not validate:
        #     with open('SurfGraph_train_datas.pkl', 'rb') as f:
        #         self.data = pickle.load(f)
        # else:
        #     with open('SurfGraph_val_datas.pkl', 'rb') as f:
        #         self.data = pickle.load(f)
        print("train" if not validate else "val", len(self.data))
        return

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
        surf_ncs, surf_pos, ff_edges = data['surf_ncs'], data['surf_bbox_wcs'], data['ff_edges']

        # Make bbox range larger
        surf_pos = surf_pos * self.bbox_scaled

        # Randomly shuffle the sequence
        random_indices = np.random.permutation(surf_pos.shape[0])
        surf_pos = surf_pos[random_indices]   # num_faces*6
        surf_ncs = surf_ncs[random_indices]   # num_faces*32*32*3

        # Pad data
        # surf_pos, surf_mask = pad_zero(surf_pos, self.max_face, return_mask=True)
        # surf_ncs = pad_zero(surf_ncs, self.max_face)

        if data_class is not None:
            return (
                torch.FloatTensor(surf_pos),
                torch.FloatTensor(surf_ncs),
                torch.from_numpy(ff_edges),
                # torch.BoolTensor(surf_mask),
                torch.LongTensor([data_class + 1])  # add 1, class 0 = uncond (furniture)
            )
        else:
            return (
                torch.FloatTensor(surf_pos),
                torch.FloatTensor(surf_ncs),
                torch.from_numpy(ff_edges),
                # torch.BoolTensor(surf_mask),
            )  # abc or deepcad


class EdgeGraphData(torch.utils.data.Dataset):
    """ Edge Feature Dataloader """

    def __init__(self, input_data, input_list, validate=False, aug=False, args=None):
        self.max_face = args.max_face
        self.max_edge = args.max_edge
        self.bbox_scaled = args.bbox_scaled
        self.aug = aug
        self.data = []
        # Load data
        self.data = load_data(input_data, input_list, validate, args)
        # Inflate furniture x50 times for training
        if len(self.data) < 2000 and not validate:
            self.data = self.data * 4
        return

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

        surf_ncs, edge_ncs, edgeFace_adj, surf_pos, edge_pos = (
            data['surf_ncs'],         # nf*32*32*3
            data['edge_ncs'],         # ne*32*3
            data['edgeFace_adj'],     # ne*2
            data['surf_bbox_wcs'],    # nf*6
            data['edge_bbox_wcs']     # ne*6
        )

        # Increase value range
        surf_pos = surf_pos * self.bbox_scaled
        edge_pos = edge_pos * self.bbox_scaled

        edge_surf_ncs = surf_ncs[edgeFace_adj]   # ne*2*32*32*3
        edge_surf_pos = surf_pos[edgeFace_adj]   # ne*2*6

        if data_class is not None:
            return (
                torch.FloatTensor(edge_ncs),
                torch.FloatTensor(edge_pos),
                torch.FloatTensor(edge_surf_ncs),
                torch.FloatTensor(edge_surf_pos),
                torch.LongTensor([data_class + 1])  # add 1, class 0 = uncond (furniture)
            )
        else:
            return (
                torch.FloatTensor(edge_ncs),
                torch.FloatTensor(edge_pos),
                torch.FloatTensor(edge_surf_ncs),
                torch.FloatTensor(edge_surf_pos),   # uncond deepcad/abc
            )


