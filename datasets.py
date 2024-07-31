import os
import torch
import math
import pickle
import numpy as np
import random
from tqdm import tqdm
from multiprocessing.pool import Pool
from utils import pad_zero


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
    faceEdge_adj, face_bbox, edge_bbox, fe_topo = (data['faceEdge_adj'], data['face_bbox_wcs'],
                                                   data['edge_bbox_wcs'], data['fe_topo'])

    # Skip over max edge-classes
    if fe_topo.max() >= edge_classes:
        return None, None

    # Skip over max size data
    if len(face_bbox) > max_face:
        return None, None

    for face_edges in faceEdge_adj:
        if len(face_edges) > max_edge:
            return None, None

    # Skip faces too close to each other
    face_bbox = face_bbox * scaled_value  # make bbox difference larger

    _face_bbox_ = face_bbox.reshape(len(face_bbox), 2, 3)
    non_repeat = _face_bbox_[:1]
    for bbox in _face_bbox_:
        diff = np.max(np.max(np.abs(non_repeat - bbox), -1), -1)
        same = diff < threshold_value
        if same.sum() >= 1:
            continue  # repeat value
        else:
            non_repeat = np.concatenate([non_repeat, bbox[np.newaxis, :, :]], 0)
    if len(non_repeat) != len(_face_bbox_):
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


class FaceVaeData(torch.utils.data.Dataset):
    """ Face VAE Dataloader """

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
                face_uv = data['face_ncs']
                datas.append(face_uv)
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
        face_uv = self.data[index]
        if np.random.rand() > 0.5 and self.aug:
            for axis in ['x', 'y', 'z']:
                angle = random.choice([90, 180, 270])
                face_uv = rotate_point_cloud(face_uv.reshape(-1, 3), angle, axis).reshape(32, 32, 3)
        return torch.FloatTensor(face_uv)


class EdgeVaeData(torch.utils.data.Dataset):
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


class FaceEdgeTopoData(torch.utils.data.Dataset):
    """ Face Bounding Box and Edge-Face Topology Dataloader """

    def __init__(self, input_data, input_list, validate=False, aug=False, args=None):
        self.max_face = args.max_face
        self.max_edge = args.max_edge
        self.bbox_scaled = args.bbox_scaled
        self.aug = aug
        # Load data
        self.data = load_data(input_data, input_list, validate, args)

        # Inflate furniture x50 times for training
        # if len(self.data) < 2000 and not validate:
        #     self.data = self.data * 3

        print("train" if not validate else "val", len(self.data))

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
        face_bbox, fe_topo = data['face_bbox_wcs'], data['fe_topo']   # num_faces*6, num_faces*num_faces

        # Make bbox range larger
        face_bbox = face_bbox * self.bbox_scaled

        # Randomly shuffle the sequence
        random_indices = np.random.permutation(face_bbox.shape[0])
        face_bbox = face_bbox[random_indices]
        fe_topo = fe_topo[random_indices, :]
        fe_topo = fe_topo[:, random_indices]

        face_bbox, mask = pad_zero(face_bbox, max_len=self.max_face+1, dim=0)  # max_faces*6, max_faces, max_faces
        fe_topo, _ = pad_zero(fe_topo, max_len=self.max_face+1, dim=1)   # max_faces*max_faces

        if data_class is not None:
            return (
                torch.FloatTensor(face_bbox),   # max_faces*6, max_faces
                torch.from_numpy(fe_topo),      # max_faces*max_faces
                torch.from_numpy(mask),         # max_faces
                torch.LongTensor([data_class + 1])  # add 1, class 0 = uncond (furniture)
            )
        else:
            return (
                torch.FloatTensor(face_bbox),
                torch.from_numpy(fe_topo),
                torch.from_numpy(mask),  # max_faces
            )  # abc or deepcad


class FaceGeomData(torch.utils.data.Dataset):
    def __init__(self, input_data, input_list, validate=False, aug=False, args=None):
        self.max_face = args.max_face
        self.max_edge = args.max_edge
        self.bbox_scaled = args.bbox_scaled
        self.aug = aug
        self.data = []
        # Load data
        self.data = load_data(input_data, input_list, validate, args)
        # Inflate furniture x50 times for training
        # if len(self.data) < 2000 and not validate:
        #     self.data = self.data * 50

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

        face_ncs, fe_topo, face_bbox = (
            data['face_ncs'],          # nf*32*32*3
            data['fe_topo'],           # nf*nf
            data['face_bbox_wcs'],     # nf*6
        )

        # Increase value range
        face_bbox = face_bbox * self.bbox_scaled    # nf*6

        # Randomly shuffle the sequence
        random_indices = np.random.permutation(face_bbox.shape[0])
        face_ncs = face_ncs[random_indices]
        face_bbox = face_bbox[random_indices]
        fe_topo = fe_topo[random_indices, :]
        fe_topo = fe_topo[:, random_indices]

        face_bbox, mask = pad_zero(face_bbox, max_len=self.max_face + 1, dim=0)  # max_faces*6, max_faces
        face_ncs, _ = pad_zero(face_ncs, max_len=self.max_face + 1, dim=0)
        fe_topo, _ = pad_zero(fe_topo, max_len=self.max_face + 1, dim=1)  # max_faces*max_faces

        if data_class is not None:
            return (
                torch.FloatTensor(face_ncs),       # max_faces*32*32*3
                torch.FloatTensor(face_bbox),      # max_faces*6
                torch.from_numpy(fe_topo),        # max_faces*max_faces
                torch.from_numpy(mask),           # # max_faces
                torch.LongTensor([data_class + 1])  # add 1, class 0 = uncond (furniture)
            )
        else:
            return (
                torch.FloatTensor(face_ncs),  # max_faces*32*32*3
                torch.FloatTensor(face_bbox),  # max_faces*6
                torch.from_numpy(fe_topo),  # max_faces*max_faces
                torch.from_numpy(mask),  # # max_faces     # uncond deepcad/abc
            )


class EdgeGeomData(torch.utils.data.Dataset):
    """ Edge Feature Dataloader """

    def __init__(self, input_data, input_list, validate=False, aug=False, args=None):
        self.max_face = args.max_face
        self.max_edge = args.max_edge
        self.max_num_edge = args.max_num_edge
        self.bbox_scaled = args.bbox_scaled
        self.aug = aug
        self.data = []
        # Load data
        self.data = load_data(input_data, input_list, validate, args)
        # Inflate furniture x50 times for training
        # if len(self.data) < 2000 and not validate:
        #     self.data = self.data * 25

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

        face_ncs, edge_ncs, edgeFace_adj, face_bbox, edge_bbox = (
            data['face_ncs'],         # nf*32*32*3
            data['edge_ncs'],         # ne*32*3
            data['edgeFace_adj'],     # ne*2
            data['face_bbox_wcs'],    # nf*6
            data['edge_bbox_wcs']     # ne*6
        )

        # Increase value range
        face_bbox = face_bbox * self.bbox_scaled    # nf*32*32*3
        edge_bbox = edge_bbox * self.bbox_scaled    # ne*6

        edge_face_ncs = face_ncs[edgeFace_adj]   # ne*2*32*32*3
        edge_face_bbox = face_bbox[edgeFace_adj]   # ne*2*6

        random_indices = np.random.permutation(edge_bbox.shape[0])
        edge_bbox = edge_bbox[random_indices]
        edge_face_ncs = edge_face_ncs[random_indices]
        edge_face_bbox = edge_face_bbox[random_indices]
        edge_ncs = edge_ncs[random_indices]    # ne*32*3

        edge_bbox, mask = pad_zero(edge_bbox, max_len=self.max_num_edge+1, dim=0)
        edge_face_ncs, _ = pad_zero(edge_face_ncs, max_len=self.max_num_edge+1, dim=0)
        edge_face_bbox, _ = pad_zero(edge_face_bbox, max_len=self.max_num_edge+1, dim=0)
        edge_ncs, _ = pad_zero(edge_ncs, max_len=self.max_num_edge+1, dim=0)

        if data_class is not None:
            return (
                torch.FloatTensor(edge_ncs),
                torch.FloatTensor(edge_bbox),
                torch.FloatTensor(edge_face_ncs),
                torch.FloatTensor(edge_face_bbox),
                torch.from_numpy(mask),  # max_num_edge
                torch.LongTensor([data_class + 1])  # add 1, class 0 = uncond (furniture)
            )
        else:
            return (
                torch.FloatTensor(edge_ncs),
                torch.FloatTensor(edge_bbox),
                torch.FloatTensor(edge_face_ncs),
                torch.FloatTensor(edge_face_bbox),
                torch.from_numpy(mask),  # max_num_edge
            )

