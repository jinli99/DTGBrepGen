import os
import torch
import math
import pickle
import numpy as np
import random
from tqdm import tqdm
from multiprocessing.pool import Pool
from utils import pad_and_stack, pad_zero, sort_bbox_multi, check_step_ok


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
    if not check_step_ok(data, max_face=max_face, max_edge=max_edge, edge_classes=edge_classes):
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


class FaceBboxData(torch.utils.data.Dataset):
    """ Face Bounding Box Dataloader """

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

        fef_adj, face_bbox = (
            data['fef_adj'],           # nf*nf
            data['face_bbox_wcs'],     # nf*6
        )

        # Increase value range
        face_bbox = sort_bbox_multi(face_bbox)
        face_bbox = face_bbox * self.bbox_scaled    # nf*6

        # Randomly shuffle the sequence
        random_indices = np.random.permutation(face_bbox.shape[0])
        face_bbox = face_bbox[random_indices]
        fef_adj = fef_adj[random_indices, :]
        fef_adj = fef_adj[:, random_indices]

        face_bbox, mask = pad_zero(face_bbox, max_len=self.max_face + 1, dim=0)  # max_faces*6, max_faces
        fef_adj, _ = pad_zero(fef_adj, max_len=self.max_face + 1, dim=1)         # max_faces*max_faces

        if data_class is not None:
            return (
                torch.FloatTensor(face_bbox),       # max_faces*6
                torch.from_numpy(fef_adj),          # max_faces*max_faces
                torch.from_numpy(mask),             # max_faces
                torch.LongTensor([data_class + 1])  # add 1, class 0 = uncond (furniture)
            )
        else:
            return (
                torch.FloatTensor(face_bbox),  # max_faces*6
                torch.from_numpy(fef_adj),     # max_faces*max_faces
                torch.from_numpy(mask),        # max_faces     # uncond deepcad/abc
            )


class VertGeomData(torch.utils.data.Dataset):
    def __init__(self, input_data, input_list, validate=False, aug=False, args=None):
        self.max_face = args.max_face
        self.max_edge = args.max_edge
        self.max_vertex = args.max_vertex
        self.max_vertexFace = args.max_vertexFace
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

        vert_geom, face_bbox, edgeFace_adj, vv_list, vertFace_adj = (
            data['corner_unique'],  # nv*3
            data['face_bbox_wcs'],  # nf*6
            data['edgeFace_adj'],   # ne*2
            data['vv_list'],        # list[(v1, v2, edge_idx), ...]
            data['vertexFace']
        )

        nv = data['corner_unique'].shape[0]
        assert data['edgeVert_adj'].max() + 1 == nv

        vv_list = np.array(vv_list)  # ne*3  array[[v1, v2, edge_idx], ...]
        indices = vv_list[:, :2].astype(int)  # ne*2
        vv_adj = np.zeros((data['corner_unique'].shape[0], data['corner_unique'].shape[0]), dtype=int)  # nv*nv
        vv_adj[indices[:, 0], indices[:, 1]] = 1
        vv_adj[indices[:, 1], indices[:, 0]] = 1    # nv*nv

        # Increase value range
        face_bbox = sort_bbox_multi(face_bbox)
        face_bbox *= self.bbox_scaled      # nf*6
        vert_geom *= self.bbox_scaled      # nv*3

        vertFace_bbox = [face_bbox[i] for i in vertFace_adj]   # list[nf*6, ...]
        vertFace_bbox, vertFace_mask = pad_and_stack(vertFace_bbox, self.max_vertexFace)     # nv*nf*6. nv*nf

        # Randomly shuffle the sequence
        random_indices = np.random.permutation(vert_geom.shape[0])
        vert_geom = vert_geom[random_indices]              # nv*3
        vv_adj = vv_adj[random_indices, :]
        vv_adj = vv_adj[:, random_indices]                     # nv*nv
        vertFace_bbox = vertFace_bbox[random_indices]      # nv*nf*6
        vertFace_mask = vertFace_mask[random_indices]                # nv*nf

        vert_geom, mask = pad_zero(vert_geom, max_len=self.max_vertex + 1, dim=0)  # max_vertices*3, max_vertices
        vv_adj, _ = pad_zero(vv_adj, max_len=self.max_vertex + 1, dim=1)
        vertFace_bbox, _ = pad_zero(vertFace_bbox, max_len=self.max_vertex + 1, dim=0)
        vertFace_mask, _ = pad_zero(vertFace_mask, max_len=self.max_vertex + 1, dim=0)

        if data_class is not None:
            return (
                torch.FloatTensor(vert_geom),       # max_vertices*3
                torch.from_numpy(vv_adj),           # max_vertices*max_vertices
                torch.FloatTensor(vertFace_bbox),   # max_vertices*vf*6
                torch.from_numpy(mask),             # max_vertices
                torch.from_numpy(vertFace_mask),    # max_vertices*nf
                torch.LongTensor([data_class + 1])  # add 1, class 0 = uncond (furniture)
            )
        else:
            return (
                torch.FloatTensor(vert_geom),  # max_vertices*3
                torch.from_numpy(vv_adj),  # max_vertices*max_vertices
                torch.FloatTensor(vertFace_bbox),  # max_vertices*nf*6
                torch.from_numpy(mask),  # max_vertices
                torch.from_numpy(vertFace_mask),  # max_vertices*nf     # uncond deepcad/abc
            )


class FaceGeomData(torch.utils.data.Dataset):
    def __init__(self, input_data, input_list, validate=False, aug=False, args=None):
        self.max_face = args.max_face
        self.max_edge = args.max_edge
        self.bbox_scaled = args.bbox_scaled
        self.aug = aug
        self.data = []
        self.edge_classes = args.edge_classes
        # Load data
        self.data = load_data(input_data, input_list, validate, args)

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
            data['face_ctrs'].reshape(-1, 48),         # nf*(16*3)
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

        face_ctrs, edge_ctrs, edgeFace_adj, face_bbox, vert_geom, edgeVert_adj = (
            data['face_ctrs'].reshape(-1, 48),         # nf*(16*3)
            data['edge_ctrs'].reshape(-1, 12),         # ne*(4*3)
            data['edgeFace_adj'],                      # ne*2
            data['face_bbox_wcs'],                     # nf*6
            data['vert_wcs'],                          # nv*3
            data['edgeVert_adj']                       # ne*2
        )

        # Increase value range
        face_bbox = sort_bbox_multi(face_bbox)
        face_bbox = face_bbox * self.bbox_scaled      # nf*6
        face_ctrs *= self.bbox_scaled                 # nf*48
        edge_ctrs *= self.bbox_scaled                 # ne*12
        vert_geom *= self.bbox_scaled                 # nv*3

        edgeFace_geom = face_ctrs[edgeFace_adj]       # ne*2*48
        edgeFace_bbox = face_bbox[edgeFace_adj]       # ne*2*6
        edgeVert_geom = vert_geom[edgeVert_adj]       # ne*2*3

        random_indices = np.random.permutation(edge_ctrs.shape[0])
        edgeFace_geom = edgeFace_geom[random_indices]
        edgeFace_bbox = edgeFace_bbox[random_indices]
        edge_geom = edge_ctrs[random_indices]              # ne*12
        edgeVert_geom = edgeVert_geom[random_indices]      # ne*2*3

        edgeFace_geom, mask = pad_zero(edgeFace_geom, max_len=self.max_num_edge, dim=0)
        edgeFace_bbox, _ = pad_zero(edgeFace_bbox, max_len=self.max_num_edge, dim=0)
        edge_geom, _ = pad_zero(edge_geom, max_len=self.max_num_edge, dim=0)
        edgeVert_geom, _ = pad_zero(edgeVert_geom, max_len=self.max_num_edge, dim=0)
        mask = mask.sum(keepdims=True)

        if data_class is not None:
            return (
                torch.FloatTensor(edge_geom),              # max_num_edge*12
                torch.FloatTensor(edgeFace_geom),          # max_num_edge*2*48
                torch.FloatTensor(edgeFace_bbox),          # max_num_edge*2*6
                torch.FloatTensor(edgeVert_geom),          # max_num_edge*2*3
                torch.from_numpy(mask),                    # 1
                torch.LongTensor([data_class + 1])         # add 1, class 0 = uncond (furniture)
            )
        else:
            return (
                torch.FloatTensor(edge_geom),              # max_num_edge*32*3
                torch.FloatTensor(edgeFace_geom),          # max_num_edge*2*32*32*3
                torch.FloatTensor(edgeFace_bbox),          # max_num_edge*2*6
                torch.FloatTensor(edgeVert_geom),          # max_num_edge*2*3
                torch.from_numpy(mask),                    # 1
            )
