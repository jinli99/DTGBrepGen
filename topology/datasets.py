import os
import pickle
import copy
import torch
import numpy as np
from tqdm import tqdm
from utils import check_step_ok, pad_zero
from itertools import chain


def opposite_idx(idx):
    return idx - 1 if idx % 2 else idx + 1


def compute_topoSeq(faceEdge_adj, edgeFace_adj, edgeVert_adj):

    nv = edgeVert_adj.max() + 1

    # assign new face idx
    sorted_faces = [(sorted(edges), idx) for idx, edges in enumerate(faceEdge_adj)]
    sorted_faces.sort(key=lambda x: x[0])
    new_face_idx = [x[1] for x in sorted_faces]
    face_id_inverse = np.zeros(len(new_face_idx), dtype=int)
    for i, idx in enumerate(new_face_idx):
        face_id_inverse[idx] = i
    edgeFace_adj = face_id_inverse[edgeFace_adj]
    faceEdge_adj = [faceEdge_adj[i] for i in new_face_idx]

    topo_seq = []
    loop_end_flag = -1
    vert_set = [set() for _ in range(nv)]
    corner_flag = [-1 for _ in range(2 * edgeVert_adj.shape[0])]
    for idx in range(len(faceEdge_adj)):

        face_seq = []
        edges_rest = copy.deepcopy(faceEdge_adj[idx].tolist())
        edge = min(edges_rest)
        edges_rest.remove(edge)
        current_corner = 2 * edge + 1
        face_seq.append(current_corner - 1)
        loop_start_edge = edge
        while edges_rest:
            current_vert = corner_flag[current_corner]
            if current_vert == -1:
                v1, v2 = edgeVert_adj[edge]
                assert corner_flag[opposite_idx(current_corner)] == -1
                corner_flag[current_corner] = v1
                corner_flag[opposite_idx(current_corner)] = v2
                vert_set[v1].add(current_corner)
                vert_set[v2].add(opposite_idx(current_corner))
                current_vert = v1

            # find next edge
            for next_edge in edges_rest:
                temp = edgeVert_adj[next_edge].tolist()
                if current_vert in temp:
                    opposite_vert = temp[0] if current_vert == temp[1] else temp[1]
                    if corner_flag[2 * next_edge] == current_vert:
                        face_seq.append(2 * next_edge)
                        current_corner = 2 * next_edge + 1
                    elif corner_flag[2 * next_edge + 1] == current_vert:
                        face_seq.append(2 * next_edge + 1)
                        current_corner = 2 * next_edge
                    else:
                        corner_flag[2 * next_edge] = current_vert
                        corner_flag[2 * next_edge + 1] = opposite_vert
                        vert_set[current_vert].add(2 * next_edge)
                        vert_set[opposite_vert].add(2 * next_edge + 1)
                        face_seq.append(2 * next_edge)
                        current_corner = 2 * next_edge + 1
                    edges_rest.remove(next_edge)
                    if not edges_rest:
                        if corner_flag[current_corner] not in edgeVert_adj[loop_start_edge].tolist():
                            return 0
                        else:
                            face_seq.append(loop_end_flag)
                    edge = next_edge
                    break
            else:
                if current_vert not in edgeVert_adj[loop_start_edge].tolist():
                    return 0
                face_seq.append(loop_end_flag)
                if edges_rest:
                    edge = min(edges_rest)
                    edges_rest.remove(edge)
                    current_corner = 2 * edge + 1
                    face_seq.append(current_corner - 1)
                    loop_start_edge = edge

        topo_seq.append(face_seq)

    return topo_seq, faceEdge_adj, edgeFace_adj


def create_topo_datasets(data_type='train'):

    def create(path):

        if not check_step_ok(path):
            return 0

        with open(os.path.join('../data_process/furniture_parsed', path), 'rb') as f:
            datas = pickle.load(f)

        data = {'name': path.replace('/', '_').replace('.pkl', '')}

        topo_seq, faceEdge_adj, edgeFace_adj = compute_topoSeq(datas['faceEdge_adj'],
                                                               datas['edgeFace_adj'],
                                                               datas['edgeCorner_adj'])

        data['topo_seq'] = topo_seq
        data['faceEdge_adj'] = faceEdge_adj
        data['edgeFace_adj'] = edgeFace_adj
        data['edgeVert_adj'] = datas['edgeCorner_adj']
        data['fe_topo'] = datas['fe_topo']

        os.makedirs(os.path.join('../data_process/topoDatasets/furniture', data_type), exist_ok=True)
        with open(os.path.join('../data_process/topoDatasets/furniture', data_type, data['name']+'.pkl'), 'wb') as f:
            pickle.dump(data, f)
        return 1

    with open('../data_process/furniture_data_split_6bit.pkl', 'rb') as tf:
        files = pickle.load(tf)[data_type]

    valid = 0
    for file in tqdm(files):
        valid += create(file)

    print(valid)


class TopoSeqDataset(torch.utils.data.Dataset):
    def __init__(self, path, data_aug=False):
        data = os.listdir(path)
        self.data = [os.path.join(path, i) for i in data]
        max_num_edge = 0
        max_seq_length = 0
        for file in self.data:
            with open(file, "rb") as tf:
                data = pickle.load(tf)
                length = [len(i) for i in data['topo_seq']]
                max_seq_length = max(sum(length) + len(length), max_seq_length)
                max_num_edge = max(max_num_edge, data['edgeFace_adj'].shape[0])
        self.max_seq_length = max_seq_length
        self.max_num_edge = max_num_edge
        self.data_aug = data_aug

    def swap_in_sublist(self, sublist, swap_prob=0.1):
        arr = np.array(sublist)

        if self.data_aug and np.random.random() < swap_prob:
            idx1, idx2 = np.random.choice(len(arr)-1, 2, replace=False)

            arr[idx1], arr[idx2] = arr[idx2], arr[idx1]

        return arr.tolist()

    @staticmethod
    def shuffle_edge_idx(faceEdge_adj, edgeFace_adj, edgeVert_adj):

        total_edges = edgeFace_adj.shape[0]

        new_edge_ids = np.random.permutation(total_edges)

        faceEdge_adj = [new_edge_ids[face_edges] for face_edges in faceEdge_adj]

        inverse_idx_map = [-1] * len(new_edge_ids)
        for i, idx in enumerate(new_edge_ids):
            inverse_idx_map[idx] = i

        edgeFace_adj = edgeFace_adj[inverse_idx_map]
        edgeVert_adj = edgeVert_adj[inverse_idx_map]

        topo_seq, faceEdge_adj, edgeFace_adj = compute_topoSeq(faceEdge_adj, edgeFace_adj, edgeVert_adj)

        return edgeFace_adj, topo_seq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with open(self.data[idx], "rb") as tf:
            data = pickle.load(tf)

        # data augment
        shuffle_prob = 0.5
        if self.data_aug and np.random.random() < shuffle_prob:
            edgeFace_adj, topo_seq = self.shuffle_edge_idx(data['faceEdge_adj'],
                                                           data['edgeFace_adj'],
                                                           data['edgeVert_adj'])
        else:
            edgeFace_adj, topo_seq = data['edgeFace_adj'], data['topo_seq']    # ne*2, List[List[int]]
        topo_seq = [self.swap_in_sublist(sublist) for sublist in topo_seq]

        assert edgeFace_adj.shape[0] <= self.max_num_edge
        edgeFace_adj, edge_mask = pad_zero(edgeFace_adj, max_len=self.max_num_edge)   # max_num_edge*2, max_num_edge

        topo_seq = np.expand_dims(np.array(list(chain.from_iterable(sublist + [-2] for sublist in topo_seq))), axis=-1)
        topo_seq, seq_mask = pad_zero(topo_seq, max_len=self.max_seq_length)        # max_seq_length*1, max_seq_length
        return (torch.from_numpy(edgeFace_adj),            # max_num_edge*2
                torch.from_numpy(edge_mask),               # max_num_edge
                torch.from_numpy(topo_seq).squeeze(-1),    # max_seq_length
                torch.from_numpy(seq_mask).squeeze(-1)     # max_seq_length
                )


class FaceEdgeDataset(torch.utils.data.Dataset):
    def __init__(self, path, args):
        data = os.listdir(path)
        self.data = [os.path.join(path, i) for i in data]
        self.max_face = args.max_face
        self.max_edge = args.edge_classes - 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        with open(self.data[idx], "rb") as tf:
            data = pickle.load(tf)
        fe_topo = data['fe_topo']                                            # nf*nf
        edge_counts = np.sum(fe_topo, axis=1)                                # nf
        sorted_ids = np.argsort(edge_counts)[::-1]                           # nf
        fe_topo = fe_topo[sorted_ids][:, sorted_ids]
        assert np.all(fe_topo == fe_topo.transpose(0, 1))
        fe_topo, mask = pad_zero(fe_topo, max_len=self.max_face, dim=1)      # max_face*max_face, max_face
        return torch.from_numpy(fe_topo), torch.from_numpy(mask)


if __name__ == '__main__':
    create_topo_datasets(data_type='train')
    create_topo_datasets(data_type='test')
