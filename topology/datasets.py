import os
import pickle
import copy
import torch
import numpy as np
from tqdm import tqdm
from utils import check_step_ok, pad_zero, load_data_with_prefix, calculate_y
from itertools import chain


# furniture class labels
text2int = {'bathtub': 0, 'bed': 1, 'bench': 2, 'bookshelf': 3, 'cabinet': 4, 'chair': 5, 'couch': 6, 'lamp': 7,
            'sofa': 8, 'table': 9}


def opposite_idx(idx):
    return idx - 1 if idx % 2 else idx + 1


def assign_idx(faceEdge_adj, edgeFace_adj, edgeVert_adj, fef_adj):
    """
    Args:
    - faceEdge_adj: List[List[int]]
    - edgeFace_adj: [ne, 2]
    - edgeVert_adj: [ne, 2]
    - fef_adj: [nf, nf]
    """
    assert np.array_equal(np.sort(np.unique(edgeFace_adj.flatten())), np.arange(fef_adj.shape[0]))

    num_edges_per_face = np.array([len(edges) for edges in faceEdge_adj])
    sorted_face_ids = np.argsort(num_edges_per_face)

    new_face_idx = np.zeros_like(sorted_face_ids)
    for new_idx, old_idx in enumerate(sorted_face_ids):
        new_face_idx[old_idx] = new_idx

    sorted_edges = []
    for i, (face1, face2) in enumerate(edgeFace_adj):
        new_face1, new_face2 = new_face_idx[face1], new_face_idx[face2]
        sorted_faces = sorted([new_face1, new_face2])
        sorted_edges.append((sorted_faces, i))

    sorted_edges.sort()

    sorted_edge_ids = [edge_id for _, edge_id in sorted_edges]

    new_edge_idx = np.zeros_like(sorted_edge_ids)
    for new_idx, old_idx in enumerate(sorted_edge_ids):
        new_edge_idx[old_idx] = new_idx

    edgeFace_adj = new_face_idx[edgeFace_adj]
    edgeFace_adj = edgeFace_adj[sorted_edge_ids]
    edgeVert_adj = edgeVert_adj[sorted_edge_ids]
    faceEdge_adj = [faceEdge_adj[i] for i in sorted_face_ids]
    faceEdge_adj = [new_edge_idx[i].tolist() for i in faceEdge_adj]
    fef_adj = fef_adj[sorted_face_ids][:, sorted_face_ids]

    return faceEdge_adj, edgeFace_adj, edgeVert_adj, fef_adj


def compute_topoSeq(faceEdge_adj, edgeVert_adj):

    nv = edgeVert_adj.max() + 1

    topo_seq = []
    loop_end_flag = -1
    vert_set = [set() for _ in range(nv)]
    corner_flag = [-1 for _ in range(2 * edgeVert_adj.shape[0])]
    for idx in range(len(faceEdge_adj)):

        face_seq = []
        edges_rest = copy.deepcopy(faceEdge_adj[idx])
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

    return topo_seq


def create_topo_datasets(data_type='train', option='deepcad'):

    def create(path):

        if option == 'furniture':
            with open(os.path.join('data_process/GeomDatasets/furniture_parsed', path), 'rb') as f:
                datas = pickle.load(f)

                if not check_step_ok(datas, max_face=32):
                    return 0
        elif option == 'deepcad':
            with open(os.path.join('data_process/GeomDatasets/deepcad_parsed', path), 'rb') as f:
                datas = pickle.load(f)

                if not check_step_ok(datas, max_face=30, max_edge=20):
                    return 0
        else:
            assert option == 'abc'
            with open(os.path.join('data_process/GeomDatasets/abc_parsed', path), 'rb') as f:
                datas = pickle.load(f)

                if not check_step_ok(datas, max_face=50, max_edge=30):
                    return 0

        if option == 'furniture':
            data = {'name': path.replace('/', '_').replace('.pkl', '')}

            faceEdge_adj, edgeFace_adj, edgeVert_adj, fef_adj = assign_idx(datas['faceEdge_adj'],
                                                                           datas['edgeFace_adj'],
                                                                           datas['edgeVert_adj'],
                                                                           datas['fef_adj'])

            topo_seq = compute_topoSeq(faceEdge_adj, edgeVert_adj)

            data['topo_seq'] = topo_seq
            data['faceEdge_adj'] = faceEdge_adj
            data['edgeFace_adj'] = edgeFace_adj
            data['edgeVert_adj'] = edgeVert_adj
            data['fef_adj'] = fef_adj

            os.makedirs(os.path.join('data_process/TopoDatasets/furniture', data_type), exist_ok=True)
            with open(os.path.join('data_process/TopoDatasets/furniture', data_type, data['name'] + '.pkl'), 'wb') as f:
                pickle.dump(data, f)
        elif option == 'deepcad':
            data = {'name': path.split('/')[-1].replace('.pkl', '')}

            faceEdge_adj, edgeFace_adj, edgeVert_adj, fef_adj = assign_idx(datas['faceEdge_adj'],
                                                                           datas['edgeFace_adj'],
                                                                           datas['edgeVert_adj'],
                                                                           datas['fef_adj'])

            topo_seq = compute_topoSeq(faceEdge_adj, edgeVert_adj)

            data['topo_seq'] = topo_seq
            data['faceEdge_adj'] = faceEdge_adj
            data['edgeFace_adj'] = edgeFace_adj
            data['edgeVert_adj'] = edgeVert_adj
            data['fef_adj'] = fef_adj
            data['pc'] = datas['pc']

            os.makedirs(os.path.join('data_process/TopoDatasets/deepcad', data_type, path.split('/')[-2]), exist_ok=True)
            with open(os.path.join('data_process/TopoDatasets/deepcad', data_type,  path.split('/')[-2], data['name']+'.pkl'), 'wb') as f:
                pickle.dump(data, f)
        else:
            assert option == 'abc'
            data = {'name': path.split('/')[-1].replace('.pkl', '')}

            faceEdge_adj, edgeFace_adj, edgeVert_adj, fef_adj = assign_idx(datas['faceEdge_adj'],
                                                                           datas['edgeFace_adj'],
                                                                           datas['edgeVert_adj'],
                                                                           datas['fef_adj'])

            topo_seq = compute_topoSeq(faceEdge_adj, edgeVert_adj)

            data['topo_seq'] = topo_seq
            data['faceEdge_adj'] = faceEdge_adj
            data['edgeFace_adj'] = edgeFace_adj
            data['edgeVert_adj'] = edgeVert_adj
            data['fef_adj'] = fef_adj

            os.makedirs(os.path.join('data_process/TopoDatasets/abc', data_type, path.split('/')[-2]), exist_ok=True)
            with open(os.path.join('data_process/TopoDatasets/abc', data_type,  path.split('/')[-2], data['name']+'.pkl'), 'wb') as f:
                pickle.dump(data, f)

        return 1

    with open(os.path.join('data_process', option+'_data_split_6bit.pkl'), 'rb') as tf:
        if data_type == 'train':
            files = pickle.load(tf)['train']
        else:
            files = pickle.load(tf)
            files = files['test']

    valid = 0
    for file in tqdm(files):
        valid += create(file)

    print(valid)


class EdgeVertDataset(torch.utils.data.Dataset):
    def __init__(self, path, args):
        self.data = load_data_with_prefix(path, '.pkl')
        # max_num_edge = 0
        # max_seq_length = 0
        # for file in self.data:
        #     with open(file, "rb") as tf:
        #         data = pickle.load(tf)
        #         length = [len(i) for i in data['topo_seq']]
        #         max_seq_length = max(sum(length) + len(length), max_seq_length)
        #         max_num_edge = max(max_num_edge, data['edgeFace_adj'].shape[0])
        self.max_seq_length = args.max_seq_length
        self.max_num_edge_topo = args.max_num_edge_topo
        self.aug = True
        self.use_cf = args.use_cf
        self.use_pc = args.use_pc

    def swap_in_sublist(self, sublist, swap_prob=0.1):
        arr = np.array(sublist)

        if self.aug and np.random.random() < swap_prob:
            idx1, idx2 = np.random.choice(len(arr)-1, 2, replace=False)

            arr[idx1], arr[idx2] = arr[idx2], arr[idx1]

        return arr.tolist()

    @staticmethod
    def shuffle_idx(faceEdge_adj, edgeFace_adj, edgeVert_adj):

        sorted_face_ids = np.random.permutation(len(faceEdge_adj))
        new_face_idx = np.zeros_like(sorted_face_ids)
        for new_idx, old_idx in enumerate(sorted_face_ids):
            new_face_idx[old_idx] = new_idx

        sorted_edges = []
        for i, (face1, face2) in enumerate(edgeFace_adj):
            new_face1, new_face2 = new_face_idx[face1], new_face_idx[face2]
            sorted_faces = sorted([new_face1, new_face2])
            sorted_edges.append((sorted_faces, i))

        sorted_edges.sort()

        sorted_edge_ids = [edge_id for _, edge_id in sorted_edges]

        new_edge_idx = np.zeros_like(sorted_edge_ids)
        for new_idx, old_idx in enumerate(sorted_edge_ids):
            new_edge_idx[old_idx] = new_idx

        edgeFace_adj = new_face_idx[edgeFace_adj]
        edgeFace_adj = edgeFace_adj[sorted_edge_ids]
        edgeVert_adj = edgeVert_adj[sorted_edge_ids]
        faceEdge_adj = [faceEdge_adj[i] for i in sorted_face_ids]
        faceEdge_adj = [new_edge_idx[i].tolist() for i in faceEdge_adj]

        topo_seq = compute_topoSeq(faceEdge_adj, edgeVert_adj)

        return edgeFace_adj, topo_seq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with open(self.data[idx], "rb") as tf:
            data = pickle.load(tf)

        # data augment
        shuffle_prob = 2
        if self.aug and np.random.random() < shuffle_prob:
            edgeFace_adj, topo_seq = self.shuffle_idx(data['faceEdge_adj'],
                                                      data['edgeFace_adj'],
                                                      data['edgeVert_adj'])
        else:
            edgeFace_adj, topo_seq = data['edgeFace_adj'], data['topo_seq']    # ne*2, List[List[int]]
        topo_seq = [self.swap_in_sublist(sublist) for sublist in topo_seq]

        assert edgeFace_adj.shape[0] <= self.max_num_edge_topo
        edgeFace_adj, edge_mask = pad_zero(edgeFace_adj, max_len=self.max_num_edge_topo)   # max_num_edge*2, max_num_edge

        share_id = calculate_y(edgeFace_adj) * edge_mask

        topo_seq = np.expand_dims(np.array(list(chain.from_iterable(sublist + [-2] for sublist in topo_seq))), axis=-1)
        topo_seq, seq_mask = pad_zero(topo_seq, max_len=self.max_seq_length)        # max_seq_length*1, max_seq_length

        edge_mask = edge_mask.sum(keepdims=True)               # 1
        seq_mask = seq_mask.sum(keepdims=True)                 # 1

        if self.use_cf:
            data_class = text2int[data['name'].split('_')[0]] + 1
            return (torch.from_numpy(edgeFace_adj),            # max_num_edge*2
                    torch.from_numpy(edge_mask),               # 1
                    torch.from_numpy(share_id),                # max_num_edge
                    torch.from_numpy(topo_seq).squeeze(-1),    # max_seq_length
                    torch.from_numpy(seq_mask),                # 1
                    torch.LongTensor([data_class])             # 1
                    )
        else:
            return (torch.from_numpy(edgeFace_adj),            # max_num_edge*2
                    torch.from_numpy(edge_mask),               # 1
                    torch.from_numpy(share_id),                # max_num_edge
                    torch.from_numpy(topo_seq).squeeze(-1),    # max_seq_length
                    torch.from_numpy(seq_mask),                # 1
                    )


class FaceEdgeDataset(torch.utils.data.Dataset):
    def __init__(self, path, args):
        self.data = load_data_with_prefix(path, '.pkl')
        self.max_face = args.max_face
        self.max_edge = args.edge_classes - 1
        self.use_cf = args.use_cf
        self.use_pc = args.use_pc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        with open(self.data[idx], "rb") as tf:
            data = pickle.load(tf)
        fef_adj = data['fef_adj']                                            # nf*nf
        edge_counts = np.sum(fef_adj, axis=1)                                # nf
        sorted_ids = np.argsort(edge_counts)                                 # nf
        fef_adj = fef_adj[sorted_ids][:, sorted_ids]
        assert np.all(fef_adj == fef_adj.transpose(0, 1))
        fef_adj, mask = pad_zero(fef_adj, max_len=self.max_face, dim=1)      # max_face*max_face, max_face

        if self.use_cf:
            data_class = text2int[data['name'].split('_')[0]] + 1
            return torch.from_numpy(fef_adj), torch.from_numpy(mask), torch.LongTensor([data_class])
        else:
            return torch.from_numpy(fef_adj), torch.from_numpy(mask)


if __name__ == '__main__':
    # create_topo_datasets(data_type='train', option='abc')
    create_topo_datasets(data_type='test', option='abc')
