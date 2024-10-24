import os
import pickle
import numpy as np
import torch
from itertools import chain
from topology.datasets import opposite_idx
from typing import List
from collections import defaultdict
from tqdm import tqdm
from model import FaceEdgeModel, TopoSeqModel
from utils import pad_zero

text2int = {'uncond':0,
            'bathtub':1,
            'bed':2,
            'bench':3,
            'bookshelf':4,
            'cabinet':5,
            'chair':6,
            'couch':7,
            'lamp':8,
            'sofa':9,
            'table':10
            }


class Edge:
    def __init__(self, idx, face1, face2):
        self.idx = idx
        self.faces = [face1, face2]
        self.verts = [2 * idx, 2 * idx + 1]  # Initialize with unique vertex ids


class Face:
    def __init__(self, idx):
        self.idx = idx
        self.loops = []                     # Each loop is a list of edge ids


class PieceEdges:
    def __init__(self):
        self.id = -1
        self.edge_seq = []
        self.vert_seq = []
        self.loop = False

    def compute_idx(self):
        self.id = min(self.edge_seq)

    def loop_seq(self, merge_list_all):
        assert self.loop
        vert_seq = [2*self.id]
        while True:
            current_vert = vert_seq[-1]
            next_vert = merge_list_all[current_vert//2][opposite_idx(current_vert) % 2][0]
            assert next_vert//2 in self.edge_seq
            if next_vert == vert_seq[0]:
                break
            else:
                vert_seq.append(next_vert)
        assert len(vert_seq) == len(self.edge_seq)
        return vert_seq

    def min_seq(self, merge_list_all):
        vert_seq = [2 * self.id]
        while True:
            current_vert = vert_seq[-1]
            next_vert = merge_list_all[current_vert//2][opposite_idx(current_vert) % 2]
            if next_vert:
                vert_seq.append(next_vert[0])
            else:
                break
        rest_seq = [2 * self.id]
        while True:
            current_vert = rest_seq[0]
            next_vert = merge_list_all[current_vert//2][current_vert % 2]
            if next_vert:
                rest_seq.insert(0, opposite_idx(next_vert[0]))
            else:
                break
        rest_seq.pop()
        return vert_seq, rest_seq


class SeqGenerator:
    def __init__(self, edgeFace_adj):
        self.topo_seq = None
        self.edgeFace_adj = edgeFace_adj
        self.num_faces = edgeFace_adj.max() + 1
        self.faceEdge_adj: List[List[int]] = [[] for _ in range(self.num_faces)]
        for edge_id, (face1, face2) in enumerate(self.edgeFace_adj):
            self.faceEdge_adj[face1].append(edge_id)
            self.faceEdge_adj[face2].append(edge_id)

        self.edges = [Edge(i, face1, face2) for i, (face1, face2) in enumerate(self.edgeFace_adj)]
        self.faces = [Face(i) for i in range(self.num_faces)]
        self.vert_flag = np.arange(2*self.edgeFace_adj.shape[0])
        self.set_flag = {i: {i} for i in range(2*self.edgeFace_adj.shape[0])}
        self.edgeVert_adj = None

    @staticmethod
    def edge_choice(topo_seq, mask, model, class_label):
        """
        Args:
            topo_seq: List[int]
            mask: List[int]
            model: Pre-trained model
            class_label:
        Returns:
        """
        device = next(model.parameters()).device
        with torch.no_grad():
            # b*ns
            topo_seq = torch.tensor(topo_seq, dtype=torch.long, device=device).unsqueeze(0)
            seq_mask = torch.ones((topo_seq.shape[0], topo_seq.shape[1]), device=device, dtype=torch.bool)    # b*ns
            logits = model.sample(topo_seq, seq_mask, mask, class_label)      # len(mask)
            logits = torch.softmax(logits, dim=0)                             # len(mask)
            return logits

    def check_topology_constraint(self, vert1, vert2):

        merged_set = self.set_flag[self.vert_flag[vert1]] | self.set_flag[self.vert_flag[vert2]]

        """ Two vertices on the same edge cannot be merged """
        for v in merged_set:
            if v % 2 == 0:
                if v + 1 in merged_set:
                    return False
            else:
                if v - 1 in merged_set:
                    return False

        """At most two points on the same face can be merged into the same point"""
        face_merge_count = defaultdict(int)
        for v in merged_set:
            edge_idx = v // 2
            for face in self.edges[edge_idx].faces:
                face_merge_count[face] += 1
                if face_merge_count[face] > 2:
                    return False

        return True

    def merge_verts(self, vert1, vert2):

        set1 = self.vert_flag[vert1]
        set2 = self.vert_flag[vert2]

        if set1 == set2:
            return

        if min(self.set_flag[set1]) < min(self.set_flag[set2]):
            set1, set2 = set2, set1

        self.set_flag[set1].update(self.set_flag[set2])

        self.vert_flag[self.vert_flag == set2] = set1

        del self.set_flag[set2]

    def find_merge_vert(self, vert, face_id):
        edges = self.faceEdge_adj[face_id].copy()
        vert_rest = [2*i for i in edges] + [2*i+1 for i in edges]
        merge_list = []
        for vert_id in vert_rest:
            if vert_id != vert and self.vert_flag[vert_id] == self.vert_flag[vert]:
                merge_list.append(vert_id)

        return merge_list

    def generate_face_topology(self, face_id, topo_seq, model, class_label):
        face = self.faces[face_id]
        edges_rest = self.faceEdge_adj[face_id].copy()

        """Connect already connected edges"""
        piece_edges = []
        merge_list_all = {}
        for edge in edges_rest:
            merge_list_all[edge] = [self.find_merge_vert(2 * edge, face_id),
                                    self.find_merge_vert(2 * edge + 1, face_id)]

        processed_edges = []
        for edge in edges_rest:

            loop_flag = False

            if edge in processed_edges:
                continue

            processed_edges.append(edge)

            piece_edge = PieceEdges()
            piece_edge.edge_seq = [edge]
            piece_edge.vert_seq = [2 * edge]

            current_edge = edge
            while merge_list_all[current_edge][piece_edge.vert_seq[0] % 2]:
                idx = merge_list_all[current_edge][piece_edge.vert_seq[0] % 2]
                assert len(idx) == 1
                idx = idx[0]
                current_edge = idx // 2
                if current_edge == piece_edge.edge_seq[-1]:
                    piece_edge.loop = True
                    loop_flag = True
                    break
                else:
                    piece_edge.edge_seq.insert(0, current_edge)
                    piece_edge.vert_seq.insert(0, opposite_idx(idx))
                    processed_edges.append(current_edge)

            if not loop_flag:
                current_edge = edge
                while merge_list_all[current_edge][opposite_idx(piece_edge.vert_seq[-1]) % 2]:
                    idx = merge_list_all[current_edge][opposite_idx(piece_edge.vert_seq[-1]) % 2]
                    assert len(idx) == 1
                    idx = idx[0]
                    current_edge = idx // 2
                    if current_edge == piece_edge.edge_seq[0]:
                        piece_edge.loop = True
                        break
                    else:
                        piece_edge.edge_seq.append(current_edge)
                        piece_edge.vert_seq.append(idx)
                        processed_edges.append(current_edge)

            piece_edges.append(piece_edge)

        for piece_edge in piece_edges:
            piece_edge.compute_idx()

        piece_edges = sorted(piece_edges, key=lambda x: x.id)

        """Connect piece edges"""
        loops = []
        while piece_edges:
            piece_edge = piece_edges.pop(0)

            if piece_edge.loop:
                loop_seq = piece_edge.loop_seq(merge_list_all)
                loops.append(piece_edge.edge_seq)
                topo_seq += loop_seq + [-1]
                continue

            start_vert = piece_edge.vert_seq[0]
            vert_seq, rest_seq = piece_edge.min_seq(merge_list_all)
            if rest_seq:
                assert rest_seq[0] == start_vert
            topo_seq += vert_seq
            current_loop = vert_seq.copy()

            # form a loop
            loop_flag = False
            while True:
                mask = [item+2 for pair in zip([i.vert_seq[0] for i in piece_edges],
                                               [opposite_idx(i.vert_seq[-1]) for i in piece_edges])
                        for item in pair] + [start_vert+2, 1]
                logits = self.edge_choice(topo_seq, mask, model, class_label)
                indices = [i for i in range(len(logits))]

                # find next edge
                while len(logits) > 0:

                    sampled_index = torch.multinomial(logits, num_samples=1).item()
                    idx = indices[sampled_index]
                    indices.pop(sampled_index)
                    logits = torch.cat([logits[:sampled_index], logits[sampled_index+1:]])
                    if len(logits) > 0:
                        logits /= logits.sum()

                    if idx >= len(mask) - 2:
                        if len(current_loop) + len(rest_seq) < 2:
                            continue
                        if piece_edges and sum([len(i.edge_seq) for i in piece_edges]) < 2:
                            continue
                        if not self.check_topology_constraint(start_vert, opposite_idx(current_loop[-1])):
                            continue
                        self.merge_verts(start_vert, opposite_idx(current_loop[-1]))
                        topo_seq += rest_seq + [-1]
                        loops.append([i // 2 for i in current_loop] + [i // 2 for i in rest_seq])
                        loop_flag = True
                        break
                    else:
                        if not self.check_topology_constraint(opposite_idx(current_loop[-1]), mask[idx]-2):
                            continue
                        self.merge_verts(opposite_idx(current_loop[-1]), mask[idx]-2)
                        if not idx % 2:
                            topo_seq += piece_edges[idx // 2].vert_seq
                            current_loop += piece_edges[idx // 2].vert_seq
                        else:
                            topo_seq += [opposite_idx(i) for i in piece_edges[idx // 2].vert_seq[::-1]]
                            current_loop += [opposite_idx(i) for i in piece_edges[idx // 2].vert_seq[::-1]]
                        piece_edges.pop(idx // 2)
                        break
                else:
                    return False, None

                if loop_flag:
                    break

        face.loops = loops
        return True, topo_seq + [-2]

    def generate(self, model, class_label):
        topo_seq = []
        for face_id in range(self.num_faces):
            flag, topo_seq = self.generate_face_topology(face_id, topo_seq, model, class_label)
            if not flag:
                # print("Failed to generate topology in face %d!" % face_id)
                return False
        self.topo_seq = topo_seq
        self.compute_edgeVert()
        return self.check_total_topology()

    def compute_edgeVert(self):

        new_id_map = {}
        next_id = 0

        for set_id in self.set_flag.keys():
            assert set_id not in new_id_map
            new_id_map[set_id] = next_id
            next_id += 1

        edgeVert_adj = np.zeros((len(self.edges), 2), dtype=int)
        for edge in self.edges:
            v1, v2 = edge.verts
            new_v1 = new_id_map[self.vert_flag[v1]]
            new_v2 = new_id_map[self.vert_flag[v2]]
            edge.verts = [new_v1, new_v2]
            edgeVert_adj[edge.idx] = np.array([new_v1, new_v2])

        self.edgeVert_adj = edgeVert_adj

        self.vert_flag = None
        self.set_flag = None

    def check_total_topology(self):

        """Two vertices on the same edge cannot be merged"""
        if not np.abs(self.edgeVert_adj[:, 0] - self.edgeVert_adj[:, 1]).min() > 0:
            print("\033[31mTwo vertices on the same edge cannot be merged: FAIL\033[0m")
            return False

        """Each point is connected to at least two edges"""
        num_vertices = np.max(self.edgeVert_adj) + 1
        vertex_edge_count = np.zeros(num_vertices, dtype=int)
        np.add.at(vertex_edge_count, self.edgeVert_adj.flatten(), 1)
        isolated_vertices = np.where(vertex_edge_count < 2)[0]
        if len(isolated_vertices) > 0:
            print("\033[31mEach point is connected to at least two edges: FAIL\033[0m")
            return False

        """The number of edges of each face is equal to the number of vertices"""
        for face_edges in self.faceEdge_adj:
            num_edges = len(face_edges)
            vertices = set()
            for edge_id in face_edges:
                vertices.update(self.edgeVert_adj[edge_id])
            num_vertices = len(vertices)
            if num_edges != num_vertices:
                print("\033[31mThe number of edges of each face is equal to the number of vertices: FAIL\033[0m")
                return False

        return True


"""test seq_model !"""
def test_valid():

    cf = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TopoSeqModel(max_num_edge=96, max_seq_length=260, max_face=32, use_cf=cf)
    model.load_state_dict(torch.load('checkpoints/furniture/topo_Seq/epoch_3000.pt'), strict=False)
    model = model.to(device).eval()

    topo_unique = np.array(0)

    def topo_generate(path, topo_unique):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        # save encoder information
        edgeFace_adj = torch.from_numpy(data['edgeFace_adj']).unsqueeze(0).to(device)

        # new_edge_ids = torch.randperm(edgeFace_adj.size(0), device=edgeFace_adj.device)
        # inverse_idx_map = torch.empty_like(new_edge_ids)
        # for i, idx in enumerate(new_edge_ids):
        #     inverse_idx_map[idx] = i
        # edgeFace_adj = edgeFace_adj[inverse_idx_map]

        if cf:
            class_label = torch.LongTensor([text2int[data['name'].split('_')[0]]]).to(device).reshape(-1, 1)
        else:
            class_label = None
        model.save_cache(edgeFace_adj=edgeFace_adj,
                         edge_mask=torch.ones((edgeFace_adj.shape[0], edgeFace_adj.shape[1]), device=device, dtype=torch.bool),
                         class_label=class_label)
        for try_time in range(10):
            generator = SeqGenerator(data['edgeFace_adj'])
            if generator.generate(model, class_label):
                print("Construct Brep Topology succeed at time %d!" % try_time)
                if generator.topo_seq != list(chain.from_iterable(sublist + [-2] for sublist in data['topo_seq'])):
                    topo_unique += 1
                model.clear_cache()
                return 1
        print("Construct Brep Topology Failed!")
        model.clear_cache()
        return 0

    path_ = 'data_process/TopoDatasets/furniture/test'
    files = os.listdir(path_)
    files = [os.path.join(path_, i) for i in files]

    valid = 0
    for file in tqdm(files):
        valid += topo_generate(file, topo_unique)

    print(valid, topo_unique)

    # def topo_generate(path):
    #
    #     if not check_step_ok(path):
    #         return 0
    #
    #     with open(os.path.join('data_process/furniture_parsed', path), 'rb') as f:
    #         data = pickle.load(f)
    #
    #     generator = BrepGenerator(data['edgeFace_adj'])
    #
    #     # save encoder information
    #     edgeFace_adj = torch.from_numpy(generator.edgeFace_adj).unsqueeze(0).to(device)
    #     model.save_cache(edgeFace_adj=edgeFace_adj, edge_mask=torch.ones((edgeFace_adj.shape[0], edgeFace_adj.shape[1]),
    #                                                                      device=device, dtype=torch.bool))
    #
    #     if generator.generate(model):
    #         print("Construct Brep Topology succeed !")
    #         model.clear_cache()
    #         return 1
    #     print("Construct Brep Topology Failed!")
    #     model.clear_cache()
    #     return 0
    #
    # with open('data_process/furniture_data_split_6bit.pkl', 'rb') as tf:
    #     files = pickle.load(tf)['train']
    #
    # valid = 0
    # for file in tqdm(files):
    #     valid += topo_generate(file)
    #
    # print(valid)

    # path = '../data_process/topoDatasets/furniture/test'
    # datas = os.listdir(path)
    # datas = [os.path.join(path, i) for i in datas]
    # for idx in range(7, len(datas)):
    #     with open(datas[idx], "rb") as tf:
    #         data = pickle.load(tf)
    #     edgeFace_adj, topo_seq = torch.from_numpy(data['edgeFace_adj']), data['topo_seq']  # ne*2, List[List[int]]
    #     topo_seq = torch.tensor(
    #         list(chain.from_iterable(sublist + [-2] for sublist in topo_seq)), device=device).unsqueeze(0)        # b*ns
    #     edgeFace_adj = edgeFace_adj.unsqueeze(0).to(device)      # b*ne*2
    #     edge_mask = torch.ones((edgeFace_adj.shape[0], edgeFace_adj.shape[1]), device=device, dtype=torch.bool)   # b*ne
    #     seq_mask = torch.ones((topo_seq.shape[0], topo_seq.shape[1]), device=device, dtype=torch.bool)            # b*ns
    #
    #     with torch.no_grad():
    #         logits = model(edgeFace_adj, edge_mask, topo_seq, seq_mask)     # b*ns*(ne+2)
    #
    #     probs = torch.softmax(logits, dim=-1)  # b*ns*(ne+2)
    #     pre_seq = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.size(0), probs.size(1)) - 2
    #     jing = (pre_seq[:, :-1] - topo_seq[:, 1:]).abs()
    #     print(idx, jing.max().item())
    #     pass


def test_faceEdge():

    path_ = 'data_process/TopoDatasets/furniture/test'
    files = os.listdir(path_)
    files = [os.path.join(path_, i) for i in files]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    faceEdge_model = FaceEdgeModel(nf=32, num_categories=5)
    faceEdge_model.load_state_dict(torch.load('checkpoints/furniture/topo_faceEdge/epoch_550.pt'), strict=False)
    faceEdge_model = faceEdge_model.to(device).eval()

    seq_model = TopoSeqModel(max_num_edge=144, max_seq_length=391)
    seq_model.load_state_dict(torch.load('checkpoints/furniture/topo_Seq/epoch_3000.pt'), strict=False)
    seq_model = seq_model.to(device).eval()

    valid = 0
    with torch.no_grad():
        for file in tqdm(files):

            with open(file, "rb") as tf:
                data = pickle.load(tf)
            fef_adj = data['fef_adj']                                            # nf*nf
            assert np.all(fef_adj == fef_adj.transpose(0, 1))
            fef_adj, mask = pad_zero(fef_adj, max_len=50, dim=1)                 # max_face*max_face, max_face
            fef_adj = torch.from_numpy(fef_adj).to(device).unsqueeze(0)          # 1*nf*nf
            mask = torch.from_numpy(mask).to(device).unsqueeze(0)                # 1*nf

            # true topology
            # edge_indices = torch.triu(fef_adj.squeeze(0), diagonal=1).nonzero(as_tuple=False)
            # num_edges = fef_adj[0, edge_indices[:, 0], edge_indices[:, 1]]
            # edgeFace_adj = edge_indices.repeat_interleave(num_edges, dim=0)      # ne*2
            # seq_model.save_cache(edgeFace_adj=edgeFace_adj.unsqueeze(0),
            #                      edge_mask=torch.ones((1, edgeFace_adj.shape[0]), device=device, dtype=torch.bool))
            # for try_time in range(10):
            #     generator = SeqGenerator(edgeFace_adj.cpu().numpy())
            #     if generator.generate(seq_model):
            #         print("Construct Brep Topology succeed at time %d!" % try_time)
            #         seq_model.clear_cache()
            #         valid += 1
            #         break
            # else:
            #     print("Construct Brep Topology Failed!")
            #     seq_model.clear_cache()

            # prediction topology
            upper_indices = torch.triu_indices(fef_adj.shape[1], fef_adj.shape[1], offset=1)
            fef_adj_upper = fef_adj[:, upper_indices[0], upper_indices[1]]   # 1*seq_len
            adj, mu, logvar = faceEdge_model(fef_adj_upper)                  # 1*seq_len*m, 1*latent_dim, 1*latent_dim
            kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            recon_loss = torch.nn.functional.cross_entropy(adj.reshape(-1, adj.shape[-1]),
                                                           fef_adj_upper.reshape(-1),
                                                           reduction='mean')
            adj_sample = torch.distributions.Categorical(logits=adj).sample()           # 1*seq_len
            print(kl_divergence.item(), recon_loss.item(), (adj_sample - fef_adj_upper).abs().sum().item())
            adj = faceEdge_model.sequence_to_matrix(adj_sample).squeeze(0)              # nf*nf
            non_zero_mask = torch.any(adj != 0, dim=1)
            adj = adj[non_zero_mask][:, non_zero_mask]                                  # nf*nf
            edge_counts = torch.sum(adj, dim=1)
            if edge_counts.sum() > seq_model.max_num_edge or edge_counts.max() > seq_model.max_edge:
                continue
            sorted_ids = torch.argsort(edge_counts)
            adj = adj[sorted_ids][:, sorted_ids]
            edge_indices = torch.triu(adj, diagonal=1).nonzero(as_tuple=False)
            num_edges = adj[edge_indices[:, 0], edge_indices[:, 1]]
            edgeFace_adj = edge_indices.repeat_interleave(num_edges, dim=0)      # ne*2
            seq_model.save_cache(edgeFace_adj=edgeFace_adj.unsqueeze(0),
                                 edge_mask=torch.ones((1, edgeFace_adj.shape[0]), device=device, dtype=torch.bool))
            for try_time in range(10):
                generator = SeqGenerator(edgeFace_adj.cpu().numpy())
                if generator.generate(seq_model):
                    print("Construct Brep Topology succeed at time %d!" % try_time)
                    seq_model.clear_cache()
                    valid += 1
                    break
            else:
                print("Construct Brep Topology Failed!")
                seq_model.clear_cache()
            pass

        print(valid)


def topo_sample():

    num_sample = 320
    batch = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    faceEdge_model = FaceEdgeModel(nf=32, num_categories=5, use_cf=True)
    faceEdge_model.load_state_dict(torch.load('checkpoints/furniture/topo_faceEdge/epoch_2000.pt'), strict=False)
    faceEdge_model = faceEdge_model.to(device).eval()

    seq_model = TopoSeqModel(max_num_edge=96, max_seq_length=260, max_face=32, use_cf=True)
    seq_model.load_state_dict(torch.load('checkpoints/furniture/topo_Seq/epoch_3000.pt'), strict=False)
    seq_model = seq_model.to(device).eval()

    valid = 0
    for _ in tqdm(range(0, num_sample, batch), desc='Processing Samples', total=(num_sample // batch)):

        with torch.no_grad():

            class_label = torch.randint(1, 11, (batch, 1), device=device)
            adj_batch = faceEdge_model.sample(num_samples=batch, class_label=class_label)  # b*nf*nf

            for i in range(batch):
                adj = adj_batch[i]                                                                     # nf*nf
                non_zero_mask = torch.any(adj != 0, dim=1)
                adj = adj[non_zero_mask][:, non_zero_mask]                                             # nf*nf
                edge_counts = torch.sum(adj, dim=1)                                                    # nf
                if edge_counts.max() > seq_model.max_edge:
                    continue
                sorted_ids = torch.argsort(edge_counts)                                                # nf
                adj = adj[sorted_ids][:, sorted_ids]
                edge_indices = torch.triu(adj, diagonal=1).nonzero(as_tuple=False)
                num_edges = adj[edge_indices[:, 0], edge_indices[:, 1]]
                edgeFace_adj = edge_indices.repeat_interleave(num_edges, dim=0)                        # ne*2

                # save encoder information
                seq_model.save_cache(edgeFace_adj=edgeFace_adj.unsqueeze(0),
                                     edge_mask=torch.ones((1, edgeFace_adj.shape[0]), device=device, dtype=torch.bool),
                                     class_label=class_label[[i]])
                for try_time in range(10):
                    generator = SeqGenerator(edgeFace_adj.cpu().numpy())
                    if generator.generate(seq_model, class_label[[i]]):
                        print("Construct Brep Topology succeed at time %d!" % try_time)
                        seq_model.clear_cache()
                        valid += 1
                        break
                else:
                    print("Construct Brep Topology Failed!")
                    seq_model.clear_cache()

    print(valid)


if __name__ == '__main__':

    # test_valid()
    topo_sample()
    # test_faceEdge()
