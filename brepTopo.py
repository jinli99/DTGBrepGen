import numpy as np
import random
import pickle
import os
from typing import List, Optional
from tqdm import tqdm
from itertools import combinations
from multiprocessing.pool import Pool
from collections import defaultdict
from utils import check_step_ok


class Edge:
    def __init__(self, idx, face1, face2):
        self.idx = idx
        self.faces = [face1, face2]
        self.verts = [2 * idx, 2 * idx + 1]  # Initialize with unique vertex ids
        self.self_loop = False


class Face:
    def __init__(self, idx):
        self.idx = idx
        self.loops = []  # Each loop is a list of edge ids


class BrepGenerator:
    def __init__(self, edgeFace_adj):
        self.edgeFace_adj = edgeFace_adj
        self.num_faces = edgeFace_adj.max() + 1
        self.edges = [Edge(i, face1, face2) for i, (face1, face2) in enumerate(edgeFace_adj)]
        self.faces = [Face(i) for i in range(self.num_faces)]
        self.faceEdge_adj: List[List[int]] = [[] for _ in range(self.num_faces)]
        for edge_id, (face1, face2) in enumerate(edgeFace_adj):
            self.faceEdge_adj[face1].append(edge_id)
            self.faceEdge_adj[face2].append(edge_id)
        self.vert_flag = np.arange(2*edgeFace_adj.shape[0])
        self.set_flag = {i: {i} for i in range(2*edgeFace_adj.shape[0])}
        self.edgeVert_adj = None

    def edge_choice(self, edge, edges_rest, global_feature):
        """This should be a trained model. For this example, we use a simple random selection as a placeholder"""

        probabilities = np.random.random(len(edges_rest))
        probabilities /= probabilities.sum()
        return list(zip(edges_rest, probabilities, range(len(edges_rest))))

    def check_topology_constraint(self, vert1, vert2):

        merged_set = self.set_flag[self.vert_flag[vert1]] | self.set_flag[self.vert_flag[vert2]]

        """ Two vertices on the same edge cannot be merged """
        for v in merged_set:
            if v % 2 == 0:
                if v + 1 in merged_set and not self.edges[v//2].self_loop:
                    return False
            else:
                if v - 1 in merged_set and not self.edges[v//2].self_loop:
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

        if len(self.set_flag[set1]) < len(self.set_flag[set2]):
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

    def check_loop_end(self, current_loop, edges_rest, start_vert, last_edge, connect_vert, face_id):

        if len(current_loop) < 2:
            return False
        if len(edges_rest) < 2:
            return False
        if self.find_merge_vert(self.edges[current_loop[0]].verts[start_vert], face_id):
            return False
        if not self.check_topology_constraint(self.edges[last_edge].verts[connect_vert],
                                              self.edges[current_loop[0]].verts[start_vert]):
            return False
        return True

    @staticmethod
    def compute_piece_edgeNum(piece_edges):
        return sum([len(x) for x in piece_edges])

    def generate_face_topology(self, face_id):
        face = self.faces[face_id]
        edges_rest = self.faceEdge_adj[face_id].copy()
        global_feature = self.compute_global_feature()

        """Connect already connected edges"""
        piece_edges = []
        start_end_verts = []
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

            piece_edge = [edge]
            start_end_vert = [2 * edge, 2 * edge + 1]

            current_edge = edge
            while merge_list_all[current_edge][start_end_vert[0] % 2]:
                idx = merge_list_all[current_edge][start_end_vert[0] % 2]
                assert len(idx) == 1
                idx = idx[0]
                current_edge = idx // 2
                if current_edge == piece_edge[-1]:
                    face.loops.append(piece_edge)
                    loop_flag = True
                    break
                else:
                    piece_edge.insert(0, current_edge)
                    start_end_vert[0] = 2 * current_edge + 1 - idx % 2
                    processed_edges.append(current_edge)

            if not loop_flag:
                current_edge = edge
                while merge_list_all[current_edge][start_end_vert[1] % 2]:
                    idx = merge_list_all[current_edge][start_end_vert[1] % 2]
                    assert len(idx) == 1
                    idx = idx[0]
                    current_edge = idx // 2
                    if current_edge == piece_edge[0]:
                        face.loops.append(piece_edge)
                        loop_flag = True
                        break
                    else:
                        piece_edge.append(current_edge)
                        start_end_vert[1] = 2 * current_edge + 1 - idx % 2
                        processed_edges.append(current_edge)

            if not loop_flag:
                piece_edges.append(piece_edge)
                start_end_verts.append(start_end_vert)

        if not piece_edges:
            return True

        """Connect piece edges"""
        edges_rest = [(edge_piece, idx) for idx, edge_piece in enumerate(piece_edges)]
        loops = []

        while edges_rest:
            current_loop = []
            end_vert = None  # Use odd vertex for the first edge
            start_vert = None
            loop_end = False
            while True:
                if not current_loop:
                    # If it's the first edge of the loop, randomly choose one
                    piece_edge, start_idx = edges_rest.pop(random.randint(0, len(edges_rest) - 1))
                    current_loop.append(piece_edge)
                    start_vert = start_end_verts[start_idx][0]
                    end_vert = start_end_verts[start_idx][1]
                else:

                    # find new vertex
                    last_piece = current_loop[-1]
                    choices = self.edge_choice(last_piece, edges_rest + [(-1, None)], global_feature)

                    for piece_edge, _, rest_idx in sorted(choices, key=lambda x: x[1], reverse=True):
                        origin_idx = piece_edge[-1]
                        piece_edge = piece_edge[0]

                        if piece_edge == -1:
                            if self.compute_piece_edgeNum(current_loop) < 2:
                                continue
                            if sum([len(x) for x, _ in edges_rest]) < 2:
                                continue
                            if self.check_topology_constraint(end_vert, start_vert):
                                loop_end = True
                                break
                        else:
                            # Try connecting with odd vertex first, then even vertex
                            for connect_vert in [0, 1]:
                                if self.check_topology_constraint(end_vert,
                                                                  start_end_verts[origin_idx][connect_vert]):
                                    self.merge_verts(end_vert, start_end_verts[origin_idx][connect_vert])
                                    current_loop.append(piece_edge if connect_vert == 0 else piece_edge[::-1])
                                    del edges_rest[rest_idx]
                                    end_vert = start_end_verts[origin_idx][1 - connect_vert]
                                    break
                            else:
                                continue
                            break
                    else:
                        print("Construct Topology Failed!")
                        return False

                if loop_end or not edges_rest:
                    # If loop is ended or all edges are used, end the current loop
                    if self.compute_piece_edgeNum(current_loop) == 1:
                        assert not edges_rest and start_vert == end_vert-1
                        self.edges[start_vert//2].self_loop = True
                    if self.check_topology_constraint(start_vert, end_vert):
                        self.merge_verts(start_vert, end_vert)
                        loops.append([item for sublist in current_loop for item in sublist])
                    else:
                        print("Construct Topology Failed!")
                        return False
                    break

        face.loops += loops
        return True

    # def generate_face_topology(self, face_id, try_times=5):
    #     face = self.faces[face_id]
    #     edges_rest = self.faceEdge_adj[face_id].copy()
    #     global_feature = self.compute_global_feature()
    #
    #     """Connect already connected edges"""
    #     piece_edges = []
    #     start_end_verts = []
    #     merge_list_all = {}
    #     for edge in edges_rest:
    #         merge_list_all[edge] = [self.find_merge_vert(2*edge, face_id), self.find_merge_vert(2*edge+1, face_id)]
    #
    #     processed_edges = []
    #     for edge in edges_rest:
    #
    #         loop_flag = False
    #
    #         if edge in processed_edges:
    #             continue
    #
    #         processed_edges.append(edge)
    #
    #         piece_edge = [edge]
    #         start_end_vert = [2*edge, 2*edge+1]
    #
    #         current_edge = edge
    #         while merge_list_all[current_edge][start_end_vert[0] % 2]:
    #             idx = merge_list_all[current_edge][start_end_vert[0] % 2]
    #             assert len(idx) == 1
    #             idx = idx[0]
    #             current_edge = idx // 2
    #             if current_edge == piece_edge[-1]:
    #                 face.loops.append(piece_edge)
    #                 loop_flag = True
    #                 break
    #             else:
    #                 piece_edge.insert(0, current_edge)
    #                 start_end_vert[0] = 2*current_edge + 1 - idx % 2
    #                 processed_edges.append(current_edge)
    #
    #         if not loop_flag:
    #             current_edge = edge
    #             while merge_list_all[current_edge][start_end_vert[1] % 2]:
    #                 idx = merge_list_all[current_edge][start_end_vert[1] % 2]
    #                 assert len(idx) == 1
    #                 idx = idx[0]
    #                 current_edge = idx // 2
    #                 if current_edge == piece_edge[0]:
    #                     face.loops.append(piece_edge)
    #                     loop_flag = True
    #                     break
    #                 else:
    #                     piece_edge.append(current_edge)
    #                     start_end_vert[1] = 2*current_edge + 1 - idx % 2
    #                     processed_edges.append(current_edge)
    #
    #         if not loop_flag:
    #             piece_edges.append(piece_edge)
    #             start_end_verts.append(start_end_vert)
    #
    #     if not piece_edges:
    #         return True
    #
    #     """Connect piece edges"""
    #     topo_succeed = False
    #     for try_time in range(try_times):
    #
    #         edges_rest = [(edge_piece, idx) for idx, edge_piece in enumerate(piece_edges)]
    #         loops = []
    #         time_succeed = True
    #
    #         while edges_rest:
    #             current_loop = []
    #             end_vert = None  # Use odd vertex for the first edge
    #             start_vert = None
    #             loop_end = False
    #             while True:
    #                 if not current_loop:
    #                     # If it's the first edge of the loop, randomly choose one
    #                     piece_edge, start_idx = edges_rest.pop(random.randint(0, len(edges_rest) - 1))
    #                     current_loop.append(piece_edge)
    #                     start_vert = start_end_verts[start_idx][0]
    #                     end_vert = start_end_verts[start_idx][1]
    #                 else:
    #
    #                     # find new vertex
    #                     last_piece = current_loop[-1]
    #                     choices = self.edge_choice(last_piece, edges_rest + [(-1, None)], global_feature)
    #
    #                     for piece_edge, _, rest_idx in sorted(choices, key=lambda x: x[1], reverse=True):
    #                         origin_idx = piece_edge[-1]
    #                         piece_edge = piece_edge[0]
    #
    #                         if piece_edge == -1:
    #                             if len(current_loop) < 2:
    #                                 continue
    #                             if len(edges_rest) < 2:
    #                                 continue
    #                             if self.check_topology_constraint(end_vert, start_vert):
    #                                 loop_end = True
    #                                 break
    #                         else:
    #                             # Try connecting with odd vertex first, then even vertex
    #                             for connect_vert in [0, 1]:
    #                                 if self.check_topology_constraint(end_vert,
    #                                                                   start_end_verts[origin_idx][connect_vert]):
    #                                     self.merge_verts(end_vert, start_end_verts[origin_idx][connect_vert])
    #                                     current_loop.append(piece_edge if connect_vert == 0 else piece_edge[::-1])
    #                                     del edges_rest[rest_idx]
    #                                     end_vert = start_end_verts[origin_idx][1 - connect_vert]
    #                                     break
    #                             else:
    #                                 continue
    #                             break
    #                     else:
    #                         print("Time %d Construct Topology Failed!" % try_time)
    #                         time_succeed = False
    #
    #                 if not time_succeed:
    #                     break
    #
    #                 if loop_end or not edges_rest:
    #                     # If loop is ended or all edges are used, end the current loop
    #                     assert len(current_loop) >= 2
    #                     if self.check_topology_constraint(start_vert, end_vert):
    #                         self.merge_verts(start_vert, end_vert)
    #                         loops.append([item for sublist in current_loop for item in sublist])
    #                     else:
    #                         time_succeed = False
    #                     break
    #
    #             if not time_succeed:
    #                 break
    #             elif not edges_rest:
    #                 topo_succeed = True
    #
    #         if topo_succeed:
    #             face.loops += loops
    #             return True
    #         else:
    #             print("Time %d Construct Topology Failed!" % try_time)
    #
    #     return False


    # def generate_face_topology(self, face_id):
    #     face = self.faces[face_id]
    #     edges_rest = self.faceEdge_adj[face_id].copy()
    #     global_feature = self.compute_global_feature()
    #
    #     while edges_rest:     # Ensure there are at least 2 edges to form a loop
    #         current_loop = []
    #         connect_vert = 1  # Use odd vertex for the first edge
    #         start_vert = 0
    #         loop_end = False
    #         while True:
    #             if not current_loop:
    #                 # If it's the first edge of the loop, randomly choose one
    #                 edge = random.choice(edges_rest)
    #                 current_loop.append(edge)
    #                 edges_rest.remove(edge)
    #             else:
    #                 last_edge = current_loop[-1]
    #
    #                 # Merge marked vertices first
    #                 merge_list = self.find_merge_vert(self.edges[last_edge].verts[connect_vert], face_id)
    #                 if len(merge_list) > 1:
    #                     print("Construct Topology Failed!")
    #                     return False
    #                 elif len(merge_list) == 1:
    #                     edge = merge_list[0] // 2
    #                     assert self.check_topology_constraint(self.edges[last_edge].verts[connect_vert],
    #                                                           self.edges[edge].verts[merge_list[0] % 2])
    #                     if edge == current_loop[0]:
    #                         loop_end = True
    #                     else:
    #                         current_loop.append(edge)
    #                         edges_rest.remove(edge)
    #                         connect_vert = 1 - merge_list[0] % 2
    #                 else:
    #                     # find new vertex
    #                     choices = self.edge_choice(last_edge, edges_rest + [-1], global_feature)
    #                     for edge, _, _ in sorted(choices, key=lambda x: x[1], reverse=True):
    #                         if edge == -1:
    #                             if self.check_loop_end(
    #                                     current_loop, edges_rest, start_vert, last_edge, connect_vert, face_id):
    #                                 loop_end = True
    #                                 break
    #                         else:
    #                             # Try connecting with odd vertex first, then even vertex
    #                             for connect_vert_ in [1, 0]:
    #                                 if self.check_topology_constraint(self.edges[last_edge].verts[connect_vert],
    #                                                                   self.edges[edge].verts[connect_vert_]):
    #                                     self.merge_verts(self.edges[last_edge].verts[connect_vert],
    #                                                      self.edges[edge].verts[connect_vert_])
    #                                     current_loop.append(edge)
    #                                     edges_rest.remove(edge)
    #                                     connect_vert = 1 - connect_vert_
    #                                     break
    #                             else:
    #                                 continue
    #                             break
    #                     else:
    #                         print("Construct Topology Failed!")
    #                         return False
    #
    #             if loop_end or not edges_rest:
    #                 # If loop is ended or all edges are used, end the current loop
    #                 assert len(current_loop) >= 2
    #                 self.merge_verts(self.edges[current_loop[-1]].verts[connect_vert], self.edges[current_loop[0]].verts[start_vert])
    #                 face.loops.append(current_loop)
    #                 break

    def compute_global_feature(self):
        return 0

    def generate(self):
        for face_id in range(self.num_faces):
            if not self.generate_face_topology(face_id):
                print("Failed to generate topology in face %d!" % face_id)
                return False

        self.compute_edgeVert()
        self.split_loop_edge()
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

    def split_loop_edge(self):
        def update_loops(loops, edge_idx, new_id):
            for i, loop in enumerate(loops):
                if loop == [edge_idx]:
                    loops[i] = [edge_idx, new_id]
                    return
            raise AssertionError("Edge not found in any loop")

        for edge in self.edges:
            if not edge.self_loop:
                continue
            new_vert_id = self.edgeVert_adj.max() + 1
            new_edge_id = len(self.edges)
            assert self.edgeVert_adj[edge.idx, 0] == self.edgeVert_adj[edge.idx, 1]
            face1, face2 = edge.faces
            self.edges.append(Edge(new_edge_id, face1, face2))
            self.edgeFace_adj = np.append(self.edgeFace_adj, np.array([[face1, face2]]), axis=0)
            self.edgeVert_adj[edge.idx, 1] = new_vert_id
            self.edgeVert_adj = np.append(
                self.edgeVert_adj, np.array([[self.edgeVert_adj[edge.idx, 0], new_vert_id]]), axis=0)
            self.faceEdge_adj[face1].append(new_edge_id)
            self.faceEdge_adj[face2].append(new_edge_id)
            update_loops(self.faces[face1].loops, edge.idx, new_edge_id)
            update_loops(self.faces[face2].loops, edge.idx, new_edge_id)
            edge.self_loop = False

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


class TopoManager:
    def __init__(self, edgeFace_adj: Optional[np.ndarray] = None,
                 faceEdge_adj: Optional[List[List[int]]] = None,
                 edgeVert_adj: Optional[np.ndarray] = None):
        self.edgeFace_adj = edgeFace_adj
        self.faceEdge_adj = faceEdge_adj
        self.edgeVert_adj = edgeVert_adj


def test_valid():
    def topo_generate(path):

        if not check_step_ok(path):
            return 0

        with open(os.path.join('data_process/furniture_parsed', path), 'rb') as f:
            data = pickle.load(f)
        for try_time in range(5):
            generator = BrepGenerator(data['edgeCorner_adj'])
            if generator.generate():
                print("Construct Brep Topology succeed at time %d!" % try_time)
                return 1
        print("Construct Brep Topology Failed!")
        return 0

    with open('data_process/furniture_data_split_6bit.pkl', 'rb') as tf:
        files = pickle.load(tf)['train']

    valid = 0
    for file in tqdm(files):
        valid += topo_generate(file)

    print(valid)


def main():
    # """Usage example"""
    # # edgeFace_adj = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])  # This is an example of a tetrahedron
    #
    # edgeFace_adj = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 5], [2, 5], [3, 5], [4, 5], [1, 2], [2, 3], [3, 4],
    #                          [1, 4]])
    #
    # for try_time in range(5):
    #     generator = BrepGenerator(edgeFace_adj)
    #     if generator.generate():
    #         print("Construct Brep Topology succeed at time %d!" % try_time)
    #         break
    # else:
    #     print("Construct Brep Topology Failed!")

    test_valid()


if __name__ == "__main__":
    main()
