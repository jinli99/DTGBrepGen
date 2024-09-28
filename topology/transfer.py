import numpy as np
import pickle
from collections import defaultdict

"""
nf: number of faces (int)
    - The total number of faces in the brep structure.

ne: number of edges (int)
    - The total number of edges, where each edge connects two vertices and may be shared by two faces.

nv: number of vertices (int)
    - The total number of vertices, where each vertex is a point in 3D space that defines the corners of edges.

edgeFace_adj: numpy.array[ne, 2]
    - This is a 2D array where each row represents an edge, and the two columns correspond to the IDs of the two faces connected by that edge.
    - For example, `edgeFace_adj[i] = [f1, f2]` means the `i`-th edge connects face `f1` and face `f2`.

faceEdge_adj: List[List[int]]
    - This is a nested list where each inner list contains the edge IDs that form the boundary of a specific face.
    - For example, `faceEdge_adj[i] = [e1, e2, e3]` means face `i` is formed by edges `e1`, `e2`, and `e3`.

edgeVert_adj: numpy.array[ne, 2]
    - A 2D array where each row represents an edge, and the two columns represent the IDs of the two vertices that define the edge's endpoints.
    - For example, `edgeVert_adj[i] = [v1, v2]` means the `i`-th edge is bounded by vertices `v1` and `v2`.

vertEdge_adj: List[List[int]]
    - A nested list where each inner list contains the edge IDs connected to a particular vertex.
    - For example, `vertEdge_adj[i] = [e1, e2]` means vertex `i` is connected to edges `e1` and `e2`.

faceVert_adj: List[List[int]]
    - A nested list where each inner list contains the vertex IDs that form the boundary of a specific face.
    - For example, `faceVert_adj[i] = [v1, v2, v3]` means face `i` is defined by vertices `v1`, `v2`, and `v3`.

vertFace_adj: List[List[int]]
    - A nested list where each inner list contains the face IDs that are adjacent to a particular vertex.
    - For example, `vertFace_adj[i] = [f1, f2]` means vertex `i` is connected to faces `f1` and `f2`.

fef_adj: numpy.array[nf, nf]
    - A symmetric 2D array representing the number of shared edges between pairs of faces.
    - For example, `fef_adj[i, j]` gives the number of edges shared by face `i` and face `j`, and because the matrix is symmetric, `fef_adj[i, j] == fef_adj[j, i]`.

"""


def face_edge_trans(edgeFace_adj=None, faceEdge_adj=None):
    """
    Transform between edgeFace_adj and faceEdge_adj representations.

    Args:
    - edgeFace_adj (numpy.array): [ne, 2] array, where each edge is shared by two faces.
    - faceEdge_adj (List[List[int]]): List of edges for each face.

    Returns:
    - If edgeFace_adj is provided: computes and returns faceEdge_adj.
    - If faceEdge_adj is provided: computes and returns edgeFace_adj.
    """

    if edgeFace_adj is not None:
        # edgeFace_adj to faceEdge_adj
        nf = edgeFace_adj.max() + 1
        # ne = edgeFace_adj.shape[0]
        faceEdge_adj = [[] for _ in range(nf)]
        for edge_id, face in enumerate(edgeFace_adj):
            faceEdge_adj[face[0]].append(edge_id)
            faceEdge_adj[face[1]].append(edge_id)
        return faceEdge_adj

    else:
        assert faceEdge_adj is not None
        edge_to_faces = defaultdict(set)
        for face_index, face_edges in enumerate(faceEdge_adj):
            for edge_id in face_edges:
                edge_to_faces[edge_id].add(face_index)
        edgeFace_adj = []
        for edge_id, faces in edge_to_faces.items():
            edgeFace_adj.append(sorted(faces))
        edgeFace_adj_numpy = np.array(edgeFace_adj)
        return edgeFace_adj_numpy


def edge_vert_trans(edgeVert_adj=None, vertEdge_adj=None):
    """
    Transform between edgeVert_adj and vertEdge_adj representations.

    Args:
    - edgeVert_adj (numpy.array): [ne, 2] array, where each edge is bounded by two vertices.
    - vertEdge_adj (List[List[int]]): List of edges for each vertex.

    Returns:
    - If edgeVert_adj is provided: computes and returns vertEdge_adj.
    - If vertEdge_adj is provided: computes and returns edgeVert_adj.
    """

    if edgeVert_adj is not None:
        # edgeVert_adj to vertEdge_adj
        nv = edgeVert_adj.max() + 1
        # ne = edgeFace_adj.shape[0]

        vertEdge_adj = [[] for _ in range(nv)]
        for egde_id, vert in enumerate(edgeVert_adj):
            vertEdge_adj[vert[0]].append(egde_id)
            vertEdge_adj[vert[1]].append(egde_id)

        return vertEdge_adj
    else:
        # vertEdge_adj to edgeVert_adj
        assert vertEdge_adj is not None
        ne = float('-inf')
        edge_to_Vert = defaultdict(list)
        for Vert_index, Vert_edges in enumerate(vertEdge_adj):
            for edge_id in Vert_edges:
                edge_to_Vert[edge_id].append(Vert_index)
                if edge_id > ne:
                    ne = edge_id
        ne += 1
        edgeVert_adj = [None] * ne
        for edge_id, Vert in edge_to_Vert.items():
            edgeVert_adj[edge_id] = sorted(Vert)
        edgeVert_adj_numpy = np.array(edgeVert_adj)
        return edgeVert_adj_numpy


def face_vert_trans(faceVert_adj=None, vertFace_adj=None):
    """
    Transform between faceVert_adj and vertFace_adj representations.

    Args:
    - faceVert_adj (List[List[int]]): List of vertices for each face.
    - vertFace_adj (List[List[int]]): List of faces for each vertex.

    Returns:
    - If faceVert_adj is provided: computes and returns vertFace_adj.
    - If vertFace_adj is provided: computes and returns faceVert_adj.
    """

    if faceVert_adj is not None:
        nv = float('-inf')
        vert_to_face = defaultdict(list)
        for face_index, faces_vert in enumerate(faceVert_adj):
            for vert_id in faces_vert:
                vert_to_face[vert_id].append(face_index)
                if vert_id > nv:
                    nv = vert_id
        nv += 1
        vertFace_adj = [None] * nv
        for vert_id, face in vert_to_face.items():
            vertFace_adj[vert_id] = sorted(face)
        return vertFace_adj
    else:
        assert vertFace_adj is not None
        nf = float('-inf')
        face_to_Vert = defaultdict(list)
        for Vert_index, Vert_faces in enumerate(vertFace_adj):
            for face_id in Vert_faces:
                face_to_Vert[face_id].append(Vert_index)
                if face_id > nf:
                    nf = face_id
        nf += 1
        faceVert_adj = [None] * nf
        for face_id, Vert in face_to_Vert.items():
            faceVert_adj[face_id] = sorted(Vert)
        return faceVert_adj


def faceVert_from_edgeVert(faceEdge_adj, edgeVert_adj):
    """
    Compute the faceVert_adj from the given faceEdge_adj and edgeVert_adj.

    Args:
    - faceEdge_adj (List[List[int]]): List of edges for each face.
    - edgeVert_adj (numpy.array): [ne, 2] array, where each edge is bounded by two vertices.

    Returns:
    - faceVert_adj (List[List[int]]): List of vertices for each face.
    """
    face_to_Vert = defaultdict(list)
    for face_index, face_edge in enumerate(faceEdge_adj):
        for edge_id in face_edge:
            edge_vert = edgeVert_adj[edge_id]
            face_to_Vert[face_index].extend(edge_vert.tolist())
    faceVert_adj = []
    for vert in face_to_Vert.values():
        faceVert_adj.append(sorted(set(vert)))
    return faceVert_adj


def fef_from_faceEdge(faceEdge_adj=None, edgeFace_adj=None):
    """
    Compute the fef_adj from the given faceEdge_adj or edgeFace_adj.

    Args:
    - faceEdge_adj (List[List[int]]): List of edges for each face.
    - edgeFace_adj (numpy.array): [ne, 2] array, where each edge is shared by two faces.

    Returns:
    - fef_adj (numpy.array): [nf, nf] array, representing the number of shared edges between pairs of faces.
    """

    if edgeFace_adj is not None:
        nf = edgeFace_adj.max() + 1
        ne = edgeFace_adj.shape[0]
        fef_adj = np.zeros((nf, nf), dtype=int)
        for edge_id in range(ne):
            face1, face2 = edgeFace_adj[edge_id]
            fef_adj[face1, face2] += 1
            fef_adj[face2, face1] += 1
        return fef_adj
    else:
        assert faceEdge_adj is not None
        nf = len(faceEdge_adj)
        fef_adj = np.zeros((nf, nf), dtype=int)
        edge_to_faces = defaultdict(list)
        for face_index, face_edges in enumerate(faceEdge_adj):
            for edge_id in face_edges:
                edge_to_faces[edge_id].append(face_index)
        for faces in edge_to_faces.values():
            fef_adj[faces[0], faces[1]] += 1
            fef_adj[faces[1], faces[0]] += 1
        return fef_adj


def main():
    with open('data_process/GeomDatasets/furniture_parsed/cabinet/assembly_assembly_0080.pkl', 'rb') as f:
        data = pickle.load(f)
    keys = data.keys()
    edgeFace_adj = data['edgeFace_adj']
    faceEdge_adj = data['faceEdge_adj']
    edgeVert_adj = data['edgeVert_adj']
    vertFace_adj = data['vertexFace']
    fef_adj = data['fef_adj']

    # edgeFace_adj to faceEdge_adj
    faceEdge_adj_transfer = face_edge_trans(edgeFace_adj, None)

    faceEdge_adj_list = [array.tolist() for array in faceEdge_adj]
    are_equal = faceEdge_adj_list == faceEdge_adj_transfer
    print("Are the faceEdge_adj equal?", are_equal)

    # faceEdge_adj to edgeFace_adj
    faceEdge_adj_list = [array.tolist() for array in faceEdge_adj]
    edgeFace_adj_transfer = face_edge_trans(None, faceEdge_adj_list)

    are_equal1 = np.array_equal(edgeFace_adj, edgeFace_adj_transfer)
    print("Are the edgeFace_adj equal?", are_equal1)

    # edgeVert_adj to vertEdge_adj
    vertEdge_adj = edge_vert_trans(edgeVert_adj, None)
    edgeVert_adj_transfer = edge_vert_trans(None, vertEdge_adj)
    are_equal2 = np.array_equal(np.sort(edgeVert_adj), edgeVert_adj_transfer)
    print("Are the edgeVert_adj equal?", are_equal2)

    # vertFace_adj to faceVert_adj
    faceVert_adj = face_vert_trans(None, vertFace_adj)
    vertFace_adj_transfer = face_vert_trans(faceVert_adj, None)
    are_equal3 = vertFace_adj == vertFace_adj_transfer
    print("Are the vertFace_adj equal?", are_equal3)

    # faceVert_adj from the given faceEdge_adj and edgeVert_adj
    faceVert_adj_Compute = faceVert_from_edgeVert(faceEdge_adj, edgeVert_adj)
    are_equal4 = faceVert_adj_Compute == faceVert_adj
    print("Are the faceVert_adj equal?", are_equal4)

    # Compute the fef_adj from the given faceEdge_adj
    fef_adj_fromFaceEdge = fef_from_faceEdge(faceEdge_adj, None)
    are_equal5 = np.array_equal(fef_adj_fromFaceEdge, fef_adj)
    print("Are the fef_adj_fromFaceEdge equal?", are_equal5)

    # Compute the fef_adj from the given edgeFace_adj
    fef_adj_fromEdgeFace = fef_from_faceEdge(None, edgeFace_adj)
    are_equal6 = np.array_equal(fef_adj_fromEdgeFace, fef_adj)
    print("Are the fef_adj_fromEdgeFace equal?", are_equal6)


if __name__ == '__main__':
    main()
