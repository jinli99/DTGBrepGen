import numpy as np


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
        pass
    else:
        assert faceEdge_adj is not None
        pass


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
        pass
    else:
        assert vertEdge_adj is not None
        pass


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
        pass
    else:
        assert vertFace_adj is not None
        pass


def faceVert_from_edgeVert(faceEdge_adj, edgeVert_adj):
    """
    Compute the faceVert_adj from the given faceEdge_adj and edgeVert_adj.

    Args:
    - faceEdge_adj (List[List[int]]): List of edges for each face.
    - edgeVert_adj (numpy.array): [ne, 2] array, where each edge is bounded by two vertices.

    Returns:
    - faceVert_adj (List[List[int]]): List of vertices for each face.
    """


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
        pass
    else:
        assert faceEdge_adj is not None
        pass

