import torch
import random
import string
import numpy as np


def pad_and_stack(inputs):

    batch_size = len(inputs)
    fixed_shape = list(inputs[0].shape[1:])

    max_n = max(tensor.shape[0] for tensor in inputs)

    padded_shape = [batch_size, max_n] + fixed_shape
    padded_surfPos = torch.zeros(padded_shape, dtype=inputs[0].dtype, device=inputs[0].device)
    node_mask = torch.zeros([batch_size, max_n], dtype=torch.bool, device=inputs[0].device)

    for i, tensor in enumerate(inputs):
        current_n = tensor.shape[0]
        padded_surfPos[i, :current_n, ...] = tensor
        node_mask[i, :current_n] = True

    return padded_surfPos, node_mask


def pad_zero(x, max_len, dim=0):
    """padding for x with shape (num_faces, dim1, ...) and edge with shape(num_faces, num_faces)"""

    assert x.shape[0] <= max_len

    if dim == 0:
        total = np.zeros((max_len, *x.shape[1:]), dtype=x.dtype)
        total[:x.shape[0]] = x
        mask = np.zeros(max_len, dtype=np.bool_)
        mask[:x.shape[0]] = True
    elif dim == 1:
        assert len(x.shape) == 2
        total = np.zeros((max_len, max_len), dtype=x.dtype)
        total[:x.shape[0], :x.shape[0]] = x
        mask = np.zeros(max_len, dtype=np.bool_)
        mask[:x.shape[0]] = True
    else:
        raise ValueError

    return total, mask


def construct_edgeFace_adj(edge_face_topo, node_mask=None):   # b*n*n, b*n
    edgeFace_adj = []

    # Get the batch size and dimensions
    b, n, _ = edge_face_topo.shape

    # Create a mask to select the upper triangle without the diagonal
    triu_mask = torch.triu(torch.ones((n, n), device=node_mask.device), diagonal=1).bool()   # n*n

    # Loop through each batch
    for m in range(b):
        # Get the upper triangle elements (excluding the diagonal)
        edge_counts = edge_face_topo[m][triu_mask]

        # Get the indices of the upper triangle
        indices = torch.nonzero(triu_mask, as_tuple=False)

        # Apply node_mask if provided
        if node_mask is not None:
            valid_nodes = node_mask[m]
            valid_indices = valid_nodes[indices[edge_counts>0]].all(dim=1)
            assert valid_indices.all(), f"Invalid edges in batch {m}"

        # Repeat indices based on the edge counts
        repeated_indices = indices.repeat_interleave(edge_counts, dim=0)

        # Append the edges to the list
        edgeFace_adj.append(repeated_indices)

    return edgeFace_adj


def construct_feTopo(edgeFace_adj):    # ne*2
    num_faces = edgeFace_adj.max()+1
    fe_topo = torch.zeros((num_faces, num_faces), device=edgeFace_adj.device, dtype=edgeFace_adj.dtype)   # nf*nf
    for face_idx in edgeFace_adj:
        fe_topo[face_idx[0], face_idx[1]] += 1
        fe_topo[face_idx[1], face_idx[0]] += 1

    assert torch.equal(fe_topo, fe_topo.transpose(0, 1))

    return fe_topo    # nf*nf


def generate_random_string(length):
    characters = string.ascii_letters + string.digits  # You can include other characters if needed
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string


def sort_box(box):
    min_corner, _ = torch.min(box.view(2, 3), dim=0)
    max_corner, _ = torch.max(box.view(2, 3), dim=0)
    return torch.stack((min_corner, max_corner))


def calc_bbox_diff(bounding_boxes):
    n = bounding_boxes.shape[0]
    diff_mat = torch.zeros((n, n), device=bounding_boxes.device)

    # Helper function to compute the volume of a box
    def box_volume(box):
        lengths = box[1] - box[0]
        return torch.prod(lengths)

    # Helper function to compute the intersection volume of two boxes
    def intersection_volume(box1_, box2_):
        min_corner = torch.max(box1_[0], box2_[0])
        max_corner = torch.min(box1_[1], box2_[1])
        intersection = torch.clamp(max_corner - min_corner, min=0)
        return torch.prod(intersection)

    # Helper function to compute the distance between two boxes
    def box_distance(box1_, box2_):
        max_min_diff = torch.max(box1_[0] - box2_[1], box2_[0] - box1_[1])
        return torch.norm(torch.clamp(max_min_diff, min=0))

    for i in range(n):
        for j in range(i + 1, n):
            box1 = sort_box(bounding_boxes[i])  # 2*3
            box2 = sort_box(bounding_boxes[j])

            inter_vol = intersection_volume(box1, box2)

            if inter_vol > 0:
                vol1 = box_volume(box1)
                vol2 = box_volume(box2)
                diff_mat[i, j] = diff_mat[j, i] = - (inter_vol / torch.min(vol1, vol2))
            else:
                dist = box_distance(box1, box2)
                diff_mat[i, j] = diff_mat[j, i] = dist

    return diff_mat  # n*n


def remove_box_edge(box, edgeFace_adj):  # nf*6, ne*2
    threshold1 = 0.7
    nf = box.shape[0]

    assert edgeFace_adj.max().item() < nf

    diff_mat = calc_bbox_diff(box)  # nf*nf

    vol = [sort_box(b) for b in box]
    vol = torch.tensor([torch.prod(b[1] - b[0]) for b in vol], device=box.device)  # nf

    # Remove close bbox
    remove_box_idx = []
    true_indices = torch.where(diff_mat < -threshold1)
    if len(true_indices[0]) > 0:
        v1, v2 = vol[true_indices[0]], vol[true_indices[1]]
        remove_box_idx = torch.unique(torch.where(v1 < v2, true_indices[0], true_indices[1])).tolist()

    # Remove bbox with no edges
    remove_box_idx += list(set(range(nf)) - set(torch.unique(edgeFace_adj.view(-1)).cpu().numpy().tolist()))
    remove_box_idx = list(set(remove_box_idx))

    # Update edgeFace_adj
    mask = torch.any(torch.isin(edgeFace_adj, torch.tensor(remove_box_idx, device=edgeFace_adj.device)), dim=1)
    edgeFace_adj = edgeFace_adj[~mask]

    # Remove the edges of the two boxes that are far apart
    threshold2 = 0.4
    mask = diff_mat[edgeFace_adj[:, 0], edgeFace_adj[:, 1]] > threshold2  # ne
    edgeFace_adj = edgeFace_adj[~mask]

    # Update box
    box = torch.cat([box[i:i+1] for i in range(nf) if i not in remove_box_idx], dim=0)

    # Update edgeFace_adj
    new_id = sorted(list(set(range(nf)) - set(remove_box_idx)))
    id_map = torch.zeros(nf, dtype=torch.int64, device=edgeFace_adj.device)
    id_map[new_id] = torch.arange(len(new_id), device=edgeFace_adj.device)
    edgeFace_adj = id_map[edgeFace_adj]

    assert set(range(box.shape[0])) == set(torch.unique(edgeFace_adj.view(-1)).cpu().numpy().tolist())

    return box, edgeFace_adj


def remove_short_edge(edge_wcs, threshold=0.1):
    # Calculate the length of each edge
    edge_diff = edge_wcs[:, 1:, :] - edge_wcs[:, :-1, :]  # Differences between consecutive points
    edge_lengths = torch.norm(edge_diff, dim=2).sum(dim=1)  # Sum of distances along each edge

    # Remove short edges from edge_wcs
    keep_edge_idx = torch.where(edge_lengths >= threshold)[0]
    edge_wcs_filtered = edge_wcs[keep_edge_idx]

    return edge_wcs_filtered, keep_edge_idx


def compute_bbox_center_and_size(bbox):  # b*6
    bbox = bbox.reshape(-1, 2, 3)   # b*2*3
    min_xyz, max_xyz = bbox.min(dim=1).values, bbox.max(dim=1).values   # b*3, b*3
    center = (min_xyz + max_xyz) * 0.5   # b*3
    size = max_xyz - min_xyz    # b*3
    return center, size


def ncs2wcs(bbox, ncs):   # n*6, n32*3
    center, size = compute_bbox_center_and_size(bbox)    # n*3, n*3
    max_xyz, min_xyz = ncs.amax(dim=1), ncs.amin(dim=1)   # n*3, n*3
    ratio = (size / (max_xyz - min_xyz)).amin(-1)  # n
    ncs = (ncs - ((min_xyz + max_xyz) * 0.5).unsqueeze(1)) * ratio.unsqueeze(-1).unsqueeze(-1) + center.unsqueeze(1)
    return ncs


def xe_mask(x=None, e=None, node_mask=None, check_sym=True):
    if x is not None:
        x_mask = node_mask.unsqueeze(-1)      # bs, n, 1
        x = x * x_mask

    if e is not None:
        x_mask = node_mask.unsqueeze(-1)      # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)         # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)         # bs, 1, n, 1
        e = e * e_mask1 * e_mask2 * (~torch.eye(e.shape[1], device=e.device).bool().unsqueeze(0).unsqueeze(-1))
        if check_sym:
            if not torch.allclose(e, torch.transpose(e, 1, 2)):
                print("The max and min value of e is", e.max().item(), e.min().item())
                torch.save(e.detach().cpu(), 'bad_edge.pt')
                assert False

    return x, e


def masked_softmax(x, mask, **kwargs):
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)


def edge_reshape_mask(e, mask):
    b, n, _, m = e.shape

    # Step 1: Reshape e to (b*n*n, m)
    e_reshaped = e.view(b * n * n, m)

    # Step 2: Generate mask to remove invalid edges
    # Create a mask for the diagonal (n * n)
    diagonal_mask = torch.eye(n, dtype=torch.bool, device=e.device).unsqueeze(0).expand(b, -1, -1)

    # Create a mask for the node states (b * n)
    node_mask = mask.unsqueeze(2) & mask.unsqueeze(1)

    # Combine the diagonal mask and the node mask
    edge_mask = node_mask & ~diagonal_mask

    # Reshape edge_mask to (b * n * n)
    edge_mask_reshaped = edge_mask.view(b * n * n)

    # Step 3: Apply the mask
    e_filtered = e_reshaped[edge_mask_reshaped]

    return e_filtered


def make_edge_symmetric(e):     # b*n*n*m
    b, n, _, m = e.shape
    mask_upper = torch.triu(torch.ones(n, n), diagonal=1).bool().unsqueeze(0).unsqueeze(-1).expand(
        b, n, n, m).to(e.device)
    e[mask_upper] = e.transpose(2, 1)[mask_upper]
    assert torch.equal(e, e.transpose(1, 2))
    return e


def assert_correctly_masked(variable, node_mask):
    if (variable * (1 - node_mask.long())).abs().max().item() > 1e-4:
        print("The max and min value of variable is", variable.max().item(), variable.min().item())
        torch.save(variable.detach().cpu(), 'bad_variable.pt')
        assert False, 'Variables not masked properly.'


def assert_weak_one_hot(variable):
    assert torch.all((variable.sum(dim=-1) <= 1))
    assert torch.all((variable == 0) | (variable == 1))
