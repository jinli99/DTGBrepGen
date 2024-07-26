import torch
import random
import string


def custom_collate_fn(batch):
    return batch


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


def generate_random_string(length):
    characters = string.ascii_letters + string.digits  # You can include other characters if needed
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string


def calc_bbox_diff(bounding_boxs):
    n = bounding_boxs.shape[0]
    diff_mat = torch.zeros((n, n), device=bounding_boxs.device)

    # Helper function to ensure the box is in the correct format (min point, max point)
    def sort_box(box):
        min_corner = torch.min(box.view(2, 3), dim=0).values
        max_corner = torch.max(box.view(2, 3), dim=0).values
        return torch.stack((min_corner, max_corner))

    # Helper function to compute the volume of a box
    def box_volume(box):
        lengths = box[1] - box[0]
        return torch.prod(lengths)

    # Helper function to compute the intersection volume of two boxes
    def intersection_volume(box1, box2):
        min_corner = torch.max(box1[0], box2[0])
        max_corner = torch.min(box1[1], box2[1])
        intersection = torch.clamp(max_corner - min_corner, min=0)
        return torch.prod(intersection)

    # Helper function to compute the distance between two boxes
    def box_distance(box1, box2):
        max_min_diff = torch.max(box1[0] - box2[1], box2[0] - box1[1])
        return torch.norm(torch.clamp(max_min_diff, min=0))

    for i in range(n):
        for j in range(i+1, n):

            box1 = sort_box(bounding_boxs[i])    # 2*3
            box2 = sort_box(bounding_boxs[j])

            inter_vol = intersection_volume(box1, box2)

            if inter_vol > 0:
                vol1 = box_volume(box1)
                vol2 = box_volume(box2)
                diff_mat[i, j] = diff_mat[j, i] = - (inter_vol / (vol1 + vol2))
            else:
                dist = box_distance(box1, box2)
                diff_mat[i, j] = diff_mat[j, i] = dist

    return diff_mat    # n*n


def remove_box_edge(box, edgeFace_adj):   # nf*6, ne*2
    threshold1 = 0.1
    nf = box.shape[0]

    diff_mat = calc_bbox_diff(box)    # nf*nf

    # Remove isolated bbox
    # mat = diff_mat > threshold1
    # mat[torch.eye(nf).bool()] = True
    # remove_box_idx = torch.nonzero(mat.all(dim=-1), as_tuple=True)[0]
    #
    # mask = torch.any(torch.isin(edgeFace_adj, remove_box_idx.unsqueeze(1)), dim=1)
    # edgeFace_adj = edgeFace_adj[~mask]

    # Remove the edges of the two boxes that are far apart
    threshold2 = 0.4
    mask = diff_mat[edgeFace_adj[:, 0], edgeFace_adj[:, 1]] > threshold2   # ne
    edgeFace_adj = edgeFace_adj[~mask]

    # return remove_box_idx, edgeFace_adj
    return [], edgeFace_adj


def compute_bbox_center_and_size(bbox):  # b*6
    bbox = bbox.reshape(-1, 2, 3)   # b*2*3
    min_xyz, max_xyz = bbox.min(dim=1).values, bbox.max(dim=1).values   # b*3, b*3
    center = (min_xyz + max_xyz) * 0.5   # b*3
    size = max_xyz - min_xyz    # b*3
    # if size_type == 'max':
    #     size = (max_xyz - min_xyz).max(dim=1).values   # b
    # else:
    #     size = (min_xyz - max_xyz).min(dim=1).values
    return center, size


def ncs2wcs(bbox, ncs):   # b*6, b*32*3
    center, size = compute_bbox_center_and_size(bbox)    # b*3, b*3
    max_xyz, min_xyz = ncs.amax(dim=1), ncs.amin(dim=1)   # b*3, b*3
    ratio = (size / (max_xyz - min_xyz)).amin(-1)  # b
    ncs = (ncs - ((min_xyz + max_xyz) * 0.5).unsqueeze(1)) * ratio.unsqueeze(-1).unsqueeze(-1) + center.unsqueeze(1)
    return ncs
    # return ncs * size.unsqueeze(-1) * 0.5 + center.unsqueeze(1)


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
            assert torch.allclose(e, torch.transpose(e, 1, 2))

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


def make_edge_symmetric(e):
    b, n, _, m = e.shape
    mask_upper = torch.triu(torch.ones(n, n), diagonal=1).bool().unsqueeze(0).unsqueeze(-1).expand(
        b, n, n, m).to(e.device)
    e[mask_upper] = e.transpose(2, 1)[mask_upper]
    assert torch.equal(e, e.transpose(1, 2))
    return e


def assert_correctly_masked(variable, node_mask):
    if (variable * (1 - node_mask.long())).abs().max().item() > 1e-4:
        print("The max and min value of variable is", variable.max().item(), variable.min().item())
        assert False, 'Variables not masked properly.'


def assert_weak_one_hot(variable):
    assert torch.all((variable.sum(dim=-1) <= 1))
    assert torch.all((variable == 0) | (variable == 1))
