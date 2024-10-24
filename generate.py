import argparse
import os
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from diffusers import DDPMScheduler, PNDMScheduler
from model import EdgeGeomTransformer, VertGeomTransformer, FaceBboxTransformer, FaceEdgeModel, TopoSeqModel
from topology.topoGenerate import SeqGenerator
from topology.transfer import faceVert_from_edgeVert, face_vert_trans, fef_from_faceEdge
from utils import xe_mask, pad_zero, sort_bbox_multi, generate_random_string, make_mask, pad_and_stack
from OCC.Extend.DataExchange import write_step_file
from visualization import *
from brepBuild import Brep2Mesh, sample_bspline_curve, create_bspline_curve, joint_optimize, construct_brep

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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

int2text = {v: k for k, v in text2int.items()}

def get_args_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--FaceEdge_path', type=str, default='checkpoints/furniture/topo_faceEdge/epoch_2000.pt')
    parser.add_argument('--TopoSeq_path', type=str, default='checkpoints/furniture/topo_Seq/epoch_3000.pt')
    parser.add_argument('--faceGeom_path', type=str, default='checkpoints/furniture/geom_faceGeom/epoch_3000.pt')
    parser.add_argument('--edgeGeom_path', type=str, default='checkpoints/furniture/geom_edgeGeom/epoch_3000.pt')
    parser.add_argument('--vertGeom_path', type=str, default='checkpoints/furniture/geom_vertGeom/epoch_3000.pt')
    parser.add_argument('--faceBbox_path', type=str, default='checkpoints/furniture/geom_faceBbox/epoch_3000.pt')
    parser.add_argument('--edge_classes', type=int, default=5, help='Number of edge classes')
    parser.add_argument('--max_edge', type=int, default=30, help='maximum number of edges per face')
    parser.add_argument('--save_folder', type=str, default="samples/furniture", help='save folder.')
    parser.add_argument('--bbox_scaled', type=float, default=3, help='scaled the bbox')
    parser.add_argument("--cf", action='store_false', help='Use class condition')
    args = parser.parse_args()

    return args


def get_topology(batch, faceEdge_model, seq_model, device, labels):
    """****************Brep Topology****************"""
    edgeVert_adj = []
    edgeFace_adj = []
    faceEdge_adj = []
    vertFace_adj = []
    fef_adj = []

    valid = 0

    class_label = torch.LongTensor(labels).to(device).reshape(-1, 1)
    fail_idx = []

    with torch.no_grad():

        adj_batch = faceEdge_model.sample(num_samples=batch, class_label=class_label)              # b*nf*nf

        for i in tqdm(range(batch)):
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
            ef_adj = edge_indices.repeat_interleave(num_edges, dim=0)                              # ne*2

            # save encoder information
            seq_model.save_cache(edgeFace_adj=ef_adj.unsqueeze(0),
                                 edge_mask=torch.ones((1, ef_adj.shape[0]), device=device, dtype=torch.bool),
                                 class_label=class_label[[i]])
            for try_time in range(10):
                generator = SeqGenerator(ef_adj.cpu().numpy())
                if generator.generate(seq_model, class_label[[i]]):
                    # print("Construct Brep Topology succeed at time %d!" % try_time)
                    seq_model.clear_cache()
                    valid += 1
                    break
            else:
                # print("Construct Brep Topology Failed!")
                seq_model.clear_cache()
                fail_idx.append(i)
                continue

            ev_adj = generator.edgeVert_adj
            fe_adj = generator.faceEdge_adj

            edgeVert_adj.append(torch.from_numpy(ev_adj).to(device))
            edgeFace_adj.append(ef_adj)
            faceEdge_adj.append(fe_adj)
            fv_adj = faceVert_from_edgeVert(fe_adj, ev_adj)
            vf_adj = face_vert_trans(faceVert_adj=fv_adj)
            vertFace_adj.append(vf_adj)
            fef = fef_from_faceEdge(edgeFace_adj=ef_adj.cpu().numpy())
            fef_adj.append(torch.from_numpy(fef).to(device))

    if labels is not None:
        labels = [labels[i] for i in range(len(class_label)) if i not in fail_idx]

    print(valid)

    return {"edgeVert_adj": edgeVert_adj, "edgeFace_adj": edgeFace_adj, "faceEdge_adj": faceEdge_adj,
            "vertFace_adj": vertFace_adj, "fef_adj": fef_adj, "class_label": labels}


def get_faceBbox(fef_adj, model, pndm_scheduler, ddpm_scheduler, class_label):
    """
    Args:
        fef_adj: List of tensors, where each tensor has shape [nf, nf], representing face-edge-face topology.
        model: vertGeom denoising model
        pndm_scheduler: pndm sampling
        ddpm_scheduler: ddpm sampling
        class_label: List with length b
    Returns:
        edge_geom (torch.Tensor): [b, ne, 2] tensor, sampled edge geometry
        edge_mask (torch.Tensor): [b, ne] tensor, edge mask of the sampled edge geometry
    """

    b = len(fef_adj)
    device = fef_adj[0].device

    nf = max([i.shape[0] for i in fef_adj])
    fef_adj = [pad_zero(i, max_len=nf, dim=1) for i in fef_adj]
    face_mask = torch.stack([i[1] for i in fef_adj])                  # b*nf
    fef_adj = torch.stack([i[0] for i in fef_adj])                    # b*nf*nf

    w = 0.6
    if class_label is not None:
        class_label = torch.LongTensor(class_label+[text2int['uncond']]*b).to(device).reshape(-1,1)

    with torch.no_grad():
        face_bbox = torch.randn((b, nf, 6), device=device)                                 # b*nf*6
        face_bbox, e = xe_mask(x=face_bbox, e=fef_adj.unsqueeze(-1), node_mask=face_mask)  # b*nf*6, b*nf*nf*1
        e = e.squeeze(-1)                                                                  # b*nf*nf
        pndm_scheduler.set_timesteps(200)
        for t in pndm_scheduler.timesteps[:158]:
            timesteps = t.reshape(-1).cuda()
            if class_label is not None:
                pred = model(torch.cat([face_bbox, face_bbox], dim=0),
                             torch.cat([e, e], dim=0),
                             torch.cat([face_mask, face_mask], dim=0),
                             class_label,
                             timesteps.expand(2*face_bbox.shape[0], 1))
                pred = pred[:b] * (1 + w) - pred[b:] * w
            else:
                pred = model(face_bbox,
                             e,
                             face_mask,
                             timesteps.expand(face_bbox.shape[0], 1))
            face_bbox = pndm_scheduler.step(pred, t, face_bbox).prev_sample
            face_bbox, _ = xe_mask(x=face_bbox, node_mask=face_mask)

        ddpm_scheduler.set_timesteps(1000)
        for t in ddpm_scheduler.timesteps[-250:]:
            timesteps = t.reshape(-1).cuda()
            if class_label is not None:
                pred = model(torch.cat([face_bbox, face_bbox], dim=0),
                             torch.cat([e, e], dim=0),
                             torch.cat([face_mask, face_mask], dim=0),
                             class_label,
                             timesteps.expand(2*face_bbox.shape[0], 1))
                pred = pred[:b] * (1 + w) - pred[b:] * w
            else:
                pred = model(face_bbox,
                             e,
                             face_mask,
                             timesteps.expand(face_bbox.shape[0], 1))
            face_bbox = ddpm_scheduler.step(pred, t, face_bbox).prev_sample
            face_bbox, _ = xe_mask(x=face_bbox, node_mask=face_mask)

    return face_bbox, face_mask


def get_vertGeom(face_bbox, vertFace_adj, edgeVert_adj, model, pndm_scheduler, ddpm_scheduler, class_label):
    """
    Args:
        face_bbox: List of tensors, where each tensor has shape [nf, 6], representing face bounding boxes.
        vertFace_adj: List of List[List[int]]
        edgeVert_adj: List tensors, where each tensor has shape [ne, 2]
        model: vertGeom denoising model
        pndm_scheduler: pndm sampling
        ddpm_scheduler: ddpm sampling
        class_label: List with length b
    Returns:
        edge_geom (torch.Tensor): [b, ne, 2] tensor, sampled edge geometry
        edge_mask (torch.Tensor): [b, ne] tensor, edge mask of the sampled edge geometry
    """

    b = len(face_bbox)
    device = face_bbox[0].device
    vert_per_num = [len(i) for i in vertFace_adj]
    nv = max(vert_per_num)
    vv_adj = torch.zeros((b, nv, nv), device=device, dtype=torch.int)
    for i in range(b):
        indices = edgeVert_adj[i]                              # ne*2
        vv_adj[i, indices[:, 0], indices[:, 1]] = 1
        vv_adj[i, indices[:, 1], indices[:, 0]] = 1            # nv*nv
    vert_mask = make_mask(torch.tensor(vert_per_num, device=device).unsqueeze(-1), nv)   # b*nv

    vertFace_bbox = []
    vertFace_mask = []
    vf = max([max([len(j) for j in i]) for i in vertFace_adj])
    for i in range(b):
        temp1 = [face_bbox[i][j] for j in vertFace_adj[i]]      # [vf*6, ...]
        temp1, temp2 = pad_and_stack(temp1, max_n=vf)           # nv*vf*6, nv*vf
        vertFace_bbox.append(temp1)
        vertFace_mask.append(temp2)
    vertFace_bbox, _ = pad_and_stack(vertFace_bbox)             # b*nv*vf*6
    vertFace_mask, _ = pad_and_stack(vertFace_mask)             # b*nv*vf
    assert vertFace_mask.shape == (b, nv, vf) and vertFace_bbox.shape == (b, nv, vf, 6)

    w = 0.6
    if class_label is not None:
        class_label = torch.LongTensor(class_label+[text2int['uncond']]*b).to(device).reshape(-1,1)

    with torch.no_grad():
        vert_geom = torch.randn((b, nv, 3), device=device)      # b*nv*3
        vert_geom, e = xe_mask(x=vert_geom, e=vv_adj.unsqueeze(-1), node_mask=vert_mask)  # b*nv*3, b*nv*nv*1
        e = e.squeeze(-1)                                                                 # b*nv*nv
        pndm_scheduler.set_timesteps(200)
        for t in pndm_scheduler.timesteps[:158]:
            timesteps = t.reshape(-1).cuda()
            if class_label is not None:
                pred = model(torch.cat([vert_geom, vert_geom], dim=0),
                             torch.cat([e, e], dim=0),
                             torch.cat([vert_mask, vert_mask], dim=0),
                             torch.cat([vertFace_bbox, vertFace_bbox], dim=0),
                             torch.cat([vertFace_mask, vertFace_mask], dim=0),
                             class_label,
                             timesteps.expand(2*vert_geom.shape[0], 1))
                pred = pred[:b] * (1 + w) - pred[b:] * w
            else:
                pred = model(vert_geom,
                             e,
                             vert_mask,
                             vertFace_bbox,
                             vertFace_mask,
                             timesteps.expand(vert_geom.shape[0], 1))
            vert_geom = pndm_scheduler.step(pred, t, vert_geom).prev_sample
            vert_geom, _ = xe_mask(x=vert_geom, node_mask=vert_mask)

        ddpm_scheduler.set_timesteps(1000)
        for t in ddpm_scheduler.timesteps[-250:]:
            timesteps = t.reshape(-1).cuda()
            if class_label is not None:
                pred = model(torch.cat([vert_geom, vert_geom], dim=0),
                             torch.cat([e, e], dim=0),
                             torch.cat([vert_mask, vert_mask], dim=0),
                             torch.cat([vertFace_bbox, vertFace_bbox], dim=0),
                             torch.cat([vertFace_mask, vertFace_mask], dim=0),
                             class_label,
                             timesteps.expand(2*vert_geom.shape[0], 1))
                pred = pred[:b] * (1 + w) - pred[b:] * w
            else:
                pred = model(vert_geom,
                             e,
                             vert_mask,
                             vertFace_bbox,
                             vertFace_mask,
                             timesteps.expand(vert_geom.shape[0], 1))
            vert_geom = ddpm_scheduler.step(pred, t, vert_geom).prev_sample
            vert_geom, _ = xe_mask(x=vert_geom, node_mask=vert_mask)

    return vert_geom, vert_mask


def get_edgeGeom(face_bbox, vert_geom, edgeFace_adj, edgeVert_adj, model, pndm_scheduler, ddpm_scheduler, class_label):
    """
    Args:
        face_bbox: List of tensors, where each tensor has shape [nf, 6], representing face bounding box.
        vert_geom: List of tensors, where each tensor has shape [nv, 3], representing vertex geometries.
        edgeFace_adj: List of tensors, where each tensor has shape [ne, 2], representing edge-face topology.
        edgeVert_adj: List of tensors, where each tensor has shape [ne, 2], representing edge-vert topology.
        model: edgeGeom denoising model
        pndm_scheduler: pndm sampling
        ddpm_scheduler: ddpm sampling
        class_label: List with length b
    Returns:
        edge_geom (torch.Tensor): [b, ne, 2] tensor, sampled edge geometry
        edge_mask (torch.Tensor): [b, ne] tensor, edge mask of the sampled edge geometry
    """

    b = len(vert_geom)
    device = vert_geom[0].device
    edgeVert_geom = [vert_geom[i][edgeVert_adj[i]] for i in range(b)]      # [ne*2*3, ...]
    edgeFace_bbox = [face_bbox[i][edgeFace_adj[i]] for i in range(b)]                           # [ne*2*6, ...]

    edgeFace_bbox, edge_mask = pad_and_stack(edgeFace_bbox)                # b*ne*2*6, b*ne
    edgeVert_geom, _ = pad_and_stack(edgeVert_geom)                        # b*ne*2*3
    edgeFace_adj, _ = pad_and_stack(edgeFace_adj)                          # b*ne*2

    ne = edge_mask.shape[1]

    w = 0.6
    if class_label is not None:
        class_label = torch.LongTensor(class_label+[text2int['uncond']]*b).to(device).reshape(-1,1)

    with torch.no_grad():
        edge_geom = torch.randn((b, ne, 12), device=device)  # b*ne*12

        pndm_scheduler.set_timesteps(200)
        for t in pndm_scheduler.timesteps[:158]:
            timesteps = t.reshape(-1).cuda()
            if class_label is not None:
                pred = model(torch.cat([edge_geom, edge_geom], dim=0),
                             torch.cat([edgeFace_bbox, edgeFace_bbox], dim=0),
                             torch.cat([edgeVert_geom, edgeVert_geom], dim=0),
                             torch.cat([edge_mask, edge_mask], dim=0),
                             class_label,
                             timesteps.expand(2*edge_geom.shape[0], 1))
                pred = pred[:b] * (1 + w) - pred[b:] * w
            else:
                pred = model(edge_geom,
                             edgeFace_bbox,
                             edgeVert_geom,
                             edge_mask,
                             class_label,
                             timesteps.expand(edge_geom.shape[0], 1))
            edge_geom = pndm_scheduler.step(pred, t, edge_geom).prev_sample
            edge_geom, _ = xe_mask(x=edge_geom, node_mask=edge_mask)

        ddpm_scheduler.set_timesteps(1000)
        for t in ddpm_scheduler.timesteps[-250:]:
            timesteps = t.reshape(-1).cuda()
            if class_label is not None:
                pred = model(torch.cat([edge_geom, edge_geom], dim=0),
                             torch.cat([edgeFace_bbox, edgeFace_bbox], dim=0),
                             torch.cat([edgeVert_geom, edgeVert_geom], dim=0),
                             torch.cat([edge_mask, edge_mask], dim=0),
                             class_label,
                             timesteps.expand(2*edge_geom.shape[0], 1))
                pred = pred[:b] * (1 + w) - pred[b:] * w
            else:
                pred = model(edge_geom,
                             edgeFace_bbox,
                             edgeVert_geom,
                             edge_mask,
                             class_label,
                             timesteps.expand(edge_geom.shape[0], 1))
            edge_geom = ddpm_scheduler.step(pred, t, edge_geom).prev_sample
            edge_geom, _ = xe_mask(x=edge_geom, node_mask=edge_mask)

    return edge_geom, edge_mask


def get_brep(args):
    # nf*48, nf*6, nv*3, ne*12, ne*2, ne*2, List[List[int]]
    face_bbox, vert_geom, edge_geom, edgeFace_adj, edgeVert_adj, faceEdge_adj = args

    assert set(range(face_bbox.shape[0])) == set(np.unique(edgeFace_adj.reshape(-1)).tolist())
    assert set(range(vert_geom.shape[0])) == set(np.unique(edgeVert_adj.reshape(-1)).tolist())

    edge_ncs = []
    for ctrs in edge_geom:
        pcd = sample_bspline_curve(create_bspline_curve(ctrs.reshape(4, 3).astype(np.float64)))  # 32*3
        edge_ncs.append(pcd)
    edge_ncs = np.stack(edge_ncs)  # ne*32*3

    # joint_optimize
    edge_wcs = joint_optimize(edge_ncs, face_bbox, vert_geom, edgeVert_adj, faceEdge_adj, len(edge_ncs), len(face_bbox))

    try:
        solid = construct_brep(edge_wcs, faceEdge_adj, edgeVert_adj)
    except Exception as e:
        print('B-rep rebuild failed...')
        return False

    return solid


def main():

    args = get_args_generate()

    # Make project directory if not exist
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initial Topology Model
    faceEdge_model = FaceEdgeModel(nf=32, num_categories=5, use_cf=args.cf)
    faceEdge_model.load_state_dict(torch.load(args.FaceEdge_path), strict=False)
    faceEdge_model = faceEdge_model.to(device).eval()
    seq_model = TopoSeqModel(max_num_edge=96, max_seq_length=260, max_face=32, use_cf=args.cf)
    seq_model.load_state_dict(torch.load(args.TopoSeq_path), strict=False)
    seq_model = seq_model.to(device).eval()

    # Initial FaceBboxTransformer
    hidden_mlp_dims = {'x': 256}
    hidden_dims = {'dx': 512, 'de': 256, 'n_head': 8, 'dim_ffX': 512}
    FaceBbox_model = FaceBboxTransformer(n_layers=8,
                                         hidden_mlp_dims=hidden_mlp_dims,
                                         hidden_dims=hidden_dims,
                                         edge_classes=args.edge_classes,
                                         act_fn_in=torch.nn.ReLU(),
                                         act_fn_out=torch.nn.ReLU(),
                                         use_cf=args.cf)
    FaceBbox_model.load_state_dict(torch.load(args.faceBbox_path))
    FaceBbox_model = FaceBbox_model.to(device).eval()

    # Initial VertGeomTransformer
    hidden_mlp_dims = {'x': 256}
    hidden_dims = {'dx': 512, 'de': 256, 'n_head': 8, 'dim_ffX': 512}
    vertGeom_model = VertGeomTransformer(n_layers=4,
                                         hidden_mlp_dims=hidden_mlp_dims,
                                         hidden_dims=hidden_dims,
                                         act_fn_in=torch.nn.ReLU(),
                                         act_fn_out=torch.nn.ReLU(),
                                         use_cf=args.cf)
    vertGeom_model.load_state_dict(torch.load(args.vertGeom_path))
    vertGeom_model = vertGeom_model.to(device).eval()

    # Initial EdgeGeomTransformer
    edgeGeom_model = EdgeGeomTransformer(n_layers=8,
                                         edge_geom_dim=12,
                                         use_cf=args.cf)
    edgeGeom_model.load_state_dict(torch.load(args.edgeGeom_path))
    edgeGeom_model = edgeGeom_model.to(device).eval()

    pndm_scheduler = PNDMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start=0.0001,
        beta_end=0.02,
    )

    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start=0.0001,
        beta_end=0.02,
        clip_sample=True,
        clip_sample_range=3
    )

    total_sample = 512
    b_each = 16
    for i in tqdm(range(0, total_sample, b_each)):

        b = min(total_sample, i+b_each)-i
        if args.cf:
            class_label = torch.randint(6, 7, (b,)).tolist()
        else:
            class_label = None

        # =======================================Brep Topology=================================================== #
        datas = get_topology(b, faceEdge_model, seq_model, device, class_label)
        fef_adj, edgeVert_adj, faceEdge_adj, edgeFace_adj, vertFace_adj = (datas["fef_adj"],
                                                                           datas["edgeVert_adj"],
                                                                           datas['faceEdge_adj'],
                                                                           datas["edgeFace_adj"],
                                                                           datas["vertFace_adj"])
        class_label = datas['class_label']
        b = len(fef_adj)

        # ========================================Face Bbox====================================================== #
        face_bbox, face_mask = get_faceBbox(fef_adj, FaceBbox_model, pndm_scheduler, ddpm_scheduler, class_label)
        face_bbox = [k[l] for k, l in zip(face_bbox, face_mask)]

        face_bbox = [sort_bbox_multi(j) for j in face_bbox]                      # [nf*6, ...]

        # =======================================Vert Geometry=================================================== #
        vert_geom, vert_mask = get_vertGeom(face_bbox, vertFace_adj,
                                            edgeVert_adj, vertGeom_model,
                                            pndm_scheduler, ddpm_scheduler, class_label)
        vert_geom = [k[l] for k, l in zip(vert_geom, vert_mask)]

        # =======================================Edge Geometry=================================================== #
        edge_geom, edge_mask = get_edgeGeom(face_bbox, vert_geom,
                                            edgeFace_adj, edgeVert_adj, edgeGeom_model,
                                            pndm_scheduler, ddpm_scheduler, class_label)
        edge_geom = [k[l] for k, l in zip(edge_geom, edge_mask)]                 # [ne*12, ...]

        # =======================================Construct Brep================================================== #
        for j in range(b):
            solid = get_brep((face_bbox[j].cpu().numpy() / args.bbox_scaled,
                              vert_geom[j].cpu().numpy() / args.bbox_scaled,
                              edge_geom[j].cpu().numpy() / args.bbox_scaled,
                              edgeFace_adj[j].cpu().numpy(),
                              edgeVert_adj[j].cpu().numpy(),
                              faceEdge_adj[j]))

            if solid is False:
                continue
            write_step_file(solid, f'{args.save_folder}/{int2text[class_label[j]]}_{generate_random_string(10)}.step')

    print('write stl...')
    mesh_tool = Brep2Mesh(input_path=args.save_folder)
    mesh_tool.generate()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
