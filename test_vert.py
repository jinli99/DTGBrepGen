import argparse
import os
import torch
from tqdm import tqdm
import pickle
from model import EdgeGeomTransformer, VertGeomTransformer
from diffusers import PNDMScheduler, DDPMScheduler
from utils import (sort_bbox_multi, pad_and_stack, pad_zero, xe_mask, make_mask)
from OCC.Extend.DataExchange import write_step_file
from brepBuild import Brep2Mesh
from test_edge import get_topology, get_edgeGeom, get_brep
from topology.transfer import faceVert_from_edgeVert
from visualization import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_args_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--faceGeom_path', type=str, default='checkpoints/furniture/geom_faceGeom_01/epoch_3000.pt')
    parser.add_argument('--edgeGeom_path', type=str, default='checkpoints/furniture/geom_edgeGeom/epoch_3000.pt')
    parser.add_argument('--vertGeom_path', type=str, default='checkpoints/furniture/geom_vertGeom_01/epoch_3000.pt')
    parser.add_argument('--edge_classes', type=int, default=5, help='Number of edge classes')
    parser.add_argument('--max_edge', type=int, default=30, help='maximum number of edges per face')
    parser.add_argument('--save_folder', type=str, default="samples/furniture", help='save folder.')
    parser.add_argument('--bbox_scaled', type=float, default=3, help='scaled the bbox')
    args = parser.parse_args()

    return args


def get_vertGeom(edgeVert_adj, edgeFace_adj, vv_list, model, pndm_scheduler, ddpm_scheduler):
    """
    Args:
        edgeVert_adj: List of tensors, where each tensor has shape [ne, 2], representing edge-vert topology
        edgeFace_adj: List of tensors, where each tensor has shape [ne, 2], representing edge-face topology
        vv_list: List of List[Tuple]
        model: vertGeom denoising model
        pndm_scheduler: pndm sampling
        ddpm_scheduler: ddpm sampling
    Returns:
        edge_geom (torch.Tensor): [b, ne, 2] tensor, sampled edge geometry
        edge_mask (torch.Tensor): [b, ne] tensor, edge mask of the sampled edge geometry
    """

    b = len(edgeVert_adj)
    device = edgeVert_adj[0].device
    vert_per_num = [i.max().item()+1 for i in edgeVert_adj]
    nv = max(vert_per_num)
    vv_adj = torch.zeros((b, nv, nv), device=device, dtype=torch.int)
    for i in range(b):
        temp = torch.tensor(vv_list[i], device=device)         # tensor[[v1, v2, edge_idx], ...]
        indices = temp[:, :2]                                  # ne*2
        vv_adj[i, indices[:, 0], indices[:, 1]] = 1
        vv_adj[i, indices[:, 1], indices[:, 0]] = 1            # nv*nv
    vert_mask = make_mask(torch.tensor(vert_per_num, device=device).unsqueeze(-1), nv)   # b*nv

    edgeFace_adj, edge_mask = pad_and_stack(edgeFace_adj)      # b*ne*2, b*ne
    edgeVert_adj, _ = pad_and_stack(edgeVert_adj)              # b*ne*2

    with torch.no_grad():
        vert_geom = torch.randn((b, nv, 3), device=device)     # b*nv*3
        vert_geom, e = xe_mask(x=vert_geom, e=vv_adj.unsqueeze(-1), node_mask=vert_mask)  # b*nv*3, b*nv*nv*1
        e = e.squeeze(-1)                                                                 # b*nv*nv
        pndm_scheduler.set_timesteps(200)
        for t in pndm_scheduler.timesteps[:158]:
            timesteps = t.reshape(-1).cuda()
            pred = model(vert_geom,
                         e,
                         vert_mask,
                         edgeFace_adj,
                         edgeVert_adj,
                         edge_mask,
                         timesteps.expand(vert_geom.shape[0], 1))
            vert_geom = pndm_scheduler.step(pred, t, vert_geom).prev_sample
            vert_geom, _ = xe_mask(x=vert_geom, node_mask=vert_mask)

        ddpm_scheduler.set_timesteps(1000)
        for t in ddpm_scheduler.timesteps[-250:]:
            timesteps = t.reshape(-1).cuda()
            pred = model(vert_geom,
                         e,
                         vert_mask,
                         edgeFace_adj,
                         edgeVert_adj,
                         edge_mask,
                         timesteps.expand(vert_geom.shape[0], 1))
            vert_geom = ddpm_scheduler.step(pred, t, vert_geom).prev_sample
            vert_geom, _ = xe_mask(x=vert_geom, node_mask=vert_mask)

    return vert_geom, vert_mask


def main():
    args = get_args_generate()

    # Make project directory if not exist
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initial VertGeomTransformer
    hidden_mlp_dims = {'x': 256}
    hidden_dims = {'dx': 512, 'de': 256, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128}
    vertGeom_model = VertGeomTransformer(max_edge=args.max_edge, edge_classes=args.edge_classes, n_layers=8,
                                         hidden_mlp_dims=hidden_mlp_dims,
                                         hidden_dims=hidden_dims,
                                         act_fn_in=torch.nn.ReLU(), act_fn_out=torch.nn.ReLU())
    vertGeom_model.load_state_dict(torch.load(args.vertGeom_path))
    vertGeom_model = vertGeom_model.to(device).eval()

    # Initial EdgeGeomTransformer
    edgeGeom_model = EdgeGeomTransformer(max_edge=args.max_edge,
                                         edge_classes=args.edge_classes,
                                         n_layers=8,
                                         edge_geom_dim=12)
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

    with open('batch_file_test.pkl', 'rb') as f:
        batch_file = pickle.load(f)

    # batch_file = ['chair/partstudio_partstudio_1242.pkl']

    b_each = 32
    for i in tqdm(range(0, len(batch_file), b_each)):

        # =======================================Brep Topology=================================================== #
        datas = get_topology(batch_file[i:i + b_each], device)
        face_bbox, edgeVert_adj, faceEdge_adj, edgeFace_adj, vv_list = (datas["face_bbox"],
                                                                        datas["edgeVert_adj"],
                                                                        datas['faceEdge_adj'],
                                                                        datas["edgeFace_adj"],
                                                                        datas["vv_list"])
        b = len(face_bbox)
        face_bbox = [sort_bbox_multi(i)*args.bbox_scaled for i in face_bbox]     # [nf*6, ...]

        # =======================================Vert Geometry=================================================== #
        vert_geom, vert_mask = get_vertGeom(edgeVert_adj,
                                            edgeFace_adj, vv_list, vertGeom_model,
                                            pndm_scheduler, ddpm_scheduler)
        vert_geom = [i[j] for i, j in zip(vert_geom, vert_mask)]

        # =======================================Edge Geometry=================================================== #
        edge_geom, edge_mask = get_edgeGeom(vert_geom,
                                            edgeVert_adj, edgeFace_adj, faceEdge_adj, edgeGeom_model,
                                            pndm_scheduler, ddpm_scheduler)
        edge_geom = [i[j] for i, j in zip(edge_geom, edge_mask)]                 # [ne*12, ...]

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
            save_name = datas['name'][j]
            write_step_file(solid, f'{args.save_folder}/{save_name}.step')

    print('write stl...')
    mesh_tool = Brep2Mesh(input_path=args.save_folder)
    mesh_tool.generate()


if __name__ == '__main__':
    main()
