import argparse
import os
import torch
import pickle
import numpy as np
from model import AutoencoderKLFastDecode, GraphTransformer, FaceGeomTransformer, EdgeGeomTransformer, AutoencoderKL1DFastDecode
from diffusion import GraphDiffusion, DDPM
from utils import (pad_and_stack, pad_zero, xe_mask, make_edge_symmetric, assert_weak_one_hot, ncs2wcs, generate_random_string,
                   remove_box_edge, construct_edgeFace_adj, construct_feTopo, remove_short_edge)
from dataFeature import GraphFeatures
from OCC.Extend.DataExchange import write_stl_file, write_step_file
from brep_functions import (
    scale_surf, manual_edgeVertex_topology,
    optimize_edgeVertex_topology, joint_optimize, construct_brep)
from visualization import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_args_gdm():
    parser = argparse.ArgumentParser()
    parser.add_argument('--face_vae_path', type=str, default='checkpoints/furniture/vae_face/epoch_400.pt')
    parser.add_argument('--edge_vae_path', type=str, default='checkpoints/furniture/vae_edge/epoch_400.pt')
    parser.add_argument('--feTopo_path', type=str, default='checkpoints/furniture/gdm_feTopo/epoch_3000.pt')
    parser.add_argument('--faceGeom_path', type=str, default='checkpoints/furniture/gdm_faceGeom/epoch_3000.pt')
    parser.add_argument('--edgeGeom_path', type=str, default='checkpoints/furniture/gdm_edgeGeom/epoch_1000.pt')
    parser.add_argument('--hyper_params_path', type=str, default='checkpoints/furniture/gdm_feTopo/hyper_params.pkl')
    parser.add_argument('--batch_size', type=int, default=2, help='sample batch size')
    parser.add_argument('--save_folder', type=str, default="samples/furniture", help='save folder.')
    args = parser.parse_args()

    return args


def main():
    args = get_args_gdm()

    # Make project directory if not exist
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    with open(args.hyper_params_path, "rb") as f:
        hyper_params = pickle.load(f)
    m = hyper_params['edge_classes']

    # Load surface vae
    face_vae = AutoencoderKLFastDecode(
        in_channels=3, out_channels=3,
        down_block_types=('DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'),
        up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'),
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        act_fn='silu',
        latent_channels=3,
        norm_num_groups=32,
        sample_size=512,
    )
    face_vae.load_state_dict(torch.load(args.face_vae_path), strict=False)     # inputs: points_batch*3*4*4
    face_vae = face_vae.to(device).eval()

    edge_vae = AutoencoderKL1DFastDecode(
        in_channels=3,
        out_channels=3,
        down_block_types=('DownBlock1D', 'DownBlock1D', 'DownBlock1D'),
        up_block_types=('UpBlock1D', 'UpBlock1D', 'UpBlock1D'),
        block_out_channels=(128, 256, 512),
        layers_per_block=2,
        act_fn='silu',
        latent_channels=3,
        norm_num_groups=32,
        sample_size=512
    )
    edge_vae.load_state_dict(torch.load(args.edge_vae_path), strict=False)     # inputs: points_batch*3*4
    edge_vae = edge_vae.to(device).eval()

    # Initial GraphTransformer and GraphDiffusion
    feTopo_model = GraphTransformer(n_layers=hyper_params['n_layers'], input_dims=hyper_params['input_dims'],
                                    hidden_mlp_dims=hyper_params['hidden_mlp_dims'],
                                    hidden_dims=hyper_params['hidden_dims'], output_dims=hyper_params['output_dims'],
                                    act_fn_in=torch.nn.ReLU(), act_fn_out=torch.nn.ReLU())
    feTopo_model.load_state_dict(torch.load(args.feTopo_path))
    feTopo_model = feTopo_model.to(device).eval()
    graphdiff = GraphDiffusion(500, m, hyper_params['edge_marginals'], device)

    # Initial FaceGeomTransformer
    faceGeom_model = FaceGeomTransformer(n_layers=hyper_params['n_layers'], input_dims={'x': 54, 'e': 5, 'y': 12},
                                         hidden_mlp_dims=hyper_params['hidden_mlp_dims'],
                                         hidden_dims=hyper_params['hidden_dims'],
                                         output_dims={'x': 48, 'e': 5, 'y': 0},
                                         act_fn_in=torch.nn.ReLU(), act_fn_out=torch.nn.ReLU())
    faceGeom_model.load_state_dict(torch.load(args.faceGeom_path))
    faceGeom_model = faceGeom_model.to(device).eval()

    # Initial EdgeGeomTransformer
    edgeGeom_model = EdgeGeomTransformer(n_layers=6, face_geom_dim=48, edge_geom_dim=12)
    edgeGeom_model.load_state_dict(torch.load(args.edgeGeom_path))
    edgeGeom_model = edgeGeom_model.to(device).eval()

    ddpm = DDPM(500, device)

    # Initial feature extractor
    extract_feat = GraphFeatures(hyper_params['extract_type'], hyper_params['node_distribution'].shape[0])

    """****************Face-Edge Topology****************"""
    num_nodes = torch.distributions.Categorical(
        hyper_params['node_distribution']).sample(torch.Size([args.batch_size])).to(device)  # b
    x, node_mask = pad_and_stack([torch.randn((i, hyper_params['diff_dim']), device=device) for i in num_nodes])    # b*n*6, b*n
    n = x.shape[1]
    e = torch.multinomial(hyper_params['edge_marginals'].unsqueeze(0).expand(batch_size*n*n, m),
                          num_samples=1, replacement=True).reshape(batch_size, n, n)   # b*n*n
    e = torch.nn.functional.one_hot(e, num_classes=m).to(device).float()  # b*n*n*m
    _, e = xe_mask(e=e, node_mask=node_mask, check_sym=False)
    e = make_edge_symmetric(e)   # b*n*n*m
    assert_weak_one_hot(e)

    with torch.no_grad():
        for t in range(graphdiff.T-1, -1, -1):
            # Extract features
            feat = extract_feat(e, node_mask)
            x_feat = torch.cat((x, feat[0]), dim=-1).float()  # b*n*12
            e_feat = torch.cat((e, feat[1]), dim=-1).float()  # b*n*n*m
            y_feat = torch.cat(
                (feat[2], torch.tensor([graphdiff.normalize_t(t)]*batch_size, device=device).unsqueeze(-1)), dim=-1).float()  # b*12

            # Prediction
            x_pred, e_pred, y_pred = feTopo_model(x_feat, e_feat, y_feat, node_mask)  # b*n*6, b*n*n*m, b*0
            # print("time:", t, torch.nn.functional.softmax(e_pred.mean(0).mean(0).mean(0)))

            # Sample x
            x = graphdiff.x_sample(x, x_pred, torch.tensor([t], device=device))  # b*n*6

            # Sample e
            e = graphdiff.e_sample(e_pred, e, torch.tensor([t]*batch_size, device=device).unsqueeze(-1))  # b*n*n*m

            # Mask
            x, e = xe_mask(x, e, node_mask)   # b*n*6, b*n*n*m

        x = x / hyper_params['bbox_scaled']

        # Remove faces and edges
        edgeFace = construct_edgeFace_adj(e[..., 1:].sum(-1), node_mask=node_mask)     # list[shape:ne*2,...]
        temp = [remove_box_edge(x[i][node_mask[i]], edgeFace[i]) for i in range(batch_size)]
        x = [i[0] for i in temp]   # list[shape:nf*6, ...]
        edgeFace = [i[1] for i in temp]   # list[shape:ne*2,...]

        """****************Face Geometry****************"""
        face_bbox, node_mask = pad_and_stack(x)    # b*nf*6, b*nf
        face_bbox *= hyper_params['bbox_scaled']
        fe_topo = [pad_zero(construct_feTopo(i).cpu().numpy(), max_len=node_mask.shape[1], dim=1)[0] for i in edgeFace]   # list[shape:nf*nf, ...]
        fe_topo = torch.from_numpy(np.stack(fe_topo)).to(device)    # b*nf*nf
        x = torch.randn((fe_topo.shape[0], fe_topo.shape[1], 48), device=device)    # b*nf*48
        e = torch.nn.functional.one_hot(fe_topo, num_classes=m)  # b*n*n*m
        x, e = xe_mask(x=x, e=e, node_mask=node_mask)    # b*nf*48, b*n*n*m
        feat = extract_feat(e, node_mask)
        for t in range(ddpm.T - 1, -1, -1):

            # Extract features
            x_feat = torch.cat((x, feat[0]), dim=-1).float()  # b*nf*54
            e_feat = torch.cat((e, feat[1]), dim=-1).float()  # b*nf*nf*m
            y_feat = torch.cat(
                (feat[2], torch.tensor([ddpm.normalize_t(t)] * batch_size, device=device).unsqueeze(-1)),
                dim=-1).float()  # b*12

            # Predict start
            pred_noise = faceGeom_model(x_feat, e_feat, y_feat, face_bbox, node_mask)  # b*nf*54

            # Sample x
            x = ddpm.p_sample(x, pred_noise, torch.tensor([t], device=device))   # b*nf*48
            x, _ = xe_mask(x=x, node_mask=node_mask)    # b*nf*48

        """****************Edge Geometry****************"""
        face_bbox_geom = torch.cat((x, face_bbox), dim=-1)   # b*nf*54
        edge_faceInfo = [face[adj] for face, adj in zip(face_bbox_geom, edgeFace)]   # list[shape:ne*2*54]
        edge_faceInfo, edge_mask = pad_and_stack(edge_faceInfo)            # b*ne*2*54, b*ne
        ne = edge_faceInfo.shape[1]
        edge_bbox_geom = torch.randn((batch_size, ne, 18), device=device)    # b*ne*18
        for t in range(ddpm.T - 1, -1, -1):

            pred_noise = edgeGeom_model(
                edge_bbox_geom, edge_faceInfo, edge_mask, torch.tensor(
                    [ddpm.normalize_t(t)]*batch_size, device=device).unsqueeze(-1))

            edge_bbox_geom = ddpm.p_sample(edge_bbox_geom, pred_noise, torch.tensor([t], device=device))    # b*ne*18

            edge_bbox_geom, _ = xe_mask(x=edge_bbox_geom, node_mask=edge_mask)

        """****************Construct Brep****************"""
        for i in range(batch_size):
            face_batch, edge_batch = face_bbox_geom[i][node_mask[i]], edge_bbox_geom[i][edge_mask[i]]   # nf*54, ne*18
            face_geom, face_bbox = face_batch[:, :-6], face_batch[:, -6:]/hyper_params['bbox_scaled']   # nf*48, nf*6
            edge_geom, edge_bbox = edge_batch[:, :-6], edge_batch[:, -6:]/hyper_params['bbox_scaled']   # ne*12, ne*6

            # Decode face geometry
            face_ncs = face_vae(
                face_geom.unflatten(-1, torch.Size([16, 3])).permute(0, 2, 1).unflatten(-1, torch.Size([4, 4])))   # nf*3*32*32
            face_ncs = face_ncs.permute(0, 2, 3, 1)    # nf*32*32*3

            # Decode edge geometry
            edge_ncs = edge_vae(
                edge_geom.reshape(-1, 4, 3).permute(0, 2, 1)).permute(0, 2, 1)   # ne*32*3

            # Get Edge-Vertex topology
            edge_wcs = ncs2wcs(edge_bbox, edge_ncs)    # ne*32*3
            edgeFace_adj = edgeFace[i]

            # Remove short edge
            edge_wcs, keep_edge_idx = remove_short_edge(edge_wcs, threshold=0.1)
            edgeFace_adj = edgeFace_adj[keep_edge_idx]

            face_wcs = ncs2wcs(face_bbox, face_ncs.flatten(1, 2)).unflatten(1, (32, 32))    # nf*32*32*3

            assert set(range(face_batch.shape[0])) == set(torch.unique(edgeFace_adj.view(-1)).cpu().numpy().tolist())
            unique_vertex_pnt, edgeVertex_adj, faceEdge_adj = optimize_edgeVertex_topology(edge_wcs.cpu(), edgeFace_adj.cpu())

            # joint_optimize:
            # numpy(nf*32*32*3), numpy(ne*32*3), numpy(nf*6), numpy(nv*3),
            # numpy(ne*2), len(list[[edge_id1, ...]...])=nf, int, int
            face_wcs, edge_wcs = joint_optimize(face_ncs.numpy(), edge_ncs.numpy(),
                                                face_bbox.cpu().numpy(), unique_vertex_pnt.numpy(),
                                                edgeVertex_adj.numpy(), faceEdge_adj, len(edge_ncs), len(face_batch))

            try:
                solid = construct_brep(face_wcs, edge_wcs, faceEdge_adj, edgeVertex_adj.numpy())
            except Exception as e:
                print('B-rep rebuild failed...')
                continue

            random_string = generate_random_string(15)
            write_step_file(solid, f'{args.save_folder}/{random_string}_{i}.step')
            write_stl_file(solid, f'{args.save_folder}/{random_string}_{i}.stl', linear_deflection=0.001,
                           angular_deflection=0.5)

            print(1)


if __name__ == '__main__':
    main()
