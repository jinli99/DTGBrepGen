import argparse
import os
import torch
import pickle
from model import AutoencoderKLFastDecode, GraphTransformer, EdgeTransformer, AutoencoderKL1DFastDecode
from diffusion import GraphDiffusion, EdgeDiffusion
from utils import pad_and_stack, xe_mask, make_edge_symmetric, assert_weak_one_hot, ncs2wcs, generate_random_string, remove_box_edge
from dataFeature import GraphFeatures
from OCC.Extend.DataExchange import write_stl_file, write_step_file
from brep_functions import (
    scale_surf, construct_edgeFace_adj, manual_edgeVertex_topology,
    optimize_edgeVertex_topology, joint_optimize, construct_brep)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_args_gdm():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surf_vae_path', type=str, default='checkpoints/furniture/vae_surf/epoch_200.pt')
    parser.add_argument('--edge_vae_path', type=str, default='checkpoints/furniture/vae_edge/epoch_400.pt')
    parser.add_argument('--surf_gdm_path', type=str, default='checkpoints/furniture/gdm_surf/test1/epoch_100.pt')
    parser.add_argument('--edge_gdm_path', type=str, default='checkpoints/furniture/gdm_edge/epoch_100.pt')
    parser.add_argument('--hyper_params_path', type=str, default='checkpoints/furniture/gdm_surf/test1/hyper_params.pkl')
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
    surf_vae = AutoencoderKLFastDecode(
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
    surf_vae.load_state_dict(torch.load(args.surf_vae_path), strict=False)     # inputs: points_batch*3*4*4
    surf_vae = surf_vae.to(device).eval()

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
    surf_model = GraphTransformer(n_layers=hyper_params['n_layers'], input_dims=hyper_params['input_dims'],
                                  hidden_mlp_dims=hyper_params['hidden_mlp_dims'],
                                  hidden_dims=hyper_params['hidden_dims'], output_dims=hyper_params['output_dims'],
                                  act_fn_in=torch.nn.ReLU(), act_fn_out=torch.nn.ReLU())
    surf_model.load_state_dict(torch.load(args.surf_gdm_path))
    surf_model = surf_model.to(device).eval()
    diff = GraphDiffusion(m, hyper_params['edge_marginals'], device)

    # Initial EdgeTransformer and EdgeDiffusion
    edge_model = EdgeTransformer(n_layers=12, surf_geom_dim=48, edge_geom_dim=12)
    edge_model.load_state_dict(torch.load(args.edge_gdm_path))
    edge_model = edge_model.to(device).eval()
    edge_diff = EdgeDiffusion(device)

    # Initial feature extractor
    extract_feat = GraphFeatures(hyper_params['extract_type'], hyper_params['node_distribution'].shape[0])

    num_nodes = torch.distributions.Categorical(
        hyper_params['node_distribution']).sample(torch.Size([args.batch_size])).to(device)  # b
    x, node_mask = pad_and_stack([torch.randn((i, hyper_params['diff_dim']), device=device) for i in num_nodes])    # b*n*54, b*n
    n = x.shape[1]
    e = torch.multinomial(hyper_params['edge_marginals'].unsqueeze(0).expand(batch_size*n*n, m),
                          num_samples=1, replacement=True).reshape(batch_size, n, n)   # b*n*n
    e = torch.nn.functional.one_hot(e, num_classes=m).to(device).float()  # b*n*n*m
    _, e = xe_mask(e=e, node_mask=node_mask, check_sym=False)
    e = make_edge_symmetric(e)   # b*n*n*m
    assert_weak_one_hot(e)

    with torch.no_grad():
        # with torch.cuda.amp.autocast():

            # Diffusion sampling process, surface geometry and surface bbox
            for t in range(diff.T-1, -1, -1):
                # Extract features
                feat = extract_feat(e, node_mask)
                x_feat = torch.cat((x, feat[0]), dim=-1).float()  # b*n*60
                e_feat = torch.cat((e, feat[1]), dim=-1).float()  # b*n*n*m
                y_feat = torch.cat(
                    (feat[2], torch.tensor([diff.normalize_t(t)]*batch_size, device=device).unsqueeze(-1)), dim=-1).float()  # b*12

                # Prediction
                x_pred, e_pred, y_pred = surf_model(x_feat, e_feat, y_feat, node_mask)  # b*n*54, b*n*n*m, b*0
                # print("time:", t, torch.nn.functional.softmax(e_pred.mean(0).mean(0).mean(0)))

                # Sample x
                x = diff.x_p_sample(x, x_pred, torch.tensor([t], device=device))  # b*n*54

                # Sample e
                e = diff.e_p_sample(e_pred, e, torch.tensor([t]*batch_size, device=device).unsqueeze(-1))  # b*n*n*m

                # Mask
                x, e = xe_mask(x, e, node_mask)   # b*n*54, b*n*n*m

            # Remove faces and edges
            edgeFace_adj = construct_edgeFace_adj(e[..., 1:].sum(-1), node_mask=node_mask)     # list[shape:ne*2,...]
            temp = [remove_box_edge(x[i, :, -6:].detach().cpu(), edgeFace_adj[i].detach().cpu()) for i in range(batch_size)]
            remove_box_idx_list = [i[0] for i in temp]
            edgeFace_adj = [i[1] for i in temp]   # list[shape:ne*2,...]
            for i, remove_box_idx in enumerate(remove_box_idx_list):
                node_mask[i][remove_box_idx] = False
            e = None   # e is invalid!!!!!!
            x, _ = xe_mask(x=x, node_mask=node_mask)

            # Sample Edge geometry and bbox
            edge_surfInfo = [x_batch[adj] for x_batch, adj in zip(x, edgeFace_adj)]   # list[shape:ne*2*54]
            # edgeFace_adj, edge_mask = pad_and_stack(edgeFace_adj)      # b*ne*2, b*ne
            edge_surfInfo, edge_mask = pad_and_stack(edge_surfInfo)            # b*ne*2*54, b*ne
            ne = edge_surfInfo.shape[1]
            edge = torch.randn((batch_size, ne, 18), device=device)    # b*ne*18
            for t in range(edge_diff.T - 1, -1, -1):
                pred_noise = edge_model(edge, edge_surfInfo, edge_mask, torch.tensor([edge_diff.normalize_t(t)]*batch_size, device=device).unsqueeze(-1))
                edge = edge_diff.p_sample(edge, pred_noise, torch.tensor([t], device=device))    # b*ne*18
                edge, _ = xe_mask(x=edge, node_mask=edge_mask)

            """********Construct Brep********"""
            for i in range(batch_size):
                x_batch, edge_batch = x[i][node_mask[i]], edge[i][edge_mask[i]]        # nf*54, ne*18
                face_geom, face_bbox = x_batch[:, :-6], x_batch[:, -6:]/hyper_params['bbox_scaled']   # nf*48, nf*6
                edge_geom, edge_bbox = edge_batch[:, :-6], edge_batch[:, -6:]/hyper_params['bbox_scaled']   # ne*12, ne*6

                # Decode surface geometry
                surf_ncs = surf_vae(
                    face_geom.unflatten(-1, torch.Size([16, 3])).permute(0, 2, 1).unflatten(-1, torch.Size([4, 4])))   # nf*3*32*32
                surf_ncs = surf_ncs.permute(0, 2, 3, 1).detach().cpu()    # nf*32*32*3

                # Get Edge-Vertex Topology
                edge_ncs = edge_vae(
                    edge_geom.reshape(-1, 4, 3).permute(0, 2, 1)).permute(0, 2, 1).detach().cpu()   # ne*32*3
                edge_wcs = ncs2wcs(edge_bbox.detach().cpu(), edge_ncs)    # ne*32*3
                assert edgeFace_adj[i].amax() == x_batch.shape[0] - 1
                assert torch.all(torch.isin(torch.arange(x_batch.shape[0], device=edgeFace_adj[i].device), edgeFace_adj[i].reshape(-1)))
                unique_vertex_pnt, edgeVertex_adj, faceEdge_adj = optimize_edgeVertex_topology(
                    edge_wcs.detach(), edgeFace_adj[i].detach().cpu())

                """joint_optimize: 
                numpy(nf*32*32*3), numpy(ne*32*3), numpy(nf*6), numpy(nv*3), 
                numpy(ne*2), len(list[[edge_id1, ...]...])=nf, int, int """
                surf_wcs, edge_wcs = joint_optimize(surf_ncs.numpy(), edge_ncs.numpy(),
                                                    face_bbox.detach().cpu().numpy(), unique_vertex_pnt.numpy(),
                                                    edgeVertex_adj.numpy(), faceEdge_adj, len(edge_ncs), len(x_batch))

                try:
                    solid = construct_brep(surf_wcs, edge_wcs, faceEdge_adj, edgeVertex_adj.numpy())
                except Exception as e:
                    print('B-rep rebuild failed...')
                    continue

                random_string = generate_random_string(15)
                write_step_file(solid, f'{args.save_folder}/{random_string}_{i}.step')
                write_stl_file(solid, f'{args.save_folder}/{random_string}_{i}.stl', linear_deflection=0.001,
                               angular_deflection=0.5)

                print(1)

    # # Decode surface geometry points
    # with torch.no_grad():
    #     with torch.cuda.amp.autocast():
    #
    #         bbox = (x[..., -6:]/hyper_params['bbox_scaled']).detach().cpu()   # b*n*6
    #         surf_wcs = scale_surf(bbox, surf_ncs, node_mask.detach().cpu())   # list[n*(32*32)*3]


if __name__ == '__main__':
    main()
