import argparse
import os
import random
from tqdm import tqdm
import torch
import pickle
import numpy as np
from model import (AutoencoderKLFastDecode, FaceGeomTransformer, EdgeGeomTransformer,
                   AutoencoderKLFastEncode, AutoencoderKL1DFastDecode)
from diffusion import GraphDiffusion, DDPM
from utils import (pad_and_stack, pad_zero, xe_mask, generate_random_string,
                   remove_box_edge, construct_faceEdge, sort_bbox_multi)
from dataFeature import GraphFeatures
from OCC.Extend.DataExchange import write_stl_file, write_step_file
from brep_functions import joint_optimize, construct_brep
from visualization import *
from generate import get_faceGeom, get_faceGeom_fvf, get_edgeGeom

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_args_gdm():
    parser = argparse.ArgumentParser()
    parser.add_argument('--face_vae_path', type=str, default='checkpoints/furniture/vae_face/epoch_400.pt')
    parser.add_argument('--edge_vae_path', type=str, default='checkpoints/furniture/vae_edge/epoch_400.pt')
    parser.add_argument('--faceBbox_path', type=str, default='checkpoints/furniture/gdm_faceBbox/epoch_3000.pt')
    parser.add_argument('--faceGeom_path', type=str, default='checkpoints/furniture/gdm_faceGeom/epoch_3000.pt')
    parser.add_argument('--vertexGeom_path', type=str, default='checkpoints/furniture/gdm_vertexGeom/epoch_3000.pt')
    parser.add_argument('--edgeGeom_path', type=str, default='checkpoints/furniture/gdm_edgeGeom/epoch_2000.pt')
    parser.add_argument('--hyper_params_path', type=str, default='checkpoints/furniture/gdm_faceBbox/hyper_params.pkl')
    parser.add_argument('--batch_size', type=int, default=8, help='sample batch size')
    parser.add_argument('--save_folder', type=str, default="samples/furniture", help='save folder.')
    args = parser.parse_args()

    return args


def get_topology(files, device):
    """****************Brep Topology****************"""
    edgeVert_adj = []
    edgeFace_adj = []
    faceEdge_adj = []
    vertFace_adj = []
    face_bbox = []
    vert_geom = []
    for file in files:
        with open(os.path.join('data_process/furniture_parsed', file), 'rb') as tf:
            data = pickle.load(tf)
        edgeVert_adj.append(data['edgeCorner_adj'])  # [ne*2, ...]
        edgeFace_adj.append(torch.from_numpy(data['edgeFace_adj']).to(device))
        faceEdge_adj.append(data['faceEdge_adj'])
        face_bbox.append(torch.from_numpy(data['face_bbox_wcs']).to(device))  # [nf*6, ...]
        vert_geom.append(torch.from_numpy(data['corner_unique']).to(device))  # [nv*3,   ]
        vertFace_adj.append(data['vertexFace'])  # [[f1, f2, ...], ...]

    face_bbox, face_mask = pad_and_stack(face_bbox)     # b*nf*6, b*nf

    vert_geom, vert_mask = pad_and_stack(vert_geom)     # b*nv*3, b*nv

    return {"face_mask": face_mask, "edgeVert_adj": edgeVert_adj, "edgeFace_adj": edgeFace_adj,
            "face_bbox": face_bbox, "vert_geom": vert_geom, "vert_mask": vert_mask, "vertFace_adj": vertFace_adj,
            "faceEdge_adj": faceEdge_adj}


def main():
    args = get_args_gdm()

    # Make project directory if not exist
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Load pretrained surface vae (fast encode version)
    face_vae_encoder = AutoencoderKLFastEncode(in_channels=3,
                                               out_channels=3,
                                               down_block_types=('DownEncoderBlock2D', 'DownEncoderBlock2D',
                                                                 'DownEncoderBlock2D', 'DownEncoderBlock2D'),
                                               up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D',
                                                               'UpDecoderBlock2D',
                                                               'UpDecoderBlock2D'),
                                               block_out_channels=(128, 256, 512, 512),
                                               layers_per_block=2,
                                               act_fn='silu',
                                               latent_channels=3,
                                               norm_num_groups=32,
                                               sample_size=512)
    face_vae_encoder.load_state_dict(torch.load(args.face_vae_path), strict=False)
    face_vae_encoder = face_vae_encoder.to(device).eval()

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

    # Initial FaceGeomTransformer
    faceGeom_model = FaceGeomTransformer(n_layers=8, input_dims={'x': 54, 'e': 5, 'y': 12},
                                         hidden_mlp_dims=hyper_params['hidden_mlp_dims'],
                                         hidden_dims=hyper_params['hidden_dims'],
                                         output_dims={'x': 48, 'e': 5, 'y': 0},
                                         act_fn_in=torch.nn.ReLU(), act_fn_out=torch.nn.ReLU())
    faceGeom_model.load_state_dict(torch.load(args.faceGeom_path))
    faceGeom_model = faceGeom_model.to(device).eval()

    # Initial EdgeGeomTransformer
    edgeGeom_model = EdgeGeomTransformer(n_layers=8, face_geom_dim=48, edge_geom_dim=12)
    edgeGeom_model.load_state_dict(torch.load(args.edgeGeom_path))
    edgeGeom_model = edgeGeom_model.to(device).eval()

    ddpm = DDPM(500, device)

    # Initial feature extractor
    extract_feat = GraphFeatures(hyper_params['extract_type'], hyper_params['node_distribution'].shape[0])

    with open('batch_file_test.pkl', 'rb') as f:
        batch_file = pickle.load(f)

    b_each = 32
    for i in tqdm(range(0, len(batch_file), b_each)):

        """****************Brep Topology****************"""
        datas = get_topology(batch_file[i:i + b_each], device)
        faceEdge_adj, vertFace_adj, face_bbox, face_mask, edgeVert_adj, edgeFace_adj, vert_geom, vert_mask = (
            datas['faceEdge_adj'],
            datas["vertFace_adj"],
            datas["face_bbox"],
            datas["face_mask"],
            datas["edgeVert_adj"],
            datas["edgeFace_adj"],
            datas["vert_geom"],
            datas["vert_mask"])
        b = face_bbox.shape[0]
        face_bbox = sort_bbox_multi(face_bbox.reshape(-1, 6)).reshape(b, -1, 6)
        face_bbox *= hyper_params['bbox_scaled']
        vert_geom *= hyper_params['bbox_scaled']

        """****************Face Geometry****************"""
        face_geom = get_faceGeom_fvf(faceEdge_adj, edgeVert_adj, vert_geom, face_bbox, face_mask,
                                     extract_feat, ddpm, edgeFace_adj, device, faceGeom_model)      # b*nf*48

        """****************Edge Geometry****************"""
        face_bbox_geom = torch.cat([face_geom, face_bbox], dim=-1)                # b*nf*54
        edge_geom, edge_mask = get_edgeGeom(
            ddpm, face_bbox_geom, edgeFace_adj, vert_geom, edgeVert_adj, device, edgeGeom_model)

        """****************Construct Brep****************"""
        for j in range(b):
            face_cad, edge_geom_cad = face_bbox_geom[j][face_mask[j]], edge_geom[j][edge_mask[j]]  # nf*54, ne*12
            face_geom_cad, face_bbox_cad = face_cad[:, :-6], face_cad[:, -6:] / hyper_params[
                'bbox_scaled']    # nf*48, nf*6
            vert_geom_cad = vert_geom[j][vert_mask[j]] / hyper_params['bbox_scaled']  # nv*3

            # Decode face geometry
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    face_ncs = face_vae(
                        face_geom_cad.unflatten(-1, torch.Size([16, 3])).permute(0, 2, 1).unflatten(-1, torch.Size(
                            [4, 4])))  # nf*3*32*32
                    face_ncs = face_ncs.permute(0, 2, 3, 1)  # nf*32*32*3

                    # face_wcs = ncs2wcs(face_bbox_each, face_ncs.flatten(1, 2)).unflatten(1, (32, 32))  # nf*32*32*3
                    # continue

                    # Decode edge geometry
                    edge_ncs = edge_vae(
                        edge_geom_cad.reshape(-1, 4, 3).permute(0, 2, 1)).permute(0, 2, 1)  # ne*32*3

            # Get Edge-vert topology
            # edge_wcs = ncs2wcs(edge_bbox, edge_ncs)         # ne*32*3
            edgeFace_cad = edgeFace_adj[j]  # ne*2
            edgeVert_cad = edgeVert_adj[j]  # ne*2
            faceEdge_cad = construct_faceEdge(edgeFace_cad)  # [(edge1, edge2,...), ...]

            assert set(range(face_cad.shape[0])) == set(torch.unique(edgeFace_cad.view(-1)).cpu().numpy().tolist())
            assert set(range(vert_geom_cad.shape[0])) == set(np.unique(edgeVert_cad.reshape(-1)).tolist())

            # joint_optimize:
            # numpy(nf*32*32*3), numpy(ne*32*3), numpy(nf*6), numpy(nv*3),
            # numpy(ne*2), len(list[[edge_id1, ...]...])=nf, int, int
            face_wcs, edge_wcs = joint_optimize(face_ncs.cpu().numpy(), edge_ncs.cpu().numpy(),
                                                face_bbox_cad.cpu().numpy(), vert_geom_cad.cpu().numpy(),
                                                edgeVert_cad, faceEdge_cad, len(edge_ncs), len(face_ncs))

            try:
                solid = construct_brep(face_wcs, edge_wcs, faceEdge_cad, edgeVert_cad)
            except Exception as e:
                print('B-rep rebuild failed...')
                continue

            random_string = generate_random_string(15)
            write_step_file(solid, f'{args.save_folder}/{random_string}_{i}.step')
            write_stl_file(solid, f'{args.save_folder}/{random_string}_{i}.stl', linear_deflection=0.001,
                           angular_deflection=0.5)


if __name__ == '__main__':
    main()
