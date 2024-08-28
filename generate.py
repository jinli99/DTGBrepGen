import argparse
import os
import random
import torch
import pickle
import numpy as np
from model import (AutoencoderKLFastDecode, FaceBboxTransformer, FaceGeomTransformer, EdgeGeomTransformer,
                   VertGeomTransformer, AutoencoderKL1DFastDecode)
from diffusion import GraphDiffusion, DDPM
from utils import (pad_and_stack, pad_zero, xe_mask, make_edge_symmetric, assert_weak_one_hot, ncs2wcs, generate_random_string,
                   remove_box_edge, construct_edgeFace_adj, construct_feTopo, remove_short_edge, reconstruct_vv_adj,
                   construct_vertFace, construct_faceEdge, construct_faceVert, check_step_ok, sort_bbox_multi, construct_fvf_geom)
from dataFeature import GraphFeatures
from OCC.Extend.DataExchange import write_stl_file, write_step_file
from brep_functions import (
    scale_surf, joint_optimize, construct_brep)
from visualization import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_args_gdm():
    parser = argparse.ArgumentParser()
    parser.add_argument('--face_vae_path', type=str, default='checkpoints/furniture/vae_face/epoch_400.pt')
    parser.add_argument('--edge_vae_path', type=str, default='checkpoints/furniture/vae_edge/epoch_400.pt')
    parser.add_argument('--faceBbox_path', type=str, default='checkpoints/furniture/gdm_faceBbox/epoch_3000.pt')
    parser.add_argument('--faceGeom_path', type=str, default='checkpoints/furniture/gdm_faceGeom_03/epoch_3000.pt')
    parser.add_argument('--vertGeom_path', type=str, default='checkpoints/furniture/gdm_vertGeom/epoch_3000.pt')
    parser.add_argument('--edgeGeom_path', type=str, default='checkpoints/furniture/gdm_edgeGeom/epoch_2000.pt')
    parser.add_argument('--hyper_params_path', type=str, default='checkpoints/furniture/gdm_faceBbox/hyper_params.pkl')
    parser.add_argument('--batch_size', type=int, default=8, help='sample batch size')
    parser.add_argument('--save_folder', type=str, default="samples/furniture", help='save folder.')
    args = parser.parse_args()

    return args


def get_ok_step(batch_size, mode='train'):
    with open("data_process/furniture_data_split_6bit.pkl", 'rb') as f:
        train_data = pickle.load(f)[mode]
    batch_file = []
    for i in range(batch_size):
        while True:
            path = random.choice(train_data)
            if check_step_ok(path):
                batch_file.append(path)
                break

    return batch_file


def get_topology(files, device):
    """****************Brep Topology****************"""
    e = []
    edgeVert_adj = []
    edgeFace_adj = []
    vv_list = []
    for file in files:
        with open(os.path.join('data_process/furniture_parsed', file), 'rb') as tf:
            data = pickle.load(tf)
        e.append(data['fe_topo'])                    # [nf*nf, ***]
        edgeVert_adj.append(data['edgeCorner_adj'])  # [ne*2, ...]
        vv_list.append(data['vv_list'])              # list[(v1, v2, edge_idx), ...]
        edgeFace_adj.append(torch.from_numpy(data['edgeFace_adj']).to(device))

    nf = max([len(i) for i in e])
    e = [pad_zero(i, max_len=nf, dim=1) for i in e]                                 # [(nf*nf, nf), ...]
    face_mask = torch.stack([torch.from_numpy(i[1]) for i in e], dim=0).to(device)  # b*nf
    e = torch.stack([torch.from_numpy(i[0]) for i in e], dim=0).to(device)          # b*nf*nf

    return {"e": e, "face_mask": face_mask, "edgeVert_adj": edgeVert_adj,
            "edgeFace_adj": edgeFace_adj, "vv_list": vv_list}


def get_manual_topology(b, device):
    """****************Brep Topology****************"""
    e = []
    edgeVert_adj = []
    edgeFace_adj = []
    vv_list = []
    for i in range(b):
        e.append(np.array([[0, 1, 0, 1, 1, 1], [1, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 1], [1, 0, 1, 0, 1, 1], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]]))                    # [nf*nf, ***]
        edgeVert_adj.append(np.array([[0, 1], [1, 2], [2, 3], [3, 0], [1, 4], [4, 5], [2, 5], [5, 7], [6, 7], [4, 6], [0, 6], [3, 7]]))    # [ne*2, ...]
        vv_list.append([(v1, v2, edge_id) for edge_id, (v1, v2) in enumerate(edgeVert_adj[i])])                # list[(v1, v2, edge_idx), ...]
        edgeFace_adj.append(torch.from_numpy(np.array([[0, 5], [0, 1], [0, 4], [0, 3], [1, 5], [1, 2], [1, 4], [2, 4], [2, 3], [2, 5], [3, 5], [3, 4]])).to(device))

    nf = max([len(i) for i in e])
    e = [pad_zero(i, max_len=nf, dim=1) for i in e]                                 # [(nf*nf, nf), ...]
    face_mask = torch.stack([torch.from_numpy(i[1]) for i in e], dim=0).to(device)  # b*nf
    e = torch.stack([torch.from_numpy(i[0]) for i in e], dim=0).to(device)          # b*nf*nf

    return {"e": e, "face_mask": face_mask, "edgeVert_adj": edgeVert_adj,
            "edgeFace_adj": edgeFace_adj, "vv_list": vv_list}


def get_faceBbox(m, e, face_mask, ddpm, extract_feat, faceBbox_model, device):
    """****************Face Bbox****************"""

    b, nf = e.shape[:2]
    x = torch.randn((b, nf, 6), device=device, dtype=torch.float)     # b*nf*6
    e = torch.nn.functional.one_hot(e, num_classes=m)  # b*nf*nf*m
    x, e = xe_mask(x=x, e=e, node_mask=face_mask, check_sym=False)
    e = make_edge_symmetric(e)    # b*n*n*m
    assert_weak_one_hot(e)

    with torch.no_grad():
        # Extract features
        feat = extract_feat(e, face_mask)
        for t in range(ddpm.T-1, -1, -1):
            x_feat = torch.cat((x, feat[0]), dim=-1).float()  # b*n*12
            e_feat = torch.cat((e, feat[1]), dim=-1).float()  # b*n*n*m
            y_feat = torch.cat(
                (feat[2], torch.tensor([ddpm.normalize_t(t)]*b, device=device).unsqueeze(-1)), dim=-1).float()  # b*12

            # Prediction
            x_pred = faceBbox_model(x_feat, e_feat, y_feat, face_mask)  # b*n*6

            # Sample x
            x = ddpm.p_sample(x, x_pred, torch.tensor([t], device=device))  # b*n*6

            # Mask
            x, _ = xe_mask(x, node_mask=face_mask)   # b*n*6

    return x    # b*n*6


def get_vertGeom(vv_list, ddpm, face_bbox, extract_feat, edgeVert_adj, edgeFace_adj, device, vertGeom_model):
    with torch.no_grad():
        b = face_bbox.shape[0]
        vv_adj = [reconstruct_vv_adj(edgeVert_adj[i].max() + 1, np.array(vv_list[i])) for i in
                  range(b)]  # [nv*nv ,...]
        nv = max([len(i) for i in vv_adj])
        vertFace_adj = [construct_vertFace(vv_adj[i].shape[0], edgeVert_adj[i], edgeFace_adj[i].cpu().numpy()) for i in
                        range(b)]  # list[[[face_1, face_2,...], ...],...]
        vertFace_info = [[face_bbox[k][j] for j in i] for k, i in enumerate(vertFace_adj)]  # [[vf1*6, vf2*6, ...], ...]
        vf = max([max([len(j) for j in i]) for i in vertFace_info])
        vertFace_info = [pad_and_stack(i, max_n=vf) for i in vertFace_info]  # [(nv*vf*6, nv*vf), ...]
        vFace_mask, assert_mask = pad_and_stack([i[1] for i in vertFace_info], max_n=nv)  # b*nv*vf, b*nv
        vertFace_info, _ = pad_and_stack([i[0] for i in vertFace_info], max_n=nv)  # b*nv*vf*6

        vv_adj = [pad_zero(i, max_len=nv, dim=1) for i in vv_adj]  # [(nv*nv, nv), ...]
        vert_mask = torch.stack([torch.from_numpy(i[1]) for i in vv_adj]).to(device)      # b*nv
        assert torch.all(assert_mask == vert_mask)
        vv_adj = torch.stack([torch.from_numpy(i[0]) for i in vv_adj], dim=0).to(device)  # b*nv*nv
        vv_adj = torch.nn.functional.one_hot(vv_adj, num_classes=2)  # b*nv*nv*2

        vert_geom = torch.randn((b, nv, 3), device=device, dtype=torch.float)
        vert_geom, e = xe_mask(x=vert_geom, e=vv_adj, node_mask=vert_mask)  # b*nv*3, b*nv*nv*2
        feat = extract_feat(e, vert_mask)
        for t in range(ddpm.T - 1, -1, -1):
            # Extract features
            x_t_feat = torch.cat((vert_geom, feat[0]), dim=-1).float()  # b*nv*9
            e_t_feat = torch.cat((e, feat[1]), dim=-1).float()  # b*nv*nv*2
            y_feat = torch.cat(
                (feat[2], torch.tensor([ddpm.normalize_t(t)] * b, device=device).unsqueeze(-1)),
                dim=-1).float()  # b*12

            # Predict start
            pred_noise = vertGeom_model(x_t_feat, e_t_feat, vertFace_info, y_feat, vert_mask, vFace_mask)  # b*nv*3

            # Sample x
            vert_geom = ddpm.p_sample(vert_geom, pred_noise, torch.tensor([t], device=device))  # b*nv*3
            vert_geom, _ = xe_mask(x=vert_geom, node_mask=vert_mask)  # b*nv*3

        return vert_geom, vert_mask, vertFace_adj


def get_faceGeom(vertFace_adj, vert_geom, face_bbox, face_mask, extract_feat, ddpm, edgeFace_adj, device, faceGeom_model):
    with torch.no_grad():
        b, nf = face_bbox.shape[:2]
        faceVert_adj = [construct_faceVert(i) for i in vertFace_adj]  # [[[v1, v2, ...],...],...]
        faceVert_geom = [[vert_geom[k][j] for j in i] for k, i in enumerate(faceVert_adj)]  # [[fv*3, fv*3, ...], ...]
        fv = max([max([len(j) for j in i]) for i in faceVert_geom])
        faceVert_geom = [pad_and_stack(i, max_n=fv) for i in faceVert_geom]  # [(nf*fv*3, nf*fv), ...]
        faceVert_mask, assert_mask = pad_and_stack([i[1] for i in faceVert_geom], max_n=nf)  # b*nf*fv, b*nf
        assert torch.all(assert_mask == face_mask)
        faceVert_geom, _ = pad_and_stack([i[0] for i in faceVert_geom], max_n=nf)  # b*nf*fv*3
        fe_topo = [pad_zero(construct_feTopo(i).cpu().numpy(), max_len=face_mask.shape[1], dim=1)[0] for i in
                   edgeFace_adj]  # list[shape:nf*nf, ...]
        fe_topo = torch.from_numpy(np.stack(fe_topo)).to(device)  # b*nf*nf
        x = torch.randn((fe_topo.shape[0], fe_topo.shape[1], 48), device=device)  # b*nf*48
        e = torch.nn.functional.one_hot(fe_topo, num_classes=5)  # b*n*n*m
        x, e = xe_mask(x=x, e=e, node_mask=face_mask)  # b*nf*48, b*n*n*m
        feat = extract_feat(e, face_mask)
        for t in range(ddpm.T - 1, -1, -1):
            # Extract features
            x_feat = torch.cat((x, feat[0]), dim=-1).float()  # b*nf*54
            e_feat = torch.cat((e, feat[1]), dim=-1).float()  # b*nf*nf*m
            y_feat = torch.cat(
                (feat[2], torch.tensor([ddpm.normalize_t(t)] * b, device=device).unsqueeze(-1)),
                dim=-1).float()  # b*12

            # Predict start
            pred_noise = faceGeom_model(x_feat, e_feat, y_feat, face_bbox, faceVert_geom, faceVert_mask,
                                        face_mask)  # b*nf*48

            # Sample x
            x = ddpm.p_sample(x, pred_noise, torch.tensor([t], device=device))  # b*nf*48
            x, _ = xe_mask(x=x, node_mask=face_mask)  # b*nf*48

        return x


# add fvf information
def get_faceGeom_fvf(faceEdge_adj, edgeVert_adj, vert_geom, face_bbox, face_mask, extract_feat, ddpm, edgeFace_adj, device, faceGeom_model):
    with torch.no_grad():
        b, nf = face_bbox.shape[:2]
        fvf_mask = [pad_zero(construct_feTopo(i).cpu().numpy(), max_len=nf, dim=1)[0] for i in
                    edgeFace_adj]  # list[shape:nf*nf, ...]
        fvf_mask = torch.from_numpy(np.stack(fvf_mask)).to(device)  # b*nf*nf

        # b*nf*nf*(m-1)*2*3
        fvf_geom = torch.stack([construct_fvf_geom(faceEdge_adj[i], edgeVert_adj[i], vert_geom[i], fvf_mask[i], 4, nf=nf) for i in range(b)])

        x = torch.randn((b, nf, 48), device=device)  # b*nf*48
        e = torch.nn.functional.one_hot(fvf_mask, num_classes=5)  # b*n*n*m
        x, e = xe_mask(x=x, e=e, node_mask=face_mask)  # b*nf*48, b*n*n*m
        feat = extract_feat(e, face_mask)
        for t in range(ddpm.T - 1, -1, -1):
            # Extract features
            x_feat = torch.cat((x, feat[0]), dim=-1).float()   # b*nf*54
            e_feat = fvf_geom.clone().detach().float()                # b*nf*nf*fv*2*3
            y_feat = torch.cat(
                (feat[2], torch.tensor([ddpm.normalize_t(t)] * b, device=device).unsqueeze(-1)),
                dim=-1).float()  # b*12

            # Predict start
            pred_noise = faceGeom_model(x_feat, e_feat, y_feat, face_bbox, fvf_mask, face_mask)  # b*n*48

            # Sample x
            x = ddpm.p_sample(x, pred_noise, torch.tensor([t], device=device))  # b*nf*48
            x, _ = xe_mask(x=x, node_mask=face_mask)  # b*nf*48

        return x


def get_edgeGeom(ddpm, face_bbox_geom, edgeFace_adj, vert_geom, edgeVert_adj, device, edgeGeom_model):
    with torch.no_grad():
        b = face_bbox_geom.shape[0]
        edgeFace_info = [face[adj] for face, adj in zip(face_bbox_geom, edgeFace_adj)]  # list[shape:ne*2*54]
        edgeFace_info, edge_mask = pad_and_stack(edgeFace_info)  # b*ne*2*54, b*ne
        ne = edgeFace_info.shape[1]
        edge_geom = torch.randn((b, ne, 12), device=device)      # b*ne*12
        edge_geom, _ = xe_mask(x=edge_geom, node_mask=edge_mask)
        edgeVert_geom = [vert[torch.from_numpy(adj).to(device)] for vert, adj in
                         zip(vert_geom, edgeVert_adj)]  # [ne*2*3, ...]
        edgeVert_geom, assert_mask = pad_and_stack(edgeVert_geom)  # b*ne*2*3, b*ne
        assert torch.all(assert_mask == edge_mask)
        for t in range(ddpm.T - 1, -1, -1):
            pred_noise = edgeGeom_model(
                edge_geom, edgeFace_info, edgeVert_geom, edge_mask, torch.tensor(
                    [ddpm.normalize_t(t)] * b, device=device).unsqueeze(-1))  # b*ne*12

            edge_geom = ddpm.p_sample(edge_geom, pred_noise, torch.tensor([t], device=device))  # b*ne*12

            edge_geom, _ = xe_mask(x=edge_geom, node_mask=edge_mask)

        return edge_geom, edge_mask


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
    hidden_mlp_dims = {'x': 256, 'e': 128, 'y': 128}
    hidden_dims = {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128}
    faceBbox_model = FaceBboxTransformer(n_layers=5, input_dims={'x': 12, 'e': m, 'y': 12},
                                         hidden_mlp_dims=hidden_mlp_dims,hidden_dims=hidden_dims, output_dims={'x': 6},
                                         act_fn_in=torch.nn.ReLU(), act_fn_out=torch.nn.ReLU())
    faceBbox_model.load_state_dict(torch.load(args.faceBbox_path))
    faceBbox_model = faceBbox_model.to(device).eval()

    # Initial FaceGeomTransformer
    faceGeom_model = FaceGeomTransformer(n_layers=8, input_dims={'x': 54, 'e': 5, 'y': 12},
                                         hidden_mlp_dims=hyper_params['hidden_mlp_dims'],
                                         hidden_dims=hyper_params['hidden_dims'],
                                         output_dims={'x': 48, 'e': 5, 'y': 0},
                                         act_fn_in=torch.nn.ReLU(), act_fn_out=torch.nn.ReLU())
    faceGeom_model.load_state_dict(torch.load(args.faceGeom_path))
    faceGeom_model = faceGeom_model.to(device).eval()

    # Initial vertGeomTransformer
    vertGeom_model = VertGeomTransformer(n_layers=8, input_dims={'x': 9, 'y': 12}, hidden_mlp_dims=hidden_mlp_dims,
                                         hidden_dims=hidden_dims, output_dims={'x': 3},
                                         act_fn_in=torch.nn.ReLU(), act_fn_out=torch.nn.ReLU())
    vertGeom_model.load_state_dict(torch.load(args.vertGeom_path))
    vertGeom_model = vertGeom_model.to(device).eval()

    # Initial EdgeGeomTransformer
    edgeGeom_model = EdgeGeomTransformer(n_layers=8, face_geom_dim=48, edge_geom_dim=12)
    edgeGeom_model.load_state_dict(torch.load(args.edgeGeom_path))
    edgeGeom_model = edgeGeom_model.to(device).eval()

    ddpm = DDPM(500, device)

    # Initial feature extractor
    extract_feat = GraphFeatures(hyper_params['extract_type'], hyper_params['node_distribution'].shape[0])

    # batch_file = get_ok_step(50, mode='test')
    # with open('batch_file_test.pkl', 'wb') as f:
    #     pickle.dump(batch_file, f)

    with open('batch_file_test.pkl', 'rb') as f:
        batch_file = pickle.load(f)

    b = 8
    for i in range(0, 8, b):

        """****************Brep Topology****************"""
        datas = get_topology(batch_file[i:i + b], device)
        e, face_mask, edgeVert_adj, edgeFace_adj, vv_list = datas["e"], datas["face_mask"],  datas["edgeVert_adj"], datas["edgeFace_adj"], datas["vv_list"]

        """******************Face Bbox******************"""
        face_bbox = get_faceBbox(m, e, face_mask, ddpm, extract_feat, faceBbox_model, device)   # b*nf*6
        face_bbox = sort_bbox_multi(face_bbox.reshape(-1, 6)).reshape(b, -1, 6)
        face_bbox = face_bbox / hyper_params['bbox_scaled']

        # Remove faces and edges
        temp = [remove_box_edge(face_bbox[i][face_mask[i]], edgeFace_adj[i]) for i in range(batch_size)]
        face_bbox = [i[0] for i in temp]                 # list[shape:nf*6, ...]
        edgeFace_adj = [i[1] for i in temp]              # list[shape:ne*2,...]
        face_bbox, face_mask = pad_and_stack(face_bbox)  # b*nf*6, b*nf
        face_bbox *= hyper_params['bbox_scaled']

        """****************vert Geometry****************"""
        vert_geom, vert_mask, vertFace_adj = get_vertGeom(
            vv_list, ddpm, face_bbox, extract_feat, edgeVert_adj, edgeFace_adj, device, vertGeom_model)

        """****************Face Geometry****************"""
        face_geom = get_faceGeom(vertFace_adj, vert_geom, face_bbox, face_mask, extract_feat, ddpm, edgeFace_adj, device,
                                 faceGeom_model)    # b*nf*48

        """****************Edge Geometry****************"""
        face_bbox_geom = torch.cat([face_geom, face_bbox], dim=-1)                # b*nf*54
        edge_geom, edge_mask = get_edgeGeom(
            ddpm, face_bbox_geom, edgeFace_adj, vert_geom, edgeVert_adj, device, edgeGeom_model)

        """****************Construct Brep****************"""
        for j in range(batch_size):
            face_cad, edge_geom_cad = face_bbox_geom[j][face_mask[j]], edge_geom[j][edge_mask[j]]             # nf*54, ne*12
            face_geom_cad, face_bbox_cad = face_cad[:, :-6], face_cad[:, -6:]/hyper_params['bbox_scaled']     # nf*48, nf*6
            vert_geom_cad = vert_geom[j][vert_mask[j]] / hyper_params['bbox_scaled']                          # nv*3

            # Decode face geometry
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    face_ncs = face_vae(
                        face_geom_cad.unflatten(-1, torch.Size([16, 3])).permute(0, 2, 1).unflatten(-1, torch.Size([4, 4])))   # nf*3*32*32
                    face_ncs = face_ncs.permute(0, 2, 3, 1)    # nf*32*32*3

                    # face_wcs = ncs2wcs(face_bbox_each, face_ncs.flatten(1, 2)).unflatten(1, (32, 32))  # nf*32*32*3
                    # continue

                    # Decode edge geometry
                    edge_ncs = edge_vae(
                        edge_geom_cad.reshape(-1, 4, 3).permute(0, 2, 1)).permute(0, 2, 1)   # ne*32*3

            # Get Edge-vert topology
            # edge_wcs = ncs2wcs(edge_bbox, edge_ncs)         # ne*32*3
            edgeFace_cad = edgeFace_adj[j]                    # ne*2
            edgeVert_cad = edgeVert_adj[j]                    # ne*2
            faceEdge_cad = construct_faceEdge(edgeFace_cad)   # [(edge1, edge2,...), ...]

            # Remove short edge
            # edge_wcs, keep_edge_idx = remove_short_edge(edge_wcs, threshold=0.1)
            # edgeFace_adj = edgeFace_adj[keep_edge_idx]

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
