import argparse
import os

import pickle
from model import (AutoencoderKLFastDecode, FaceGeomTransformer, EdgeGeomTransformer,
                   VertGeomTransformer, AutoencoderKLFastEncode, AutoencoderKL1DFastDecode)
from diffusion import DDPM
from utils import (pad_and_stack, pad_zero, xe_mask, generate_random_string, construct_feTopo, reconstruct_vv_adj, construct_vertFace, construct_faceEdge, construct_faceVert)
from geometry.dataFeature import GraphFeatures
from OCC.Extend.DataExchange import write_stl_file, write_step_file
from brepBuild import (
    joint_optimize, construct_brep)
from visualization import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_args_gdm():
    parser = argparse.ArgumentParser()
    parser.add_argument('--face_vae_path', type=str, default='checkpoints/furniture/vae_face/epoch_400.pt')
    parser.add_argument('--edge_vae_path', type=str, default='checkpoints/furniture/vae_edge/epoch_400.pt')
    parser.add_argument('--faceBbox_path', type=str, default='checkpoints/furniture/gdm_faceBbox/epoch_3000.pt')
    parser.add_argument('--faceGeom_path', type=str, default='checkpoints/furniture/gdm_faceGeom/epoch_3000.pt')
    parser.add_argument('--vertexGeom_path', type=str, default='checkpoints/furniture/gdm_vertGeom/epoch_3000.pt')
    parser.add_argument('--edgeGeom_path', type=str, default='checkpoints/furniture/gdm_edgeGeom/epoch_1000.pt')
    parser.add_argument('--hyper_params_path', type=str, default='checkpoints/furniture/gdm_faceBbox/hyper_params.pkl')
    parser.add_argument('--batch_size', type=int, default=8, help='sample batch size')
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

    # Load pretrained surface vae (fast encode version)
    face_vae_encoder = AutoencoderKLFastEncode(in_channels=3,
                                       out_channels=3,
                                       down_block_types=('DownEncoderBlock2D', 'DownEncoderBlock2D',
                                                         'DownEncoderBlock2D', 'DownEncoderBlock2D'),
                                       up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D',
                                                       'UpDecoderBlock2D'),
                                       block_out_channels=(128, 256, 512, 512),
                                       layers_per_block=2,
                                       act_fn='silu',
                                       latent_channels=3,
                                       norm_num_groups=32,
                                       sample_size=512,
                                       )
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

    # Initial GraphTransformer and GraphDiffusion
    hidden_mlp_dims = {'x': 256, 'e': 128, 'y': 128}
    hidden_dims = {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128}

    # Initial FaceGeomTransformer
    faceGeom_model = FaceGeomTransformer(n_layers=8, input_dims={'x': 54, 'e': 5, 'y': 12},
                                         hidden_mlp_dims=hyper_params['hidden_mlp_dims'],
                                         hidden_dims=hyper_params['hidden_dims'],
                                         output_dims={'x': 48, 'e': 5, 'y': 0},
                                         act_fn_in=torch.nn.ReLU(), act_fn_out=torch.nn.ReLU())
    faceGeom_model.load_state_dict(torch.load(args.faceGeom_path))
    faceGeom_model = faceGeom_model.to(device).eval()

    # Initial VertGeomTransformer
    vertexGeom_model = VertGeomTransformer(n_layers=8, input_dims={'x': 9, 'y': 12}, hidden_mlp_dims=hidden_mlp_dims,
                                             hidden_dims=hidden_dims, output_dims={'x': 3},
                                             act_fn_in=torch.nn.ReLU(), act_fn_out=torch.nn.ReLU())
    vertexGeom_model.load_state_dict(torch.load(args.vertexGeom_path))
    vertexGeom_model = vertexGeom_model.to(device).eval()

    # Initial EdgeGeomTransformer
    edgeGeom_model = EdgeGeomTransformer(n_layers=6, face_geom_dim=48, edge_geom_dim=12)
    edgeGeom_model.load_state_dict(torch.load(args.edgeGeom_path))
    edgeGeom_model = edgeGeom_model.to(device).eval()

    ddpm = DDPM(500, device)

    # Initial feature extractor
    extract_feat = GraphFeatures(hyper_params['extract_type'], hyper_params['node_distribution'].shape[0])

    """****************Brep Topology****************"""
    e = []
    edgeVertex = []
    edgeFace = []
    vv_list = []
    face_bbox = []
    vertexFace = []
    with open("data_process/furniture_data_split_6bit.pkl", 'rb') as f:
        train_data = pickle.load(f)['train']
    for i in range(batch_size):
        while True:
            path = random.choice(train_data)
            with open(os.path.join('data_process/furniture_parsed', path), 'rb') as f:
                data = pickle.load(f)
                if len(data['faceEdge_adj']) <= 50 and data['fef_adj'].max() < m and max([len(j) for j in data['faceEdge_adj']]) <= 30:
                    e.append(data['fef_adj'])                     # [nf*nf, ***]
                    edgeVertex.append(data['edgeVert_adj'])     # [ne*2, ...]
                    vv_list.append(data['vv_list'])               # list[(v1, v2, edge_idx), ...]
                    edgeFace.append(torch.from_numpy(data['edgeFace_adj']).to(device))
                    face_bbox.append(torch.from_numpy(data['face_bbox_wcs']).to(device))      # [nf*6, ...]
                    vertexFace.append(data['vertexFace'])     # [[f1, f2, ...], ...]
                    break

    face_bbox, node_mask = pad_and_stack(face_bbox)           # b*nf*6, b*nf
    face_bbox *= hyper_params['bbox_scaled']

    with torch.no_grad():

        """****************Vertex Geometry****************"""
        nf = face_bbox.shape[1]
        vv_adj = [reconstruct_vv_adj(edgeVertex[i].max()+1, np.array(vv_list[i])) for i in range(batch_size)]   # [nv*nv ,...]
        nv = max([len(i) for i in vv_adj])
        vertexFace = [construct_vertFace(
            vv_adj[i].shape[0], edgeVertex[i], edgeFace[i].cpu().numpy()) for i in range(batch_size)]    # list[[[face_1, face_2,...], ...],...]
        vertex_faceInfo = [[face_bbox[k][j] for j in i] for k, i in enumerate(vertexFace)]    # [[vf1*6, vf2*6, ...], ...]
        vf = max([max([len(j) for j in i]) for i in vertex_faceInfo])
        vertex_faceInfo = [pad_and_stack(i, max_n=vf) for i in vertex_faceInfo]               # [(nv*vf*6, nv*vf), ...]
        vFace_mask, assert_mask = pad_and_stack([i[1] for i in vertex_faceInfo], max_n=nv)    # b*nv*vf, b*nv
        vertex_faceInfo, _ = pad_and_stack([i[0] for i in vertex_faceInfo], max_n=nv)         # b*nv*vf*6

        vv_adj = [pad_zero(i, max_len=nv, dim=1) for i in vv_adj]    # [(nv*nv, nv), ...]
        vertex_mask = torch.stack([torch.from_numpy(i[1]) for i in vv_adj]).to(device)     # b*nv
        assert torch.all(assert_mask == vertex_mask)
        vv_adj = torch.stack([torch.from_numpy(i[0]) for i in vv_adj], dim=0).to(device)   # b*nv*nv
        vv_adj = torch.nn.functional.one_hot(vv_adj, num_classes=2)       # b*nv*nv*2

        vertex_geom = torch.randn((batch_size, nv, 3), device=device, dtype=torch.float)
        vertex_geom, e = xe_mask(x=vertex_geom, e=vv_adj, node_mask=vertex_mask)  # b*nv*3, b*nv*nv*2
        feat = extract_feat(e, vertex_mask)
        for t in range(ddpm.T - 1, -1, -1):

            # Extract features
            x_t_feat = torch.cat((vertex_geom, feat[0]), dim=-1).float()  # b*nv*9
            e_t_feat = torch.cat((e, feat[1]), dim=-1).float()  # b*nv*nv*2
            y_feat = torch.cat(
                (feat[2], torch.tensor([ddpm.normalize_t(t)] * batch_size, device=device).unsqueeze(-1)),
                dim=-1).float()  # b*12

            # Predict start
            pred_noise = vertexGeom_model(x_t_feat, e_t_feat, vertex_faceInfo, y_feat, vertex_mask, vFace_mask)  # b*nv*3

            # Sample x
            vertex_geom = ddpm.p_sample(vertex_geom, pred_noise, torch.tensor([t], device=device))   # b*nv*3
            vertex_geom, _ = xe_mask(x=vertex_geom, node_mask=vertex_mask)    # b*nv*3

        """****************Face Geometry****************"""
        faceVertex = [construct_faceVert(i) for i in vertexFace]   # [[[v1, v2, ...],...],...]
        fv_geom = [[vertex_geom[k][j] for j in i] for k, i in enumerate(faceVertex)]   # [[fv*3, fv*3, ...], ...]
        fv = max([max([len(j) for j in i]) for i in fv_geom])
        fv_geom = [pad_and_stack(i, max_n=fv) for i in fv_geom]  # [(nf*fv*3, nf*fv), ...]
        fv_mask, assert_mask = pad_and_stack([i[1] for i in fv_geom], max_n=nf)  # b*nf*fv, b*nf
        assert torch.all(assert_mask == node_mask)
        fv_geom, _ = pad_and_stack([i[0] for i in fv_geom], max_n=nf)  # b*nf*fv*3
        fef_adj = [pad_zero(construct_feTopo(i).cpu().numpy(), max_len=node_mask.shape[1], dim=1)[0] for i in edgeFace]   # list[shape:nf*nf, ...]
        fef_adj = torch.from_numpy(np.stack(fef_adj)).to(device)    # b*nf*nf
        x = torch.randn((fef_adj.shape[0], fef_adj.shape[1], 48), device=device)    # b*nf*48
        e = torch.nn.functional.one_hot(fef_adj, num_classes=m)  # b*n*n*m
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
            pred_noise = faceGeom_model(x_feat, e_feat, y_feat, face_bbox, fv_geom, fv_mask, node_mask)  # b*nf*48

            # Sample x
            x = ddpm.p_sample(x, pred_noise, torch.tensor([t], device=device))   # b*nf*48
            x, _ = xe_mask(x=x, node_mask=node_mask)    # b*nf*48

        """****************Edge Geometry****************"""
        vertex_geom /= hyper_params['bbox_scaled']
        face_bbox_geom = torch.cat((x, face_bbox), dim=-1)  # b*nf*54
        edge_faceInfo = [face[adj] for face, adj in zip(face_bbox_geom, edgeFace)]  # list[shape:ne*2*54]
        edge_faceInfo, edge_mask = pad_and_stack(edge_faceInfo)  # b*ne*2*54, b*ne
        ne = edge_faceInfo.shape[1]
        edge_geom = torch.randn((batch_size, ne, 12), device=device)  # b*ne*12
        edge_geom, _ = xe_mask(x=edge_geom, node_mask=edge_mask)
        edge_vertex = [vertex[torch.from_numpy(adj).to(device)] for vertex, adj in
                       zip(vertex_geom, edgeVertex)]  # [ne*2*3, ...]
        edge_vertex, assert_mask = pad_and_stack(edge_vertex)  # b*ne*2*3, b*ne
        assert torch.all(assert_mask == edge_mask)
        for t in range(ddpm.T - 1, -1, -1):
            # self.model(e_t, edge_faceInfo, edge_vertex, edge_mask, t)  # b*ne*12
            pred_noise = edgeGeom_model(
                edge_geom, edge_faceInfo, edge_vertex, edge_mask, torch.tensor(
                    [ddpm.normalize_t(t)] * batch_size, device=device).unsqueeze(-1))  # b*ne*12

            edge_geom = ddpm.p_sample(edge_geom, pred_noise, torch.tensor([t], device=device))  # b*ne*12

            edge_geom, _ = xe_mask(x=edge_geom, node_mask=edge_mask)

    """****************Construct Brep****************"""
    for i in range(batch_size):
        face_batch, edge_geom_each = face_bbox_geom[i][node_mask[i]], edge_geom[i][edge_mask[i]]              # nf*54, ne*12
        face_geom_each, face_bbox_each = face_batch[:, :-6], face_batch[:, -6:]/hyper_params['bbox_scaled']   # nf*48, nf*6
        vertex_geom_each = vertex_geom[i][vertex_mask[i]]    # nv*3

        # Decode face geometry
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                face_ncs = face_vae(
                    face_geom_each.unflatten(-1, torch.Size([16, 3])).permute(0, 2, 1).unflatten(-1, torch.Size([4, 4])))   # nf*3*32*32
                face_ncs = face_ncs.permute(0, 2, 3, 1)    # nf*32*32*3
                # face_wcs = ncs2wcs(face_bbox, face_ncs.flatten(1, 2)).unflatten(1, (32, 32))  # nf*32*32*3

                # Decode edge geometry
                edge_ncs = edge_vae(
                    edge_geom_each.reshape(-1, 4, 3).permute(0, 2, 1)).permute(0, 2, 1)   # ne*32*3

        # # Get Edge-Vertex topology
        # edge_wcs = ncs2wcs(edge_bbox, edge_ncs)    # ne*32*3
        edgeFace_each = edgeFace[i]         # ne*2
        edgeVertex_each = edgeVertex[i]     # ne*2
        faceEdge_each = construct_faceEdge(edgeFace_each)   # [(edge1, edge2,...), ...]

        # # Remove short edge
        # edge_wcs, keep_edge_idx = remove_short_edge(edge_wcs, threshold=0.1)
        # edgeFace_adj = edgeFace_adj[keep_edge_idx]

        assert set(range(face_batch.shape[0])) == set(torch.unique(edgeFace_each.view(-1)).cpu().numpy().tolist())
        assert set(range(vertex_geom_each.shape[0])) == set(np.unique(edgeVertex_each.reshape(-1)).tolist())

        # joint_optimize:
        # numpy(nf*32*32*3), numpy(ne*32*3), numpy(nf*6), numpy(nv*3),
        # numpy(ne*2), len(list[[edge_id1, ...]...])=nf, int, int
        face_wcs, edge_wcs = joint_optimize(face_ncs.cpu().numpy(), edge_ncs.cpu().numpy(),
                                            face_bbox_each.cpu().numpy(), vertex_geom_each.cpu().numpy(),
                                            edgeVertex_each, faceEdge_each, len(edge_ncs), len(face_ncs))

        try:
            solid = construct_brep(face_wcs, edge_wcs, faceEdge_each, edgeVertex_each)
        except Exception as e:
            print('B-rep rebuild failed...')
            continue

        random_string = generate_random_string(15)
        write_step_file(solid, f'{args.save_folder}/{random_string}_{i}.step')
        write_stl_file(solid, f'{args.save_folder}/{random_string}_{i}.stl', linear_deflection=0.001,
                       angular_deflection=0.5)


if __name__ == '__main__':
    main()
