import argparse
import os
from tqdm import tqdm
import pickle
from model import EdgeGeomTransformer
from geometry.diffusion import DDPM
from topology.transfer import face_edge_trans
from utils import (generate_random_string, sort_bbox_multi, pad_and_stack, xe_mask)
from OCC.Extend.DataExchange import write_step_file
from brepBuild import (joint_optimize, construct_brep, Brep2Mesh,
                       create_bspline_surface, create_bspline_curve,
                       sample_bspline_surface, sample_bspline_curve)
from visualization import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_args_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edgeGeom_path', type=str, default='checkpoints/furniture/geom_edgeGeom/epoch_3000.pt')
    parser.add_argument('--save_folder', type=str, default="samples/furniture", help='save folder.')
    parser.add_argument('--bbox_scaled', type=float, default=3, help='scaled the bbox')
    args = parser.parse_args()

    return args


def get_topology(files, device):
    """****************Brep Topology****************"""
    edgeVert_adj = []
    edgeFace_adj = []
    face_geom = []
    face_bbox = []
    vert_geom = []

    for file in files:
        with open(os.path.join('data_process/GeomDatasets/furniture_parsed', file), 'rb') as tf:
            data = pickle.load(tf)

        edgeVert_adj.append(torch.from_numpy(data['edgeVert_adj']))                               # [ne*2, ...]
        edgeFace_adj.append(torch.from_numpy(data['edgeFace_adj']).to(device))
        face_geom.append(torch.FloatTensor(data['face_ctrs']).to(device).reshape(-1, 48))         # [nf*48, ...]
        face_bbox.append(torch.FloatTensor(data['face_bbox_wcs']).to(device))                     # [nf*6, ...]
        vert_geom.append(torch.FloatTensor(data['vert_wcs']).to(device))                     # [nv*3,   ]

    return {"edgeVert_adj": edgeVert_adj, "edgeFace_adj": edgeFace_adj,
            "face_geom": face_geom, "face_bbox": face_bbox, "vert_geom": vert_geom}


def get_edgeGeom(face_geom, face_bbox, vert_geom, edgeVert_adj, edgeFace_adj, model, ddpm):
    """
    Args:
        face_geom: List of tensors, where each tensor has shape [nf, 48], representing face geometries.
        face_bbox: List of tensors, where each tensor has shape [nf, 6], representing face bounding boxes.
        vert_geom: List of tensors, where each tensor has shape [nv, 3], representing vertex geometries.
        edgeVert_adj: List of tensors, where each tensor has shape [ne, 2], representing edge-vert topology
        edgeFace_adj: List of tensors, where each tensor has shape [ne, 2], representing edge-face topology
        model: edgeGeom denoising model
        ddpm: ddpm sampling
    Returns:
        edge_geom (torch.Tensor): [b, ne, 2] tensor, sampled edge geometry
        edge_mask (torch.Tensor): [b, ne] tensor, edge mask of the sampled edge geometry
    """

    b = len(face_geom)
    device = face_geom[0].device
    edgeFace_geom = [face_geom[i][edgeFace_adj[i]] for i in range(b)]      # [ne*2*48, ...]
    edgeFace_bbox = [face_bbox[i][edgeFace_adj[i]] for i in range(b)]      # [ne*2*6, ...]
    edgeVert_geom = [vert_geom[i][edgeVert_adj[i]] for i in range(b)]      # [ne*2*3, ...]

    edgeFace_geom, edge_mask = pad_and_stack(edgeFace_geom)                # b*ne*2*48, b*ne
    edgeFace_bbox, _ = pad_and_stack(edgeFace_bbox)                        # b*ne*2*6
    edgeVert_geom, _ = pad_and_stack(edgeVert_geom)                        # b*ne*2*3

    ne = edge_mask.shape[1]
    edgeFace_info = torch.cat((edgeFace_geom, edgeFace_bbox), dim=-1)    # b*ne*2*54

    edge_geom = torch.randn((b, ne, 12), device=device)          # b*ne*12
    with torch.no_grad():
        for t in range(ddpm.T - 1, -1, -1):
            pred_noise = model(edge_geom,
                               edgeFace_info,
                               edgeVert_geom,
                               edge_mask,
                               torch.tensor([ddpm.normalize_t(t)] * b, device=device).unsqueeze(-1))  # b*ne*12

            edge_geom = ddpm.p_sample(edge_geom, pred_noise, torch.tensor([t], device=device))  # b*ne*12

            edge_geom, _ = xe_mask(x=edge_geom, node_mask=edge_mask)                                  # b*ne*12

    return edge_geom, edge_mask


def main():
    args = get_args_generate()

    # Make project directory if not exist
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initial EdgeGeomTransformer
    edgeGeom_model = EdgeGeomTransformer(n_layers=8, face_geom_dim=48, edge_geom_dim=12)
    edgeGeom_model.load_state_dict(torch.load(args.edgeGeom_path))
    edgeGeom_model = edgeGeom_model.to(device).eval()

    ddpm = DDPM(500, device)

    with open('batch_file_test.pkl', 'rb') as f:
        batch_file = pickle.load(f)

    b_each = 32
    for i in tqdm(range(110, len(batch_file), b_each)):

        """****************Brep Topology****************"""
        datas = get_topology(batch_file[i:i + b_each], device)
        face_geom, face_bbox, edgeVert_adj, edgeFace_adj, vert_geom = (datas["face_geom"],
                                                                       datas["face_bbox"],
                                                                       datas["edgeVert_adj"],
                                                                       datas["edgeFace_adj"],
                                                                       datas["vert_geom"])
        b = len(face_bbox)
        face_bbox = [sort_bbox_multi(i)*args.bbox_scaled for i in face_bbox]     # [nf*6, ...]
        face_geom = [i*args.bbox_scaled for i in face_geom]                      # [nf*48, ...]
        vert_geom = [i*args.bbox_scaled for i in vert_geom]                      # [nf*3, ...]

        """****************Edge Geometry****************"""
        edge_geom, edge_mask = get_edgeGeom(face_geom, face_bbox, vert_geom,
                                            edgeVert_adj, edgeFace_adj, edgeGeom_model, ddpm)
        edge_geom = [i[j] for i, j in zip(edge_geom, edge_mask)]

        """****************Construct Brep****************"""
        for j in range(b):

            face_geom_cad = face_geom[j].cpu().numpy() / args.bbox_scaled    # nf*48
            face_bbox_cad = face_bbox[j].cpu().numpy() / args.bbox_scaled    # nf*6
            vert_geom_cad = vert_geom[j].cpu().numpy() / args.bbox_scaled    # nv*3
            edge_geom_cad = edge_geom[j].cpu().numpy() / args.bbox_scaled    # ne*12
            edgeFace_cad = edgeFace_adj[j].cpu().numpy()                     # ne*2
            edgeVert_cad = edgeVert_adj[j].cpu().numpy()                     # ne*2
            faceEdge_cad = face_edge_trans(edgeFace_adj=edgeFace_cad)        # List[List[int]]

            assert set(range(face_geom_cad.shape[0])) == set(np.unique(edgeFace_cad.reshape(-1)).tolist())
            assert set(range(vert_geom_cad.shape[0])) == set(np.unique(edgeVert_cad.reshape(-1)).tolist())

            face_ncs = []
            for ctrs in face_geom_cad:
                pcd = sample_bspline_surface(create_bspline_surface(ctrs.reshape(16, 3).astype(np.float64)))   # 32*32*3
                face_ncs.append(pcd)
            face_ncs = np.stack(face_ncs)                                                                   # nf*32*32*3

            edge_ncs = []
            for ctrs in edge_geom_cad:
                pcd = sample_bspline_curve(create_bspline_curve(ctrs.reshape(4, 3).astype(np.float64)))     # 32*3
                edge_ncs.append(pcd)
            edge_ncs = np.stack(edge_ncs)                                                                   # ne*32*3

            # joint_optimize
            face_wcs, edge_wcs = joint_optimize(face_ncs, edge_ncs,
                                                face_bbox_cad, vert_geom_cad,
                                                edgeVert_cad, faceEdge_cad, len(edge_ncs), len(face_ncs))

            try:
                solid = construct_brep(face_wcs, edge_wcs, faceEdge_cad, edgeVert_cad)
            except Exception as e:
                print('B-rep rebuild failed...')
                continue

            random_string = generate_random_string(15)
            write_step_file(solid, f'{args.save_folder}/{random_string}_{i}.step')

    print('write stl...')
    mesh_tool = Brep2Mesh(input_path=args.save_folder)
    mesh_tool.generate()


if __name__ == '__main__':
    main()
