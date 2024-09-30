import argparse
import os
from tqdm import tqdm
import pickle
from model import FaceGeomTransformer, EdgeGeomTransformer
from test_faceEdge import get_edgeGeom
from geometry.diffusion import DDPM
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
    parser.add_argument('--faceGeom_path', type=str, default='checkpoints/furniture/geom_faceGeom/epoch_3000.pt')
    parser.add_argument('--edgeGeom_path', type=str, default='checkpoints/furniture/geom_edgeGeom/epoch_3000.pt')
    parser.add_argument('--save_folder', type=str, default="samples/jing", help='save folder.')
    parser.add_argument('--bbox_scaled', type=float, default=3, help='scaled the bbox')
    args = parser.parse_args()

    return args


def get_topology(files, device):
    """****************Brep Topology****************"""
    edgeVert_adj = []
    edgeFace_adj = []
    faceEdge_adj = []
    face_bbox = []
    face_geom = []
    vert_geom = []
    edge_geom = []

    for file in files:
        with open(os.path.join('data_process/GeomDatasets/furniture_parsed', file), 'rb') as tf:
            data = pickle.load(tf)

        edgeVert_adj.append(torch.from_numpy(data['edgeVert_adj']))                               # [ne*2, ...]
        edgeFace_adj.append(torch.from_numpy(data['edgeFace_adj']).to(device))
        faceEdge_adj.append(data['faceEdge_adj'])                                                 # List[List[int]]
        face_bbox.append(torch.FloatTensor(data['face_bbox_wcs']).to(device))                     # [nf*6, ...]
        face_geom.append(torch.FloatTensor(data['face_ctrs']).to(device).reshape(-1, 48))         # [nf*48, ...]
        vert_geom.append(torch.FloatTensor(data['vert_wcs']).to(device))                          # [nv*3,   ]
        edge_geom.append(torch.FloatTensor(data['edge_ctrs']).to(device).reshape(-1, 12))         # [ne*12,   ]

    return {"edgeVert_adj": edgeVert_adj, "edgeFace_adj": edgeFace_adj, "faceEdge_adj": faceEdge_adj,
            "face_bbox": face_bbox, "face_geom": face_geom, "vert_geom": vert_geom, "edge_geom": edge_geom}


def get_faceGeom(face_bbox, vert_geom, edge_geom, edgeVert_adj, faceEdge_adj, model, ddpm):
    """
    Args:
        face_bbox: List of tensors, where each tensor has shape [nf, 6], representing face bounding boxes.
        vert_geom: List of tensors, where each tensor has shape [nv, 3], representing vertex geometries.
        edge_geom: List of tensors, where each tensor has shape [ne, 12], representing edge geometries.
        edgeVert_adj: List of tensors, where each tensor has shape [ne, 2], representing edge-vert topology
        faceEdge_adj: List of list, where each list has shape List[List[int]], representing face-edge topology
        model: faceGeom denoising model
        ddpm: ddpm sampling
    Returns:
        face_geom: List of tensors, where each tensor has shape [nf, 48], representing face geometries.
    """

    b = len(face_bbox)
    device = face_bbox[0].device

    face_bbox, face_mask = pad_and_stack(face_bbox)          # b*nf*6, b*nf
    nf = face_bbox.shape[1]

    faceVert_geom = []                                       # List[List[fv*3, ...]]
    fv = 0
    for i in range(b):
        faceVert_geom.append([vert_geom[i][torch.unique(edgeVert_adj[i][j].flatten())] for j in faceEdge_adj[i]])
        fv = max(fv, max([j.shape[0] for j in faceVert_geom[-1]]))
    faceVert_geom = [pad_and_stack(i, max_n=fv) for i in faceVert_geom]         # List[(Tensor(nf, fv, 3)), nf*fv]
    faceVert_mask, _ = pad_and_stack([i[1] for i in faceVert_geom], max_n=nf)   # b*nf*fv
    faceVert_geom, _ = pad_and_stack([i[0] for i in faceVert_geom], max_n=nf)   # b*nf*fv*3

    faceEdge_geom = []                                                          # List[List[fe*12, ...]]
    fe = 0
    for i in range(b):
        faceEdge_geom.append([edge_geom[i][j] for j in faceEdge_adj[i]])
        fe = max(fe, max([j.shape[0] for j in faceEdge_geom[-1]]))
    faceEdge_geom = [pad_and_stack(i, max_n=fe) for i in faceEdge_geom]         # List[(Tensor(nf, fe, 12)), nf*fe]
    faceEdge_mask, _ = pad_and_stack([i[1] for i in faceEdge_geom], max_n=nf)   # b*nf*fe
    faceEdge_geom, _ = pad_and_stack([i[0] for i in faceEdge_geom], max_n=nf)   # b*nf*fe*12

    with torch.no_grad():
        face_geom = torch.randn((b, nf, 48), device=device)                     # b*nf*48
        for t in range(ddpm.T - 1, -1, -1):

            # Predict start
            pred_noise = model(face_geom,
                               face_bbox,
                               faceVert_geom,
                               faceEdge_geom,
                               face_mask,
                               faceVert_mask,
                               faceEdge_mask,
                               torch.tensor([ddpm.normalize_t(t)] * b, device=device).unsqueeze(-1))   # b*n*48

            # Sample x
            face_geom = ddpm.p_sample(face_geom, pred_noise, torch.tensor([t], device=device))    # b*nf*48
            face_geom, _ = xe_mask(x=face_geom, node_mask=face_mask)                                    # b*nf*48

    return face_geom, face_mask


def main():
    args = get_args_generate()

    # Make project directory if not exist
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initial FaceGeomTransformer
    faceGeom_model = FaceGeomTransformer(n_layers=12, face_geom_dim=48)
    faceGeom_model.load_state_dict(torch.load(args.faceGeom_path))
    faceGeom_model = faceGeom_model.to(device).eval()

    # Initial EdgeGeomTransformer
    edgeGeom_model = EdgeGeomTransformer(n_layers=12, face_geom_dim=48, edge_geom_dim=12)
    edgeGeom_model.load_state_dict(torch.load(args.edgeGeom_path))
    edgeGeom_model = edgeGeom_model.to(device).eval()

    ddpm = DDPM(500, device)

    with open('batch_file_test.pkl', 'rb') as f:
        batch_file = pickle.load(f)

    b_each = 32
    for i in tqdm(range(0, len(batch_file), b_each)):

        """****************Brep Topology****************"""
        datas = get_topology(batch_file[i:i + b_each], device)
        face_bbox, edgeVert_adj, edgeFace_adj, faceEdge_adj, vert_geom = (datas["face_bbox"],
                                                                          datas["edgeVert_adj"],
                                                                          datas["edgeFace_adj"],
                                                                          datas["faceEdge_adj"],
                                                                          datas["vert_geom"])
        b = len(face_bbox)
        face_bbox = [sort_bbox_multi(i)*args.bbox_scaled for i in face_bbox]     # [nf*6, ...]
        vert_geom = [i*args.bbox_scaled for i in vert_geom]                      # [nf*3, ...]
        face_geom = [i * args.bbox_scaled for i in datas['face_geom']]  # [nf*48, ...]

        """****************Edge Geometry****************"""
        edge_geom, edge_mask = get_edgeGeom(face_geom, face_bbox, vert_geom,
                                            edgeVert_adj, edgeFace_adj, edgeGeom_model, ddpm)
        edge_geom = [i[j] for i, j in zip(edge_geom, edge_mask)]

        """****************Face Geometry****************"""
        face_geom, face_mask = get_faceGeom(face_bbox, vert_geom, edge_geom, edgeVert_adj,
                                            faceEdge_adj, faceGeom_model, ddpm)  # b*nf*48, b*nf
        face_geom = [i[j] for i, j in zip(face_geom, face_mask)]

        """****************Construct Brep****************"""
        for j in range(b):

            face_geom_cad = face_geom[j].cpu().numpy() / args.bbox_scaled    # nf*48
            face_bbox_cad = face_bbox[j].cpu().numpy() / args.bbox_scaled    # nf*6
            vert_geom_cad = vert_geom[j].cpu().numpy() / args.bbox_scaled    # nv*3
            edge_geom_cad = edge_geom[j].cpu().numpy() / args.bbox_scaled    # ne*12
            edgeFace_cad = edgeFace_adj[j].cpu().numpy()                     # ne*2
            edgeVert_cad = edgeVert_adj[j].cpu().numpy()                     # ne*2
            faceEdge_cad = faceEdge_adj[j]                                   # List[List[int]]

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
