import argparse
import os
from tqdm import tqdm
import pickle
from model import EdgeGeomTransformer
from diffusers import PNDMScheduler, DDPMScheduler
from utils import (sort_bbox_multi, pad_and_stack, xe_mask)
from OCC.Extend.DataExchange import write_step_file
from brepBuild import Brep2Mesh, sample_bspline_curve, create_bspline_curve, joint_optimize, construct_brep
from topology.transfer import faceVert_from_edgeVert
from visualization import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_args_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--faceGeom_path', type=str, default='checkpoints/furniture/geom_faceGeom_01/epoch_3000.pt')
    parser.add_argument('--edgeGeom_path', type=str, default='checkpoints/furniture/geom_edgeGeom/epoch_3000.pt')
    parser.add_argument('--edge_classes', type=int, default=5, help='Number of edge classes')
    parser.add_argument('--max_edge', type=int, default=30, help='maximum number of edges per face')
    parser.add_argument('--save_folder', type=str, default="samples/furniture", help='save folder.')
    parser.add_argument('--bbox_scaled', type=float, default=3, help='scaled the bbox')
    args = parser.parse_args()

    return args


def get_topology(files, device):
    """****************Brep Topology****************"""
    edgeVert_adj = []
    edgeFace_adj = []
    faceEdge_adj = []
    vv_list = []
    face_bbox = []
    face_geom = []
    vert_geom = []
    edge_geom = []
    file_name = []

    for file in files:
        with open(os.path.join('data_process/GeomDatasets/furniture_parsed', file), 'rb') as tf:
            data = pickle.load(tf)

        edgeVert_adj.append(torch.from_numpy(data['edgeVert_adj']).to(device))                    # [ne*2, ...]
        edgeFace_adj.append(torch.from_numpy(data['edgeFace_adj']).to(device))
        faceEdge_adj.append(data['faceEdge_adj'])                                                 # List[List[int]]
        vv_list.append(data['vv_list'])                                                           # List[List(tuple)]
        face_bbox.append(torch.FloatTensor(data['face_bbox_wcs']).to(device))                     # [nf*6, ...]
        vert_geom.append(torch.FloatTensor(data['vert_wcs']).to(device))                          # [nv*3,   ]
        edge_geom.append(torch.FloatTensor(data['edge_ctrs']).to(device).reshape(-1, 12))         # [ne*12,   ]
        file_name.append(file[:-4].replace('/', '_'))

    return {"edgeVert_adj": edgeVert_adj, "edgeFace_adj": edgeFace_adj, "faceEdge_adj": faceEdge_adj,
            "face_bbox": face_bbox, "face_geom": face_geom, "vert_geom": vert_geom, "edge_geom": edge_geom,
            'name': file_name, 'vv_list': vv_list}


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


def get_edgeGeom(vert_geom, edgeVert_adj, edgeFace_adj, faceEdge_adj, model, pndm_scheduler, ddpm_scheduler):
    """
    Args:
        vert_geom: List of tensors, where each tensor has shape [nv, 3], representing vertex geometries.
        edgeVert_adj: List of tensors, where each tensor has shape [ne, 2], representing edge-vert topology
        edgeFace_adj: List of tensors, where each tensor has shape [ne, 2], representing edge-face topology
        faceEdge_adj: List of list, representing face-edge topology
        model: edgeGeom denoising model
        pndm_scheduler: pndm sampling
        ddpm_scheduler: ddpm sampling
    Returns:
        edge_geom (torch.Tensor): [b, ne, 2] tensor, sampled edge geometry
        edge_mask (torch.Tensor): [b, ne] tensor, edge mask of the sampled edge geometry
    """

    b = len(vert_geom)
    device = vert_geom[0].device
    edgeVert_geom = [vert_geom[i][edgeVert_adj[i]] for i in range(b)]      # [ne*2*3, ...]
    faceVert_adj = [faceVert_from_edgeVert(faceEdge_adj=i, edgeVert_adj=j.cpu().numpy()) for i, j in zip(faceEdge_adj, edgeVert_adj)]
    face_bbox = []
    for i in range(b):
        temp = [vert_geom[i][j] for j in faceVert_adj[i]]                                       # [fv*3, ...]
        face_bbox.append(torch.stack([torch.cat([j.min(0)[0], j.max(0)[0]]) for j in temp]))    # nf*6
    edgeFace_bbox = [face_bbox[i][edgeFace_adj[i]] for i in range(b)]                           # [ne*2*6, ...]

    edgeFace_bbox, edge_mask = pad_and_stack(edgeFace_bbox)                # b*ne*2*6, b*ne
    edgeVert_geom, _ = pad_and_stack(edgeVert_geom)                        # b*ne*2*3
    edgeFace_adj, _ = pad_and_stack(edgeFace_adj)                          # b*ne*2

    ne = edge_mask.shape[1]

    with torch.no_grad():
        edge_geom = torch.randn((b, ne, 12), device=device)  # b*ne*12

        pndm_scheduler.set_timesteps(200)
        for t in pndm_scheduler.timesteps[:158]:
            timesteps = t.reshape(-1).cuda()
            pred = model(edge_geom,
                         edgeFace_bbox,
                         edgeVert_geom,
                         edge_mask,
                         edgeFace_adj,
                         timesteps.expand(edge_geom.shape[0], 1))
            edge_geom = pndm_scheduler.step(pred, t, edge_geom).prev_sample
            edge_geom, _ = xe_mask(x=edge_geom, node_mask=edge_mask)

        ddpm_scheduler.set_timesteps(1000)
        for t in ddpm_scheduler.timesteps[-250:]:
            timesteps = t.reshape(-1).cuda()
            pred = model(edge_geom,
                         edgeFace_bbox,
                         edgeVert_geom,
                         edge_mask,
                         edgeFace_adj,
                         timesteps.expand(edge_geom.shape[0], 1))
            edge_geom = ddpm_scheduler.step(pred, t, edge_geom).prev_sample
            edge_geom, _ = xe_mask(x=edge_geom, node_mask=edge_mask)

    return edge_geom, edge_mask


def main():
    args = get_args_generate()

    # Make project directory if not exist
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    batch_file = ['chair/partstudio_partstudio_1242.pkl']

    b_each = 32
    for i in tqdm(range(0, len(batch_file), b_each)):

        # =======================================Brep Topology=================================================== #
        datas = get_topology(batch_file[i:i + b_each], device)
        face_bbox, edgeVert_adj, faceEdge_adj, edgeFace_adj, vert_geom = (datas["face_bbox"],
                                                                          datas["edgeVert_adj"],
                                                                          datas['faceEdge_adj'],
                                                                          datas["edgeFace_adj"],
                                                                          datas["vert_geom"])
        b = len(face_bbox)
        face_bbox = [sort_bbox_multi(i)*args.bbox_scaled for i in face_bbox]     # [nf*6, ...]
        vert_geom = [i*args.bbox_scaled for i in vert_geom]                      # [nv*3, ...]

        # =======================================Edge Geometry================================================= #
        edge_geom, edge_mask = get_edgeGeom(vert_geom,
                                            edgeVert_adj, edgeFace_adj, faceEdge_adj, edgeGeom_model,
                                            pndm_scheduler, ddpm_scheduler)
        edge_geom = [i[j] for i, j in zip(edge_geom, edge_mask)]                 # [ne*12, ...]

        # =======================================Construct Brep================================================ #
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
