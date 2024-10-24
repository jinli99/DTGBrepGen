import argparse
import os
import torch
from tqdm import tqdm
import pickle
from utils import (pad_and_stack, xe_mask)
from OCC.Extend.DataExchange import write_step_file
from brepBuild import Brep2Mesh, construct_brep
from visualization import *
import warnings
import torch.multiprocessing as mp

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
    faceEdge_adj = []
    edge_wcs = []
    file_name = []

    for file in files:
        with open(os.path.join('data_process/GeomDatasets/furniture_parsed', file), 'rb') as tf:
            data = pickle.load(tf)
        edge_wcs.append(data['edge_wcs'])                                                         # [ne*32*3]
        edgeVert_adj.append(torch.from_numpy(data['edgeVert_adj']).to(device))                    # [ne*2, ...]
        faceEdge_adj.append(data['faceEdge_adj'])                                                 # List[List[int]]
        file_name.append(file[:-4].replace('/', '_'))

    return {"edge_wcs": edge_wcs, "edgeVert_adj": edgeVert_adj, "faceEdge_adj": faceEdge_adj, 'name': file_name}


def get_brep(edge_wcs, faceEdge_adj, edgeVert_adj):

    # draw_edge(edge_wcs)         # display edges
    try:
        solid = construct_brep(edge_wcs, faceEdge_adj, edgeVert_adj)
    except Exception as e:
        print('B-rep rebuild failed...')
        return False

    return solid


def get_edgeGeom(face_bbox, vert_geom, edgeFace_adj, edgeVert_adj, model, pndm_scheduler, ddpm_scheduler):
    """
    Args:
        face_bbox: List of tensors, where each tensor has shape [nf, 6], representing face bounding box.
        vert_geom: List of tensors, where each tensor has shape [nv, 3], representing vertex geometries.
        edgeFace_adj: List of tensors, where each tensor has shape [ne, 2], representing edge-face topology.
        edgeVert_adj: List of tensors, where each tensor has shape [ne, 2], representing edge-vert topology.
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

    with open('batch_file_test.pkl', 'rb') as f:
        batch_file = pickle.load(f)

    # batch_file = batch_file[:8]

    b_each = 32
    for i in tqdm(range(0, len(batch_file), b_each)):

        # =======================================Brep Topology=================================================== #
        datas = get_topology(batch_file[i:i + b_each], device)
        edge_wcs, faceEdge_adj, edgeVert_adj = (datas["edge_wcs"],
                                                datas['faceEdge_adj'],
                                                datas["edgeVert_adj"])
        b = len(edge_wcs)

        # =======================================Construct Brep================================================ #
        for j in range(b):
            solid = get_brep(edge_wcs[j], faceEdge_adj[j], edgeVert_adj[j].cpu().numpy())

            if solid is False:
                continue
            save_name = datas['name'][j]
            write_step_file(solid, f'{args.save_folder}/{save_name}.step')

    print('write stl...')
    mesh_tool = Brep2Mesh(input_path=args.save_folder)
    mesh_tool.generate()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
