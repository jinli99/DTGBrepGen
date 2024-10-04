import argparse
import os
from tqdm import tqdm
import pickle
from model import EdgeGeomTransformer, FaceGeomTransformer
from diffusers import PNDMScheduler, DDPMScheduler
from test_face import get_faceGeom, get_topology, get_brep
from utils import (sort_bbox_multi, pad_and_stack, xe_mask)
from OCC.Extend.DataExchange import write_step_file
from brepBuild import Brep2Mesh
from visualization import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_args_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--faceGeom_path', type=str, default='checkpoints/furniture/geom_faceGeom_01/epoch_3000.pt')
    parser.add_argument('--edgeGeom_path', type=str, default='checkpoints/furniture/geom_edgeGeom/epoch_3000.pt')
    parser.add_argument('--save_folder', type=str, default="samples/furniture", help='save folder.')
    parser.add_argument('--bbox_scaled', type=float, default=3, help='scaled the bbox')
    args = parser.parse_args()

    return args


def get_edgeGeom(face_bbox, vert_geom, edgeVert_adj, edgeFace_adj, model, pndm_scheduler, ddpm_scheduler):
    """
    Args:
        face_bbox: List of tensors, where each tensor has shape [nf, 6], representing face bounding boxes.
        vert_geom: List of tensors, where each tensor has shape [nv, 3], representing vertex geometries.
        edgeVert_adj: List of tensors, where each tensor has shape [ne, 2], representing edge-vert topology
        edgeFace_adj: List of tensors, where each tensor has shape [ne, 2], representing edge-face topology
        model: edgeGeom denoising model
        pndm_scheduler: pndm sampling
        ddpm_scheduler: ddpm sampling
    Returns:
        edge_geom (torch.Tensor): [b, ne, 2] tensor, sampled edge geometry
        edge_mask (torch.Tensor): [b, ne] tensor, edge mask of the sampled edge geometry
    """

    b = len(face_bbox)
    device = face_bbox[0].device
    edgeFace_bbox = [face_bbox[i][edgeFace_adj[i]] for i in range(b)]      # [ne*2*6, ...]
    edgeVert_geom = [vert_geom[i][edgeVert_adj[i]] for i in range(b)]      # [ne*2*3, ...]

    edgeFace_bbox, edge_mask = pad_and_stack(edgeFace_bbox)                # b*ne*2*6, b*ne
    edgeVert_geom, _ = pad_and_stack(edgeVert_geom)                        # b*ne*2*3

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
            edge_geom = pndm_scheduler.step(pred, t, edge_geom).prev_sample
            edge_geom, _ = xe_mask(x=edge_geom, node_mask=edge_mask)

    return edge_geom, edge_mask


def main():
    args = get_args_generate()

    # Make project directory if not exist
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initial EdgeGeomTransformer
    edgeGeom_model = EdgeGeomTransformer(n_layers=8, edge_geom_dim=12)
    edgeGeom_model.load_state_dict(torch.load(args.edgeGeom_path))
    edgeGeom_model = edgeGeom_model.to(device).eval()

    # Initial FaceGeomTransformer
    faceGeom_model = FaceGeomTransformer(n_layers=8, face_geom_dim=48)
    faceGeom_model.load_state_dict(torch.load(args.faceGeom_path))
    faceGeom_model = faceGeom_model.to(device).eval()

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

    # batch_file = ['chair/partstudio_partstudio_0759.pkl']

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
        edge_geom, edge_mask = get_edgeGeom(face_bbox, vert_geom,
                                            edgeVert_adj, edgeFace_adj, edgeGeom_model,
                                            pndm_scheduler, ddpm_scheduler)
        edge_geom = [i[j] for i, j in zip(edge_geom, edge_mask)]                 # [ne*12, ...]

        # =======================================Face Geometry================================================= #
        face_geom, face_mask = get_faceGeom(face_bbox, vert_geom, edge_geom,
                                            edgeVert_adj, faceEdge_adj, faceGeom_model,
                                            pndm_scheduler, ddpm_scheduler)      # b*nf*48, b*nf
        face_geom = [i[j] for i, j in zip(face_geom, face_mask)]

        # =======================================Construct Brep================================================ #
        for j in range(b):
            solid = get_brep((face_geom[j].cpu().numpy() / args.bbox_scaled,
                              face_bbox[j].cpu().numpy() / args.bbox_scaled,
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
