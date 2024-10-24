import argparse
import os
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import pickle
from model import EdgeGeomTransformer, VertGeomTransformer
from diffusers import PNDMScheduler, DDPMScheduler
from utils import sort_bbox_multi
from OCC.Extend.DataExchange import write_step_file
from brepBuild import Brep2Mesh
from test_geom import get_topology
from generate import get_edgeGeom, get_vertGeom, get_brep, text2int
from visualization import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_args_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--faceGeom_path', type=str, default='checkpoints/furniture/geom_faceGeom_01/epoch_3000.pt')
    parser.add_argument('--edgeGeom_path', type=str, default='checkpoints/furniture/geom_edgeGeom/epoch_3000.pt')
    parser.add_argument('--vertGeom_path', type=str, default='checkpoints/furniture/geom_vertGeom/epoch_3000.pt')
    parser.add_argument('--edge_classes', type=int, default=5, help='Number of edge classes')
    parser.add_argument('--max_edge', type=int, default=30, help='maximum number of edges per face')
    parser.add_argument('--save_folder', type=str, default="samples/furniture", help='save folder.')
    parser.add_argument('--bbox_scaled', type=float, default=3, help='scaled the bbox')
    parser.add_argument("--cf", action='store_false', help='Use class condition')
    args = parser.parse_args()

    return args


def main():
    args = get_args_generate()

    # Make project directory if not exist
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initial VertGeomTransformer
    hidden_mlp_dims = {'x': 256}
    hidden_dims = {'dx': 512, 'de': 256, 'n_head': 8, 'dim_ffX': 512}
    vertGeom_model = VertGeomTransformer(n_layers=8,
                                         hidden_mlp_dims=hidden_mlp_dims,
                                         hidden_dims=hidden_dims,
                                         act_fn_in=torch.nn.ReLU(),
                                         act_fn_out=torch.nn.ReLU(),
                                         use_cf=args.cf)
    vertGeom_model.load_state_dict(torch.load(args.vertGeom_path))
    vertGeom_model = vertGeom_model.to(device).eval()

    # Initial EdgeGeomTransformer
    edgeGeom_model = EdgeGeomTransformer(n_layers=8,
                                         edge_geom_dim=12,
                                         use_cf=args.cf)
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

    # batch_file = ['chair/partstudio_partstudio_1242.pkl']

    b_each = 16
    for i in tqdm(range(0, len(batch_file), b_each)):

        # =======================================Brep Topology=================================================== #
        datas = get_topology(batch_file[i:i + b_each], device)
        face_bbox, edgeVert_adj, faceEdge_adj, edgeFace_adj, vertFace_adj = (datas["face_bbox"],
                                                                             datas["edgeVert_adj"],
                                                                             datas['faceEdge_adj'],
                                                                             datas["edgeFace_adj"],
                                                                             datas["vertFace_adj"])
        b = len(face_bbox)
        face_bbox = [sort_bbox_multi(i)*args.bbox_scaled for i in face_bbox]     # [nf*6, ...]

        if args.cf:
            class_label = [text2int[i.split('_')[0]] for i in datas['name']]
        else:
            class_label = None

        # =======================================Vert Geometry=================================================== #
        vert_geom, vert_mask = get_vertGeom(face_bbox, vertFace_adj,
                                            edgeVert_adj, vertGeom_model,
                                            pndm_scheduler, ddpm_scheduler, class_label)
        vert_geom = [i[j] for i, j in zip(vert_geom, vert_mask)]

        # =======================================Edge Geometry=================================================== #
        edge_geom, edge_mask = get_edgeGeom(face_bbox, vert_geom,
                                            edgeFace_adj, edgeVert_adj, edgeGeom_model,
                                            pndm_scheduler, ddpm_scheduler, class_label)
        edge_geom = [i[j] for i, j in zip(edge_geom, edge_mask)]                 # [ne*12, ...]

        # =======================================Construct Brep================================================== #
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
    mp.set_start_method('spawn')
    main()
