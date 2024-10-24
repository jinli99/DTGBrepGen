import argparse
import os
import torch
import pickle
import random
from tqdm import tqdm
import torch.multiprocessing as mp
from diffusers import DDPMScheduler, PNDMScheduler
from model import EdgeGeomTransformer, VertGeomTransformer, FaceBboxTransformer
from utils import check_step_ok, sort_bbox_multi
from OCC.Extend.DataExchange import write_step_file
from visualization import *
from brepBuild import Brep2Mesh
from generate import get_brep, get_faceBbox, get_vertGeom, get_edgeGeom, text2int

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_args_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--faceGeom_path', type=str, default='checkpoints/furniture/geom_faceGeom_01/epoch_3000.pt')
    parser.add_argument('--edgeGeom_path', type=str, default='checkpoints/furniture/geom_edgeGeom/epoch_3000.pt')
    parser.add_argument('--vertGeom_path', type=str, default='checkpoints/furniture/geom_vertGeom/epoch_3000.pt')
    parser.add_argument('--faceBbox_path', type=str, default='checkpoints/furniture/geom_faceBbox/epoch_3000.pt')
    parser.add_argument('--edge_classes', type=int, default=5, help='Number of edge classes')
    parser.add_argument('--max_edge', type=int, default=30, help='maximum number of edges per face')
    parser.add_argument('--save_folder', type=str, default="samples/furniture", help='save folder.')
    parser.add_argument('--bbox_scaled', type=float, default=3, help='scaled the bbox')
    parser.add_argument("--cf", action='store_false', help='Use class condition')
    args = parser.parse_args()

    return args


def get_ok_step(batch_size, mode='train'):
    with open("data_process/furniture_data_split_6bit.pkl", 'rb') as f:
        if mode == 'train':
            datas = pickle.load(f)[mode]
        else:
            assert mode == 'test'
            datas = pickle.load(f)
            datas = datas['test'] + datas['val']
    datas = random.sample(datas, len(datas))
    batch_file = []
    while len(batch_file) < batch_size and datas:
        file = datas.pop(0)
        with open(os.path.join('data_process/GeomDatasets/furniture_parsed', file), 'rb') as tf:
            if check_step_ok(pickle.load(tf)):
                batch_file.append(file)

    return batch_file


def get_topology(files, device):
    """****************Brep Topology****************"""
    edgeVert_adj = []
    edgeFace_adj = []
    faceEdge_adj = []
    vertFace_adj = []
    vv_list = []
    fef_adj = []
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
        vertFace_adj.append(data['vertFace_adj'])
        faceEdge_adj.append(data['faceEdge_adj'])                                                 # List[List[int]]
        vv_list.append(data['vv_list'])                                                           # List[List(tuple)]
        fef_adj.append(torch.from_numpy(data['fef_adj']).to(device))                              # [nf*nf, ...]
        face_bbox.append(torch.FloatTensor(data['face_bbox_wcs']).to(device))                     # [nf*6, ...]
        vert_geom.append(torch.FloatTensor(data['vert_wcs']).to(device))                          # [nv*3,   ]
        edge_geom.append(torch.FloatTensor(data['edge_ctrs']).to(device).reshape(-1, 12))         # [ne*12,   ]
        file_name.append(file[:-4].replace('/', '_'))

    return {"edgeVert_adj": edgeVert_adj, "edgeFace_adj": edgeFace_adj, "faceEdge_adj": faceEdge_adj,
            "face_bbox": face_bbox, "face_geom": face_geom, "vert_geom": vert_geom, "edge_geom": edge_geom,
            'name': file_name, 'vv_list': vv_list, "vertFace_adj": vertFace_adj, "fef_adj": fef_adj}


def main():

    # batch_file = get_ok_step(150, mode='test')
    # with open('batch_file_test.pkl', 'wb') as f:
    #     pickle.dump(batch_file, f)
    # return

    args = get_args_generate()

    # Make project directory if not exist
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initial FaceBboxTransformer
    hidden_mlp_dims = {'x': 256}
    hidden_dims = {'dx': 512, 'de': 256, 'n_head': 8, 'dim_ffX': 512}
    FaceBbox_model = FaceBboxTransformer(n_layers=8,
                                         hidden_mlp_dims=hidden_mlp_dims,
                                         hidden_dims=hidden_dims,
                                         edge_classes=args.edge_classes,
                                         act_fn_in=torch.nn.ReLU(),
                                         act_fn_out=torch.nn.ReLU(),
                                         use_cf=args.cf)
    FaceBbox_model.load_state_dict(torch.load(args.faceBbox_path))
    FaceBbox_model = FaceBbox_model.to(device).eval()

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

    batch_file = ['chair/partstudio_partstudio_0104.pkl']

    b_each = 16
    for i in tqdm(range(0, len(batch_file), b_each)):

        # =======================================Brep Topology=================================================== #
        datas = get_topology(batch_file[i:i + b_each], device)
        fef_adj, edgeVert_adj, faceEdge_adj, edgeFace_adj, vv_list, vertFace_adj = (datas["fef_adj"],
                                                                                    datas["edgeVert_adj"],
                                                                                    datas['faceEdge_adj'],
                                                                                    datas["edgeFace_adj"],
                                                                                    datas["vv_list"],
                                                                                    datas["vertFace_adj"])
        b = len(fef_adj)

        if args.cf:
            class_label = [text2int[i.split('_')[0]] for i in datas['name']]
        else:
            class_label = None

        # ========================================Face Bbox====================================================== #
        face_bbox, face_mask = get_faceBbox(fef_adj, FaceBbox_model, pndm_scheduler, ddpm_scheduler, class_label)
        face_bbox = [i[j] for i, j in zip(face_bbox, face_mask)]

        face_bbox = [sort_bbox_multi(i) for i in face_bbox]                      # [nf*6, ...]

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
