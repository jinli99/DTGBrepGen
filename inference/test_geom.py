import os
import yaml
import torch
import pickle
import random
from tqdm import tqdm
from argparse import Namespace
import torch.multiprocessing as mp
from diffusers import DDPMScheduler, PNDMScheduler
from model import EdgeGeomTransformer, VertGeomTransformer, FaceBboxTransformer
from utils import check_step_ok, sort_bbox_multi
from OCC.Extend.DataExchange import write_step_file
from inference.brepBuild import Brep2Mesh
from inference.generate_noface import get_brep, get_faceBbox, get_vertGeom, get_edgeGeom, text2int, process_batch

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_ok_step(batch_size, mode='test', dataset_name='deepcad'):
    with open(os.path.join('data_process', dataset_name+'_data_split_6bit.pkl'), 'rb') as f:
        if mode == 'train':
            datas = pickle.load(f)[mode]
        else:
            assert mode == 'test'
            datas = pickle.load(f)
            datas = datas['test']
    datas = random.sample(datas, len(datas))
    batch_file = []
    while len(batch_file) < batch_size and datas:
        file = datas.pop(0)
        with open(os.path.join('data_process/GeomDatasets', dataset_name+'_parsed', file), 'rb') as tf:
            if check_step_ok(pickle.load(tf), max_face=30, max_edge=20):
                batch_file.append(file)

    return batch_file


def get_topology(files, device, dataset_name):
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
        with open(os.path.join('data_process/GeomDatasets', dataset_name+'_parsed', file), 'rb') as tf:
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


def main(args):

    # dataset_name = 'deepcad'
    # test_files = get_ok_step(500, mode='test', dataset_name=dataset_name)
    # with open(os.path.join('inference', dataset_name+'_test.pkl'), 'wb') as f:
    #     pickle.dump(test_files, f)
    # return

    # Make project directory if not exist
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initial FaceBboxTransformer
    FaceBbox_model = FaceBboxTransformer(n_layers=args.FaceBboxModel['n_layers'],
                                         hidden_mlp_dims=args.FaceBboxModel['hidden_mlp_dims'],
                                         hidden_dims=args.FaceBboxModel['hidden_dims'],
                                         edge_classes=args.edge_classes,
                                         act_fn_in=torch.nn.ReLU(),
                                         act_fn_out=torch.nn.ReLU(),
                                         use_cf=args.use_cf)
    FaceBbox_model.load_state_dict(torch.load(args.faceBbox_path))
    FaceBbox_model = FaceBbox_model.to(device).eval()

    # Initial VertGeomTransformer
    vertGeom_model = VertGeomTransformer(n_layers=args.VertGeomModel['n_layers'],
                                         hidden_mlp_dims=args.VertGeomModel['hidden_mlp_dims'],
                                         hidden_dims=args.VertGeomModel['hidden_dims'],
                                         act_fn_in=torch.nn.ReLU(),
                                         act_fn_out=torch.nn.ReLU(),
                                         use_cf=args.use_cf)
    vertGeom_model.load_state_dict(torch.load(args.vertGeom_path))
    vertGeom_model = vertGeom_model.to(device).eval()

    # Initial EdgeGeomTransformer
    edgeGeom_model = EdgeGeomTransformer(n_layers=args.EdgeGeomModel['n_layers'],
                                         edge_geom_dim=args.EdgeGeomModel['edge_geom_dim'],
                                         d_model=args.EdgeGeomModel['d_model'],
                                         nhead=args.EdgeGeomModel['nhead'],
                                         use_cf=args.use_cf)
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

    with open(os.path.join('inference', args.name+'_test.pkl'), 'rb') as f:
        batch_file = pickle.load(f)

    # batch_file = ['0003/00030362_8ff9449eccc64885baa58838_step_004.pkl']*4

    b_each = 16 if args.name == 'furniture' else 32

    for i in tqdm(range(0, len(batch_file), b_each)):

        # =======================================Brep Topology=================================================== #
        datas = get_topology(batch_file[i:i + b_each], device, name)
        fef_adj, edgeVert_adj, faceEdge_adj, edgeFace_adj, vv_list, vertFace_adj = (datas["fef_adj"],
                                                                                    datas["edgeVert_adj"],
                                                                                    datas['faceEdge_adj'],
                                                                                    datas["edgeFace_adj"],
                                                                                    datas["vv_list"],
                                                                                    datas["vertFace_adj"])
        b = len(fef_adj)

        if args.use_cf:
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
        success_count = process_batch(face_bbox, vert_geom, edge_geom,
                                      edgeFace_adj, edgeVert_adj, faceEdge_adj,
                                      args, class_label)
        print(f"Successfully processed {success_count} items out of {b}")

    print('write stl...')
    mesh_tool = Brep2Mesh(input_path=args.save_folder)
    mesh_tool.generate()


if __name__ == '__main__':
    mp.set_start_method('spawn')

    name = 'furniture'
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file).get(name, {})
    config['edgeGeom_path'] = os.path.join('checkpoints', name, 'geom_edgeGeom/epoch_3000.pt')
    config['vertGeom_path'] = os.path.join('checkpoints', name, 'geom_vertGeom/epoch_3000.pt')
    config['faceBbox_path'] = os.path.join('checkpoints', name, 'geom_faceBbox/epoch_3000.pt')
    config['save_folder'] = os.path.join('samples', name)
    # config['save_folder'] = os.path.join('bug')
    config['name'] = name
    config['parallel'] = True

    main(args=Namespace(**config))
