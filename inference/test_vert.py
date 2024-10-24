import os
import yaml
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from argparse import Namespace
import pickle
from model import EdgeGeomTransformer, VertGeomTransformer
from diffusers import PNDMScheduler, DDPMScheduler
from utils import sort_bbox_multi
from OCC.Extend.DataExchange import write_step_file
from brepBuild import Brep2Mesh
from inference.test_geom import get_topology
from inference.generate import get_edgeGeom, get_vertGeom, get_brep, text2int
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main(args):

    # Make project directory if not exist
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initial VertGeomTransformer
    vertGeom_model = VertGeomTransformer(n_layers=args.VertGeomModel['n_layer'],
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

    with open('furniture_test.pkl', 'rb') as f:
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

    name = 'deepcad'
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file).get(name, {})
    config['edgeGeom_path'] = os.path.join('checkpoints', name, 'geom_edgeGeom/epoch_3000.pt')
    config['vertGeom_path'] = os.path.join('checkpoints', name, 'geom_vertGeom/epoch_3000.pt')
    config['save_folder'] = os.path.join('samples', name)

    main(args=Namespace(**config))
