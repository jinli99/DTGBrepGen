import os
import pickle
import random
import numpy as np
from tqdm import tqdm
from utils import check_step_ok
from OCC.Extend.DataExchange import write_stl_file, write_step_file
from visualization import *
from brepBuild import (create_bspline_surface, create_bspline_curve,
                       sample_bspline_surface, sample_bspline_curve, joint_optimize, construct_brep_fit)

test_files = ['data_process/GeomDatasets/furniture_parsed/cabinet/partstudio_partstudio_3851.pkl']
for file in tqdm(test_files):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        face_ctrs = data['face_ctrs']
        edge_ctrs = data['edge_ctrs']

        face_ncs = []
        for i, ctrs in enumerate(face_ctrs):

            ncs = sample_bspline_surface(create_bspline_surface(ctrs))      # 32*32*3
            face_ncs.append(ncs)
        face_ncs = np.stack(face_ncs)   # nf*32*32*3

        edge_ncs = []
        for ctrs in edge_ctrs:
            ncs = sample_bspline_curve(create_bspline_curve(ctrs))
            edge_ncs.append(ncs)
        edge_ncs = np.stack(edge_ncs)    # ne*32*3

        face_wcs, edge_wcs = joint_optimize(face_ncs, edge_ncs,
                                            data['face_bbox_wcs'], data['vert_wcs'],
                                            data['edgeVert_adj'], data['faceEdge_adj'], len(edge_ncs), len(face_ncs))

        try:
            solid = construct_brep_fit(face_wcs, edge_wcs, data['faceEdge_adj'], data['edgeVert_adj'])
        except Exception as e:
            print('B-rep rebuild failed...')
            continue

        parts = file.split('/')
        result = f"{parts[-2]}_{parts[-1]}"
        save_name = result[:-4]
        write_step_file(solid, f'samples/{save_name}.step')
        write_stl_file(solid, f'samples/{save_name}.stl', linear_deflection=0.001, angular_deflection=0.5)


# import argparse
# import os
# from model import FaceGeomTransformer
# from test_face import get_faceGeom, get_topology
# from diffusers import DDPMScheduler, PNDMScheduler
# from utils import sort_bbox_multi
# from OCC.Extend.DataExchange import write_step_file
# from brepBuild import (joint_optimize, construct_brep, Brep2Mesh,
#                        create_bspline_surface, create_bspline_curve,
#                        sample_bspline_surface, sample_bspline_curve)
# from visualization import *
# import warnings
#
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)
#
#
# def get_args_generate():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--faceGeom_path', type=str, default='checkpoints/furniture/geom_faceGeom_02/epoch_3000.pt')
#     parser.add_argument('--save_folder', type=str, default="samples/test", help='save folder.')
#     parser.add_argument('--bbox_scaled', type=float, default=3, help='scaled the bbox')
#     args = parser.parse_args()
#
#     return args
#
#
# def get_brep(args):
#     # nf*48, nf*6, nv*3, ne*12, ne*2, ne*2, List[List[int]]
#     face_geom, face_bbox, vert_geom, edge_geom, edgeFace_adj, edgeVert_adj, faceEdge_adj = args
#
#     assert set(range(face_geom.shape[0])) == set(np.unique(edgeFace_adj.reshape(-1)).tolist())
#     assert set(range(vert_geom.shape[0])) == set(np.unique(edgeVert_adj.reshape(-1)).tolist())
#
#     face_ncs = []
#     for ctrs in face_geom:
#         pcd = sample_bspline_surface(create_bspline_surface(ctrs.reshape(16, 3).astype(np.float64)))  # 32*32*3
#         face_ncs.append(pcd)
#     face_ncs = np.stack(face_ncs)  # nf*32*32*3
#
#     edge_ncs = []
#     for ctrs in edge_geom:
#         pcd = sample_bspline_curve(create_bspline_curve(ctrs.reshape(4, 3).astype(np.float64)))  # 32*3
#         edge_ncs.append(pcd)
#     edge_ncs = np.stack(edge_ncs)  # ne*32*3
#
#     # joint_optimize
#     face_wcs, edge_wcs = joint_optimize(face_ncs, edge_ncs,
#                                         face_bbox, vert_geom,
#                                         edgeVert_adj, faceEdge_adj, len(edge_ncs), len(face_ncs))
#
#     try:
#         solid = construct_brep(face_wcs, edge_wcs, faceEdge_adj, edgeVert_adj)
#     except Exception as e:
#         print('B-rep rebuild failed...')
#         return False
#
#     return solid
#
# def main():
#     args = get_args_generate()
#
#     # Make project directory if not exist
#     if not os.path.exists(args.save_folder):
#         os.makedirs(args.save_folder)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # Initial FaceGeomTransformer
#     faceGeom_model = FaceGeomTransformer(n_layers=8, face_geom_dim=48)
#     faceGeom_model.load_state_dict(torch.load(args.faceGeom_path))
#     faceGeom_model = faceGeom_model.to(device).eval()
#
#     pndm_scheduler = PNDMScheduler(
#         num_train_timesteps=1000,
#         beta_schedule='linear',
#         prediction_type='epsilon',
#         beta_start=0.0001,
#         beta_end=0.02,
#     )
#
#     ddpm_scheduler = DDPMScheduler(
#         num_train_timesteps=1000,
#         beta_schedule='linear',
#         prediction_type='epsilon',
#         beta_start=0.0001,
#         beta_end=0.02,
#         clip_sample=True,
#         clip_sample_range=3
#     )
#
#     """****************Brep Topology****************"""
#     datas = get_topology(['lamp/partstudio_0394.pkl'], device)
#     face_bbox, edgeVert_adj, edgeFace_adj, faceEdge_adj, vert_geom, edge_geom = (datas["face_bbox"],
#                                                                                  datas["edgeVert_adj"],
#                                                                                  datas["edgeFace_adj"],
#                                                                                  datas["faceEdge_adj"],
#                                                                                  datas["vert_geom"],
#                                                                                  datas['edge_geom'])
#     b = len(face_bbox)
#     face_bbox = [sort_bbox_multi(i)*args.bbox_scaled for i in face_bbox]     # [nf*6, ...]
#     vert_geom = [i*args.bbox_scaled for i in vert_geom]                      # [nv*3, ...]
#     edge_geom = [i * args.bbox_scaled for i in edge_geom]                    # [ne*12, ...]
#
#     """****************Face Geometry****************"""
#     face_geom, face_mask = get_faceGeom(face_bbox, vert_geom, edge_geom, edgeVert_adj,
#                                         faceEdge_adj, faceGeom_model, pndm_scheduler, ddpm_scheduler)  # b*nf*48, b*nf
#     face_geom = [i[j] for i, j in zip(face_geom, face_mask)]
#
#     """****************Construct Brep****************"""
#     args_list = [(face_geom[j].cpu().numpy() / args.bbox_scaled,
#                   face_bbox[j].cpu().numpy() / args.bbox_scaled,
#                   vert_geom[j].cpu().numpy() / args.bbox_scaled,
#                   edge_geom[j].cpu().numpy() / args.bbox_scaled,
#                   edgeFace_adj[j].cpu().numpy(),
#                   edgeVert_adj[j].cpu().numpy(),
#                   faceEdge_adj[j]) for j in range(b)]
#     solid = get_brep(args_list[0])
#     save_name = datas['name'][0]
#     write_step_file(solid, f'{args.save_folder}/{save_name}.step')
#
#     print('write stl...')
#     mesh_tool = Brep2Mesh(input_path=args.save_folder)
#     mesh_tool.generate()
#
#
# if __name__ == '__main__':
#     main()


# import os
# import pickle
# from utils import check_step_ok, load_data_with_prefix
# from OCC.Core.STEPControl import STEPControl_Reader
# from OCC.Core.TopAbs import TopAbs_FACE
# from OCC.Core.TopExp import TopExp_Explorer
# from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
# from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface
# from tqdm import tqdm
#
#
# def read_step_file(step_file_path):
#     reader = STEPControl_Reader()
#     status = reader.ReadFile(step_file_path)
#     if status == 1:
#         reader.TransferRoots()
#         shape = reader.OneShape()
#         return shape
#     else:
#         raise Exception("Failed to read STEP file.")
#
# def count_face_types(step_file_path, face_type_count):
#     shape = read_step_file(step_file_path)
#
#     # 遍历所有面
#     explorer = TopExp_Explorer(shape, TopAbs_FACE)
#     while explorer.More():
#         face = explorer.Current()
#
#         # 适配面来获得其几何信息
#         surface_adaptor = BRepAdaptor_Surface(face, True)
#         surface_type = surface_adaptor.GetType()
#
#         # 通过面类型来计数
#         if surface_type == GeomAbs_Plane:
#             face_type_count['Plane'] += 1
#         elif surface_type == GeomAbs_Cylinder:
#             face_type_count['Cylinder'] += 1
#         elif surface_type == GeomAbs_Cone:
#             face_type_count['Cone'] += 1
#         elif surface_type == GeomAbs_Sphere:
#             face_type_count['Sphere'] += 1
#         elif surface_type == GeomAbs_Torus:
#             face_type_count['Torus'] += 1
#         elif surface_type == GeomAbs_BSplineSurface:
#             face_type_count['BSplineSurface'] += 1
#         else:
#             face_type_count['Other'] += 1
#
#         explorer.Next()
#
#     return face_type_count
#
#
# face_type_count = {
#     'Plane': 0,
#     'Cylinder': 0,
#     'Cone': 0,
#     'Sphere': 0,
#     'Torus': 0,
#     'BSplineSurface': 0,
#     'Other': 0
# }
# test_files = load_data_with_prefix('samples/bug', '.step')
# for data in tqdm(test_files):
#     face_type_count = count_face_types(data, face_type_count)
# print(face_type_count)
