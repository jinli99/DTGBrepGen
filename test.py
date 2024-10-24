import os
import pickle
import random
import shutil
import trimesh
import yaml
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
from utils import check_step_ok, load_data_with_prefix
from OCC.Extend.DataExchange import write_stl_file, write_step_file
from visualization import *
from brepBuild import (create_bspline_surface, create_bspline_curve,
                       sample_bspline_surface, sample_bspline_curve, joint_optimize, construct_brep)
from collections import defaultdict

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
#             print(step_file_path)
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
# test_files = load_data_with_prefix('samples/11', '.step')
# for data in tqdm(test_files):
#     face_type_count = count_face_types(data, face_type_count)
# print(face_type_count)


def main():
    # test_files = ['data_process/GeomDatasets/deepcad_parsed/0007/00070019_9321559d3d404be585a05ab2_step_003.pkl',
    #               'data_process/GeomDatasets/deepcad_parsed/0012/00120032_e7e0b2ca270f985feb34bccb_step_000.pkl']
    # for file in tqdm(test_files):
    #     with open(file, 'rb') as f:
    #         data = pickle.load(f)
    #         edge_ctrs = data['edge_ctrs']
    #
    #         edge_ncs = []
    #         for ctrs in edge_ctrs:
    #             ncs = sample_bspline_curve(create_bspline_curve(ctrs))
    #             edge_ncs.append(ncs)
    #         edge_ncs = np.stack(edge_ncs)  # ne*32*3
    #
    #         edge_wcs = joint_optimize(edge_ncs,
    #                                   data['face_bbox_wcs'], data['vert_wcs'],
    #                                   data['edgeVert_adj'], data['faceEdge_adj'], len(edge_ncs),
    #                                   len(data['face_bbox_wcs']))
    #
    #         try:
    #             solid = construct_brep(edge_wcs, data['faceEdge_adj'], data['edgeVert_adj'])
    #         except Exception as e:
    #             print('B-rep rebuild failed...')
    #             continue
    #
    #         parts = file.split('/')
    #         result = f"{parts[-2]}_{parts[-1]}"
    #         save_name = result[:-4]
    #         write_step_file(solid, f'samples/{save_name}.step')
    #         write_stl_file(solid, f'samples/{save_name}.stl', linear_deflection=0.001, angular_deflection=0.5)

    # files = load_data_with_prefix('data_process/GeomDatasets/deepcad_parsed', '.pkl')
    # pc_path = '/home/jing/PythonProjects/BrepGDM/comparison/datas/deepcad/reference_test'
    # for file in tqdm(files):
    #     with open(file, 'rb') as f:
    #         data = pickle.load(f)
    #     if 'pc' in data and data['pc'] is not None:
    #         continue
    #     label = file.split('/')[-2]
    #     name = file.split('/')[-1][:-4] + '.ply'
    #     pc_file = os.path.join(pc_path, label, name)
    #     if os.path.exists(pc_file):
    #         mesh = trimesh.load(pc_file)
    #         points = np.array(mesh.vertices)
    #         data['pc'] = points
    #     else:
    #         data['pc'] = None
    #     with open(file, 'wb') as f:
    #         pickle.dump(data, f)

    # with open('data_process/deepcad_data_split_6bit.pkl', 'rb') as f:
    #     data = pickle.load(f)['train']
    # valid = 0
    # for i in data:
    #     with open(os.path.join('data_process/GeomDatasets/deepcad_parsed', i), 'rb') as f:
    #         file = pickle.load(f)
    #     if check_step_ok(file):
    #         valid += 1
    # print(valid)

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # 根据 dataset_name 获取对应的参数
    # dataset_config = config['datasets'].get(dataset_name)
    print(1)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
