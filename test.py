import sys
from collections import Counter
import yaml
import os
import shutil
import torch.multiprocessing as mp
from tqdm import tqdm
import pickle
import numpy as np
from OCC.Extend.DataExchange import write_step_file, write_stl_file
from inference.brepBuild import sample_bspline_curve, create_bspline_curve, joint_optimize, construct_brep
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
    GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface, GeomAbs_SurfaceOfRevolution,
    GeomAbs_SurfaceOfExtrusion, GeomAbs_OffsetSurface,
    GeomAbs_OtherSurface
)
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from utils import load_data_with_prefix


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


def get_surface_type(surface_type):
    """Convert GeomAbs surface type to string"""
    types = {
        GeomAbs_Plane: "Plane",
        GeomAbs_Cylinder: "Cylinder",
        GeomAbs_Cone: "Cone",
        GeomAbs_Sphere: "Sphere",
        GeomAbs_Torus: "Torus",
        GeomAbs_BezierSurface: "Bezier",
        GeomAbs_BSplineSurface: "BSpline",
        GeomAbs_SurfaceOfRevolution: "Revolution",
        GeomAbs_SurfaceOfExtrusion: "Extrusion",
        GeomAbs_OffsetSurface: "Offset",
        GeomAbs_OtherSurface: "Other"
    }
    return types.get(surface_type, "Unknown")


def analyze_step_file_faces(step_file_path):
    """
    Analyze and print surface type for each face in a STEP file
    Args:
        step_file_path: path to the STEP file
    """
    # Read STEP file
    reader = STEPControl_Reader()
    reader.ReadFile(step_file_path)
    reader.TransferRoots()
    shape = reader.OneShape()

    # Explore faces
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_index = 1

    while explorer.More():
        face = explorer.Current()

        # Get surface type
        surf = BRepAdaptor_Surface(face)
        surface_type = get_surface_type(surf.GetType())

        print(f"Face {face_index}: {surface_type}")

        face_index += 1
        explorer.Next()


def count_faces_in_step(filepath):
    """加载一个STEP文件并统计面数量。"""
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filepath)
    if status != 1:
        print(f"Failed to load {filepath}")
        return None

    step_reader.TransferRoots()
    shape = step_reader.OneShape()

    # 统计面数量
    face_count = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face_count += 1
        explorer.Next()
    return face_count


def last():
    with open('/home/jing/PythonProjects/BrepGDM/Pics/Srcs/vis_overview/table_SP6URjYZo8.pkl', 'rb') as f:
        data = pickle.load(f)
    solid = construct_brep(data['edge_wcs'], data['faceEdge_adj'], data['edgeVert_adj'])
    write_step_file(solid, 'jing.step')


def main():
    # test_files = ['00638827_fcef2331e2f6a654e9365b06_step_000.pkl']
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
    #         save_name = file[:-4]
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

    # step_file = "/home/jing/PythonProjects/BrepGDM/samples/deepcad_results/test_edge/0034_00349931_d3200ad7e4b39853237b1471_step_022.step"
    # print(f"Analyzing surface types for each face in: {step_file}")
    # print("-" * 50)
    # analyze_step_file_faces(step_file)

    """统计多个STEP文件中的面数量分布。"""
    step_files = load_data_with_prefix('/home/jing/PythonProjects/BrepGDM/comparison/datas/abc/BrepGen_3500/', '.step')
    face_counts = []
    for filepath in tqdm(step_files):
        face_count = count_faces_in_step(filepath)
        if face_count is not None:
            face_counts.append(face_count)
    distribution = Counter(face_counts)
    for num_faces, count in sorted(distribution.items()):
        print(f"{num_faces} faces: {count} file(s)")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    # main()

    last()
