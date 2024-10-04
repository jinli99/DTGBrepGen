# import os
# import pickle
# import random
# import numpy as np
# from tqdm import tqdm
# from utils import check_step_ok
# from OCC.Extend.DataExchange import write_stl_file, write_step_file
# from visualization import *
# from brepBuild import (create_bspline_surface, create_bspline_curve,
#                        sample_bspline_surface, sample_bspline_curve, joint_optimize, construct_brep)
#
# test_files = ['data_process/GeomDatasets/furniture_parsed/lamp/partstudio_0394.pkl']
# for file in tqdm(test_files):
#     with open(file, 'rb') as f:
#         data = pickle.load(f)
#         face_ctrs = data['face_ctrs']
#         edge_ctrs = data['edge_ctrs']
#
#         face_ncs = []
#         for i, ctrs in enumerate(face_ctrs):
#
#             ncs = sample_bspline_surface(create_bspline_surface(ctrs))      # 32*32*3
#             face_ncs.append(ncs)
#         face_ncs = np.stack(face_ncs)   # nf*32*32*3
#
#         edge_ncs = []
#         for ctrs in edge_ctrs:
#             ncs = sample_bspline_curve(create_bspline_curve(ctrs))
#             edge_ncs.append(ncs)
#         edge_ncs = np.stack(edge_ncs)    # ne*32*3
#
#         face_wcs, edge_wcs = joint_optimize(face_ncs, edge_ncs,
#                                             data['face_bbox_wcs'], data['vert_wcs'],
#                                             data['edgeVert_adj'], data['faceEdge_adj'], len(edge_ncs), len(face_ncs))
#
#         try:
#             solid = construct_brep(face_wcs, edge_wcs, data['faceEdge_adj'], data['edgeVert_adj'])
#         except Exception as e:
#             print('B-rep rebuild failed...')
#             continue
#
#         parts = file.split('/')
#         result = f"{parts[-2]}_{parts[-1]}"
#         save_name = result[:-4]
#         write_step_file(solid, f'samples/{save_name}.step')
#         write_stl_file(solid, f'samples/{save_name}.stl', linear_deflection=0.001, angular_deflection=0.5)


import os
import pickle
from utils import check_step_ok, load_data_with_prefix
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface
from tqdm import tqdm


def read_step_file(step_file_path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_file_path)
    if status == 1:
        reader.TransferRoots()
        shape = reader.OneShape()
        return shape
    else:
        raise Exception("Failed to read STEP file.")

def count_face_types(step_file_path, face_type_count):
    shape = read_step_file(step_file_path)

    # 遍历所有面
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()

        # 适配面来获得其几何信息
        surface_adaptor = BRepAdaptor_Surface(face, True)
        surface_type = surface_adaptor.GetType()

        # 通过面类型来计数
        if surface_type == GeomAbs_Plane:
            face_type_count['Plane'] += 1
        elif surface_type == GeomAbs_Cylinder:
            face_type_count['Cylinder'] += 1
        elif surface_type == GeomAbs_Cone:
            face_type_count['Cone'] += 1
        elif surface_type == GeomAbs_Sphere:
            face_type_count['Sphere'] += 1
        elif surface_type == GeomAbs_Torus:
            face_type_count['Torus'] += 1
        elif surface_type == GeomAbs_BSplineSurface:
            print(step_file_path)
            face_type_count['BSplineSurface'] += 1
        else:
            face_type_count['Other'] += 1

        explorer.Next()

    return face_type_count


face_type_count = {
    'Plane': 0,
    'Cylinder': 0,
    'Cone': 0,
    'Sphere': 0,
    'Torus': 0,
    'BSplineSurface': 0,
    'Other': 0
}
test_files = load_data_with_prefix('samples/11', '.step')
for data in tqdm(test_files):
    face_type_count = count_face_types(data, face_type_count)
print(face_type_count)


# import pickle
# import numpy as np
# import os
# from tqdm import tqdm
# from utils import check_step_ok
#
# test_files = []
# for root, dirs, files in os.walk('data_process/GeomDatasets/furniture_parsed'):
#     for file in files:
#         with open(os.path.join(root, file), 'rb') as f:
#             data = pickle.load(f)
#             if check_step_ok(data):
#                 test_files.append(os.path.join(root, file))
# print(len(test_files))
#
# for file in tqdm(test_files):
#     with open(file, 'rb') as f:
