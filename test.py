import os
import pickle
import random
import numpy as np
from utils import check_step_ok
from OCC.Extend.DataExchange import write_stl_file, write_step_file
from brepBuild import (create_bspline_surface, create_bspline_curve,
                       sample_bspline_surface, sample_bspline_curve, joint_optimize, construct_brep)
from data_process.brep_process import sort_face_ctrs


testFiles = []
for root, dirs, files in os.walk('data_process/GeomDatasets/furniture_parsed'):
    for file in files:
        if file.endswith('.pkl'):
            with open(os.path.join(root, file), 'rb') as f:
                data = pickle.load(f)
            if check_step_ok(data):
                testFiles.append(os.path.join(root, file))

print(f'total: {len(testFiles)}')

# datas = random.sample(testFiles, len(testFiles))
datas = testFiles
for file in [datas[500]]:
    with open(file, 'rb') as f:
        data = pickle.load(f)
    face_ctrs = data['face_ctrs']
    edge_ctrs = data['edge_ctrs']

    face_ncs = []
    for i, ctrs in enumerate(face_ctrs):
        if i < 5:
            ctrs = ctrs.reshape(4, 4, 3)
            ctrs = np.swapaxes(ctrs, 0, 1)
            ctrs = ctrs.reshape(-1, 3)
        else:
            ctrs = ctrs.reshape(4, 4, 3)
            ctrs = ctrs[::-1, ...]
            ctrs = ctrs.reshape(-1, 3)

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
        solid = construct_brep(face_wcs, edge_wcs, data['faceEdge_adj'], data['edgeVert_adj'])
    except Exception as e:
        print('B-rep rebuild failed...')
        continue

    parts = file.split('/')
    result = f"{parts[-2]}_{parts[-1]}"
    save_name = result[:-4]
    write_step_file(solid, f'samples/test/{save_name}.step')
    write_stl_file(solid, f'samples/test/{save_name}.stl', linear_deflection=0.001, angular_deflection=0.5)
