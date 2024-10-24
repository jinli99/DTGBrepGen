import os
import argparse
import random
from multiprocessing.pool import Pool
from occwl.io import load_step
from OCC.Extend.DataExchange import write_stl_file
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.gp import gp_Vec, gp_Trsf
from tqdm import tqdm
import multiprocessing
from pathlib import Path
import pickle
import trimesh
from trimesh.sample import sample_surface
from plyfile import PlyData, PlyElement
import numpy as np
from utils import check_step_ok, load_data_with_prefix
from functools import partial


def write_ply(points, filename, text=False):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)


def find_files(folder, extension):
    return sorted([Path(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith(extension)])


class SamplePoints:
    """
    Perform sampleing of points.
    """

    def __init__(self):
        """
        Constructor.
        """
        parser = self.get_parser()
        self.options = parser.parse_args()

    def get_parser(self):
        """
        Get parser of tool.

        :return: parser
        """
        parser = argparse.ArgumentParser(description='Scale a set of meshes stored as OFF files.')
        parser.add_argument('--in_dir', type=str, default='/home/jing/PythonProjects/BrepGDM/comparison/datas/deepcad/reference_test/0000', help='Path to input directory.')
        parser.add_argument('--out_dir', type=str, default='/home/jing/PythonProjects/BrepGDM/comparison/datas/deepcad/reference_test/0000', help='Path to output directory; files within are overwritten!')
        return parser

    def run_parallel(self, path):
        fileName = os.path.join(self.options.out_dir, path.split('/')[-1][:-4])

        N_POINTS = 2000
        out_mesh = trimesh.load(path)
        out_pc, _ = sample_surface(out_mesh, N_POINTS)
        save_path = os.path.join(fileName + '.ply')
        write_ply(out_pc, save_path)
        return

    def run(self):
        """
        Run simplification.
        """
        if not os.path.exists(self.options.out_dir):
            os.makedirs(self.options.out_dir)

        shape_paths = load_data_with_prefix(self.options.in_dir,
                                            '.stl')  # + load_data_with_prefix(self.options.in_dir, '.obj')

        # shape_paths = shape_paths
        # for path in shape_paths:
        #     self.run_parallel(path)

        num_cpus = multiprocessing.cpu_count()
        convert_iter = multiprocessing.Pool(num_cpus).imap(self.run_parallel, shape_paths)
        for _ in tqdm(convert_iter, total=len(shape_paths)):
            pass


def step2stl(file, option='deepcad'):

    with open(os.path.join('data_process/GeomDatasets', option+'_parsed', file), 'rb') as f:
        data = pickle.load(f)
        if not check_step_ok(data):
            return 0

    parts = file.split('/')
    temp = parts[1].split('_', 1)

    if option == 'furniture':
        step_path = os.path.join('/home/jing/Datasets/Furniture/', parts[0], temp[0], temp[1].split('.')[0]+'.step')
    elif option == 'deepcad':
        step_path = os.path.join('/home/jing/Datasets/DeepCAD/', parts[0], parts[1].split('.')[0] + '.step')
    else:
        assert option == 'abc'

    solid = load_step(step_path)[0]
    solid = solid.split_all_closed_faces(num_splits=0)
    solid = solid.split_all_closed_edges(num_splits=0)
    bbox = Bnd_Box()
    brepbndlib_Add(solid._shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    max_dim = max(xmax - xmin, ymax - ymin, zmax - zmin)
    scale_factor = 2.0 / max_dim

    trsf = gp_Trsf()
    trsf.SetTranslationPart(gp_Vec(-(xmin + xmax) / 2.0, -(ymin + ymax) / 2.0, -(zmin + zmax) / 2.0))
    solid = BRepBuilderAPI_Transform(solid._shape, trsf).Shape()

    trsf = gp_Trsf()
    trsf.SetScaleFactor(scale_factor)
    solid = BRepBuilderAPI_Transform(solid, trsf).Shape()
    bbox = Bnd_Box()
    brepbndlib_Add(solid, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    assert max(xmin, ymin, zmin, xmax, ymax, zmax) < 1.1 and min(xmin, ymin, zmin, xmax, ymax, zmax) > -1.1

    try:
        if option == 'furniture':
            write_stl_file(
                solid,
                os.path.join('comparison/datas/furniture/reference', parts[0] + '_' + parts[1].split('.')[0] + '.stl'),
                linear_deflection=0.001, angular_deflection=0.5)
        elif option == 'deepcad':
            os.makedirs(os.path.join('comparison/datas/deepcad/reference', parts[0]), exist_ok=True)
            write_stl_file(
                solid,
                os.path.join('comparison/datas/deepcad/reference', parts[0], parts[1].split('.')[0] + '.stl'),
                linear_deflection=0.001, angular_deflection=0.5)
        return 1
    except Exception as e:
        return 0


def get_reference_stl():
    with open('data_process/deepcad_data_split_6bit.pkl', 'rb') as tf:
        files = pickle.load(tf)['train']

    option = 'deepcad'

    valid = 0
    process_with_option = partial(step2stl, option=option)
    convert_iter = Pool(os.cpu_count()).imap(process_with_option, files)
    for status in tqdm(convert_iter, total=len(files)):
        valid += status

    # valid = 0
    # for file in tqdm(files):
    #     valid += step2stl(file, option=option)

    print(valid)


if __name__ == '__main__':
    app = SamplePoints()
    app.run()

    # get_reference_stl()
