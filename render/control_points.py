import ast
import copy
import os

import numpy as np
from tqdm import tqdm
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import topods_Face, topods_Edge, topods_Vertex
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace
from OCC.Core.gp import gp_Pnt, gp_Ax1, gp_Dir, gp_Trsf, gp_Vec
import pickle

from inference.brepBuild import create_bspline_surface
from render.myx3dom_render import myX3DomRenderer
from utils import load_data_with_prefix


def read_step(filename):
    step_reader = STEPControl_Reader()
    step_reader.ReadFile(filename)
    step_reader.TransferRoot()
    shape = step_reader.Shape()
    return shape


def diffusion_ctrs():

    from diffusers import DDPMScheduler
    import torch
    import numpy as np

    files = load_data_with_prefix('/home/jing/PythonProjects/BrepGDM/Supp/ctrs/', 'pkl')
    for file in tqdm(files):
        print(file)
        with open(file, 'rb') as f:
            data = pickle.load(f)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_0 = torch.from_numpy(data['face_ctrs']).to(device)*3       # nf*16*3
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='linear',
            prediction_type='epsilon',
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,
        )

        np.save(os.path.join(os.path.dirname(file), '0.npy'), (x_0.cpu().numpy())/3)
        times = [time for time in range(5, 101, 5)] + [150, 200, 250, 300, 350, 400, 450, 500, 999]

        for time in times:
            t = torch.tensor([time], dtype=torch.long, device=device)  # b
            noise = torch.randn(x_0.shape).to(device)
            x = noise_scheduler.add_noise(x_0, noise, t)    # nf*16*3
            np.save(os.path.join(os.path.dirname(file), f'{time}.npy'), (x.cpu().numpy())/3)


def draw_diffusion_ctrs():
    import numpy as np
    from inference.brepBuild import create_bspline_surface

    name = '/home/jing/PythonProjects/BrepGDM/Supp/ctrs/0clZHMJMgp6hzKqa'
    file_path = '/home/jing/PythonProjects/BrepGDM/Supp/ctrs/0clZHMJMgp6hzKqa/htmlscolor.txt'
    save_path = '/home/jing/PythonProjects/BrepGDM/Supp/ctrs/0clZHMJMgp6hzKqa/htmls'
    ctrs = np.load(os.path.join(name, '700.npy'))    # nf*16*3

    axisz = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    anglez = -30 * (3.14159 / 180)
    rotationz = gp_Trsf()
    rotationz.SetRotation(axisz, anglez)
    axisx = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0))
    anglex = 30 * (3.14159 / 180)
    rotationx = gp_Trsf()
    rotationx.SetRotation(axisx, anglex)
    radius = 0.03
    my_renderer = myX3DomRenderer(display_axes_plane=False, path=save_path)
    colors = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    colors = [ast.literal_eval(line.strip()) for line in lines]

    # # create face control lines
    # for k, face_ctrs in enumerate(ctrs):
    #     # color = (np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8))
    #     # colors.append(color)
    #
    #     color = colors[k]
    #     surf = create_bspline_surface(face_ctrs)
    #     face = BRepBuilderAPI_MakeFace(surf, 1e-6).Face()
    #     face = BRepBuilderAPI_Transform(face, rotationz).Shape()
    #     face = BRepBuilderAPI_Transform(face, rotationx).Shape()
    #     # Display face
    #     translation = gp_Trsf()
    #     translation.SetTranslation(gp_Vec(0, 0, 0))
    #     face = BRepBuilderAPI_Transform(face, translation).Shape()
    #     my_renderer.myDisplayShape(face, color=color, export_edges=True, transparency=0)
    #
    #     # sphere_face = []
    #     # line_face = []
    #     # for ctrs in face_ctrs:
    #     #     sphere = BRepPrimAPI_MakeSphere(gp_Pnt(ctrs[0], ctrs[1], ctrs[2]), radius).Shape()
    #     #     sphere = BRepBuilderAPI_Transform(sphere, rotationz).Shape()
    #     #     sphere_face.append(BRepBuilderAPI_Transform(sphere, rotationx).Shape())
    #     # face_ctrs = face_ctrs.reshape(4, 4, 3)
    #     # for i in range(4):
    #     #     for j in range(3):
    #     #         pt1 = gp_Pnt(face_ctrs[i, j, 0], face_ctrs[i, j, 1], face_ctrs[i, j, 2])
    #     #         pt2 = gp_Pnt(face_ctrs[i, j + 1, 0], face_ctrs[i, j + 1, 1], face_ctrs[i, j + 1, 2])
    #     #         line = BRepBuilderAPI_MakeEdge(pt1, pt2).Edge()
    #     #         line = BRepBuilderAPI_Transform(line, rotationz).Shape()
    #     #         line_face.append(BRepBuilderAPI_Transform(line, rotationx).Shape())
    #     # for j in range(4):
    #     #     for i in range(3):
    #     #         pt1 = gp_Pnt(face_ctrs[i, j, 0], face_ctrs[i, j, 1], face_ctrs[i, j, 2])
    #     #         pt2 = gp_Pnt(face_ctrs[i + 1, j, 0], face_ctrs[i + 1, j, 1], face_ctrs[i + 1, j, 2])
    #     #         line = BRepBuilderAPI_MakeEdge(pt1, pt2).Edge()
    #     #         line = BRepBuilderAPI_Transform(line, rotationz).Shape()
    #     #         line_face.append(BRepBuilderAPI_Transform(line, rotationx).Shape())
    #     #
    #     # # Display face control points
    #     # for sphere in sphere_face:
    #     #     my_renderer.myDisplayShape(sphere, color=color, export_edges=False)
    #     # for line in line_face:
    #     #     my_renderer.myDisplayShape(line, line_color=color, export_edges=False, line_width=5)
    #
    # # with open(file_path, 'w') as file:  # 'w' 表示写入模式，文件不存在会自动创建
    # #     for line in colors:
    # #         file.write(str(line) + '\n')


    # Display total brep
    shape = read_step(os.path.join(name, name.split('/')[-1] + '.step'))
    shape = BRepBuilderAPI_Transform(shape, rotationz).Shape()
    shape = BRepBuilderAPI_Transform(shape, rotationx).Shape()
    translation = gp_Trsf()
    translation.SetTranslation(gp_Vec(6, 0, 0))
    shape = BRepBuilderAPI_Transform(shape, translation).Shape()
    exp_face = TopExp_Explorer(shape, TopAbs_FACE)
    index = 0
    while exp_face.More():
        face = topods_Face(exp_face.Current())
        my_renderer.myDisplayShape(face, export_edges=True, color=colors[index])
        exp_face.Next()
        index += 1


    my_renderer.render()


# def record(name, i, save_base_path, colors, radius, scale_trsf, rotationz, rotationx):
#
#     ctrs_file = os.path.join(name, f"{i}.npy")
#
#     ctrs = np.load(ctrs_file)
#     save_path = os.path.join(save_base_path, str(i))
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#
#     my_renderer = myX3DomRenderer(display_axes_plane=False, path=save_path)
#     save_colors = []
#
#     for k, face_ctrs in enumerate(ctrs):
#         if colors is None:
#             color = (np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8))
#             save_colors.append(color)
#         else:
#             color = colors[k]
#         surf = create_bspline_surface(face_ctrs)
#         face = BRepBuilderAPI_MakeFace(surf, 1e-6).Face()
#         face = BRepBuilderAPI_Transform(face, scale_trsf).Shape()
#         face = BRepBuilderAPI_Transform(face, rotationz).Shape()
#         face = BRepBuilderAPI_Transform(face, rotationx).Shape()
#         translation = gp_Trsf()
#         translation.SetTranslation(gp_Vec(0, 0, 0))
#         face = BRepBuilderAPI_Transform(face, translation).Shape()
#         my_renderer.myDisplayShape(face, color=color, export_edges=True, transparency=0)
#
#
#
#         # sphere_face = []
#         # line_face = []
#         # for ctrs in face_ctrs:
#         #     sphere = BRepPrimAPI_MakeSphere(gp_Pnt(ctrs[0], ctrs[1], ctrs[2]), radius).Shape()
#         #     sphere = BRepBuilderAPI_Transform(sphere, scale_trsf).Shape()
#         #     sphere = BRepBuilderAPI_Transform(sphere, rotationz).Shape()
#         #     sphere_face.append(BRepBuilderAPI_Transform(sphere, rotationx).Shape())
#         # face_ctrs = face_ctrs.reshape(4, 4, 3)
#         # for i in range(4):
#         #     for j in range(3):
#         #         pt1 = gp_Pnt(face_ctrs[i, j, 0], face_ctrs[i, j, 1], face_ctrs[i, j, 2])
#         #         pt2 = gp_Pnt(face_ctrs[i, j + 1, 0], face_ctrs[i, j + 1, 1], face_ctrs[i, j + 1, 2])
#         #         line = BRepBuilderAPI_MakeEdge(pt1, pt2).Edge()
#         #         line = BRepBuilderAPI_Transform(line, scale_trsf).Shape()
#         #         line = BRepBuilderAPI_Transform(line, rotationz).Shape()
#         #         line_face.append(BRepBuilderAPI_Transform(line, rotationx).Shape())
#         # for j in range(4):
#         #     for i in range(3):
#         #         pt1 = gp_Pnt(face_ctrs[i, j, 0], face_ctrs[i, j, 1], face_ctrs[i, j, 2])
#         #         pt2 = gp_Pnt(face_ctrs[i + 1, j, 0], face_ctrs[i + 1, j, 1], face_ctrs[i + 1, j, 2])
#         #         line = BRepBuilderAPI_MakeEdge(pt1, pt2).Edge()
#         #         line = BRepBuilderAPI_Transform(line, scale_trsf).Shape()
#         #         line = BRepBuilderAPI_Transform(line, rotationz).Shape()
#         #         line_face.append(BRepBuilderAPI_Transform(line, rotationx).Shape())
#         #
#         # # Display face control points
#         # for sphere in sphere_face:
#         #     my_renderer.myDisplayShape(sphere, color=color, export_edges=False)
#         # for line in line_face:
#         #     my_renderer.myDisplayShape(line, line_color=color, export_edges=False, line_width=5)
#
#     if len(save_colors) > 0:
#         with open(save_base_path+"color.txt", 'w') as file:  # 'w' 表示写入模式，文件不存在会自动创建
#             for line in save_colors:
#                 file.write(str(line) + '\n')
#     my_renderer.render()
#     return my_renderer
#
# def draw_diffusion_ctrs():
#     name = '/home/jing/PythonProjects/BrepGDM/Supp/ctrs/0clZHMJMgp6hzKqa'
#     file_path = '/home/jing/PythonProjects/BrepGDM/Supp/ctrs/0clZHMJMgp6hzKqa/htmlscolor.txt'
#     save_base_path = '/home/jing/PythonProjects/BrepGDM/Supp/ctrs/0clZHMJMgp6hzKqa/htmls'
#     save_base_path2 = '/home/jing/PythonProjects/BrepGDM/Supp/ctrs/0clZHMJMgp6hzKqa/htmls2'
#
#     scale_trsf = gp_Trsf()
#     scale_trsf.SetScale(gp_Pnt(0, 0, 0), 2)
#
#     axisz = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
#     anglez = -30 * (3.14159 / 180)
#     rotationz = gp_Trsf()
#     rotationz.SetRotation(axisz, anglez)
#     axisx = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0))
#     anglex = 30 * (3.14159 / 180)
#     rotationx = gp_Trsf()
#     rotationx.SetRotation(axisx, anglex)
#     radius = 0.03
#
#
#     colors = []
#
#     if os.path.exists(file_path):
#         with open(file_path, 'r') as file:
#             lines = file.readlines()
#         colors = [ast.literal_eval(line.strip()) for line in lines]
#     else:
#         colors = None
#
#     for i in range(0, 100, 5):
#         my_renderer = record(name, i, save_base_path2, colors, radius, scale_trsf, rotationz, rotationx)
#
#     for i in range(100, 501, 50):
#         my_renderer = record(name, i, save_base_path2, colors, radius, scale_trsf, rotationz, rotationx)
#
#     my_renderer = record(name, 999, save_base_path2, colors, radius, scale_trsf, rotationz, rotationx)
#     # record(name, 0, save_base_path, colors, radius, scale_trsf, rotationz, rotationx)


def main():

    # diffusion_ctrs()
    draw_diffusion_ctrs()


if __name__ == "__main__":
    main()
