import copy
import math
import os
from OCC import VERSION
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Display.WebGl.x3dom_renderer import X3DomRenderer, HTMLHeader, HTMLBody, HEADER
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Ax1, gp_Pnt, gp_Dir
from myx3dom_render import myX3DomRenderer
from utils import load_data_with_prefix
import random


def render_web(step_folder, color, scale_factor, angle_x, angle_z, translation_step_x, translation_step_z, set_x=None, set_z=None, line_width=3.0):

    my_renderer = myX3DomRenderer(display_axes_plane=False)
    # my_renderer = x3dom_renderer.X3DomRenderer(display_axes_plane=False)

    files = load_data_with_prefix(step_folder, '.step')

    temp = files[10:20]
    files[10:20] = files[:10]
    files[:10] = temp
    temp = scale_factor[10:20]
    scale_factor[10:20] = scale_factor[:10]
    scale_factor[:10] = temp
    temp = angle_x[10:20]
    angle_x[10:20] = angle_x[:10]
    angle_x[:10] = temp
    temp = angle_z[10:20]
    angle_z[10:20] = angle_z[:10]
    angle_z[:10] = temp
    # if set_x is not None:
    #     temp = set_x[10:20]
    #     set_x[10:20] = set_x[:10]
    #     set_x[:10] = temp
    #     temp = set_z[10:20]
    #     set_z[10:20] = set_z[:10]
    #     set_z[:10] = temp

    shapes_per_row = 5
    current_offset_x = 0
    current_offset_z = 0
    shape_count = 0

    for i, file in enumerate(files):
        print(f"Loading: {file}")
        step_reader = STEPControl_Reader()
        status = step_reader.ReadFile(file)

        if status == IFSelect_RetDone:
            step_reader.TransferRoots()
            shape = step_reader.OneShape()

            axisx = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0))
            anglex = angle_x[i] * (3.14159 / 180)
            rotationx = gp_Trsf()
            rotationx.SetRotation(axisx, anglex)
            rotated_shapex = BRepBuilderAPI_Transform(shape, rotationx).Shape()

            axisz = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
            anglez = angle_z[i] * (3.14159 / 180)
            rotationz = gp_Trsf()
            rotationz.SetRotation(axisz, anglez)
            rotated_shapez = BRepBuilderAPI_Transform(rotated_shapex, rotationz).Shape()

            scale_trsf = gp_Trsf()
            scale_trsf.SetScale(gp_Pnt(0, 0, 0), scale_factor[i])
            scaled_shape = BRepBuilderAPI_Transform(rotated_shapez, scale_trsf).Shape()

            translation = gp_Trsf()
            if set_x is not None:
                translation.SetTranslation(gp_Vec(set_x[i], 0, set_z[i]))
            else:
                translation.SetTranslation(gp_Vec(current_offset_x, 0, current_offset_z))
            transformed_shape = BRepBuilderAPI_Transform(scaled_shape, translation).Shape()

            my_renderer.myDisplayShape(transformed_shape, color=(color.Red(), color.Green(), color.Blue()),
                                     export_edges=True, line_width=line_width)

            shape_count += 1

            if shape_count % shapes_per_row == 0:
                current_offset_z += translation_step_z
                current_offset_x = 0
            else:
                current_offset_x += translation_step_x

        else:
            print(f"Error: Failed to read the STEP file: {file}")

    my_renderer.render()

rgb_colors = {'ours': Quantity_Color(50 / 255.0, 160 / 255.0, 190 / 255.0, Quantity_TOC_RGB),
              'brepgen': Quantity_Color(190 / 255.0, 60 / 255.0, 50 / 255.0, Quantity_TOC_RGB),
              'deepcad': Quantity_Color(50 / 255.0, 150 / 255.0, 50 / 255.0, Quantity_TOC_RGB),
              'fail': Quantity_Color(147 / 255.0, 112 / 255.0, 219 / 255.0, Quantity_TOC_RGB),
              'bug': Quantity_Color(153 / 255.0, 128 / 255.0, 76 / 255.0, Quantity_TOC_RGB)}

# ==============01================
# x_deepcad = [0, 2.7, 5, 7.5, 10,
#              0, 2.5, 5, 7.2, 9.8,
#              0, 3.2, 4.5, 7.5, 10,
#              0, 2.5, 5, 7.5, 10,]
# z_deepcad = [0, -1, 0, 0, 0,
#              2, 2.5, 2.6, 2.1, 2.5,
#              5, 5.4, 4.6, 5, 5,
#              7.5, 7.8, 7.5, 7.5, 7.5]

# ===============02================
x_deepcad = [0, 2.9, 5, 7.5, 10,
             0, 2.2, 5, 7.2, 9.8,
             0, 3.2, 4.5, 7.5, 10,
             0, 2.5, 5, 7.5, 10,]
z_deepcad = [0, -0.5, -0.2, 0, 0,
             2, 2.3, 2.4, 2.8, 2.1,
             5, 5.4, 6, 5, 5,
             7.5, 7.8, 7.5, 7.5, 7.5]

scale_factor_deepcad = [1.2, 2.2, 0.8, 1.2, 1.4,
                        1.4, 1.3, 1.5, 1.2, 1.0,
                        1.3, 1.5, 0.8, 1.2, 1.1,
                        1.1, 0.8, 1.0, 1.2, 1.6]
angle_x_deepcad = [90, 0, 90, 45, 0,
                   90, 0, 0, 0, 90,
                   0, 0, 0, 0, 0,
                   0, 0, 0, 90, 0,]
angle_z_deepcad = [30, 0, 30, 30, 30,
                   30, 30, 30, 30, 30,
                   120, 30, 30, 30, 30,
                   30, 30, 30, 0, 0]

scale_factor_brepgen = [1.1, 1., 1.1, 1.0, 1.0,
                        1.1, 1.1, 1.0, 1.1, 0.8,
                        1, 1, 1.0, 1, 0.8,
                        1.1, 1., 0.9, 0.9, 1.0]
angle_x_brepgen = [0, 0, 0, 0, 0,
                   90, 0, 0, 0, 0,
                   0, 0, 0, 90, 0,
                   90, 90, 0, 5, 0,]
angle_z_brepgen = [-30, 0, -30, -30, 30,
                   30, 30, 30, 30, 30,
                   30, -60, -60, -30, -30,
                   30, 0, -30, 0, 30,]

scale_factor_ours = [1.1, 1.0, 1.3, 0.9, 0.9,
                     1.3, 1.0, 1.0, 1.0, 1.1,
                     1.3, 1.2, 0.9, 1.0, 1.0,
                     1.0, 1.3, 1.0, 0.9, 0.9]
angle_x_ours = [30, 0, 0, 0, 90,
                90, 0, 0, 0, 90,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 90,]
angle_z_ours = [30, -95, -30, -30, -30,
                30, -90, 30, -45, 30,
                -30, -30, -30, -30, -30,
                30, -30, 30, 30, 30,]

scale_factor_fail = [1.0, 1.0, 1.0]
angle_x_fail = [0, 15, 0]
angle_z_fail = [-30, 0, -60]

scale_factor_bug = [1.0, 1.0, 1.0]
angle_x_bug = [0, 0, 90]
angle_z_bug = [0, 0, 0]

# render_web("Pics/Srcs/vis_bug", rgb_colors['ours'], scale_factor_bug, angle_x_bug, angle_z_bug, translation_step_x=2.5, translation_step_z=2.5, line_width=3.0)
# render_web("Pics/Srcs/vis_fail", rgb_colors['ours'], scale_factor_fail, angle_x_fail, angle_z_fail, translation_step_x=2.5, translation_step_z=2.5, set_x=[0, 2.5, 5], set_z=[0, 0, 0], line_width=3.0)
# render_web("Pics/Srcs/vis_ours", rgb_colors['ours'], scale_factor_ours, angle_x_ours, angle_z_ours, translation_step_x=3, translation_step_z=3, line_width=3.0)
# render_web("Pics/Srcs/vis_brepgen", rgb_colors['brepgen'], scale_factor_brepgen, angle_x_brepgen, angle_z_brepgen, translation_step_x = 3, translation_step_z=3)
render_web("Pics/Srcs/vis_deepcad", rgb_colors['deepcad'], scale_factor_deepcad, angle_x_deepcad, angle_z_deepcad,
           translation_step_x=2.5, translation_step_z=2.5, set_x=x_deepcad, set_z=z_deepcad, line_width=3.0)

# def vis_furniture(step_folder, scale_factor_furniture, angle_x_furniture, angle_y_furniture, angle_z_furniture):
#     my_renderer = myX3DomRenderer(display_axes_plane=False)
#     translation_step_x = 3.2
#     translation_step_z = 3
#
#     files = [f for f in os.listdir(step_folder) if f.endswith('.step')]
#     categories = {}
#
#     # 将文件按类别分组
#     for file in files:
#         category = file.split('_')[0]
#         if category not in categories:
#             categories[category] = []
#         categories[category].append(os.path.join(step_folder, file))
#
#     category_colors = {}
#     for category in categories.keys():
#         # 生成随机RGB颜色，避免太浅的颜色
#         r = random.uniform(0.2, 0.8)
#         g = random.uniform(0.2, 0.8)
#         b = random.uniform(0.2, 0.8)
#         category_colors[category] = Quantity_Color(r, g, b, Quantity_TOC_RGB)
#
#     # 处理每个类别
#     for category, file_list in sorted(categories.items()):
#         if len(file_list) != 2:
#             print(f"Warning: Category {category} does not have exactly 2 files")
#             continue
#
#         col_idx = list(sorted(categories.keys())).index(category)
#         color = category_colors[category]
#
#         for row_idx, file in enumerate(file_list):
#             print(f"Loading: {file}")
#
#             step_reader = STEPControl_Reader()
#             status = step_reader.ReadFile(file)
#
#             if status == IFSelect_RetDone:
#                 step_reader.TransferRoots()
#                 shape = step_reader.OneShape()
#
#                 current_offset_x = col_idx * translation_step_x
#                 current_offset_z = row_idx * translation_step_z
#
#                 scale_trsf = gp_Trsf()
#                 scale_trsf.SetScale(gp_Pnt(0, 0, 0), scale_factor_furniture[row_idx][col_idx])
#                 scaled_shape = BRepBuilderAPI_Transform(shape, scale_trsf).Shape()
#
#                 rotation_trsfy = gp_Trsf()
#                 rotation_axisy = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0))
#                 rotation_trsfy.SetRotation(rotation_axisy, math.radians(angle_y_furniture[row_idx][col_idx]))
#                 rotated_shapey = BRepBuilderAPI_Transform(scaled_shape, rotation_trsfy).Shape()
#
#                 rotation_trsfz = gp_Trsf()
#                 rotation_axisz = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
#                 rotation_trsfz.SetRotation(rotation_axisz, math.radians(angle_z_furniture[row_idx][col_idx]))
#                 rotated_shapez = BRepBuilderAPI_Transform(rotated_shapey, rotation_trsfz).Shape()
#
#                 rotation_trsfx = gp_Trsf()
#                 rotation_axisx = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0))
#                 rotation_trsfx.SetRotation(rotation_axisx, math.radians(angle_x_furniture[row_idx][col_idx]))
#
#                 rotated_shapex = BRepBuilderAPI_Transform(rotated_shapez, rotation_trsfx).Shape()
#                 translation = gp_Trsf()
#                 translation.SetTranslation(gp_Vec(current_offset_x, 0, current_offset_z))
#                 transformed_shape = BRepBuilderAPI_Transform(rotated_shapex, translation).Shape()
#                 my_renderer.myDisplayShape(transformed_shape, color=(color.Red(), color.Green(), color.Blue()), export_edges=True, line_width=3.0)
#
#             else:
#                 print(f"Error: Failed to read the STEP file: {file}")
#     my_renderer.render()


scale_factor_furniture = [[1.5, 1.2, 1.5, 1.0, 1.0, 1.0, 1.1, 0.9, 1.3, 0.9],
                          [1.4, 1.0, 1.5, 1.2, 1.0, 1.0, 1.2, 1.3, 1.3, 1.2]]

angle_x_furniture = [[20, 20, 20, 20, 20, 20, 20, 5, 20, 10],
                     [20, 20, 25, 20, 20, 20, 20, 10, 20, 20]]

angle_y_furniture = [[0, 0, 0, 0, 0, 0, 0, -3, 0, -3],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, -8]]

angle_z_furniture = [[65, -30, -35, -20, -60, -60, -60, 0, -60, 30],
                     [60, -20, -25, -30, -60, -60, -60, -60, -60, 10]]

# vis_furniture("Pics/Srcs/vis_furniture", scale_factor_furniture, angle_x_furniture, angle_y_furniture, angle_z_furniture)
