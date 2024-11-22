import os
import shutil
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.AIS import AIS_Shape
from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Pnt, gp_Ax1, gp_Dir
from OCC.Display.SimpleGui import init_display
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB, Quantity_NOC_WHITE, Quantity_NOC_BLACK
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.Graphic3d import Graphic3d_NOM_PLASTIC, Graphic3d_MaterialAspect, Graphic3d_NOM_NEON_PHC
from utils import load_data_with_prefix
from comparison.metrics import check_brep_validity, check_mesh_validity
from tqdm import tqdm
from OCC.Core.Graphic3d import Graphic3d_NOM_NEON_PHC
from PIL import Image
import numpy as np
import math
import random


def rgb_color(r, g, b):
    return Quantity_Color(r, g, b, Quantity_TOC_RGB)


def get_face_count(step_file):
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_file)

    if status == 1:
        step_reader.TransferRoot()
        shape = step_reader.Shape()

        face_count = 0
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face_count += 1
            explorer.Next()

        return face_count
    else:
        print("Failed to read STEP file.")
        return None


def filter_step(input_path, save_path, min_face=10, max_face=20):
    files = load_data_with_prefix(input_path, '.step')
    for file in tqdm(files):
        if not check_brep_validity(file):
            continue
        if not check_mesh_validity(file[:-5] + '.stl'):
            continue
        face_count = get_face_count(step_file=file)
        if face_count is not None and min_face <= face_count <= max_face:
            shutil.copy(file, os.path.join(save_path, os.path.basename(file)))
            shutil.copy(file[:-5] + '.stl', os.path.join(save_path, os.path.basename(file)[:-5] + '.stl'))


def capture_png(step_folder, color, save_name):
    display, start_display, add_menu, add_function_to_menu = init_display(size=(4000, 3000), display_triedron=True)
    display.View.SetBgGradientColors(
        Quantity_Color(Quantity_NOC_WHITE),
        Quantity_Color(Quantity_NOC_WHITE),
        2,
        True
    )

    display.View.SetProj(0, -1, 1)  # 设置投影方向
    display.View.SetUp(0, 1, 0)    # 设置向上方向

    translation_step_x = 2.2
    translation_step_y = 3
    shapes_per_row = 5

    current_offset_x = 0
    current_offset_y = 0
    shape_count = 0

    files = load_data_with_prefix(step_folder, '.step')
    files[3], files[16] = files[16], files[3]
    files[1], files[2] = files[2], files[1]

    for file in files:
        print(f"Loading: {file}")
        step_reader = STEPControl_Reader()
        status = step_reader.ReadFile(file)
        if status == IFSelect_RetDone:
            step_reader.TransferRoots()
            shape = step_reader.OneShape()
            ais_shp = AIS_Shape(shape)

            ais_shp.SetDisplayMode(1)

            translation = gp_Trsf()
            translation.SetTranslation(gp_Vec(current_offset_x, current_offset_y, 0))
            transformed_shape = BRepBuilderAPI_Transform(shape, translation).Shape()

            display.DisplayShape(transformed_shape, color=color, update=False)

            shape_count += 1

            if shape_count % shapes_per_row == 0:
                current_offset_x = 0
                current_offset_y += translation_step_y
            else:
                current_offset_x += translation_step_x
        else:
            print(f"Error: Failed to read the STEP file: {file}")

    display.FitAll()
    display.View.Dump(save_name+'.png')

    img = Image.open(save_name+'.png').convert("RGBA")
    datas = img.getdata()
    newData = [(item[0], item[1], item[2], 0) if item[0] == 255 and item[1] == 255 and item[2] == 255 else item for item
               in datas]
    img.putdata(newData)
    img.save(save_name+'.png', "PNG")
    print("Image saved with transparent background as 'capture_png.png'")

    start_display()


def capture_png_deepcad(step_folder, color, save_name):
    display, start_display, add_menu, add_function_to_menu = init_display(size=(4000, 3000), display_triedron=False)
    display.View.SetBgGradientColors(
        Quantity_Color(Quantity_NOC_WHITE),
        Quantity_Color(Quantity_NOC_WHITE),
        2,
        True
    )

    display.View.SetProj(0, -1.5, 1)
    display.View.SetUp(0, 1, 0)

    # material = Graphic3d_MaterialAspect(Graphic3d_NOM_NEON_PHC)
    # material.SetShininess(0.3)

    current_offset_x = [0, 1.5, 3.2, 4, 5.6,
                        0, 1.5, 3.1, 4, 5.3,
                        0, 1.6, 2.8, 4.3, 5.4,
                        0, 1.5, 2.8, 4.3, 5.5]
    current_offset_y = [0, 0.1, -0.6, 0.4, 0.5,
                        2.2, 2, 2.8, 2, 2.1,
                        4.8, 3.9, 4.5, 4.4, 4.6,
                        6.2, 6, 6.2, 6.4, 6]
    scale_factor = [1, 0.6, 1.5, 0.5, 0.8,
                    0.8, 0.7, 0.8, 0.7, 0.5,
                    0.7, 0.8, 0.5, 0.7, 0.6,
                    0.7, 0.7, 0.6, 0.7, 0.8]
    angle = [-30, 40, -30, -30, -30,
             -30, -30, -30, -30, -30,
             50, -30, -30, -30, -30,
             -30, -30, -30, -30, -30]

    files = load_data_with_prefix(step_folder, '.step')
    files[3], files[16] = files[16], files[3]
    files[1], files[2] = files[2], files[1]

    for i, file in enumerate(files):
        print(f"Loading: {file}")
        step_reader = STEPControl_Reader()
        status = step_reader.ReadFile(file)
        if status == IFSelect_RetDone:
            step_reader.TransferRoots()
            shape = step_reader.OneShape()
            ais_shp = AIS_Shape(shape)

            ais_shp.SetDisplayMode(1)

            scale_trsf = gp_Trsf()
            scale_trsf.SetScale(gp_Pnt(0, 0, 0), scale_factor[i])
            scaled_shape = BRepBuilderAPI_Transform(shape, scale_trsf).Shape()
            rotation_trsf = gp_Trsf()
            rotation_axis = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
            rotation_trsf.SetRotation(rotation_axis, math.radians(angle[i]))
            rotated_shape = BRepBuilderAPI_Transform(scaled_shape, rotation_trsf).Shape()
            translation = gp_Trsf()
            translation.SetTranslation(gp_Vec(current_offset_x[i], current_offset_y[i], 0))
            transformed_shape = BRepBuilderAPI_Transform(rotated_shape, translation).Shape()

            ais_shape = display.DisplayShape(
                transformed_shape,
                update=False,
                transparency=0.0,
            )[0]

            # 设置颜色和材质
            display.Context.SetColor(ais_shape, color, False)
            # display.Context.SetMaterial(ais_shape, material, False)

        else:
            print(f"Error: Failed to read the STEP file: {file}")

    display.FitAll()
    display.View.Dump(save_name+'.png')

    img = Image.open(save_name+'.png').convert("RGBA")
    datas = img.getdata()
    newData = [(item[0], item[1], item[2], 0) if item[0] == 255 and item[1] == 255 and item[2] == 255 else item for item
               in datas]
    img.putdata(newData)
    img.save(save_name+'.png', "PNG")
    print("Image saved with transparent background as 'capture_png.png'")

    start_display()


def capture_png_brepgen(step_folder, color, save_name):
    display, start_display, add_menu, add_function_to_menu = init_display(size=(4000, 3000), display_triedron=False)
    display.View.SetBgGradientColors(
        Quantity_Color(Quantity_NOC_WHITE),
        Quantity_Color(Quantity_NOC_WHITE),
        2,
        True
    )

    display.View.SetProj(0, -1.5, 1)
    display.View.SetUp(0, 1, 0)

    material = Graphic3d_MaterialAspect(Graphic3d_NOM_NEON_PHC)
    material.SetShininess(0.3)

    current_offset_x = [0, 2, 3.6, 5.4, 7,
                        0, 2, 3.6, 5.4, 7,
                        0.2, 2., 3.5, 5.3, 6.9,
                        0.2, 2.1, 3.7, 5.3, 6.8]
    current_offset_y = [0, 0, -0.2, 0, 0,
                        2.4, 2.4, 2.4, 2, 2.2,
                        4.3, 4.3, 4.3, 4.4, 4.5,
                        6.5, 6.2, 6.6, 6.5, 6.5]
    scale_factor = [0.8, 0.8, 0.7, 0.7, 0.7,
                    0.55, 0.6, 0.6, 0.7, 0.6,
                    0.62, 0.45, 0.52, 0.7, 0.65,
                    0.6, 0.5, 0.5, 0.55, 0.55]
    angle = [-30, -30, -30, -30, -30,
             -30, -30, -30, -30, -30,
             -30, -30, -30, -30, -30,
             -30, -30, -30, -30, -30]

    files = load_data_with_prefix(step_folder, '.step')
    files[3], files[16] = files[16], files[3]
    files[1], files[2] = files[2], files[1]

    for i, file in enumerate(files):
        print(f"Loading: {file}")
        step_reader = STEPControl_Reader()
        status = step_reader.ReadFile(file)
        if status == IFSelect_RetDone:
            step_reader.TransferRoots()
            shape = step_reader.OneShape()
            ais_shp = AIS_Shape(shape)

            ais_shp.SetDisplayMode(1)

            scale_trsf = gp_Trsf()
            scale_trsf.SetScale(gp_Pnt(0, 0, 0), scale_factor[i])
            scaled_shape = BRepBuilderAPI_Transform(shape, scale_trsf).Shape()
            rotation_trsf = gp_Trsf()
            rotation_axis = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
            rotation_trsf.SetRotation(rotation_axis, math.radians(angle[i]))
            rotated_shape = BRepBuilderAPI_Transform(scaled_shape, rotation_trsf).Shape()
            translation = gp_Trsf()
            translation.SetTranslation(gp_Vec(current_offset_x[i], current_offset_y[i], 0))
            transformed_shape = BRepBuilderAPI_Transform(rotated_shape, translation).Shape()

            ais_shape = display.DisplayShape(
                transformed_shape,
                update=False,
                transparency=0.0,
            )[0]

            # 设置颜色和材质
            # display.Context.SetColor(ais_shape, color, False)
            display.Context.SetMaterial(ais_shape, material, False)

        else:
            print(f"Error: Failed to read the STEP file: {file}")

    display.FitAll()
    display.View.Dump(save_name+'.png')

    # img = Image.open(save_name+'.png').convert("RGBA")
    # datas = img.getdata()
    # newData = [(item[0], item[1], item[2], 0) if item[0] == 255 and item[1] == 255 and item[2] == 255 else item for item
    #            in datas]
    # img.putdata(newData)
    # img.save(save_name+'.png', "PNG")
    # print("Image saved with transparent background as 'capture_png.png'")

    start_display()


def capture_png_ours(step_folder, color, save_name):
    display, start_display, add_menu, add_function_to_menu = init_display(size=(4000, 3000), display_triedron=False)
    display.View.SetBgGradientColors(
        Quantity_Color(Quantity_NOC_WHITE),
        Quantity_Color(Quantity_NOC_WHITE),
        2,
        True
    )

    display.View.SetProj(0, -1.5, 1)
    display.View.SetUp(0, 1, 0)

    # material = Graphic3d_MaterialAspect(Graphic3d_NOM_PLASTIC)
    # material = Graphic3d_MaterialAspect(Graphic3d_NOM_NEON_PHC)
    # material.SetShininess(0.3)

    current_offset_x = [0, 2, 3.6, 5.4, 7.2,
                        0, 2, 3.6, 5.4, 7,
                        0, 2., 3.5, 5.1, 6.9,
                        0, 1.8, 3.5, 5.1, 6.8]
    current_offset_y = [0, 0, -0.2, 0, 0,
                        2.4, 2.6, 2.4, 2.3, 2.4,
                        4.5, 4.7, 4.7, 4.4, 4.6,
                        6.6, 7.1, 6.9, 6.7, 6.7]
    scale_factor = [0.8, 0.8, 0.6, 0.8, 0.6,
                    0.8, 0.6, 0.7, 0.65, 0.7,
                    0.7, 0.8, 0.52, 0.65, 0.65,
                    0.6, 0.45, 0.5, 0.55, 0.55]
    angle = [-30, -30, -120, -30, -30,
             -30, -30, -120, -30, -30,
             -30, -30, -30, -30, -30,
             -30, -30, -30, -30, -30]

    files = load_data_with_prefix(step_folder, '.step')
    files[3], files[16] = files[16], files[3]
    files[1], files[2] = files[2], files[1]

    for i, file in enumerate(files):
        print(f"Loading: {file}")
        step_reader = STEPControl_Reader()
        status = step_reader.ReadFile(file)
        if status == IFSelect_RetDone:
            step_reader.TransferRoots()
            shape = step_reader.OneShape()
            ais_shp = AIS_Shape(shape)

            ais_shp.SetDisplayMode(1)

            scale_trsf = gp_Trsf()
            scale_trsf.SetScale(gp_Pnt(0, 0, 0), scale_factor[i])
            scaled_shape = BRepBuilderAPI_Transform(shape, scale_trsf).Shape()
            rotation_trsf = gp_Trsf()
            rotation_axis = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
            rotation_trsf.SetRotation(rotation_axis, math.radians(angle[i]))
            rotated_shape = BRepBuilderAPI_Transform(scaled_shape, rotation_trsf).Shape()
            translation = gp_Trsf()
            translation.SetTranslation(gp_Vec(current_offset_x[i], current_offset_y[i], 0))
            transformed_shape = BRepBuilderAPI_Transform(rotated_shape, translation).Shape()

            ais_shape = display.DisplayShape(
                transformed_shape,
                update=False,
                transparency=0.0,
            )[0]

            # 设置颜色和材质
            display.Context.SetColor(ais_shape, color, False)
            # display.Context.SetMaterial(ais_shape, material, False)

        else:
            print(f"Error: Failed to read the STEP file: {file}")

    display.FitAll()
    display.View.Dump(save_name+'.png')

    img = Image.open(save_name+'.png').convert("RGBA")
    datas = img.getdata()
    newData = [(item[0], item[1], item[2], 0) if item[0] == 255 and item[1] == 255 and item[2] == 255 else item for item
               in datas]
    img.putdata(newData)
    img.save(save_name+'.png', "PNG")
    print("Image saved with transparent background as 'capture_png.png'")

    start_display()


def vis_furniture(step_folder, save_path='furniture.png'):
    display, start_display, add_menu, add_function_to_menu = init_display(size=(4000, 3000), display_triedron=False)
    # 设置透明背景
    display.View.SetBgGradientColors(
        Quantity_Color(Quantity_NOC_WHITE),
        Quantity_Color(Quantity_NOC_WHITE),
        2,
        True
    )
    display.set_bg_gradient_color([255, 255, 255], [255, 255, 255])

    # 设置正视图
    display.View.SetProj(0, -1.5, 1)
    display.View.SetUp(0, 1, 0)

    # 设置布局参数
    translation_step_x = 3  # 列间距
    translation_step_y = 4.8  # 行间距

    # 获取所有step文件并按类别分组
    files = [f for f in os.listdir(step_folder) if f.endswith('.step')]
    categories = {}

    material = Graphic3d_MaterialAspect(Graphic3d_NOM_PLASTIC)
    material.SetShininess(0.3)

    # 将文件按类别分组
    for file in files:
        category = file.split('_')[0]
        if category not in categories:
            categories[category] = []
        categories[category].append(os.path.join(step_folder, file))

    # 为每个类别生成随机颜色
    category_colors = {}
    for category in categories.keys():
        # 生成随机RGB颜色，避免太浅的颜色
        r = random.uniform(0.2, 0.8)
        g = random.uniform(0.2, 0.8)
        b = random.uniform(0.2, 0.8)
        category_colors[category] = Quantity_Color(r, g, b, Quantity_TOC_RGB)

    # 处理每个类别
    for category, file_list in sorted(categories.items()):
        if len(file_list) != 2:
            print(f"Warning: Category {category} does not have exactly 2 files")
            continue

        col_idx = list(sorted(categories.keys())).index(category)
        color = category_colors[category]

        for row_idx, file in enumerate(file_list):
            print(f"Loading: {file}")

            step_reader = STEPControl_Reader()
            status = step_reader.ReadFile(file)

            if status == IFSelect_RetDone:
                step_reader.TransferRoots()
                shape = step_reader.OneShape()

                current_offset_x = col_idx * translation_step_x
                current_offset_y = row_idx * translation_step_y

                scale_trsf = gp_Trsf()
                scale_trsf.SetScale(gp_Pnt(0, 0, 0), 1)
                scaled_shape = BRepBuilderAPI_Transform(shape, scale_trsf).Shape()
                rotation_trsf = gp_Trsf()
                rotation_axis = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
                rotation_trsf.SetRotation(rotation_axis, math.radians(-30))
                rotated_shape = BRepBuilderAPI_Transform(scaled_shape, rotation_trsf).Shape()
                translation = gp_Trsf()
                translation.SetTranslation(gp_Vec(current_offset_x, current_offset_y, 0))
                transformed_shape = BRepBuilderAPI_Transform(rotated_shape, translation).Shape()

                ais_shape = display.DisplayShape(
                    transformed_shape,
                    update=False,
                    transparency=0.0,
                )[0]

                # 设置颜色和材质
                display.Context.SetColor(ais_shape, color, False)
                display.Context.SetMaterial(ais_shape, material, False)
            else:
                print(f"Error: Failed to read the STEP file: {file}")

    display.FitAll()

    if save_path:
        display.View.Update()
        display.View.Dump(save_path, True)

    display.FitAll()
    start_display()


def vis_bug(step_folder):

    display, start_display, add_menu, add_function_to_menu = init_display(size=(4000, 3000))
    display.View.SetBgGradientColors(
        Quantity_Color(Quantity_NOC_WHITE),
        Quantity_Color(Quantity_NOC_WHITE),
        2,
        True
    )
    display.set_bg_gradient_color([255, 255, 255], [255, 255, 255])
    material = Graphic3d_MaterialAspect(Graphic3d_NOM_PLASTIC)
    material.SetShininess(0.3)

    step_files = load_data_with_prefix(step_folder, '.step')

    offset = 3

    for i, step_file in enumerate(step_files):

        step_reader = STEPControl_Reader()
        step_reader.ReadFile(step_file)
        step_reader.TransferRoots()
        shape = step_reader.OneShape()

        transform = gp_Trsf()
        transform.SetTranslation(gp_Vec(i * offset, 0, 0))
        transformed_shape = BRepBuilderAPI_Transform(shape, transform).Shape()
        ais_shape = display.DisplayShape(
            transformed_shape,
            update=False,
            transparency=0.0,
        )[0]
        display.Context.SetColor(ais_shape, Quantity_Color(0.6, 0.5, 0.3, Quantity_TOC_RGB), False)
        display.Context.SetMaterial(ais_shape, material, False)

    start_display()


def crop_png(path):

    png_files = [f for f in os.listdir(path) if f.endswith('.png')]

    min_x, min_y, max_x, max_y = None, None, None, None

    crop_folder = os.path.join(path, 'crop')
    os.makedirs(crop_folder, exist_ok=True)

    for file in png_files:
        image_path = os.path.join(path, file)
        image = Image.open(image_path).convert("RGBA")
        image_np = np.array(image)

        alpha_channel = image_np[:, :, 3]

        non_zero_rows = np.any(alpha_channel != 0, axis=1)
        non_zero_cols = np.any(alpha_channel != 0, axis=0)

        y_min, y_max = np.where(non_zero_rows)[0][[0, -1]]
        x_min, x_max = np.where(non_zero_cols)[0][[0, -1]]

        min_x = x_min if min_x is None else min(min_x, x_min)
        min_y = y_min if min_y is None else min(min_y, y_min)
        max_x = x_max if max_x is None else max(max_x, x_max)
        max_y = y_max if max_y is None else max(max_y, y_max)

    for file in png_files:
        image_path = os.path.join(path, file)
        image = Image.open(image_path).convert("RGBA")

        cropped_image = image.crop((min_x, min_y, max_x + 1, max_y + 1))

        cropped_image.save(os.path.join(crop_folder, file))


def new_visualize_step(step_folder, num=200):
    display, start_display, add_menu, add_function_to_menu = init_display(size=(1600, 900), display_triedron=False)

    display.View.SetBgGradientColors(Quantity_Color(Quantity_NOC_WHITE), Quantity_Color(Quantity_NOC_WHITE), 2, True)
    display.View.Update()
    display.Repaint()

    translation_step_x = 3
    translation_step_y = 3
    shapes_per_row = 20

    step_files = load_data_with_prefix(step_folder, '.step')
    total_files = len(step_files)

    for batch_start in range(0, total_files, num):
        current_offset_x = 0
        current_offset_y = 0
        shape_count = 0

        display.EraseAll()

        batch_end = min(batch_start + num, total_files)
        for step_file in step_files[batch_start:batch_end]:
            print(f"Loading: {step_file}")
            step_reader = STEPControl_Reader()
            status = step_reader.ReadFile(step_file)

            if status == IFSelect_RetDone:
                try:
                    step_reader.TransferRoots()
                    shape = step_reader.OneShape()
                    ais_shp = AIS_Shape(shape)

                    color = rgb_color(0., 0., 0.)
                    ais_shp.SetColor(color)

                    translation = gp_Trsf()
                    translation.SetTranslation(gp_Vec(current_offset_x, current_offset_y, 0))
                    transformed_shape = BRepBuilderAPI_Transform(shape, translation).Shape()
                    display.DisplayShape(transformed_shape, color=color, material=Graphic3d_NOM_NEON_PHC, update=False)

                    # 修正DisplayMessage的使用方式
                    label_position = gp_Pnt(current_offset_x + 0.5, current_offset_y + 0.5, 0)
                    display.DisplayMessage(label_position, os.path.basename(step_file), message_color=(0, 0, 0))  # 简化调用

                    shape_count += 1
                    rows = shape_count // shapes_per_row
                    if shape_count % shapes_per_row == 0:
                        current_offset_x = -rows * translation_step_x
                        current_offset_y = rows * translation_step_y
                    else:
                        current_offset_x += translation_step_x
                        current_offset_y += translation_step_y
                except Exception as e:
                    continue
            else:
                print(f"Error: Failed to read the STEP file: {step_file}")

        display.FitAll()
        start_display()

        if batch_end < total_files:
            user_input = input(f"\nDisplayed {batch_end}/{total_files} files. Press Enter to continue, 'q' to quit: ")
            if user_input.lower() == 'q':
                break


new_visualize_step("/home/jing/PythonProjects/BrepGDM/Supp/furniture/")
# filter_step(input_path='/home/jing/PythonProjects/BrepGDM/samples/deepcad_results/vis_brepgen',
#             save_path='/home/jing/PythonProjects/BrepGDM/samples/deepcad_results/11', min_face=10, max_face=20)

rgb_colors = {'ours': rgb_color(50 / 255.0, 160 / 255.0, 190 / 255.0),
              'brepgen': rgb_color(190 / 255.0, 60 / 255.0, 50 / 255.0),
              'deepcad': rgb_color(50 / 255.0, 150 / 255.0, 50 / 255.0)}
name = 'ours'
# vis_furniture("/home/jing/PythonProjects/BrepGDM/Pics/fyhpics/vis_furniture/")
# capture_png_ours('/home/jing/PythonProjects/BrepGDM/samples/deepcad_results/vis_' + name,
#                  color=rgb_colors[name], save_name='deepcad_'+name)

# name = 'brepgen'
# capture_png_brepgen('/home/jing/PythonProjects/BrepGDM/samples/deepcad_results/vis_' + name,
#                     color=rgb_colors[name], save_name='deepcad_'+name)
#
# name = 'deepcad'
# capture_png_deepcad('/home/jing/PythonProjects/BrepGDM/samples/deepcad_results/vis_' + name,
#                     color=rgb_colors[name], save_name='deepcad_'+name)

# vis_furniture('samples/furniture_results/vis_ours', save_path='output.png')

# vis_bug('samples/deepcad_results/vis_bug')

# crop_png(path='Pics/deepcad')
