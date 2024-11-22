import copy
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import topods_Face, topods_Edge, topods_Vertex
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace
from OCC.Core.gp import gp_Pnt, gp_Ax1, gp_Dir, gp_Trsf
import pickle
from render.myx3dom_render import myX3DomRenderer


def read_step(filename):
    step_reader = STEPControl_Reader()
    step_reader.ReadFile(filename)
    step_reader.TransferRoot()
    shape = step_reader.Shape()
    return shape


def save_views(shape, output_dir):

    my_renderer = myX3DomRenderer(display_axes_plane=False)

    # # 1. 显示face bounding boxes
    # exp = TopExp_Explorer(shape, TopAbs_FACE)
    # while exp.More():
    #     face = topods_Face(exp.Current())
    #     bbox = Bnd_Box()
    #     brepbndlib_Add(face, bbox)
    #     xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    #     box_shape = BRepPrimAPI_MakeBox(gp_Pnt(xmin, ymin, zmin),
    #                                     gp_Pnt(xmax, ymax, zmax)).Shape()
    #     edge_exp = TopExp_Explorer(box_shape, TopAbs_EDGE)
    #     line_color = (random.uniform(0.2, 0.8), random.uniform(0.2, 0.8), random.uniform(0.2, 0.8))
    #
    #     while edge_exp.More():
    #         edge = topods_Edge(edge_exp.Current())
    #         my_renderer.myDisplayShape(edge, line_color=line_color, line_width=15.0)
    #         edge_exp.Next()
    #
    #     exp.Next()
    # my_renderer.render()

    # # 2. 只显示顶点
    # exp = TopExp_Explorer(shape, TopAbs_VERTEX)
    # color = (1, 1, 0)
    # while exp.More():
    #     vertex = topods_Vertex(exp.Current())
    #     pnt = BRep_Tool.Pnt(vertex)
    #     x, y, z = pnt.X(), pnt.Y(), pnt.Z()
    #     radius = 0.05
    #     sphere = BRepPrimAPI_MakeSphere(gp_Pnt(x, y, z), radius).Shape()
    #     my_renderer.myDisplayShape(sphere, color=color, export_edges=False)
    #     exp.Next()
    # my_renderer.render()

    # # 3. 显示顶点和边
    # # 先显示边
    # line_colors = []
    # with open('line_colors.txt', 'r') as file:
    #     for line in file:
    #         color = tuple(map(float, line.split()))
    #         line_colors.append(color)
    # exp_edge = TopExp_Explorer(shape, TopAbs_EDGE)
    # color_index = 0
    # while exp_edge.More():
    #     # line_color = (random.uniform(0.2, 0.8), random.uniform(0.2, 0.8), random.uniform(0.2, 0.8))
    #     line_color = line_colors[color_index]
    #     edge = topods_Edge(exp_edge.Current())
    #     my_renderer.myDisplayShape(edge, line_color=line_color, line_width=15.0)
    #     exp_edge.Next()
    #     color_index += 1
    # # 再显示顶点
    # exp = TopExp_Explorer(shape, TopAbs_VERTEX)
    # color = (1, 1, 0)
    # while exp.More():
    #     vertex = topods_Vertex(exp.Current())
    #     pnt = BRep_Tool.Pnt(vertex)
    #     x, y, z = pnt.X(), pnt.Y(), pnt.Z()
    #     radius = 0.05
    #     sphere = BRepPrimAPI_MakeSphere(gp_Pnt(x, y, z), radius).Shape()
    #     my_renderer.myDisplayShape(sphere, color=color, export_edges=False)
    #     exp.Next()
    # my_renderer.render()


    # 4. 显示完整几何
    # 显示面
    # exp_face = TopExp_Explorer(shape, TopAbs_FACE)
    # face_index = 0
    # colors = [
    #     (0.1, 0.2, 0.9),  # 深蓝色
    #     (0.9, 0.3, 0.1),  # 鲜艳的红色
    #     (0.2, 0.8, 0.2),  # 明亮的绿色
    #     (0.7, 0.2, 0.7),  # 紫色
    #     (0.6, 0.3, 0.9),  # 紫红色
    #     (0.8, 0.3, 0.5),  # 桃红色
    #     (0.1, 0.7, 0.6),  # 青色
    #     (50 / 255.0, 150 / 255.0, 50 / 255.0),  # 苔藓绿色
    #     (0.3, 0.2, 0.8),  # 靛蓝色
    #     (0.9, 0.5, 0.6),  # 淡粉色
    #     (0.5, 0.1, 0.6),  # 深紫色
    #     (0.6, 0.6, 0.2),  # 油橄榄色
    #     (0.2, 0.5, 0.9)  # 天空蓝
    # ]
    #
    # while exp_face.More():
    #     face = topods_Face(exp_face.Current())
    #     my_renderer.myDisplayShape(face, color=colors[face_index], export_edges=False, transparency=0)
    #     face_index += 1
    #     exp_face.Next()
    #
    # # 先显示边
    # line_colors = []
    # with open('render/line_colors.txt', 'r') as file:
    #     for line in file:
    #         color = tuple(map(float, line.split()))
    #         line_colors.append(color)
    # exp_edge = TopExp_Explorer(shape, TopAbs_EDGE)
    # color_index = 0
    # while exp_edge.More():
    #     line_color = line_colors[color_index % len(line_colors)]
    #     edge = topods_Edge(exp_edge.Current())
    #     my_renderer.myDisplayShape(edge, line_color=line_color, line_width=15.0)
    #     color_index += 1
    #     exp_edge.Next()
    # # 再显示顶点
    # exp = TopExp_Explorer(shape, TopAbs_VERTEX)
    # color = (1, 1, 0)
    # while exp.More():
    #     vertex = topods_Vertex(exp.Current())
    #     pnt = BRep_Tool.Pnt(vertex)
    #     x, y, z = pnt.X(), pnt.Y(), pnt.Z()
    #     radius = 0.05
    #     sphere = BRepPrimAPI_MakeSphere(gp_Pnt(x, y, z), radius).Shape()
    #     my_renderer.myDisplayShape(sphere, color=color, export_edges=False)
    #     exp.Next()
    # my_renderer.render()

    # # 5. brep-show
    my_renderer.myDisplayShape(shape, color=(50 / 255.0, 160 / 255.0, 190 / 255.0), export_edges=True, line_width=15, transparency=0.)

    my_renderer.render()


def draw_special_ctrs(shape, sphere_edge, line_edge, sphere_face, line_face):    # TopoDS_Solid

    my_renderer = myX3DomRenderer(display_axes_plane=False)

    # Display faces
    exp_face = TopExp_Explorer(shape, TopAbs_FACE)
    face_index = 0
    colors = [
        (0.1, 0.2, 0.9),  # 深蓝色
        (0.9, 0.3, 0.1),  # 鲜艳的红色
        (0.2, 0.8, 0.2),  # 明亮的绿色
        (0.7, 0.2, 0.7),  # 紫色
        (0.6, 0.3, 0.9),  # 紫红色
        (0.8, 0.3, 0.5),  # 桃红色
        (0.1, 0.7, 0.6),  # 青色
        (50 / 255.0, 150 / 255.0, 50 / 255.0),  # 苔藓绿色
        (0.3, 0.2, 0.8),  # 靛蓝色
        (0.9, 0.5, 0.6),  # 淡粉色
        (0.5, 0.1, 0.6),  # 深紫色
        (0.6, 0.6, 0.2),  # 油橄榄色
        (0.2, 0.5, 0.9)  # 天空蓝
    ]
    colors = [(0.7, 0.7, 0.7) for _ in range(len(colors))]
    colors[8] = (0., 1, 0.)
    while exp_face.More():
        face = topods_Face(exp_face.Current())
        if face_index != 8:
            my_renderer.myDisplayShape(face, color=colors[face_index], export_edges=True, transparency=0.7, line_width=15.0)
        else:
            my_renderer.myDisplayShape(face, color=colors[face_index], export_edges=True, transparency=0., line_width=15.0)
        face_index += 1
        exp_face.Next()
    # Display face control points
    color = (1, 0, 1)
    for sphere in sphere_face:
        my_renderer.myDisplayShape(sphere, color=color, export_edges=False)
    for line in line_face:
        my_renderer.myDisplayShape(line, line_color=color, export_edges=False, line_width=5.0)

    # Display edges
    # line_colors = []
    # with open('render/line_colors.txt', 'r') as file:
    #     for line in file:
    #         color = tuple(map(float, line.split()))
    #         line_colors.append(color)
    # exp_edge = TopExp_Explorer(shape, TopAbs_EDGE)
    # color_index = 0
    # copy_color = copy.deepcopy(line_colors)
    # line_colors = [(0., 0., 0.) for _ in range(len(line_colors))]
    # line_colors[26] = (0., 1., 0.)
    # line_colors[32] = (0., 1., 0.)
    # while exp_edge.More():
    #     line_color = line_colors[color_index % len(line_colors)]
    #     edge = topods_Edge(exp_edge.Current())     # TopoDS_Edge
    #     my_renderer.myDisplayShape(edge, line_color=line_color, export_edges=False, line_width=15.0)
    #     color_index += 1
    #     exp_edge.Next()
    # # Display edge control points
    # color = (1, 0, 1)
    # for sphere in sphere_edge:
    #     my_renderer.myDisplayShape(sphere, color=color, export_edges=False)
    # for line in line_edge:
    #     my_renderer.myDisplayShape(line, line_color=color, export_edges=False, line_width=5.0)

    my_renderer.render()


def diff_ctrs():

    from diffusers import DDPMScheduler
    import torch
    import numpy as np

    with open('/home/jing/PythonProjects/BrepGDM/Pics/Srcs/vis_overview/table_SP6URjYZo8.pkl', 'rb') as f:
        data = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_0 = torch.from_numpy(data['face_ctrs']).to(device)       # nf*16*3
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start=0.0001,
        beta_end=0.02,
        clip_sample=False,
    )

    np.save('Pics/Srcs/vis_overview/x_0.npy', x_0.cpu().numpy())

    t = torch.tensor([50], dtype=torch.long, device=device)  # b
    noise = torch.randn(x_0.shape).to(device)
    x_500 = noise_scheduler.add_noise(x_0, noise, t)    # nf*16*3
    np.save('Pics/Srcs/vis_overview/x_50.npy', x_500.cpu().numpy())
    t = torch.tensor([100], dtype=torch.long, device=device)  # b
    noise = torch.randn(x_0.shape).to(device)
    x_1000 = noise_scheduler.add_noise(x_0, noise, t)    # nf*16*3
    np.save('Pics/Srcs/vis_overview/x_100.npy', x_1000.cpu().numpy())


def draw_diff_ctrs():
    import numpy as np
    from inference.brepBuild import create_bspline_surface
    ctrs = np.load('Pics/Srcs/vis_overview/x_0.npy')    # nf*16*3
    colors = [
        (0.1, 0.2, 0.9),  # 深蓝色
        (0.9, 0.3, 0.1),  # 鲜艳的红色
        (0.2, 0.8, 0.2),  # 明亮的绿色
        (0.7, 0.2, 0.7),  # 紫色
        (0.6, 0.3, 0.9),  # 紫红色
        (0.8, 0.3, 0.5),  # 桃红色
        (0.1, 0.7, 0.6),  # 青色
        (50 / 255.0, 150 / 255.0, 50 / 255.0),  # 苔藓绿色
        (0.3, 0.2, 0.8),  # 靛蓝色
        (0.9, 0.5, 0.6),  # 淡粉色
        (0.5, 0.1, 0.6),  # 深紫色
        (0.6, 0.6, 0.2),  # 油橄榄色
        (0.2, 0.5, 0.9)  # 天空蓝
    ]

    # create face control lines
    axisz = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    anglez = -30 * (3.14159 / 180)
    rotationz = gp_Trsf()
    rotationz.SetRotation(axisz, anglez)
    axisx = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0))
    anglex = 30 * (3.14159 / 180)
    rotationx = gp_Trsf()
    rotationx.SetRotation(axisx, anglex)
    radius = 0.06
    my_renderer = myX3DomRenderer(display_axes_plane=False)

    for k, face_ctrs in enumerate(ctrs):

        surf = create_bspline_surface(face_ctrs)
        face = BRepBuilderAPI_MakeFace(surf, 1e-6).Face()
        face = BRepBuilderAPI_Transform(face, rotationz).Shape()
        face = BRepBuilderAPI_Transform(face, rotationx).Shape()

        sphere_face = []
        line_face = []
        for ctrs in face_ctrs:
            sphere = BRepPrimAPI_MakeSphere(gp_Pnt(ctrs[0], ctrs[1], ctrs[2]), radius).Shape()
            sphere = BRepBuilderAPI_Transform(sphere, rotationz).Shape()
            sphere_face.append(BRepBuilderAPI_Transform(sphere, rotationx).Shape())
        face_ctrs = face_ctrs.reshape(4, 4, 3)
        for i in range(4):
            for j in range(3):
                pt1 = gp_Pnt(face_ctrs[i, j, 0], face_ctrs[i, j, 1], face_ctrs[i, j, 2])
                pt2 = gp_Pnt(face_ctrs[i, j + 1, 0], face_ctrs[i, j + 1, 1], face_ctrs[i, j + 1, 2])
                line = BRepBuilderAPI_MakeEdge(pt1, pt2).Edge()
                line = BRepBuilderAPI_Transform(line, rotationz).Shape()
                line_face.append(BRepBuilderAPI_Transform(line, rotationx).Shape())
        for j in range(4):
            for i in range(3):
                pt1 = gp_Pnt(face_ctrs[i, j, 0], face_ctrs[i, j, 1], face_ctrs[i, j, 2])
                pt2 = gp_Pnt(face_ctrs[i + 1, j, 0], face_ctrs[i + 1, j, 1], face_ctrs[i + 1, j, 2])
                line = BRepBuilderAPI_MakeEdge(pt1, pt2).Edge()
                line = BRepBuilderAPI_Transform(line, rotationz).Shape()
                line_face.append(BRepBuilderAPI_Transform(line, rotationx).Shape())

        # Display face control points
        color = colors[k]
        color = (1,0,1)
        for sphere in sphere_face:
            my_renderer.myDisplayShape(sphere, color=color, export_edges=False)
        for line in line_face:
            my_renderer.myDisplayShape(line, line_color=color, export_edges=False, line_width=8)

        # my_renderer.myDisplayShape(face, color=color, export_edges=True, transparency=0)
    my_renderer.render()


def main():
    step_file = "/home/jing/PythonProjects/BrepGDM/Pics/Srcs/vis_overview/table_SP6URjYZo8.step"
    output_dir = "Pics/Overview"

    shape = read_step(step_file)

    axisz = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    anglez = -30 * (3.14159 / 180)
    rotationz = gp_Trsf()
    rotationz.SetRotation(axisz, anglez)
    rotated_shapez = BRepBuilderAPI_Transform(shape, rotationz).Shape()
    axisx = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0))
    anglex = 30 * (3.14159 / 180)
    rotationx = gp_Trsf()
    rotationx.SetRotation(axisx, anglex)
    rotated_shapex = BRepBuilderAPI_Transform(rotated_shapez, rotationx).Shape()

    # save_views(rotated_shapex, output_dir)

    # create edge control lines
    # with open('/home/jing/PythonProjects/BrepGDM/Pics/Srcs/vis_overview/table_SP6URjYZo8.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # edge_ctrs = data['edge_ctrs'][14]   # 4*3
    # face_ctrs = data['face_ctrs'][7]       # 16*3
    # sphere_edge = []
    # radius = 0.02
    # for ctrs in edge_ctrs:
    #     sphere = BRepPrimAPI_MakeSphere(gp_Pnt(ctrs[0], ctrs[1], ctrs[2]), radius).Shape()
    #     sphere = BRepBuilderAPI_Transform(sphere, rotationz).Shape()
    #     sphere_edge.append(BRepBuilderAPI_Transform(sphere, rotationx).Shape())
    # line_edge = []
    # for i in range(3):  # 3 edges connecting 4 points
    #     pt1 = gp_Pnt(edge_ctrs[i, 0], edge_ctrs[i, 1], edge_ctrs[i, 2])
    #     pt2 = gp_Pnt(edge_ctrs[i + 1, 0], edge_ctrs[i + 1, 1], edge_ctrs[i + 1, 2])
    #     line = BRepBuilderAPI_MakeEdge(pt1, pt2).Edge()
    #     line = BRepBuilderAPI_Transform(line, rotationz).Shape()
    #     line_edge.append(BRepBuilderAPI_Transform(line, rotationx).Shape())
    #
    # # create face control lines
    # sphere_face = []
    # line_face = []
    # for ctrs in face_ctrs:
    #     sphere = BRepPrimAPI_MakeSphere(gp_Pnt(ctrs[0], ctrs[1], ctrs[2]), radius).Shape()
    #     sphere = BRepBuilderAPI_Transform(sphere, rotationz).Shape()
    #     sphere_face.append(BRepBuilderAPI_Transform(sphere, rotationx).Shape())
    # face_ctrs = face_ctrs.reshape(4, 4, 3)
    # for i in range(4):
    #     for j in range(3):
    #         pt1 = gp_Pnt(face_ctrs[i, j, 0], face_ctrs[i, j, 1], face_ctrs[i, j, 2])
    #         pt2 = gp_Pnt(face_ctrs[i, j + 1, 0], face_ctrs[i, j + 1, 1], face_ctrs[i, j + 1, 2])
    #         line = BRepBuilderAPI_MakeEdge(pt1, pt2).Edge()
    #         line = BRepBuilderAPI_Transform(line, rotationz).Shape()
    #         line_face.append(BRepBuilderAPI_Transform(line, rotationx).Shape())
    # for j in range(4):
    #     for i in range(3):
    #         pt1 = gp_Pnt(face_ctrs[i, j, 0], face_ctrs[i, j, 1], face_ctrs[i, j, 2])
    #         pt2 = gp_Pnt(face_ctrs[i + 1, j, 0], face_ctrs[i + 1, j, 1], face_ctrs[i + 1, j, 2])
    #         line = BRepBuilderAPI_MakeEdge(pt1, pt2).Edge()
    #         line = BRepBuilderAPI_Transform(line, rotationz).Shape()
    #         line_face.append(BRepBuilderAPI_Transform(line, rotationx).Shape())
    #
    # draw_special_ctrs(rotated_shapex, sphere_edge, line_edge, sphere_face, line_face)

    # diff_ctrs()
    draw_diff_ctrs()


if __name__ == "__main__":
    main()
