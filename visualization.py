import os
import plotly.offline as pyo
import plotly.graph_objects as go
import numpy as np
from OCC.Display.SimpleGui import init_display
from OCC.Core.STEPControl import STEPControl_Reader
from utils import load_data_with_prefix
from OCC.Core.IFSelect import IFSelect_RetDone


def draw_bfep(bbox=None, faces=None, edges=None, points=None):
    # nf*6, nf*32*32*3, ne*32*3, nv*3
    fig = go.Figure()

    # Helper function to draw boxes
    def draw_boxes(bbox, colors):
        for i in range(bbox.shape[0]):
            box = bbox[i].reshape(2, 3)
            x = [box[0, 0], box[1, 0], box[1, 0], box[0, 0], box[0, 0], box[0, 0], box[1, 0], box[1, 0], box[1, 0], box[0, 0], box[0, 0], box[1, 0]]
            y = [box[0, 1], box[0, 1], box[1, 1], box[1, 1], box[0, 1], box[0, 1], box[0, 1], box[0, 1], box[1, 1], box[1, 1], box[1, 1], box[1, 1]]
            z = [box[0, 2], box[0, 2], box[0, 2], box[0, 2], box[0, 2], box[1, 2], box[1, 2], box[1, 2], box[1, 2], box[1, 2], box[0, 2], box[0, 2]]
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=colors[i]), name=f'box_{i}'))

    # Helper function to draw faces
    def draw_faces(faces, colors):
        for i in range(faces.shape[0]):
            face = faces[i].reshape(-1, 3)
            x, y, z = face[:, 0], face[:, 1], face[:, 2]
            fig.add_trace(go.Mesh3d(x=x, y=y, z=z, color=colors[i], opacity=0.5, name=f'face_{i}'))

    # Helper function to draw edges
    def draw_edges(edges):
        for i in range(edges.shape[0]):
            edge = edges[i]
            x, y, z = edge[:, 0], edge[:, 1], edge[:, 2]
            color = f'rgb({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)})'
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=color, width=5), name=f'edge_{i}'))

    # Helper function to draw points
    def draw_points(points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color='black', size=5), name='points'))

    # Generate random colors for boxes and faces
    num_boxes = bbox.shape[0] if bbox is not None else faces.shape[0] if faces is not None else 0
    colors = [f'rgb({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)})' for _ in range(num_boxes)]

    # Draw boxes
    if bbox is not None:
        draw_boxes(bbox, colors)

    # Draw faces
    if faces is not None:
        draw_faces(faces, colors)

    # Draw edges
    if edges is not None:
        draw_edges(edges)

    # Draw points
    if points is not None:
        draw_points(points)

    # Set the layout
    fig.update_layout(scene=dict(
        xaxis=dict(nticks=10, range=[-1.1, 1.1]),
        yaxis=dict(nticks=10, range=[-1.1, 1.1]),
        zaxis=dict(nticks=10, range=[-1.1, 1.1]),
        aspectratio=dict(x=1, y=1, z=1),
        aspectmode='cube'
    ))

    pyo.plot(fig)


def draw_edgeVert(points, edgeVert_adj):
    # nv*3, ne*2

    fig = go.Figure()

    # Helper function to draw points
    def draw_points(points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color='black', size=5), name='points'))

    # Helper function to draw edges
    def draw_edges(points, edgeVert_adj):
        for i in range(edgeVert_adj.shape[0]):
            # Get the point coordinates for the two points connected by this edge
            edge_points = points[edgeVert_adj[i]]
            x, y, z = edge_points[:, 0], edge_points[:, 1], edge_points[:, 2]
            # Generate a random color for each edge
            color = f'rgb({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)})'
            # Add a trace for each edge
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=color, width=3), name=f'edge_{i}'))

    # Draw points
    draw_points(points)

    # Draw edges
    draw_edges(points, edgeVert_adj)

    # Set the layout
    fig.update_layout(scene=dict(
        xaxis=dict(nticks=10, range=[-1.1, 1.1]),
        yaxis=dict(nticks=10, range=[-1.1, 1.1]),
        zaxis=dict(nticks=10, range=[-1.1, 1.1]),
        aspectratio=dict(x=1, y=1, z=1),
        aspectmode='cube'
    ))

    # Render the plot
    pyo.plot(fig)


def draw_labeled_points(wcs_pnts):
    # wcs_pnts is a numpy.ndarray/torch.tensor of shape (ne, 3), representing the coordinates of ne points in 3D space

    x = wcs_pnts[:, 0]
    y = wcs_pnts[:, 1]
    z = wcs_pnts[:, 2]

    # Create scatter plot with labels
    scatter_points = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers+text',
        text=[str(i) for i in range(len(wcs_pnts))],  # Labels as the point ids
        textposition='top center',
        marker=dict(size=5, color='blue')
    )

    # Create the figure
    fig = go.Figure(data=[scatter_points])

    # Display the figure
    pyo.plot(fig)


def draw_edge(edge_wcs):
    ne = edge_wcs.shape[0]

    # Create a list of colors, you can customize it as you want
    colors = [f'rgba({np.random.randint(0, 255)},{np.random.randint(0, 255)},{np.random.randint(0, 255)},0.8)' for _ in
              range(ne)]

    data = []

    for i in range(ne):
        edge = edge_wcs[i]
        x, y, z = edge[:, 0], edge[:, 1], edge[:, 2]

        scatter = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines',
            line=dict(color=colors[i], width=4),
            name=f'Edge {i + 1}'
        )

        data.append(scatter)

    layout = go.Layout(
        title='3D Edge Visualization',
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        )
    )

    fig = go.Figure(data=data, layout=layout)
    pyo.plot(fig)


def draw_points(points, random_color=False):
    # nv*3

    if random_color:
        colors = np.random.rand(points.shape[0], 3)

        scatter = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=['rgb({},{},{})'.format(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in colors]
            )
        )
    else:
        scatter = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=5
            )
        )

    fig = go.Figure(data=[scatter])

    range_min = min(-1, points.min())
    range_max = max(1, points.max())

    # Set the layout
    fig.update_layout(scene=dict(
        xaxis=dict(nticks=10, range=[range_min, range_max]),
        yaxis=dict(nticks=10, range=[range_min, range_max]),
        zaxis=dict(nticks=10, range=[range_min, range_max]),
        aspectratio=dict(x=1, y=1, z=1),
        aspectmode='cube'
    ))

    pyo.plot(fig)


def draw_ctrs(ctrs):
    """
    Draw multiple control grids with b batches of control points, their grid lines, and display IDs for each point.

    Args:
    - ctrs (numpy.array): A b * 48-dimensional numpy array representing b batches of 16 control points per grid,
                          where each point has x, y, z coordinates. The points form 4x4 grids.
    """
    b = ctrs.shape[0]  # Number of control grids
    assert ctrs.shape[1] == 48, "Each control grid should have 48 values (16 control points with x, y, z coordinates)"

    # Reshape the array into (b, 16, 3) where each grid has 16 3D points
    ctrs = ctrs.reshape(b, 16, 3)

    # Define a color palette for different grids
    color_palette = ['blue', 'green', 'red', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    fig = go.Figure()

    for batch_idx in range(b):
        # Extract the x, y, z coordinates of each control point for the current batch
        x_points = ctrs[batch_idx, :, 0]
        y_points = ctrs[batch_idx, :, 1]
        z_points = ctrs[batch_idx, :, 2]

        color = color_palette[batch_idx % len(color_palette)]  # Choose color based on the batch index

        # Add scatter plot for the control points with labels (IDs)
        fig.add_trace(go.Scatter3d(
            x=x_points,
            y=y_points,
            z=z_points,
            mode='markers+text',
            marker=dict(size=5, color=color),
            text=[f'{i}' for i in range(16)],  # Add IDs as labels
            textposition='top center',
            name=f"Control Points {batch_idx + 1}"
        ))

        # Draw grid lines along rows
        for i in range(4):  # Drawing row lines
            fig.add_trace(go.Scatter3d(
                x=x_points[i * 4:(i + 1) * 4],
                y=y_points[i * 4:(i + 1) * 4],
                z=z_points[i * 4:(i + 1) * 4],
                mode='lines',
                line=dict(color=color, width=2),
                name=f'Row {i + 1} (Grid {batch_idx + 1})'
            ))

        # Draw grid lines along columns
        for j in range(4):  # Drawing column lines
            fig.add_trace(go.Scatter3d(
                x=x_points[j::4],
                y=y_points[j::4],
                z=z_points[j::4],
                mode='lines',
                line=dict(color=color, width=2),
                name=f'Column {j + 1} (Grid {batch_idx + 1})'
            ))

    # Set the layout for the 3D plot
    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
        ),
        title="Multiple Control Grids with IDs",
        showlegend=False
    )

    # Display the plot
    pyo.plot(fig)


def vis_step(path, num=10):
    """
    可视化 path 文件夹下所有 STEP 文件，一次显示 num 个文件。

    参数：
    - path: str，包含 STEP 文件的文件夹路径
    - num: int，一次可视化的 STEP 文件数量
    """
    # 初始化显示窗口
    display, start_display, add_menu, add_function_to_menu = init_display()

    # 列出文件夹中的所有 STEP 文件
    step_files = load_data_with_prefix(path, '.step')
    # step_files.sort()  # 可选，按字母顺序排序

    # 每次展示 num 个文件
    for i in range(0, len(step_files), num):
        # 清除之前的显示
        display.EraseAll()

        # 获取当前批次的 STEP 文件
        batch_files = step_files[i:i + num]

        # 加载和显示每个文件
        for step_file in batch_files:
            step_reader = STEPControl_Reader()
            filepath = os.path.join(path, step_file)
            status = step_reader.ReadFile(filepath)

            if status == 1:
                print(f"无法读取文件 {step_file}")
                continue

            step_reader.TransferRoots()
            shape = step_reader.Shape()
            display.DisplayShape(shape, update=True)

        # 更新显示窗口
        display.FitAll()
        print(f"显示文件 {i + 1} 至 {min(i + num, len(step_files))}")

        # 等待用户关闭窗口
        start_display()


"""Visulize Step File"""
#
# # 初始化显示器
# display, start_display, add_menu, add_function_to_menu = init_display()
#
# # 创建STEP读取器
# step_reader = STEPControl_Reader()
#
# # 加载STEP文件
# step_file = 'samples/test_ef/eval/test_ef_05/2zPAZRfIxsTyvQX_0.step'  # 替换为你要加载的STEP文件路径
# status = step_reader.ReadFile(step_file)
#
# # 确保文件成功读取
# if status == IFSelect_RetDone:
#     step_reader.TransferRoots()
#     shape = step_reader.OneShape()
#     display.DisplayShape(shape, update=True)
# else:
#     print("Error: Failed to read the STEP file.")
#
# # 启动显示窗口
# start_display()


# """Visulize Stl File"""
# import open3d as o3d
# import numpy as np
#
# # STL文件路径列表
# stl_files = [
#     "comparison/point_cloud/furniture/bed/bed_YI5Gq16WRM.stl",
#     "comparison/point_cloud/furniture/bench/bench_OiThAgBD4B.stl",
#     "comparison/point_cloud/furniture/chair/chair_9azOyb3elZ.stl",
#     "comparison/point_cloud/furniture/couch/couch_cLRwqUuX7x.stl",
#     "comparison/point_cloud/furniture/sofa/sofa_gVP8WMC2DL.stl",
#     "comparison/point_cloud/furniture/table/table_nz6MRe5tlP.stl"
# ]
#
# # 平移向量列表，每个物体将沿X轴依次平移
# translations = 1.5*np.array([
#     [0, 0, 0],  # 第一个物体不平移
#     [2, 0, 0],  # 第二个物体沿X轴平移2个单位
#     [4, 0, 0],  # 第三个物体沿X轴平移4个单位
#     [6, 0, 0],  # 第四个物体沿X轴平移6个单位
#     [8, 0, 0],  # 依次类推
#     [10, 0, 0],
# ])
#
# # 创建一个可视化窗口
# vis = o3d.visualization.Visualizer()
# vis.create_window()
#
# # 设置背景为白色
# opt = vis.get_render_option()
# opt.background_color = np.asarray([1, 1, 1])  # RGB: 白色
#
# # 生成绕z轴逆时针旋转90度的旋转矩阵
# rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, np.pi / 2])
#
# # 循环读取并添加每个STL文件到场景中
# for i, stl_file in enumerate(stl_files):
#     # 读取STL文件
#     mesh = o3d.io.read_triangle_mesh(stl_file)
#
#     # 确保法向量已计算
#     mesh.compute_vertex_normals()
#
#     # 为每个网格设置相同的灰色
#     mesh.paint_uniform_color(np.random.rand(3))
#
#     # 对物体进行平移和旋转
#     translation = translations[i]
#     mesh.translate(translation)
#     mesh.rotate(rotation_matrix)
#
#     # 添加网格到场景
#     vis.add_geometry(mesh)
#
# # 更新并显示
# vis.poll_events()
# vis.update_renderer()
#
# # 运行可视化
# vis.run()
#
# # 销毁窗口
# vis.destroy_window()


def main():
    pass


if __name__ == '__main__':
    vis_step(path='samples/Transformer_2000')
