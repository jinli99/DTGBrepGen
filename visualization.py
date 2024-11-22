import os
import random
import math
import plotly.offline as pyo
import plotly.graph_objects as go
import numpy as np
from OCC.Display.SimpleGui import init_display
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.gp import gp_Trsf, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.AIS import AIS_Shape
import open3d as o3d
from utils import load_data_with_prefix


def draw_bfep(bbox=None, faces=None, edges=None, points=None, label=False, bbox_fill=False):
    fig = go.Figure()

    # Helper function to draw boxes
    def draw_boxes(bbox, colors):
        for i in range(bbox.shape[0]):
            box = bbox[i].reshape(2, 3)
            if bbox_fill:
                # Draw 6 faces of the box using Mesh3d
                vertices = np.array([
                    [box[0, 0], box[0, 1], box[0, 2]],
                    [box[1, 0], box[0, 1], box[0, 2]],
                    [box[1, 0], box[1, 1], box[0, 2]],
                    [box[0, 0], box[1, 1], box[0, 2]],
                    [box[0, 0], box[0, 1], box[1, 2]],
                    [box[1, 0], box[0, 1], box[1, 2]],
                    [box[1, 0], box[1, 1], box[1, 2]],
                    [box[0, 0], box[1, 1], box[1, 2]],
                ])
                # Define the faces of the box
                i_faces = [
                    [0, 1, 2, 3],  # Bottom face
                    [4, 5, 6, 7],  # Top face
                    [0, 1, 5, 4],  # Side face 1
                    [1, 2, 6, 5],  # Side face 2
                    [2, 3, 7, 6],  # Side face 3
                    [3, 0, 4, 7]   # Side face 4
                ]
                for face in i_faces:
                    x, y, z = vertices[face, 0], vertices[face, 1], vertices[face, 2]
                    fig.add_trace(go.Mesh3d(x=x, y=y, z=z, color=colors[i], opacity=0.5, name=f'box_{i}'))
            else:
                # Draw only edges of the box
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
            # Calculate the middle index
            mid_index = len(x) // 2
            # Only display edge ID at the middle point of the edge if label is True
            text = [''] * len(x)
            if label:
                text[mid_index] = f'E {i}'
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines+text', line=dict(color=color, width=5), text=text, textposition="top center", name=f'edge_{i}'))

    # Helper function to draw points
    def draw_points(points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        text = [f'P {i}' for i in range(points.shape[0])] if label else None  # Display point ID if label is True
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers+text', marker=dict(color='black', size=5), text=text, textposition="top center", name='points'))

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


def draw_edge(edge_wcs, save_name='temp-plot.html', auto_open=True):
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
    pyo.plot(fig, filename=save_name, auto_open=auto_open)


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


def draw_face_ctrs(ctrs):
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


def draw_edge_ctrs(ctrs):
    """
    Draw multiple edges with four control points each, display control points and connect them with lines.

    Args:
    - ctrs (numpy.array): A b * 12-dimensional numpy array representing b edges,
                          where each edge has 4 control points, and each point has x, y, z coordinates.
    """
    b = ctrs.shape[0]  # Number of edges
    assert ctrs.shape[1] == 12, "Each edge should have 12 values (4 control points with x, y, z coordinates)"

    # Reshape the array into (b, 4, 3) where each edge has 4 control points with 3D coordinates
    ctrs = ctrs.reshape(b, 4, 3)

    # Define a color palette for different edges
    color_palette = ['blue', 'green', 'red', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    fig = go.Figure()

    for edge_idx in range(b):
        # Extract the x, y, z coordinates of each control point for the current edge
        x_points = ctrs[edge_idx, :, 0]
        y_points = ctrs[edge_idx, :, 1]
        z_points = ctrs[edge_idx, :, 2]

        color = color_palette[edge_idx % len(color_palette)]  # Choose color based on the edge index

        # Add scatter plot for the control points with labels (IDs)
        fig.add_trace(go.Scatter3d(
            x=x_points,
            y=y_points,
            z=z_points,
            mode='markers+text',
            marker=dict(size=5, color=color),
            text=[f'{i}' for i in range(4)],  # Add IDs as labels for each control point
            textposition='top center',
            name=f"Edge Control Points {edge_idx + 1}"
        ))

        # Connect the control points with a line to form the edge
        fig.add_trace(go.Scatter3d(
            x=x_points,
            y=y_points,
            z=z_points,
            mode='lines',
            line=dict(color=color, width=2),
            name=f"Edge {edge_idx + 1}"
        ))

    # Set the layout for the 3D plot
    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
        ),
        title="Multiple Edges with Control Points and IDs",
        showlegend=False
    )

    # Display the plot
    pyo.plot(fig)


def visualize_step(step_folder, num=200):
    """
    Visualize Step Files in batches with random colors
    Args:
        step_folder: path to the folder containing step files
        num: number of step files to display per batch, default 100
    """
    # Get all step files
    step_files = load_data_with_prefix(step_folder, '.step')
    total_files = len(step_files)
    total_batches = math.ceil(total_files / num)

    for batch in range(total_batches):
        display, start_display, add_menu, add_function_to_menu = init_display()

        # Set white background
        display.View.SetBackgroundColor(Quantity_Color(1, 1, 1, Quantity_TOC_RGB))

        translation_step_x = 3
        translation_step_y = 3
        shapes_per_row = 10

        current_offset_x = 0
        current_offset_y = 0
        shape_count = 0

        # Calculate current batch file range
        start_idx = batch * num
        end_idx = min((batch + 1) * num, total_files)
        current_batch_files = step_files[start_idx:end_idx]

        for step_file in current_batch_files:
            step_reader = STEPControl_Reader()
            status = step_reader.ReadFile(step_file)

            if status == IFSelect_RetDone:
                step_reader.TransferRoots()
                shape = step_reader.OneShape()

                translation = gp_Trsf()
                translation.SetTranslation(gp_Vec(current_offset_x, current_offset_y, 0))
                transformed_shape = BRepBuilderAPI_Transform(shape, translation).Shape()

                # Generate random color and display shape
                random_color = generate_random_color(min_rgb=0.2, max_rgb=0.8)
                display.DisplayShape(
                    transformed_shape,
                    update=False,
                    color=random_color
                )

                shape_count += 1
                length = shape_count // shapes_per_row
                if shape_count % shapes_per_row == 0:
                    current_offset_x = -length * translation_step_x
                    current_offset_y = length * translation_step_y
                else:
                    current_offset_x += translation_step_x
                    current_offset_y += translation_step_y

        display.FitAll()

        # Display batch information
        print(f"Displaying batch {batch + 1}/{total_batches} (Files {start_idx + 1} to {end_idx} of {total_files})")
        print("Press 'q' to close current batch and continue to next batch...")

        start_display()


def generate_random_color(min_rgb=0.2, max_rgb=0.8):
    """
    Generate random RGB color within specified range to avoid too dark or too bright colors
    Args:
        min_rgb: minimum RGB value (0-1)
        max_rgb: maximum RGB value (0-1)
    Returns:
        Quantity_Color: Random color object
    """
    return Quantity_Color(
        min_rgb + random.random() * (max_rgb - min_rgb),
        min_rgb + random.random() * (max_rgb - min_rgb),
        min_rgb + random.random() * (max_rgb - min_rgb),
        Quantity_TOC_RGB
    )


def visualize_stl(stl_dir, num=100):

    stl_files = [os.path.join(stl_dir, f) for f in os.listdir(stl_dir) if f.endswith('.stl')]
    num_files = len(stl_files)

    for batch_start in range(0, num_files, num):
        batch_files = stl_files[batch_start:batch_start + num]

        translations = 1.5 * np.array([[2 * (i % 10), 2 * (i // 10), 0] for i in range(len(batch_files))])

        vis = o3d.visualization.Visualizer()
        vis.create_window()

        opt = vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])

        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, np.pi / 2])

        for i, stl_file in enumerate(batch_files):
            mesh = o3d.io.read_triangle_mesh(stl_file)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(np.random.rand(3))

            translation = translations[i]
            mesh.translate(translation)
            mesh.rotate(rotation_matrix)

            vis.add_geometry(mesh)

        vis.poll_events()
        vis.update_renderer()
        vis.run()
        vis.destroy_window()


def main():
    pass


if __name__ == '__main__':
   pass
