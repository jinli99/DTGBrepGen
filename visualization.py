import plotly.offline as pyo
import plotly.graph_objects as go
import torch
import random
import numpy as np


def draw_bbox_colored(bbox):   # n*6
    fig = go.Figure()

    # Randomly assign colors to each bounding box for visualization
    colors = [f'rgba({c[0]}, {c[1]}, {c[2]}, 0.5)' for c in torch.randint(0, 255, (bbox.shape[0], 3))]

    for i, box in enumerate(bbox):
        # Extract the coordinates of the two diagonal points
        x1, y1, z1, x2, y2, z2 = box

        # Create the 8 vertices of the bounding box
        vertices = [
            [x1, y1, z1], [x2, y1, z1], [x2, y2, z1], [x1, y2, z1],  # Bottom face
            [x1, y1, z2], [x2, y1, z2], [x2, y2, z2], [x1, y2, z2]   # Top face
        ]

        # Define the 12 edges of the bounding box
        edges = [
            [vertices[j] for j in [0, 1]], [vertices[j] for j in [1, 2]], [vertices[j] for j in [2, 3]], [vertices[j] for j in [3, 0]],  # Bottom face edges
            [vertices[j] for j in [4, 5]], [vertices[j] for j in [5, 6]], [vertices[j] for j in [6, 7]], [vertices[j] for j in [7, 4]],  # Top face edges
            [vertices[j] for j in [0, 4]], [vertices[j] for j in [1, 5]], [vertices[j] for j in [2, 6]], [vertices[j] for j in [3, 7]]   # Side edges
        ]

        for edge in edges:
            fig.add_trace(go.Scatter3d(x=[edge[0][0], edge[1][0]],
                                       y=[edge[0][1], edge[1][1]],
                                       z=[edge[0][2], edge[1][2]],
                                       mode='lines',
                                       line=dict(color='black', width=2)))

        # Define the 6 faces of the bounding box
        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],  # Top and bottom faces
            [0, 1, 5, 4], [1, 2, 6, 5],  # Side faces
            [2, 3, 7, 6], [3, 0, 4, 7]   # Other side faces
        ]

        # Add each face as a separate Mesh trace
        for face in faces:
            fig.add_trace(go.Mesh3d(
                x=[vertices[j][0] for j in face],
                y=[vertices[j][1] for j in face],
                z=[vertices[j][2] for j in face],
                color=colors[i],
                opacity=0.5,
                i=[0, 1, 2, 3],
                j=[1, 2, 3, 0],
                k=[2, 3, 0, 1]
            ))

    fig.update_layout(scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z')
    ))

    # Show the figure
    pyo.plot(fig)


def draw_bbox_feTopo(bbox, edgeFace_adj):
    """
    Draw bounding boxes and edges connecting their centers, and save the plot as an HTML file.

    Parameters:
    bbox (numpy.ndarray): nf*6 array, where each row represents a bounding box defined by its two diagonal points in 3D space.
    edgeFace_adj (numpy.ndarray): ne*2 array, where each row represents an edge connecting two bounding boxes by their indices.
    """
    # Initialize figure
    fig = go.Figure()

    # Initialize list to store bounding box centers
    bbox_centers = []

    # Add bounding boxes
    for i in range(bbox.shape[0]):
        x_min, y_min, z_min, x_max, y_max, z_max = bbox[i]

        # Define the vertices of the box
        vertices = np.array([
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max]
        ])

        # Define the 12 edges of the box
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]

        # Add edges of the bounding box
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=vertices[edge, 0],
                y=vertices[edge, 1],
                z=vertices[edge, 2],
                mode='lines',
                line=dict(color='blue', width=2)
            ))

        # Calculate the center of the bounding box
        center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
        bbox_centers.append(center)

    # Add edges connecting bounding box centers
    bbox_centers = np.array(bbox_centers)
    for edge in edgeFace_adj:
        start, end = edge
        fig.add_trace(go.Scatter3d(
            x=[bbox_centers[start, 0], bbox_centers[end, 0]],
            y=[bbox_centers[start, 1], bbox_centers[end, 1]],
            z=[bbox_centers[start, 2], bbox_centers[end, 2]],
            mode='lines',
            line=dict(color='red', width=2)
        ))

    # Set layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title='Bounding Boxes and Connecting Edges'
    )

    # Save plot to HTML file
    pyo.plot(fig)


def draw_bbox_face_edge(bbox=None, faces=None, edges=None):
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

    # Set the layout
    fig.update_layout(scene=dict(
        xaxis=dict(nticks=10, range=[-1, 1]),
        yaxis=dict(nticks=10, range=[-1, 1]),
        zaxis=dict(nticks=10, range=[-1, 1]),
        aspectratio=dict(x=1, y=1, z=1),
        aspectmode='cube'
    ))

    pyo.plot(fig)


def draw_face_topology(bbox, faces, topology, node_mask):    # n*6, n*p*3, n*n, n
    fig = go.Figure()

    # Generate random colors for each face
    colors = [f'rgba({c[0]}, {c[1]}, {c[2]}, 0.5)' for c in torch.randint(0, 255, (faces.shape[0], 3))]

    for i, box in enumerate(bbox):
        if not node_mask[i]:
            continue

        # Draw the faces as scatter points
        face = faces[i]
        fig.add_trace(go.Scatter3d(x=face[:, 0],
                                   y=face[:, 1],
                                   z=face[:, 2],
                                   mode='markers',
                                   marker=dict(color=colors[i], size=2)))

    # Draw connections based on topology
    for i in range(topology.shape[0]):
        if not node_mask[i]:
            continue
        for j in range(topology.shape[1]):
            if i != j and topology[i, j] > 0 and node_mask[j]:
                box_i = bbox[i]
                box_j = bbox[j]

                # Calculate center points of the boxes
                center_i = [(box_i[0] + box_i[3]) / 2, (box_i[1] + box_i[4]) / 2, (box_i[2] + box_i[5]) / 2]
                center_j = [(box_j[0] + box_j[3]) / 2, (box_j[1] + box_j[4]) / 2, (box_j[2] + box_j[5]) / 2]

                fig.add_trace(go.Scatter3d(x=[center_i[0], center_j[0]],
                                           y=[center_i[1], center_j[1]],
                                           z=[center_i[2], center_j[2]],
                                           mode='lines',
                                           line=dict(color='black', width=4)))  # Increased width for better distinction

    fig.update_layout(scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z')
    ))

    pyo.plot(fig)


def draw_bbox_point(bbox, points):
    """
    Draw bounding boxes and points using Plotly.

    Parameters:
    bbox (numpy.ndarray): An array of shape (b, 6) representing b bounding boxes.
                          Each bounding box is defined by two opposite corner points (x_min, y_min, z_min, x_max, y_max, z_max).
    points (numpy.ndarray): An array of shape (nv, 3) representing nv points in 3D space.
    """
    fig = go.Figure()

    # Generate random RGB colors for each bounding box
    def random_color():
        return 'rgb({}, {}, {})'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # Plot bounding boxes
    for box in bbox:
        x_min, y_min, z_min, x_max, y_max, z_max = box
        color = random_color()
        # Create the lines for the edges of the box
        lines = [
            [[x_min, x_max], [y_min, y_min], [z_min, z_min]], [[x_min, x_max], [y_max, y_max], [z_min, z_min]],
            [[x_min, x_max], [y_min, y_min], [z_max, z_max]], [[x_min, x_max], [y_max, y_max], [z_max, z_max]],
            [[x_min, x_min], [y_min, y_max], [z_min, z_min]], [[x_max, x_max], [y_min, y_max], [z_min, z_min]],
            [[x_min, x_min], [y_min, y_max], [z_max, z_max]], [[x_max, x_max], [y_min, y_max], [z_max, z_max]],
            [[x_min, x_min], [y_min, y_min], [z_min, z_max]], [[x_max, x_max], [y_min, y_min], [z_min, z_max]],
            [[x_min, x_min], [y_max, y_max], [z_min, z_max]], [[x_max, x_max], [y_max, y_max], [z_min, z_max]]
        ]
        for line in lines:
            fig.add_trace(go.Scatter3d(x=line[0], y=line[1], z=line[2], mode='lines', line=dict(color=color)))

    # Plot points
    fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers',
                               marker=dict(size=2, color='black')))

    # Set the layout
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))

    # Display the plot
    pyo.plot(fig)


def draw_edgeBox_edge(bb, ee):
    # bb is array of shape (6,), representing the coordinates of the two diagonal vertices of the box
    # ee is array of shape (32, 3), representing the coordinates of 32 points in 3D space

    bb = bb.reshape(2, 3)  # Reshape bb to (2, 3) NumPy array

    # Define the 8 vertices of the box
    x = [bb[0, 0], bb[1, 0], bb[1, 0], bb[0, 0], bb[0, 0], bb[1, 0], bb[1, 0], bb[0, 0]]
    y = [bb[0, 1], bb[0, 1], bb[1, 1], bb[1, 1], bb[0, 1], bb[0, 1], bb[1, 1], bb[1, 1]]
    z = [bb[0, 2], bb[0, 2], bb[0, 2], bb[0, 2], bb[1, 2], bb[1, 2], bb[1, 2], bb[1, 2]]

    # Define the edges of the box
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)  # Vertical edges
    ]

    # Set the color for the box edges and points
    color = 'blue'

    # Draw the edges of the box
    box_edges = []
    for edge in edges:
        x0, y0, z0 = x[edge[0]], y[edge[0]], z[edge[0]]
        x1, y1, z1 = x[edge[1]], y[edge[1]], z[edge[1]]
        box_edges.append(go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1], mode='lines', line=dict(color=color)))

    # Draw the scatter points with the same color as the box edges
    scatter_points = go.Scatter3d(x=ee[:, 0], y=ee[:, 1], z=ee[:, 2], mode='markers', marker=dict(color=color, size=5))

    # Create the figure
    fig = go.Figure(data=box_edges + [scatter_points])

    # Display the figure
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


def draw_points(points):

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

    fig = go.Figure(data=[scatter])

    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        title="3D Scatter Plot"
    )

    pyo.plot(fig)
