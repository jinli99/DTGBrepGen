import plotly.offline as pyo
import plotly.graph_objects as go
import torch


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


def draw_bbox_and_faces(bbox, faces):
    fig = go.Figure()

    # Generate random colors for each bounding box and face for visualization
    colors = [f'rgba({c[0]}, {c[1]}, {c[2]}, 1)' for c in torch.randint(0, 255, (bbox.shape[0], 3))]

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
                                       line=dict(color=colors[i], width=2)))

        # Plot the faces with the same color as the bounding box edges
        fig.add_trace(go.Scatter3d(
            x=faces[i][:, 0],
            y=faces[i][:, 1],
            z=faces[i][:, 2],
            mode='markers',
            marker=dict(color=colors[i], size=3)
        ))

    fig.update_layout(scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z')
    ))

    # Use pyo.plot to display the figure in the browser
    pyo.plot(fig)


def draw_bbox_topology(bbox, topology, node_mask):   # n*6, n*n, n
    fig = go.Figure()

    # Generate random colors for each bounding box
    colors = [f'rgba({c[0]}, {c[1]}, {c[2]}, 0.5)' for c in torch.randint(0, 255, (bbox.shape[0], 3))]

    for i, box in enumerate(bbox):
        if not node_mask[i]:
            continue

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
                                       line=dict(color=colors[i], width=2)))

    # Draw connections based on topology
    for i in range(topology.shape[0]):
        if not node_mask[i]:
            continue
        for j in range(topology.shape[1]):
            if i != j and topology[i, j] > 0 and node_mask[j]:
                box_i = bbox[i]
                box_j = bbox[j]
                color = colors[i] if i < len(colors) else 'black'

                # Calculate center points of the boxes
                center_i = [(box_i[0] + box_i[3]) / 2, (box_i[1] + box_i[4]) / 2, (box_i[2] + box_i[5]) / 2]
                center_j = [(box_j[0] + box_j[3]) / 2, (box_j[1] + box_j[4]) / 2, (box_j[2] + box_j[5]) / 2]

                fig.add_trace(go.Scatter3d(x=[center_i[0], center_j[0]],
                                           y=[center_i[1], center_j[1]],
                                           z=[center_i[2], center_j[2]],
                                           mode='lines',
                                           line=dict(color=color, width=4)))  # Increased width for better distinction

    fig.update_layout(scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z')
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


def draw_box_and_points(bb, ee):
    # bb is a tensor of shape (6,), representing the coordinates of the two diagonal vertices of the box
    # ee is a tensor of shape (32, 3), representing the coordinates of 32 points in 3D space

    bb = bb.view(2, 3).numpy()  # Reshape bb to (2, 3) NumPy array
    ee = ee.numpy()  # Convert ee to NumPy array

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
