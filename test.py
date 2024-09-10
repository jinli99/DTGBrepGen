import numpy as np
import pickle
import os


path = 'data_process/topoDatasets/furniture/eval'
files = os.listdir(path)
bad_files = []

for file in files:
    is_ok = True
    label, name = file.split('_', 1)
    with open(os.path.join('data_process/furniture_parsed', label, name), 'rb') as f:
        data = pickle.load(f)
    faceEdge_adj = data['faceEdge_adj']
    edgeVert_adj = data['edgeCorner_adj']
    for face_edges in faceEdge_adj:
        num_edges = len(face_edges)
        vertices = set()
        for edge_id in face_edges:
            vertices.update(edgeVert_adj[edge_id])
        num_vertices = len(vertices)
        if num_edges != num_vertices:
            bad_files.append(os.path.join(path, file))
            is_ok = False
            break
        if not is_ok:
            break
print(bad_files)
for file in bad_files:
    try:
        os.remove(file)
        print(f"{file} has been deleted.")
    except FileNotFoundError:
        print(f"{file} not found.")
    except Exception as e:
        print(f"Error deleting {file}: {e}")
