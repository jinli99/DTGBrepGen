import pickle
import os

keys_map = {
    'edgeVert_adj': 'edgeVert_adj',
    'fef_adj': 'fef_adj'
}
path = 'data_process/TopoDatasets'
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.pkl'):
            with open(os.path.join(root, file), 'rb') as f:
                data = pickle.load(f)
            for old_key, new_key in keys_map.items():
                if old_key in data:
                    data[new_key] = data.pop(old_key)
            with open(os.path.join(root, file), "wb") as tf:
                pickle.dump(data, tf)
