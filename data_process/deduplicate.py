import os
import random
import json
import math
import pickle
import argparse
from tqdm import tqdm
from hashlib import sha256
import numpy as np


def create_parser():
    """Create the base argument parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='furniture_parsed', help="Data folder path or CAD .pkl file")
    parser.add_argument("--bit", type=int, default=6, help='Deduplicate precision')
    parser.add_argument("--option", type=str, choices=['abc', 'deepcad', 'furniture'], default='furniture',
                        help="Choose between dataset options: [abc/deepcad/furniture]")
    return parser


def args_deduplicate_cad(known_args):
    parser = create_parser()
    return parser.parse_args(known_args)


def args_deduplicate_facEdge(known_args):
    parser = create_parser()
    parser.add_argument("--list", type=str, default='furniture_data_split_6bit.pkl', help="UID list")
    parser.add_argument("--type", type=str, choices=['face', 'edge'], default='edge', help='Process edge or face')
    return parser.parse_args(known_args)


def load_abc_pkl(root_dir, use_deepcad):
    """
    Recursively searches through a given parent directory and its subdirectories
    to find the paths of all ABC .pkl files.

    Args:
    - root_dir (str): Path to the root directory where the search begins.
    - use_deepcad (bool): Process deepcad or not

    Returns:
    - train [str]: A list containing the paths to all .pkl train data
    - val [str]: A list containing the paths to all .pkl validation data
    - test [str]: A list containing the paths to all .pkl test data
    """
    # Load DeepCAD UID
    if use_deepcad:
        with open('train_val_test_split.json', 'r') as json_file:
            deepcad_data = json.load(json_file)
        train_uid = set([uid.split('/')[1] for uid in deepcad_data['train']])
        val_uid = set([uid.split('/')[1] for uid in deepcad_data['validation']])
        test_uid = set([uid.split('/')[1] for uid in deepcad_data['test']])

    # Load ABC UID
    else:
        full_uids = []
        dirs = [f'{root_dir}/{str(i).zfill(4)}' for i in range(100)]
        for folder in dirs:
            files = os.listdir(folder)
            full_uids += files
        # 90-5-5 random split, same as deepcad
        random.shuffle(full_uids)  # randomly shuffle data
        train_uid = full_uids[0:int(len(full_uids)*0.9)]
        val_uid = full_uids[int(len(full_uids)*0.9):int(len(full_uids)*0.95)]
        test_uid = full_uids[int(len(full_uids)*0.95):]
        train_uid = set([uid.split('.')[0] for uid in train_uid])
        val_uid = set([uid.split('.')[0] for uid in val_uid])
        test_uid = set([uid.split('.')[0] for uid in test_uid])

    train = []
    val = []
    test = []
    dirs = [f'{root_dir}/{str(i).zfill(4)}' for i in range(100)]
    for folder in dirs:
        files = os.listdir(folder)
        for file in files:
            key_id = file.split('.')[0]
            if key_id in train_uid:
                train.append(file)
            elif key_id in val_uid:
                val.append(file)
            elif key_id in test_uid:
                test.append(file)
            else:
                print('unknown uid...')
                assert False
    return train, val, test


def load_furniture_pkl(root_dir):
    """
    Recursively searches through a given parent directory and its subdirectories
    to find the paths of all furniture .pkl files.

    Args:
    - root_dir (str): Path to the root directory where the search begins.

    Returns:
    - train [str]: A list containing the paths to all .pkl train data
    - val [str]: A list containing the paths to all .pkl validation data
    - test [str]: A list containing the paths to all .pkl test data
    """
    full_uids = []
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            # Check if the file ends with the specified prefix
            if filename.endswith('.pkl'):
                file_path = os.path.join(root, filename)
                full_uids.append(file_path)

    # 90-5-5 random split, similar to deepcad
    random.shuffle(full_uids)  # randomly shuffle data
    train_uid = full_uids[0:int(len(full_uids) * 0.9)]
    val_uid = full_uids[int(len(full_uids) * 0.9):int(len(full_uids) * 0.95)]
    test_uid = full_uids[int(len(full_uids) * 0.95):]

    train_uid = ['/'.join(uid.split('/')[-2:]) for uid in train_uid]
    val_uid = ['/'.join(uid.split('/')[-2:]) for uid in val_uid]
    test_uid = ['/'.join(uid.split('/')[-2:]) for uid in test_uid]

    return train_uid, val_uid, test_uid


def real2bit(data, n_bits=8, min_range=-1, max_range=1):
    """Convert vertices in [-1., 1.] to discrete values in [0, n_bits**2 - 1]."""
    range_quantize = 2**n_bits - 1
    data_quantize = (data - min_range) * range_quantize / (max_range - min_range)
    data_quantize = np.clip(data_quantize, a_min=0, a_max=range_quantize) # clip values
    return data_quantize.astype(int)


def load_pkl_data(path):
    """Load pkl data from a given path."""
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data


def hash_face_points(faces_wcs, n_bits):
    """Hash the surface sampled points."""
    face_hash_total = []
    for face in faces_wcs:
        np_bit = real2bit(face, n_bits=n_bits).reshape(-1, 3)
        data_hash = sha256(np_bit.tobytes()).hexdigest()
        face_hash_total.append(data_hash)
    return '_'.join(sorted(face_hash_total))


def save_unique_data(save_path, unique_data):
    """Save unique data to a pickle file."""
    with open(save_path, "wb") as tf:
        pickle.dump(unique_data, tf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, choices=['cad', 'facEdge'],
                        default='facEdge', help="Specify which function to call")
    args, unknown = parser.parse_known_args()

    if args.name == 'cad':
        cad_args = args_deduplicate_cad(unknown)
        print("CAD args:", cad_args)

        # Load all STEP folders
        if cad_args.option == 'furniture':
            train, val_path, test_path = load_furniture_pkl(cad_args.data)
        else:
            train, val_path, test_path = load_abc_pkl(cad_args.data, cad_args.option == 'deepcad')

        # Remove duplicate for the training set
        train_path = []
        unique_hash = set()
        total = 0

        for path_idx, uid in tqdm(enumerate(train)):
            total += 1
            if cad_args.option == 'furniture':
                path = os.path.join(cad_args.data, uid)
            else:
                path = os.path.join(cad_args.data, str(math.floor(int(uid.split('.')[0]) / 10000)).zfill(4), uid)
            data = load_pkl_data(path)

            # Hash the face sampled points
            data_hash = hash_face_points(data['face_wcs'], cad_args.bit)

            # Save non-duplicate shapes
            prev_len = len(unique_hash)
            unique_hash.add(data_hash)
            if prev_len < len(unique_hash):
                train_path.append(uid)

            if path_idx % 2000 == 0:
                print(len(unique_hash) / total)

        # Save data
        data_path = {
            'train': train_path,
            'val': val_path,
            'test': test_path,
        }
        save_unique_data(f'{cad_args.option}_data_split_{cad_args.bit}bit.pkl', data_path)

    elif args.name == 'facEdge':
        facEdge_args = args_deduplicate_facEdge(unknown)
        print("facEdge args:", facEdge_args)

        data_list = load_pkl_data(facEdge_args.list)['train']

        unique_data = []
        unique_hash = set()
        total = 0

        for path_idx, uid in tqdm(enumerate(data_list)):
            if facEdge_args.option == 'furniture':
                path = os.path.join(facEdge_args.data, uid)
            else:
                path = os.path.join(facEdge_args.data, str(math.floor(int(uid.split('.')[0]) / 10000)).zfill(4), uid)
            data = load_pkl_data(path)
            face_ncs, edge_ncs = data['face_ncs'], data['edge_ncs']
            data = edge_ncs if facEdge_args.type == 'edge' else face_ncs

            data_bits = real2bit(data, n_bits=facEdge_args.bit)

            for np_bit, np_real in zip(data_bits, data):
                total += 1

                # Reshape the array to a 2D array
                np_bit = np_bit.reshape(-1, 3)
                data_hash = sha256(np_bit.tobytes()).hexdigest()

                prev_len = len(unique_hash)
                unique_hash.add(data_hash)

                if prev_len < len(unique_hash):
                    unique_data.append(np_real)

            if path_idx % 2000 == 0:
                print(len(unique_hash) / total)

        save_unique_data(facEdge_args.list.split('.')[0] + f'_{facEdge_args.type}.pkl', unique_data)


if __name__ == "__main__":
    main()
