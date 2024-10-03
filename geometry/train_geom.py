import argparse
import os.path
import pickle
import numpy as np
import torch
import wandb
from collections import defaultdict
from geometry.datasets import FaceBboxData, FaceGeomData, VertGeomData, EdgeGeomData
from geometry.trainers import FaceBboxTrainer, FaceGeomTrainer, VertGeomTrainer, EdgeGeomTrainer
from geometry.dataFeature import GraphFeatures


def get_args_geom():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data_process/GeomDatasets/furniture_parsed',
                        help='Path to data folder')
    parser.add_argument('--train_list', type=str, default='data_process/furniture_data_split_6bit.pkl',
                        help='Path to data list')
    parser.add_argument('--face_vae', type=str, default='checkpoints/furniture/vae_face/epoch_400.pt',
                        help='Path to pretrained surface vae weights')
    parser.add_argument('--edge_vae', type=str, default='checkpoints/furniture/vae_edge/epoch_400.pt',
                        help='Path to pretrained edge vae weights')
    parser.add_argument("--option", type=str, choices=[
        'faceBbox', 'faceGeom', 'vertGeom', 'edgeGeom'], default='faceGeom',)
    parser.add_argument('--edge_classes', type=int, default=5, help='Number of edge classes')
    parser.add_argument("--extract_type", type=str, choices=['cycles', 'eigenvalues', 'all'], default='all',
                        help="Graph feature extraction type (default: all)")
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--train_epochs', type=int, default=3000, help='number of epochs to train for')
    parser.add_argument('--test_epochs', type=int, default=50, help='number of epochs to test model')
    parser.add_argument('--save_epochs', type=int, default=500, help='number of epochs to save model')
    parser.add_argument('--timesteps', type=int, default=500, help='diffusion timesteps')
    parser.add_argument('--max_face', type=int, default=50, help='maximum number of faces')
    parser.add_argument('--max_edge', type=int, default=30, help='maximum number of edges per face')
    parser.add_argument('--max_num_edge', type=int, default=100, help='maximum number of edges per brep')
    parser.add_argument('--max_vertex', type=int, default=100, help='maximum number of vertices per brep')
    parser.add_argument('--max_vertexFace', type=int, default=5, help='maximum number of faces each vertex')
    parser.add_argument('--threshold', type=float, default=0.05, help='minimum threshold between two faces')
    parser.add_argument('--bbox_scaled', type=float, default=3, help='scaled the bbox')
    parser.add_argument('--z_scaled', type=float, default=1, help='scaled the latent z')
    parser.add_argument("--gpu", type=int, nargs='+', default=[0, 1],
                        help="GPU IDs to use for training (default: [0, 1])")
    parser.add_argument("--data_aug",  action='store_true', help='Use data augmentation')
    parser.add_argument("--cf",  action='store_false', help='Use data augmentation')
    # Save dirs and reload
    parser.add_argument('--env', type=str, default="furniture_geom_faceGeom", help='environment')
    parser.add_argument('--dir_name', type=str, default="checkpoints", help='name of the log folder.')
    args = parser.parse_args()
    # saved folder
    args.save_dir = os.path.join(args.dir_name, args.env.split('_', 1)[0], args.env.split('_', 1)[1])
    return args


def compute_dataset_info(args):
    # Compute the marginal edge distribution
    with open(args.train_list, "rb") as file:
        datas = pickle.load(file)['train']

    integer_counts = defaultdict(int)
    node_distribution = torch.zeros(args.max_face+1)
    max_num_edge = 0
    max_vertex = 0
    max_vertexFace = 0
    for path in datas:
        with open(os.path.join(args.data, path), 'rb') as file:
            data = pickle.load(file)
            fef_adj = data['fef_adj']
            max_num_edge = max(max_num_edge, len(data['edgeFace_adj']))
            max_vertex = max(max_vertex, len(data['vert_wcs']))
            max_vertexFace = max(max_vertexFace, max([len(i) for i in data['vertFace_adj']]))
            assert np.array_equal(fef_adj, fef_adj.T) and np.all(np.diag(fef_adj) == 0)

            unique, counts = np.unique(fef_adj, return_counts=True)
            for value, count in zip(unique, counts):
                integer_counts[value] += count

            n = fef_adj.shape[0]
            if 0 in integer_counts:
                integer_counts[0] -= n

            if fef_adj.shape[0] <= args.max_face:
                node_distribution[fef_adj.shape[0]] += 1

    args.max_num_edge = max_num_edge
    args.max_vertex = max_vertex
    args.max_vertexFace = max_vertexFace

    assert min(integer_counts.keys()) == 0
    if args.edge_classes == -1:
        args.edge_classes = max(integer_counts.keys()) + 1
    marginal = np.zeros(args.edge_classes, dtype=float)
    for key, count in integer_counts.items():
        if key >= args.edge_classes:
            continue
        marginal[key] = count
    marginal /= np.sum(marginal)
    print("Marginal distribution of the edge classes:", marginal)

    # Compute the input dims and output dims
    for path in datas:
        with open(os.path.join(args.data, path), 'rb') as file:
            example_data = pickle.load(file)
            if example_data['face_ncs'].shape[0] <= args.max_face:
                break
    input_dims = {'x': 0, 'e': args.edge_classes, 'y': 1}
    extract_feat = GraphFeatures(args.extract_type, args.max_face)
    fef_adj = torch.from_numpy(example_data['fef_adj'])
    example_feat = extract_feat(torch.nn.functional.one_hot(
        fef_adj, num_classes=args.edge_classes).unsqueeze(0), torch.ones(1, fef_adj.shape[0], dtype=torch.bool))
    input_dims['x'] += example_feat[0].shape[-1]
    input_dims['e'] += example_feat[1].shape[-1]
    input_dims['y'] += example_feat[2].shape[-1]
    output_dims = {'x': 0, 'e': args.edge_classes, 'y': 0}

    return {'marginal': torch.from_numpy(marginal), 'max_n_nodes': args.max_face,
            'node_distribution': node_distribution/node_distribution.sum(),
            'input_dims': input_dims, 'output_dims': output_dims}


def main():

    # Parse input augments
    args = get_args_geom()

    # Make project directory if not exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Set PyTorch to use only the specified GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu))

    # Initialize dataset loader and trainer
    if args.option == 'faceBbox':
        dataset_info = compute_dataset_info(args)
        train_dataset = FaceBboxData(args.data, args.train_list, validate=False, aug=args.data_aug, args=args)
        val_dataset = FaceBboxData(args.data, args.train_list, validate=True, aug=False, args=args)
        gdm = FaceBboxTrainer(args, train_dataset, val_dataset, dataset_info)
    elif args.option == 'faceGeom':
        dataset_info = compute_dataset_info(args)
        train_dataset = FaceGeomData(args.data, args.train_list, validate=False, aug=args.data_aug, args=args)
        val_dataset = FaceGeomData(args.data, args.train_list, validate=True, aug=False, args=args)
        gdm = FaceGeomTrainer(args, train_dataset, val_dataset, dataset_info)
    elif args.option == 'vertGeom':
        dataset_info = compute_dataset_info(args)
        train_dataset = VertGeomData(args.data, args.train_list, validate=False, aug=args.data_aug, args=args)
        val_dataset = VertGeomData(args.data, args.train_list, validate=True, aug=False, args=args)
        gdm = VertGeomTrainer(args, train_dataset, val_dataset, dataset_info)
    else:
        assert args.option == 'edgeGeom'
        dataset_info = compute_dataset_info(args)
        train_dataset = EdgeGeomData(args.data, args.train_list, validate=False, aug=args.data_aug, args=args)
        val_dataset = EdgeGeomData(args.data, args.train_list, validate=True, aug=False, args=args)
        gdm = EdgeGeomTrainer(args, train_dataset, val_dataset, dataset_info)

    # Main training loop
    print('Start training...')

    # Initialize wandb
    # os.environ["WANDB_MODE"] = "offline"
    wandb.init(project='BrepGDM', dir=args.save_dir, name=args.env)

    # Main training loop
    for _ in range(args.train_epochs):
        # Train for one epoch
        gdm.train_one_epoch()

        # Evaluate model performance on validation set
        if gdm.epoch % args.test_epochs == 0:
            gdm.test_val()

        # Save model
        if gdm.epoch % args.save_epochs == 0:
            gdm.save_model()


if __name__ == "__main__":
    main()
