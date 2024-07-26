# ### Training Furniture Latent Diffusion Model (classifier-free) ###
# python ldm.py --data data_process/furniture_parsed \
#     --list data_process/furniture_data_split_6bit.pkl --option surfpos --gpu 0 1 \
#     --env furniture_ldm_surfpos --train_nepoch 3000 --test_nepoch 200 --save_nepoch 200 \
#     --max_face 50 --max_edge 30 --cf
#
# python ldm.py --data data_process/furniture_parsed \
#     --list data_process/furniture_data_split_6bit.pkl --option surfz \
#     --surfvae proj_log/furniture_vae_surf.pt --gpu 0 1 \
#     --env furniture_ldm_surfz --train_nepoch 3000 --batch_size 256 \
#     --max_face 50 --max_edge 30 --cf
#
# python ldm.py --data data_process/furniture_parsed \
#     --list data_process/furniture_data_split_6bit.pkl --option edgepos \
#     --surfvae proj_log/furniture_vae_surf.pt --gpu 0 1 \
#     --env furniture_ldm_edgepos --train_nepoch 1000 --batch_size 64 \
#     --max_face 50 --max_edge 30 --cf
#
# python ldm.py --data data_process/furniture_parsed \
#     --list data_process/furniture_data_split_6bit.pkl --option edgez \
#     --surfvae proj_log/furniture_vae_surf.pt --edgevae proj_log/furniture_vae_edge.pt --gpu 0 1 \
#     --env furniture_ldm_edgez --train_nepoch 1000 --batch_size 64 \
#     --max_face 50 --max_edge 30 --cf


import argparse
import os.path
import pickle
import numpy as np
import torch
import wandb
from collections import defaultdict
from datasets import SurfGraphData, EdgeGraphData
from trainer_gather import SurfDiffTrainer, EdgeDiffTrainer
from dataFeature import GraphFeatures
import logging


def get_args_gdm():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data_process/furniture_parsed',
                        help='Path to data folder')
    parser.add_argument('--train_list', type=str, default='data_process/furniture_data_split_6bit.pkl',
                        help='Path to data list')
    parser.add_argument('--surfvae', type=str, default='checkpoints/furniture/vae_surf/epoch_200.pt',
                        help='Path to pretrained surface vae weights')
    parser.add_argument('--edgevae', type=str, default='checkpoints/furniture/vae_edge/epoch_400.pt',
                        help='Path to pretrained edge vae weights')
    parser.add_argument("--option", type=str, choices=['surface', 'edge'], default='edge',
                        help="Choose between option [surf,edge] (default: surf)")
    parser.add_argument('--edge_classes', type=int, default=9, help='Number of edge classes')
    parser.add_argument("--extract_type", type=str, choices=['cycles', 'eigenvalues', 'all'], default='all',
                        help="Graph feature extraction type (default: surf)")
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--train_nepoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--test_nepoch', type=int, default=200, help='number of epochs to test model')
    parser.add_argument('--save_nepoch', type=int, default=50, help='number of epochs to save model')
    parser.add_argument('--max_face', type=int, default=50, help='maximum number of faces')
    parser.add_argument('--max_edge', type=int, default=30, help='maximum number of edges per face')
    parser.add_argument('--threshold', type=float, default=0.05, help='minimum threshold between two faces')
    parser.add_argument('--bbox_scaled', type=float, default=3, help='scaled the bbox')
    parser.add_argument('--z_scaled', type=float, default=1, help='scaled the latent z')
    parser.add_argument("--gpu", type=int, nargs='+', default=[0, 1],
                        help="GPU IDs to use for training (default: [0, 1])")
    parser.add_argument("--data_aug",  action='store_true', help='Use data augmentation')
    parser.add_argument("--cf",  action='store_false', help='Use data augmentation')
    # Save dirs and reload
    parser.add_argument('--env', type=str, default="furniture_gdm_edge", help='environment')
    parser.add_argument('--dir_name', type=str, default="checkpoints", help='name of the log folder.')
    args = parser.parse_args()
    # saved folder
    args.save_dir = os.path.join(args.dir_name, args.env.split('_', 1)[0], args.env.split('_', 1)[1])
    return args


def compute_dataset_info(args):
    # Compute the marginal edge distribution
    with open(args.train_list, "rb") as file:
        data = pickle.load(file)['train']

    integer_counts = defaultdict(int)
    node_distribution = torch.zeros(args.max_face+1)
    for path in data:
        with open(os.path.join(args.data, path), 'rb') as file:
            ff_edges = pickle.load(file)['ff_edges']

            assert np.array_equal(ff_edges, ff_edges.T) and np.all(np.diag(ff_edges) == 0)

            unique, counts = np.unique(ff_edges, return_counts=True)
            for value, count in zip(unique, counts):
                integer_counts[value] += count

            n = ff_edges.shape[0]
            if 0 in integer_counts:
                integer_counts[0] -= n

            if ff_edges.shape[0] <= args.max_face:
                node_distribution[ff_edges.shape[0]] += 1

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
    for path in data:
        with open(os.path.join(args.data, path), 'rb') as file:
            example_data = pickle.load(file)
            if example_data['surf_ncs'].shape[0] <= args.max_face:
                break
    input_dims = {'X': 0, 'E': args.edge_classes, 'Y': 1}
    extract_feat = GraphFeatures(args.extract_type, args.max_face)
    ff_edges = torch.from_numpy(example_data['ff_edges'])
    example_feat = extract_feat(torch.nn.functional.one_hot(
        ff_edges, num_classes=args.edge_classes).unsqueeze(0), torch.ones(1, ff_edges.shape[0], dtype=torch.bool))
    input_dims['X'] += example_feat[0].shape[-1]
    input_dims['E'] += example_feat[1].shape[-1]
    input_dims['Y'] += example_feat[2].shape[-1]
    output_dims = {'X': 0, 'E': args.edge_classes, 'Y': 0}

    return {'marginal': torch.from_numpy(marginal), 'max_n_nodes': args.max_face,
            'node_distribution': node_distribution/node_distribution.sum(),
            'input_dims': input_dims, 'output_dims': output_dims}


def main():

    # Parse input augments
    args = get_args_gdm()

    # Make project directory if not exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Set PyTorch to use only the specified GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu))

    # Initialize dataset loader and trainer
    if args.option == 'surface':
        dataset_info = compute_dataset_info(args)
        train_dataset = SurfGraphData(args.data, args.train_list, validate=False, aug=args.data_aug, args=args)
        val_dataset = SurfGraphData(args.data, args.train_list, validate=True, aug=False, args=args)
        gdm = SurfDiffTrainer(args, train_dataset, val_dataset, dataset_info)
    else:
        assert args.option == 'edge', 'please choose between surface or edge'
        dataset_info = compute_dataset_info(args)
        train_dataset = EdgeGraphData(args.data, args.train_list, validate=False, aug=args.data_aug, args=args)
        val_dataset = EdgeGraphData(args.data, args.train_list, validate=True, aug=False, args=args)
        gdm = EdgeDiffTrainer(args, train_dataset, val_dataset, dataset_info)

    # Main training loop
    print('Start training...')

    # Initialize wandb
    # os.environ["WANDB_MODE"] = "offline"
    # wandb.init(project='DiffBrep', dir=args.save_dir, name=args.env)

    # Main training loop
    for _ in range(args.train_nepoch):
        # Train for one epoch
        gdm.train_one_epoch()

        # # Evaluate model performance on validation set
        # if ldm.epoch % args.test_nepoch == 0:
        #     ldm.test_val()
        #
        # Save model
        if gdm.epoch % args.save_nepoch == 0:
            gdm.save_model()


if __name__ == "__main__":
    main()
