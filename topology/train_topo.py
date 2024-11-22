import os
import argparse
import wandb
import yaml
from topology.datasets import EdgeVertDataset, FaceEdgeDataset
from topology.trainers import EdgeVertTrainer, FaceEdgeTrainer


def get_args_topo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='furniture',
                        choices=['furniture', 'deepcad', 'abc'])
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument("--option", type=str, choices=['faceEdge', 'edgeVert'], default='edgeVert')
    parser.add_argument('--train_epochs', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--test_epochs', type=int, default=50, help='number of epochs to test model')
    parser.add_argument('--save_epochs', type=int, default=200, help='number of epochs to save model')
    parser.add_argument('--dir_name', type=str, default="checkpoints", help='name of the log folder.')
    args = parser.parse_args()
    args.env = args.name+'_topo_'+args.option
    args.save_dir = os.path.join(args.dir_name, args.env.split('_', 1)[0], args.env.split('_', 1)[1])
    return args


def main():

    # Parse input augments
    args = get_args_topo()
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file).get(args.name, {})
    for key, value in config.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # Make project directory if not exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Set PyTorch to use only the specified GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu))

    if args.option == 'faceEdge':
        train_dataset = FaceEdgeDataset(os.path.join('data_process/TopoDatasets', args.name, 'train'), args)
        val_dataset = FaceEdgeDataset(os.path.join('data_process/TopoDatasets', args.name, 'test'), args)
        topo = FaceEdgeTrainer(args, train_dataset, val_dataset)
    else:
        assert args.option == 'edgeVert'
        train_dataset = EdgeVertDataset(os.path.join('data_process/TopoDatasets', args.name, 'train'), args)
        # print(train_dataset.max_seq_length, train_dataset.max_num_edge_topo)
        val_dataset = EdgeVertDataset(os.path.join('data_process/TopoDatasets', args.name, 'test'), args)
        # print(val_dataset.max_seq_length, train_dataset.max_num_edge_topo)
        topo = EdgeVertTrainer(args, train_dataset, val_dataset)

    # Main training loop
    print('Start training...')

    # Initialize wandb
    os.environ["WANDB_MODE"] = "offline"
    wandb.init(project='BrepGDM', dir=args.save_dir, name=args.env)

    # Main training loop
    for _ in range(args.train_epochs):
        # Train for one epoch
        topo.train_one_epoch()

        # Evaluate model performance on validation set
        if topo.epoch % args.test_epochs == 0:
            topo.test_val()

        # Save model
        if topo.epoch % args.save_epochs == 0:
            topo.save_model()


if __name__ == '__main__':
    main()
