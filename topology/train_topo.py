import os
import argparse
import wandb
from topology.datasets import TopoSeqDataset, FaceEdgeDataset
from topology.trainers import TopoSeqTrainer, FaceEdgeTrainer


def get_args_topo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument("--option", type=str, choices=['Seq', 'faceEdge'], default='faceEdge')
    parser.add_argument('--max_face', type=int, default=50, help='maximum number of faces')
    parser.add_argument('--edge_classes', type=int, default=5, help='Number of edge classes')
    parser.add_argument('--train_epochs', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--test_epochs', type=int, default=50, help='number of epochs to test model')
    parser.add_argument('--save_epochs', type=int, default=200, help='number of epochs to save model')
    parser.add_argument("--gpu", type=int, nargs='+', default=[0, 1],
                        help="GPU IDs to use for training (default: [0, 1])")
    parser.add_argument('--env', type=str, default="furniture_topo_faceEdge", help='environment')
    parser.add_argument('--dir_name', type=str, default="checkpoints", help='name of the log folder.')
    args = parser.parse_args()
    args.save_dir = os.path.join(args.dir_name, args.env.split('_', 1)[0], args.env.split('_', 1)[1])
    return args


def main():

    # Parse input augments
    args = get_args_topo()

    # Make project directory if not exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Set PyTorch to use only the specified GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu))

    if args.option == 'Seq':
        train_dataset = TopoSeqDataset('data_process/TopoDatasets/furniture/train', data_aug=True)
        val_dataset = TopoSeqDataset('data_process/TopoDatasets/furniture/test')
        topo = TopoSeqTrainer(args, train_dataset, val_dataset)
    else:
        assert args.option == 'faceEdge'
        train_dataset = FaceEdgeDataset('data_process/TopoDatasets/furniture/train', args)
        val_dataset = FaceEdgeDataset('data_process/TopoDatasets/furniture/test', args)
        topo = FaceEdgeTrainer(args, train_dataset, val_dataset)

    # Main training loop
    print('Start training...')

    # Initialize wandb
    # os.environ["WANDB_MODE"] = "offline"
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
