import argparse
import os.path
import yaml
import wandb
from geometry.datasets import FaceBboxData, FaceGeomData, VertGeomData, EdgeGeomData
from geometry.trainers import FaceBboxTrainer, FaceGeomTrainer, VertGeomTrainer, EdgeGeomTrainer


def get_args_geom():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='deepcad',
                        choices=['furniture', 'deepcad', 'abc'])
    parser.add_argument('--face_vae', type=str, default='checkpoints/furniture/vae_face/epoch_400.pt',
                        help='Path to pretrained surface vae weights')
    parser.add_argument('--edge_vae', type=str, default='checkpoints/furniture/vae_edge/epoch_400.pt',
                        help='Path to pretrained edge vae weights')
    parser.add_argument("--option", type=str, choices=[
        'faceBbox', 'faceGeom', 'vertGeom', 'edgeGeom'], default='vertGeom')
    parser.add_argument("--extract_type", type=str, choices=['cycles', 'eigenvalues', 'all'], default='all',
                        help="Graph feature extraction type (default: all)")
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--train_epochs', type=int, default=3000, help='number of epochs to train for')
    parser.add_argument('--test_epochs', type=int, default=50, help='number of epochs to test model')
    parser.add_argument('--save_epochs', type=int, default=500, help='number of epochs to save model')
    parser.add_argument('--dir_name', type=str, default="checkpoints", help='name of the log folder.')

    args = parser.parse_args()
    args.data = os.path.join('data_process/GeomDatasets', args.name+'_parsed')
    args.train_list = os.path.join('data_process', args.name+'_data_split_6bit.pkl')
    args.env = args.name+'_geom_'+args.option
    args.save_dir = os.path.join(args.dir_name, args.env.split('_', 1)[0], args.env.split('_', 1)[1])
    return args


def main():

    # Parse input augments
    args = get_args_geom()
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file).get(args.name, {})
    for key, value in config.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # Make project directory if not exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Initialize dataset loader and trainer
    if args.option == 'faceBbox':
        train_dataset = FaceBboxData(args=args, validate=False)
        val_dataset = FaceBboxData(args=args, validate=True)
        gdm = FaceBboxTrainer(args, train_dataset, val_dataset)
    elif args.option == 'vertGeom':
        train_dataset = VertGeomData(args=args, validate=False)
        val_dataset = VertGeomData(args=args, validate=False)
        gdm = VertGeomTrainer(args, train_dataset, val_dataset)
    elif args.option == 'edgeGeom':
        train_dataset = EdgeGeomData(args=args, validate=False)
        val_dataset = EdgeGeomData(args=args, validate=True)
        gdm = EdgeGeomTrainer(args, train_dataset, val_dataset)
    else:
        assert args.option == 'faceGeom'
        train_dataset = FaceGeomData(args=args, validate=False)
        val_dataset = FaceGeomData(args=args, validate=False)
        gdm = FaceGeomTrainer(args, train_dataset, val_dataset)

    # Main training loop
    print('Start training...')

    # Initialize wandb
    os.environ["WANDB_MODE"] = "offline"
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
