import os
import wandb
import argparse
from datasets import SurfData, EdgeData
from trainer_gather import SurfVAETrainer, EdgeVAETrainer


### Furniture VAE Training (fintune) ###
# python vae.py --data data_process/furniture_parsed \
#     --train_list data_process/furniture_data_split_6bit_surface.pkl \
#     --val_list data_process/furniture_data_split_6bit.pkl \
#     --option surface --gpu 0 --env furniture_vae_surf --train_nepoch 200 --finetune \
#     --weight proj_log/deepcad_vae_surf.pt
#
# python vae.py --data data_process/furniture_parsed \
#     --train_list data_process/furniture_data_split_6bit_edge.pkl \
#     --val_list data_process/furniture_data_split_6bit.pkl \
#     --option edge --gpu 0 --env furniture_vae_edge --train_nepoch 200 --finetune \
#     --weight proj_log/deepcad_vae_edge.pt


def get_args_vae():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data_process/furniture_parsed',
                        help='Path to data folder')
    parser.add_argument('--train_list', type=str, default='data_process/furniture_data_split_6bit_edge.pkl',
                        help='Path to training list')
    parser.add_argument('--val_list', type=str, default='data_process/furniture_data_split_6bit.pkl',
                        help='Path to validation list')
    # Training parameters
    parser.add_argument("--option", type=str, choices=['surface', 'edge'], default='edge',
                        help="Choose between option surface or edge (default: surface)")
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
    parser.add_argument('--train_nepoch', type=int, default=400, help='number of epochs to train for')
    parser.add_argument('--save_nepoch', type=int, default=50, help='number of epochs to save model')
    parser.add_argument('--test_nepoch', type=int, default=10, help='number of epochs to test model')
    parser.add_argument("--data_aug",  action='store_true', help='Use data augmentation')
    parser.add_argument("--finetune",  action='store_true', help='Finetune from existing weights')
    parser.add_argument("--weight",  type=str, default=None, help='Weight path when finetuning')
    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help="GPU IDs to use for training (default: [0])")
    # Save dirs and reload
    parser.add_argument('--env', type=str, default="furniture_vae_edge", help='environment')
    parser.add_argument('--dir_name', type=str, default="checkpoints", help='name of the log folder.')
    args = parser.parse_args()
    # saved folder
    args.save_dir = os.path.join(args.dir_name, args.env.split('_', 1)[0], args.env.split('_', 1)[1])
    return args


def main():

    # Parse input augments
    args = get_args_vae()

    # Make project directory if not exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Set PyTorch to use only the specified GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu))

    # Initialize wandb
    wandb.init(project='DiffBrep', dir=args.save_dir, name=args.env)

    # Initialize dataset loader and trainer
    if args.option == 'surface':
        train_dataset = SurfData(args.data, args.train_list, validate=False, aug=args.data_aug)
        val_dataset = SurfData(args.data, args.val_list, validate=True, aug=False)
        vae = SurfVAETrainer(args, train_dataset, val_dataset)
    else:
        assert args.option == 'edge', 'please choose between surface or edge'
        train_dataset = EdgeData(args.data, args.train_list, validate=False, aug=args.data_aug)
        val_dataset = EdgeData(args.data, args.val_list, validate=True, aug=False)
        vae = EdgeVAETrainer(args, train_dataset, val_dataset)

    # Main training loop
    print('Start training...')

    for _ in range(args.train_nepoch):

        # Train for one epoch
        vae.train_one_epoch()

        # Evaluate model performance on validation set
        if vae.epoch % args.test_nepoch == 0:
            vae.test_val()

        # save model
        if vae.epoch % args.save_nepoch == 0:
            vae.save_model()
    return


if __name__ == "__main__":
    main()
