from tqdm import tqdm
import wandb
import os
import torch
import torch.nn as nn
import pickle
from diffusers import AutoencoderKL
from model import (AutoencoderKL1D, FaceBboxTransformer, AutoencoderKLFastEncode, AutoencoderKL1DFastEncode,
                   FaceGeomTransformer, VertexGeomTransformer, EdgeGeomTransformer)
from diffusion import GraphDiffusion, DDPM
from utils import edge_reshape_mask, assert_weak_one_hot, xe_mask
from dataFeature import GraphFeatures


class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha, gamma=2, reduction='mean'):
        """
        :param alpha: weight for each class
        :param gamma:
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # shape=(bs,)
        log_softmax = torch.log_softmax(pred, dim=1)   # shape=(bs, m)
        log_pt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # shape=(bs, 1)
        log_pt = log_pt.reshape(-1)  # shape=(bs)
        ce_loss = -log_pt
        pt = torch.exp(log_pt)  # shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # multi class focal loss，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


class FaceVaeTrainer:
    """ Face VAE Trainer """

    def __init__(self, args, train_dataset, val_dataset):
        # Initialize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = AutoencoderKL(in_channels=3,
                              out_channels=3,
                              down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D',
                                                'DownEncoderBlock2D'],
                              up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D',
                                              'UpDecoderBlock2D'],
                              block_out_channels=[128, 256, 512, 512],
                              layers_per_block=2,
                              act_fn='silu',
                              latent_channels=3,
                              norm_num_groups=32,
                              sample_size=512,
                              )

        # Load pretrained face vae (fast encode version)
        if args.finetune:
            model.load_state_dict(torch.load(args.weight))

        self.model = model.to(self.device).train()

        # Initialize optimizer
        self.network_params = list(self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4,
            weight_decay=1e-5
        )
        self.scaler = torch.cuda.amp.GradScaler()

        # Initializer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                            shuffle=True,
                                                            batch_size=args.batch_size,
                                                            num_workers=8)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                          shuffle=False,
                                                          batch_size=args.batch_size,
                                                          num_workers=8)

    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()
        loss_fn = nn.MSELoss()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train
        for face_uv in self.train_dataloader:
            with torch.cuda.amp.autocast():
                face_uv = face_uv.to(self.device).permute(0, 3, 1, 2)
                self.optimizer.zero_grad()  # zero gradient

                # Pass through VAE
                posterior = self.model.encode(face_uv).latent_dist
                z = posterior.sample()
                dec = self.model.decode(z).sample

                # Loss functions
                kl_loss = posterior.kl().mean()
                mse_loss = loss_fn(dec, face_uv)
                total_loss = mse_loss + 1e-6 * kl_loss

                # Update model
                self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 10 == 0:
                wandb.log({"Loss-mse": mse_loss, "Loss-kl": kl_loss}, step=self.iters)

            self.iters += 1
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1
        return

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval()  # set to eval
        total_loss = 0
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')
        with torch.no_grad():
            for face_uv in self.val_dataloader:
                face_uv = face_uv.to(self.device).permute(0, 3, 1, 2)

                posterior = self.model.encode(face_uv).latent_dist
                z = posterior.sample()
                dec = self.model.decode(z).sample

                loss = mse_loss(dec, face_uv).mean((1, 2, 3)).sum().item()
                total_loss += loss
                total_count += len(face_uv)

        mse = total_loss / total_count
        self.model.train()  # set to train
        wandb.log({"Val-mse": mse}, step=self.iters)
        return mse

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'epoch_' + str(self.epoch) + '.pt'))
        return


class EdgeVaeTrainer:
    """ Edge VAE Trainer """

    def __init__(self, args, train_dataset, val_dataset):
        # Initialize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = AutoencoderKL1D(
            in_channels=3,
            out_channels=3,
            down_block_types=('DownBlock1D', 'DownBlock1D', 'DownBlock1D'),
            up_block_types=('UpBlock1D', 'UpBlock1D', 'UpBlock1D'),
            block_out_channels=(128, 256, 512),
            layers_per_block=2,
            act_fn='silu',
            latent_channels=3,
            norm_num_groups=32,
            sample_size=512
        )

        # Load pretrained edge vae (fast encode version)
        if args.finetune:
            model.load_state_dict(torch.load(args.weight))

        self.model = model.to(self.device).train()

        # Initialize optimizer
        self.network_params = list(self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4,
            weight_decay=1e-5
        )
        self.scaler = torch.cuda.amp.GradScaler()

        # Initializer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                            shuffle=True,
                                                            batch_size=args.batch_size,
                                                            num_workers=8)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                          shuffle=False,
                                                          batch_size=args.batch_size,
                                                          num_workers=8)

    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()
        loss_fn = nn.MSELoss()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train
        for edge_u in self.train_dataloader:
            with torch.cuda.amp.autocast():
                edge_u = edge_u.to(self.device).permute(0, 2, 1)
                self.optimizer.zero_grad()  # zero gradient

                # Pass through VAE
                posterior = self.model.encode(edge_u).latent_dist
                z = posterior.sample()
                dec = self.model.decode(z).sample

                # Loss functions
                kl_loss = 0.5 * torch.sum(
                    torch.pow(posterior.mean, 2) + posterior.var - 1.0 - posterior.logvar,
                    dim=[1, 2],
                ).mean()
                mse_loss = loss_fn(dec, edge_u)
                total_loss = mse_loss + 1e-6 * kl_loss

                # Update model
                self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 10 == 0:
                wandb.log({"Loss-mse": mse_loss, "Loss-kl": kl_loss}, step=self.iters)

            self.iters += 1
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1
        return

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval()  # set to eval
        total_loss = 0
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')
        with torch.no_grad():
            for edge_u in self.val_dataloader:
                edge_u = edge_u.to(self.device).permute(0, 2, 1)

                posterior = self.model.encode(edge_u).latent_dist
                z = posterior.sample()
                dec = self.model.decode(z).sample

                loss = mse_loss(dec, edge_u).mean((1, 2)).sum().item()
                total_loss += loss
                total_count += len(edge_u)

        mse = total_loss / total_count
        self.model.train()  # set to train
        wandb.log({"Val-mse": mse}, step=self.iters)
        return mse

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'epoch_' + str(self.epoch) + '.pt'))
        return


class FaceBboxTrainer:
    def __init__(self, args, train_dataset, val_dataset, dataset_info):
        # Initialize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.use_cf = args.cf
        self.z_scaled = args.z_scaled
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion = DDPM(args.timesteps, self.device)
        self.extract_feat = GraphFeatures(args.extract_type, args.max_face)
        self.edge_classes = args.edge_classes

        # Initialize network
        n_layers = 5
        hidden_mlp_dims = {'x': 256, 'e': 128, 'y': 128}
        # The dimensions should satisfy dx % n_head == 0
        hidden_dims = {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128}
        input_dims, output_dims = dataset_info['input_dims'], dataset_info['output_dims']
        input_dims['x'] += 6
        output_dims['x'] += 6
        model = FaceBboxTransformer(n_layers=n_layers, input_dims=input_dims, hidden_mlp_dims=hidden_mlp_dims,
                                    hidden_dims=hidden_dims, output_dims=output_dims,
                                    act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU())
        model = nn.DataParallel(model)    # distributed training
        self.model = model.to(self.device).train()

        # Initialize optimizer
        self.network_params = list(self.model.parameters())

        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )

        self.scaler = torch.cuda.amp.GradScaler()

        # Initializer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                            shuffle=True,
                                                            batch_size=args.batch_size,
                                                            num_workers=16)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                          shuffle=False,
                                                          batch_size=args.batch_size,
                                                          num_workers=16)

    def train_one_epoch(self):
        """ Train the model for one epoch """
        self.model.train()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train
        for data in self.train_dataloader:
            with torch.cuda.amp.autocast():
                data = [x.to(self.device) for x in data]
                if self.use_cf:
                    face_bbox, fe_topo, node_mask, class_label = data   # b*n*32*32*3, b*n*6, b*n*n, b*n, b*1
                else:
                    face_bbox, fe_topo, node_mask = data   # b*n*32*32*3, b*n*6, b*n*n, b*1
                    class_label = None
                num_faces = torch.nonzero(node_mask.int().sum(0), as_tuple=True)[0][-1].cpu().item()+1
                face_bbox = face_bbox[:, :num_faces, :]        # b*n*6
                fe_topo = fe_topo[:, :num_faces, :num_faces]   # b*n*n
                node_mask = node_mask[:, :num_faces]           # b*n
                e_0 = torch.nn.functional.one_hot(fe_topo, num_classes=self.edge_classes)   # b*n*n*m

                # # Augment the surface position (see https://arxiv.org/abs/2106.15282)
                # conditions = [surfPos]
                # aug_data = []
                # for data in conditions:
                #     aug_timesteps = torch.randint(0, 15, (bsz,), device=self.device).long()
                #     aug_noise = torch.randn(data.shape).to(self.device)
                #     aug_data.append(self.noise_scheduler.add_noise(data, aug_noise, aug_timesteps))
                # surfPos = aug_data[0]

                x_0 = (face_bbox * self.z_scaled).clone().detach()   # rescaled the latent z  # b*n*6
                x_0, e_0 = xe_mask(x=x_0, e=e_0, node_mask=node_mask)

                self.optimizer.zero_grad()  # zero gradient

                # Add noise
                noise_data = self.diffusion.add_noise(x_0, node_mask)
                x_t, y = noise_data['x_t'], self.diffusion.normalize_t(noise_data['t'])  # b*n*48, b*1

                # Extract features
                with torch.cuda.amp.autocast(enabled=False):
                    feat = self.extract_feat(e_0, node_mask)
                x_t_feat = torch.cat((x_t, feat[0]), dim=-1).float()   # b*n*12
                e_t_feat = torch.cat((e_0, feat[1]), dim=-1).float()   # b*n*n*m
                y_feat = torch.cat((feat[2], y), dim=-1).float()       # b*12

                # Predict start
                pred_noise = self.model(x_t_feat, e_t_feat, y_feat, node_mask)   # b*n*6

                if torch.isnan(pred_noise).any() or torch.isinf(pred_noise).any():
                    print("Has nan!!!!")
                    torch.save(pred_noise.detach().cpu(), 'bad_noise.pt')
                    assert False

                # Loss
                face_mse_loss = torch.nn.functional.mse_loss(pred_noise[node_mask], noise_data['noise'][node_mask])

                # Update model
                self.scaler.scale(face_mse_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0)  # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 20 == 0:
                wandb.log({"Loss-noise": face_mse_loss}, step=self.iters)

            self.iters += 1
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval()  # set to eval
        total_count = 0

        progress_bar = tqdm(total=len(self.val_dataloader))
        progress_bar.set_description(f"Testing")

        total_loss = [0, 0, 0, 0, 0]

        for data in self.val_dataloader:
            with torch.no_grad():
                data = [x.to(self.device) for x in data]
                if self.use_cf:
                    face_bbox, fe_topo, node_mask, class_label = data   # b*n*32*32*3, b*n*6, b*n*n, b*1, b*1
                else:
                    face_bbox, fe_topo, node_mask = data   # b*n*32*32*3, b*n*6, b*n*n, b*1
                    class_label = None
                num_faces = torch.nonzero(node_mask.int().sum(0), as_tuple=True)[0][-1].cpu().item()+1
                face_bbox = face_bbox[:, :num_faces, :]        # b*n*6
                fe_topo = fe_topo[:, :num_faces, :num_faces]   # b*n*n
                node_mask = node_mask[:, :num_faces]           # b*n
                e_0 = torch.nn.functional.one_hot(fe_topo, num_classes=self.edge_classes)   # b*n*n*m
                x_0 = (face_bbox * self.z_scaled).clone().detach()   # rescaled the latent z  # b*n*6
                x_0, e_0 = xe_mask(x=x_0, e=e_0, node_mask=node_mask)
                b = face_bbox.shape[0]

            total_count += 1

            for idx, step in enumerate([10, 50, 100, 200, 500]):
                # Evaluate at timestep
                timesteps = torch.randint(step - 1, step, (b, 1), device=self.device).long()  # [batch, 1]

                # Add noise
                noise_data = self.diffusion.add_noise(x_0, node_mask, t=timesteps)
                x_t, y = noise_data['x_t'], self.diffusion.normalize_t(noise_data['t'])  # b*n*48, b*1

                # Extract features
                feat = self.extract_feat(e_0, node_mask)
                x_t_feat = torch.cat((x_t, feat[0]), dim=-1).float()   # b*n*12
                e_t_feat = torch.cat((e_0, feat[1]), dim=-1).float()   # b*n*n*m
                y_feat = torch.cat((feat[2], y), dim=-1).float()       # b*12

                # Predict start
                pred_noise = self.model(x_t_feat, e_t_feat, y_feat, node_mask)   # b*n*6

                if torch.isnan(pred_noise).any() or torch.isinf(pred_noise).any():
                    print("Has nan!!!!")
                    torch.save(pred_noise.detach().cpu(), 'bad_noise.pt')
                    assert False

                # Loss
                face_mse_loss = torch.nn.functional.mse_loss(pred_noise[node_mask], noise_data['noise'][node_mask])

                total_loss[idx] += face_mse_loss

            progress_bar.update(1)
        progress_bar.close()

        mse = [loss / total_count for loss in total_loss]
        self.model.train()  # set to train
        wandb.log({"Val-010": mse[0], "Val-050": mse[1], "Val-100": mse[2], "Val-200": mse[3], "Val-500": mse[4]},
                  step=self.iters)

    def save_model(self):
        torch.save(self.model.module.state_dict(), os.path.join(self.save_dir, 'epoch_'+str(self.epoch)+'.pt'))
        return


class FaceGeomTrainer:
    def __init__(self, args, train_dataset, val_dataset, dataset_info):
        # Initialize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.use_cf = args.cf
        self.z_scaled = args.z_scaled
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion = DDPM(args.timesteps, self.device)
        self.extract_feat = GraphFeatures(args.extract_type, args.max_face)
        self.edge_classes = args.edge_classes

        # Load pretrained surface vae (fast encode version)
        face_vae = AutoencoderKLFastEncode(in_channels=3,
                                           out_channels=3,
                                           down_block_types=('DownEncoderBlock2D', 'DownEncoderBlock2D',
                                                             'DownEncoderBlock2D', 'DownEncoderBlock2D'),
                                           up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D',
                                                           'UpDecoderBlock2D'),
                                           block_out_channels=(128, 256, 512, 512),
                                           layers_per_block=2,
                                           act_fn='silu',
                                           latent_channels=3,
                                           norm_num_groups=32,
                                           sample_size=512,
                                           )
        face_vae.load_state_dict(torch.load(args.face_vae), strict=False)
        face_vae = nn.DataParallel(face_vae)    # distributed inference
        self.face_vae = face_vae.to(self.device).eval()

        # Initialize network
        n_layers = 5
        hidden_mlp_dims = {'x': 256, 'e': 128, 'y': 128}
        # The dimensions should satisfy dx % n_head == 0
        hidden_dims = {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128}
        input_dims, output_dims = dataset_info['input_dims'], dataset_info['output_dims']
        example_data = train_dataset[0]
        face_ncs = example_data[0].to(self.device)   # max_faces*32*32*3
        with torch.no_grad():
            face_uv = face_ncs.permute(0, 3, 1, 2)
            face_latent = self.face_vae(face_uv)
            face_latent = face_latent.flatten(-2, -1).permute(0, 2, 1)
        input_dims['x'] += face_latent.shape[-1] * face_latent.shape[-2]
        output_dims['x'] += face_latent.shape[-1] * face_latent.shape[-2]
        model = FaceGeomTransformer(n_layers=n_layers, input_dims=input_dims, hidden_mlp_dims=hidden_mlp_dims,
                                    hidden_dims=hidden_dims, output_dims=output_dims,
                                    act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU())
        model = nn.DataParallel(model)    # distributed training
        self.model = model.to(self.device).train()

        # Initialize optimizer
        self.network_params = list(self.model.parameters())

        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )

        self.scaler = torch.cuda.amp.GradScaler()

        # Initializer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                            shuffle=True,
                                                            batch_size=args.batch_size,
                                                            num_workers=16)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                          shuffle=False,
                                                          batch_size=args.batch_size,
                                                          num_workers=16)

    def train_one_epoch(self):
        """ Train the model for one epoch """
        self.model.train()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train
        for data in self.train_dataloader:
            with torch.cuda.amp.autocast():
                data = [x.to(self.device) for x in data]
                if self.use_cf:
                    face_ncs, face_bbox, fe_topo, node_mask, class_label = data   # b*n*32*32*3, b*n*6, b*n*n, b*n, b*1
                else:
                    face_ncs, face_bbox, fe_topo, node_mask = data   # b*n*32*32*3, b*n*6, b*n*n, b*1
                    class_label = None
                num_faces = torch.nonzero(node_mask.int().sum(0), as_tuple=True)[0][-1].cpu().item()+1
                face_ncs = face_ncs[:, :num_faces, ...]        # b*n*32*32*3
                face_bbox = face_bbox[:, :num_faces, :]        # b*n*6
                fe_topo = fe_topo[:, :num_faces, :num_faces]   # b*n*n
                node_mask = node_mask[:, :num_faces]           # b*n
                e_0 = torch.nn.functional.one_hot(fe_topo, num_classes=self.edge_classes)   # b*n*n*m
                bsz = face_bbox.shape[0]

                # # Augment the surface position (see https://arxiv.org/abs/2106.15282)
                # conditions = [surfPos]
                # aug_data = []
                # for data in conditions:
                #     aug_timesteps = torch.randint(0, 15, (bsz,), device=self.device).long()
                #     aug_noise = torch.randn(data.shape).to(self.device)
                #     aug_data.append(self.noise_scheduler.add_noise(data, aug_noise, aug_timesteps))
                # surfPos = aug_data[0]

                # Pass through surface VAE to sample latent z
                with torch.no_grad():
                    face_uv = face_ncs.flatten(0, 1).permute(0, 3, 1, 2)
                    face_latent = self.face_vae(face_uv)
                    face_latent = face_latent.unflatten(0, (bsz, -1)).flatten(-2, -1).permute(0, 1, 3, 2)   # b*n*16*3

                x_0 = face_latent.flatten(-2, -1) * self.z_scaled   # rescaled the latent z  # b*n*48
                x_0, e_0 = xe_mask(x=x_0, e=e_0, node_mask=node_mask)

                self.optimizer.zero_grad()  # zero gradient

                # Add noise
                noise_data = self.diffusion.add_noise(x_0, node_mask)
                x_t, y = noise_data['x_t'], self.diffusion.normalize_t(noise_data['t'])  # b*n*48, b*1

                # Extract features
                with torch.cuda.amp.autocast(enabled=False):
                    feat = self.extract_feat(e_0, node_mask)
                x_t_feat = torch.cat((x_t, feat[0]), dim=-1).float()   # b*n*54
                e_t_feat = torch.cat((e_0, feat[1]), dim=-1).float()   # b*n*n*m
                y_feat = torch.cat((feat[2], y), dim=-1).float()       # b*12

                # Predict start
                pred_noise = self.model(x_t_feat, e_t_feat, y_feat, face_bbox, node_mask)   # b*n*54

                if torch.isnan(pred_noise).any() or torch.isinf(pred_noise).any():
                    print("Has nan!!!!")
                    torch.save(pred_noise.detach().cpu(), 'bad_noise.pt')
                    assert False

                # Loss
                face_mse_loss = torch.nn.functional.mse_loss(pred_noise[node_mask], noise_data['noise'][node_mask])

                # Update model
                self.scaler.scale(face_mse_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0)  # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 20 == 0:
                wandb.log({"Loss-noise": face_mse_loss}, step=self.iters)

            self.iters += 1
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval()  # set to eval
        total_count = 0

        progress_bar = tqdm(total=len(self.val_dataloader))
        progress_bar.set_description(f"Testing")

        total_loss = [0, 0, 0, 0, 0]

        for data in self.val_dataloader:
            with torch.no_grad():
                data = [x.to(self.device) for x in data]
                if self.use_cf:
                    face_ncs, face_bbox, fe_topo, node_mask, class_label = data   # b*n*32*32*3, b*n*6, b*n*n, b*1, b*1
                else:
                    face_ncs, face_bbox, fe_topo, node_mask = data   # b*n*32*32*3, b*n*6, b*n*n, b*1
                    class_label = None
                num_faces = torch.nonzero(node_mask.int().sum(0), as_tuple=True)[0][-1].cpu().item()+1
                face_ncs = face_ncs[:, :num_faces, ...]        # b*n*32*32*3
                face_bbox = face_bbox[:, :num_faces, :]        # b*n*6
                fe_topo = fe_topo[:, :num_faces, :num_faces]   # b*n*n
                node_mask = node_mask[:, :num_faces]           # b*n
                e_0 = torch.nn.functional.one_hot(fe_topo, num_classes=self.edge_classes)   # b*n*n*m
                _, e_0 = xe_mask(e=e_0, node_mask=node_mask)
                bsz = face_bbox.shape[0]

                # Pass through surface VAE to sample latent z
                face_uv = face_ncs.flatten(0, 1).permute(0, 3, 1, 2)
                face_latent = self.face_vae(face_uv)
                face_latent = face_latent.unflatten(0, (bsz, -1)).flatten(-2, -1).permute(0, 1, 3, 2)   # b*n*16*3

                x_0 = face_latent.flatten(-2, -1) * self.z_scaled   # rescaled the latent z  # b*n*48

            total_count += 1

            for idx, step in enumerate([10, 50, 100, 200, 500]):
                # Evaluate at timestep
                timesteps = torch.randint(step - 1, step, (bsz, 1), device=self.device).long()  # [batch, 1]

                # Add noise
                noise_data = self.diffusion.add_noise(x_0, node_mask, t=timesteps)
                x_t, y = noise_data['x_t'], self.diffusion.normalize_t(noise_data['t'])  # b*n*48, b*1

                # Extract features
                feat = self.extract_feat(e_0, node_mask)
                x_t_feat = torch.cat((x_t, feat[0]), dim=-1).float()   # b*n*54
                e_t_feat = torch.cat((e_0, feat[1]), dim=-1).float()   # b*n*n*m
                y_feat = torch.cat((feat[2], y), dim=-1).float()       # b*12

                # Predict start
                pred_noise = self.model(x_t_feat, e_t_feat, y_feat, face_bbox, node_mask)   # b*n*54

                if torch.isnan(pred_noise).any() or torch.isinf(pred_noise).any():
                    print("Has nan!!!!")
                    torch.save(pred_noise.detach().cpu(), 'bad_noise.pt')
                    assert False

                # Loss
                face_mse_loss = torch.nn.functional.mse_loss(pred_noise[node_mask], noise_data['noise'][node_mask])

                total_loss[idx] += face_mse_loss

            progress_bar.update(1)
        progress_bar.close()

        mse = [loss / total_count for loss in total_loss]
        self.model.train()  # set to train
        wandb.log({"Val-010": mse[0], "Val-050": mse[1], "Val-100": mse[2], "Val-200": mse[3], "Val-500": mse[4]},
                  step=self.iters)

    def save_model(self):
        torch.save(self.model.module.state_dict(), os.path.join(self.save_dir, 'epoch_'+str(self.epoch)+'.pt'))
        return


class VertexGeomTrainer:
    def __init__(self, args, train_dataset, val_dataset, dataset_info):
        # Initialize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.use_cf = args.cf
        self.z_scaled = args.z_scaled
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion = DDPM(args.timesteps, self.device)
        self.extract_feat = GraphFeatures(args.extract_type, args.max_face)
        self.edge_classes = args.edge_classes

        # Load pretrained surface vae (fast encode version)
        face_vae = AutoencoderKLFastEncode(in_channels=3,
                                           out_channels=3,
                                           down_block_types=('DownEncoderBlock2D', 'DownEncoderBlock2D',
                                                             'DownEncoderBlock2D', 'DownEncoderBlock2D'),
                                           up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D',
                                                           'UpDecoderBlock2D'),
                                           block_out_channels=(128, 256, 512, 512),
                                           layers_per_block=2,
                                           act_fn='silu',
                                           latent_channels=3,
                                           norm_num_groups=32,
                                           sample_size=512,
                                           )
        face_vae.load_state_dict(torch.load(args.face_vae), strict=False)
        face_vae = nn.DataParallel(face_vae)    # distributed inference
        self.face_vae = face_vae.to(self.device).eval()

        # Initialize network
        n_layers = 5
        hidden_mlp_dims = {'x': 256, 'e': 128, 'y': 128}
        # The dimensions should satisfy dx % n_head == 0
        hidden_dims = {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128}
        input_dims, output_dims = dataset_info['input_dims'], dataset_info['output_dims']
        input_dims['x'] += 3
        output_dims['x'] += 3
        model = VertexGeomTransformer(n_layers=n_layers, input_dims=input_dims, hidden_mlp_dims=hidden_mlp_dims,
                                      hidden_dims=hidden_dims, output_dims=output_dims,
                                      act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU())
        model = nn.DataParallel(model)    # distributed training
        self.model = model.to(self.device).train()

        # Initialize optimizer
        self.network_params = list(self.model.parameters())

        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )

        self.scaler = torch.cuda.amp.GradScaler()

        # Initializer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                            shuffle=True,
                                                            batch_size=args.batch_size,
                                                            num_workers=16)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                          shuffle=False,
                                                          batch_size=args.batch_size,
                                                          num_workers=16)

    def train_one_epoch(self):
        """ Train the model for one epoch """
        self.model.train()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train
        for data in self.train_dataloader:
            with torch.cuda.amp.autocast():
                data = [x.to(self.device) for x in data]
                if self.use_cf:
                    # b*nv*3, b*nv*nv, b*nv*nf*6, b*nv*nf*32*32*3, b*nv, b*nv*nf, b*1
                    vertex_geom, vv_adj, vertex_faceBbox, vertex_faceGeom, vertex_mask, vFace_mask, class_label = data
                else:
                    # b*nv*3, b*nv*nv, b*nv*nf*6, b*nv*nf*32*32*3, b*nv, b*nv*nf, b*1
                    vertex_geom, vv_adj, vertex_faceBbox, vertex_faceGeom, vertex_mask, vFace_mask = data
                    class_label = None
                nv = torch.nonzero(vertex_mask.int().sum(0), as_tuple=True)[0][-1].cpu().item()+1
                vf = torch.nonzero(vFace_mask.flatten(0, 1).int().sum(0), as_tuple=True)[0][-1].cpu().item()+1
                vertex_geom = vertex_geom[:, :nv, ...]                # b*nv*3
                vv_adj = vv_adj[:, :nv, :nv]                          # b*nv*nv
                vertex_faceBbox = vertex_faceBbox[:, :nv, :vf, ...]   # b*nv*vf*6
                vertex_faceGeom = vertex_faceGeom[:, :nv, :vf, ...]   # b*nv*vf*32*32*3
                vertex_mask = vertex_mask[:, :nv]                     # b*nv
                vFace_mask = vFace_mask[:, :nv, :vf]                  # b*nv*vf
                vv_adj = torch.nn.functional.one_hot(vv_adj, num_classes=2)   # b*nv*nv*2
                b = vertex_geom.shape[0]

                # # Augment the surface position (see https://arxiv.org/abs/2106.15282)
                # conditions = [surfPos]
                # aug_data = []
                # for data in conditions:
                #     aug_timesteps = torch.randint(0, 15, (bsz,), device=self.device).long()
                #     aug_noise = torch.randn(data.shape).to(self.device)
                #     aug_data.append(self.noise_scheduler.add_noise(data, aug_noise, aug_timesteps))
                # surfPos = aug_data[0]

                # Pass through surface VAE to sample latent z
                with torch.no_grad():
                    face_uv = torch.flatten(vertex_faceGeom, start_dim=0, end_dim=2).permute(0, 3, 1, 2)
                    face_latent = self.face_vae(face_uv)    # ?*3*4*4
                    face_latent = face_latent.unflatten(0, (b, nv, vf)).flatten(-2, -1).permute(0, 1, 2, 4, 3)   # b*nv*nf*16*3

                # rescaled the latent z,  b*nv*nf*54
                vertex_faceInfo = torch.cat(
                    (face_latent.flatten(-2, -1) * self.z_scaled, vertex_faceBbox), dim=-1).clone().detach()
                x_0, e_0 = xe_mask(x=vertex_geom, e=vv_adj, node_mask=vertex_mask)    # b*nv*3, b*nv*nv*2

                self.optimizer.zero_grad()  # zero gradient

                # Add noise
                noise_data = self.diffusion.add_noise(x_0, vertex_mask)
                x_t, y = noise_data['x_t'], self.diffusion.normalize_t(noise_data['t'])  # b*nv*3, b*1

                # Extract features
                with torch.cuda.amp.autocast(enabled=False):
                    feat = self.extract_feat(e_0, vertex_mask)
                x_t_feat = torch.cat((x_t, feat[0]), dim=-1).float()   # b*n*9
                e_t_feat = torch.cat((e_0, feat[1]), dim=-1).float()   # b*n*n*2
                y_feat = torch.cat((feat[2], y), dim=-1).float()       # b*12

                # Predict start
                pred_noise = self.model(x_t_feat, e_t_feat, vertex_faceInfo, y_feat, vertex_mask, vFace_mask)   # b*n*3

                if torch.isnan(pred_noise).any() or torch.isinf(pred_noise).any():
                    print("Has nan!!!!")
                    torch.save(pred_noise.detach().cpu(), 'bad_noise.pt')
                    assert False

                # Loss
                vertex_mse_loss = torch.nn.functional.mse_loss(pred_noise[vertex_mask], noise_data['noise'][vertex_mask])

                # Update model
                self.scaler.scale(vertex_mse_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0)  # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 20 == 0:
                wandb.log({"Loss-noise": vertex_mse_loss}, step=self.iters)

            if self.iters % 100 == 0:
                print("*****************Loss-noise****:", vertex_mse_loss.item())

            self.iters += 1
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval()  # set to eval
        total_count = 0

        progress_bar = tqdm(total=len(self.val_dataloader))
        progress_bar.set_description(f"Testing")

        total_loss = [0, 0, 0, 0, 0]

        for data in self.val_dataloader:
            with torch.no_grad():
                data = [x.to(self.device) for x in data]
                if self.use_cf:
                    # b*nv*3, b*nv*nv, b*nv*nf*6, b*nv*nf*32*32*3, b*nv, b*nv*nf, b*1
                    vertex_geom, vv_adj, vertex_faceBbox, vertex_faceGeom, vertex_mask, vFace_mask, class_label = data
                else:
                    # b*nv*3, b*nv*nv, b*nv*nf*6, b*nv*nf*32*32*3, b*nv, b*nv*nf, b*1
                    vertex_geom, vv_adj, vertex_faceBbox, vertex_faceGeom, vertex_mask, vFace_mask = data
                    class_label = None
                nv = torch.nonzero(vertex_mask.int().sum(0), as_tuple=True)[0][-1].cpu().item()+1
                nf = torch.nonzero(vFace_mask.flatten(0, 1).int().sum(0), as_tuple=True)[0][-1].cpu().item()+1
                vertex_geom = vertex_geom[:, :nv, ...]                # b*nv*3
                vv_adj = vv_adj[:, :nv, :nv]                          # b*nv*nv
                vertex_faceBbox = vertex_faceBbox[:, :nv, :nf, ...]   # b*nv*nf*6
                vertex_faceGeom = vertex_faceGeom[:, :nv, :nf, ...]   # b*nv*nf*32*32*3
                vertex_mask = vertex_mask[:, :nv]                     # b*nv
                vFace_mask = vFace_mask[:, :nv, :nf]                  # b*nv*nf
                vv_adj = torch.nn.functional.one_hot(vv_adj, num_classes=2)   # b*n*n*2
                b = vertex_geom.shape[0]

                # Pass through surface VAE to sample latent z
                face_uv = torch.flatten(vertex_faceGeom, start_dim=0, end_dim=2).permute(0, 3, 1, 2)
                face_latent = self.face_vae(face_uv)    # ?*3*4*4
                face_latent = face_latent.unflatten(0, (b, nv, nf)).flatten(-2, -1).permute(0, 1, 2, 4, 3)   # b*nv*nf*16*3

                # rescaled the latent z,  b*nv*nf*54
                vertex_faceInfo = torch.cat(
                    (face_latent.flatten(-2, -1) * self.z_scaled, vertex_faceBbox), dim=-1).clone().detach()
                x_0, e_0 = xe_mask(x=vertex_geom, e=vv_adj, node_mask=vertex_mask)    # b*nv*3, b*nv*nv*2

                total_count += 1

                for idx, step in enumerate([10, 50, 100, 200, 500]):
                    # Evaluate at timestep
                    timesteps = torch.randint(step - 1, step, (b, 1), device=self.device).long()  # [batch, 1]

                    # Add noise
                    noise_data = self.diffusion.add_noise(x_0, vertex_mask, t=timesteps)
                    x_t, y = noise_data['x_t'], self.diffusion.normalize_t(noise_data['t'])  # b*nv*3, b*1

                    # Extract features
                    feat = self.extract_feat(e_0, vertex_mask)
                    x_t_feat = torch.cat((x_t, feat[0]), dim=-1).float()  # b*n*9
                    e_t_feat = torch.cat((e_0, feat[1]), dim=-1).float()  # b*n*n*2
                    y_feat = torch.cat((feat[2], y), dim=-1).float()  # b*12

                    # Predict start
                    pred_noise = self.model(x_t_feat, e_t_feat, vertex_faceInfo, y_feat, vertex_mask, vFace_mask)   # b*n*3

                    if torch.isnan(pred_noise).any() or torch.isinf(pred_noise).any():
                        print("Has nan!!!!")
                        torch.save(pred_noise.detach().cpu(), 'bad_noise.pt')
                        assert False

                    # Loss
                    vertex_mse_loss = torch.nn.functional.mse_loss(pred_noise[vertex_mask],
                                                                   noise_data['noise'][vertex_mask])

                    total_loss[idx] += vertex_mse_loss

            progress_bar.update(1)
        progress_bar.close()

        mse = [loss / total_count for loss in total_loss]
        self.model.train()  # set to train
        wandb.log({"Val-010": mse[0], "Val-050": mse[1], "Val-100": mse[2], "Val-200": mse[3], "Val-500": mse[4]},
                  step=self.iters)

    def save_model(self):
        torch.save(self.model.module.state_dict(), os.path.join(self.save_dir, 'epoch_'+str(self.epoch)+'.pt'))
        return


class EdgeGeomTrainer:

    def __init__(self, args, train_dataset, val_dataset, dataset_info):
        # Initialize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.use_cf = args.cf
        self.z_scaled = args.z_scaled
        self.max_edge = args.max_edge
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.diffusion = DDPM(args.timesteps, self.device)

        # Initialize network

        model = EdgeGeomTransformer(n_layers=6, face_geom_dim=48, edge_geom_dim=12)
        model = nn.DataParallel(model)  # distributed training
        self.model = model.to(self.device).train()

        # Load pretrained surface vae (fast encode version)
        face_vae = AutoencoderKLFastEncode(in_channels=3,
                                           out_channels=3,
                                           down_block_types=('DownEncoderBlock2D', 'DownEncoderBlock2D',
                                                             'DownEncoderBlock2D', 'DownEncoderBlock2D'),
                                           up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D',
                                                           'UpDecoderBlock2D'),
                                           block_out_channels=(128, 256, 512, 512),
                                           layers_per_block=2,
                                           act_fn='silu',
                                           latent_channels=3,
                                           norm_num_groups=32,
                                           sample_size=512,)
        face_vae.load_state_dict(torch.load(args.face_vae), strict=False)
        face_vae = nn.DataParallel(face_vae)  # distributed inference
        self.face_vae = face_vae.to(self.device).eval()

        # Load pretrained edge vae (fast encode version)
        edge_vae = AutoencoderKL1DFastEncode(
            in_channels=3,
            out_channels=3,
            down_block_types=('DownBlock1D', 'DownBlock1D', 'DownBlock1D'),
            up_block_types=('UpBlock1D', 'UpBlock1D', 'UpBlock1D'),
            block_out_channels=(128, 256, 512),
            layers_per_block=2,
            act_fn='silu',
            latent_channels=3,
            norm_num_groups=32,
            sample_size=512)
        edge_vae.load_state_dict(torch.load(args.edge_vae), strict=False)
        edge_vae = nn.DataParallel(edge_vae)  # distributed inference
        self.edge_vae = edge_vae.to(self.device).eval()

        # Initialize optimizer
        self.network_params = list(self.model.parameters())

        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )

        self.scaler = torch.cuda.amp.GradScaler()

        # Initializer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                            shuffle=True,
                                                            batch_size=args.batch_size,
                                                            num_workers=16)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                          shuffle=False,
                                                          batch_size=args.batch_size,
                                                          num_workers=16)
        return

    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train
        for data in self.train_dataloader:
            with torch.cuda.amp.autocast():
                data = [x.to(self.device) for x in data]
                if self.use_cf:
                    edge_ncs, edge_face_ncs, edge_face_bbox, edge_vertex, edge_mask, class_label = data
                else:
                    edge_ncs, edge_face_ncs, edge_face_bbox, edge_vertex, edge_mask = data
                    class_label = None

                ne = torch.nonzero(edge_mask.int().sum(0), as_tuple=True)[0][-1].cpu().item() + 1
                edge_ncs = edge_ncs[:, :ne, ...]               # b*ne*32*3
                edge_face_ncs = edge_face_ncs[:, :ne, ...]     # b*ne*2*32*32*3
                edge_face_bbox = edge_face_bbox[:, :ne, ...]   # b*ne*2*6
                edge_vertex = edge_vertex[:, :ne, ...]         # b*ne*2*3
                edge_mask = edge_mask[:, :ne]                  # b*ne

                b = edge_ncs.shape[0]

                # Pass through surface/edge VAE to sample latent z
                with torch.no_grad():
                    face_uv = edge_face_ncs.flatten(1, 2).flatten(0, 1).permute(0, 3, 1, 2)
                    face_z = self.face_vae(face_uv)
                    face_z = face_z.unflatten(0, (b, -1)).flatten(-2, -1).permute(0, 1, 3, 2).unflatten(1, (ne, 2))   # b*ne*2*16*3

                    edge_u = edge_ncs.flatten(0, 1).permute(0, 2, 1)
                    edge_z = self.edge_vae(edge_u)
                    edge_z = edge_z.unflatten(0, (b, ne)).permute(0, 1, 3, 2)   # b*ne*4*3

                # b*ne*2*(48+6)
                edge_faceInfo = torch.cat((face_z.flatten(-2, -1) * self.z_scaled, edge_face_bbox), dim=-1)
                e_0 = edge_z.flatten(-2, -1) * self.z_scaled       # b*ne*12

                # # Augment the surface position and latent (see https://arxiv.org/abs/2106.15282)
                # conditions = [edgePos, surfPos, surfZ]
                # aug_data = []
                # for data in conditions:
                #     aug_timesteps = torch.randint(0, 15, (bsz,), device=self.device).long()
                #     aug_noise = torch.randn(data.shape).to(self.device)
                #     aug_data.append(self.noise_scheduler.add_noise(data, aug_noise, aug_timesteps))
                # edgePos, surfPos, surfZ = aug_data[0], aug_data[1], aug_data[2]

                # Zero gradient
                self.optimizer.zero_grad()

                # Add noise
                noise_data = self.diffusion.add_noise(e_0, edge_mask)
                e_t, noise, t = noise_data['x_t'], noise_data['noise'], self.diffusion.normalize_t(noise_data['t'])   # b*ne*12, b*ne*12, b*1

                # Predict noise
                pred_noise = self.model(e_t, edge_faceInfo, edge_vertex, edge_mask, t)    # b*ne*12

                if torch.isnan(pred_noise).any() or torch.isinf(pred_noise).any():
                    print("Has nan!!!!")
                    torch.save(pred_noise.detach().cpu(), 'bad_noise.pt')
                    assert False

                assert pred_noise.shape == noise.shape

                # Loss
                loss = torch.nn.functional.mse_loss(pred_noise[edge_mask], noise[edge_mask])

                # Update model
                self.scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0)  # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 20 == 0:
                wandb.log({"Loss-noise": loss},
                          step=self.iters)

            self.iters += 1
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval()  # set to eval
        total_count = 0

        progress_bar = tqdm(total=len(self.val_dataloader))
        progress_bar.set_description(f"Testing")

        total_loss = [0, 0, 0, 0, 0]

        for data in self.val_dataloader:
            with torch.no_grad():
                data = [x.to(self.device) for x in data]
                if self.use_cf:
                    edge_ncs, edge_face_ncs, edge_face_bbox, edge_vertex, edge_mask, class_label = data
                else:
                    edge_ncs, edge_face_ncs, edge_face_bbox, edge_vertex, edge_mask = data
                    class_label = None

                ne = torch.nonzero(edge_mask.int().sum(0), as_tuple=True)[0][-1].cpu().item() + 1
                edge_ncs = edge_ncs[:, :ne, ...]               # b*ne*32*3
                edge_face_ncs = edge_face_ncs[:, :ne, ...]     # b*ne*2*32*32*3
                edge_face_bbox = edge_face_bbox[:, :ne, ...]   # b*ne*2*6
                edge_vertex = edge_vertex[:, :ne, ...]         # b*ne*2*3
                edge_mask = edge_mask[:, :ne]                  # b*ne

                b = edge_ncs.shape[0]

                # Pass through surface/edge VAE to sample latent z
                face_uv = edge_face_ncs.flatten(1, 2).flatten(0, 1).permute(0, 3, 1, 2)
                face_z = self.face_vae(face_uv)
                face_z = face_z.unflatten(0, (b, -1)).flatten(-2, -1).permute(0, 1, 3, 2).unflatten(1, (ne, 2))   # b*ne*2*16*3

                edge_u = edge_ncs.flatten(0, 1).permute(0, 2, 1)
                edge_z = self.edge_vae(edge_u)
                edge_z = edge_z.unflatten(0, (b, ne)).permute(0, 1, 3, 2)   # b*ne*4*3

                # b*ne*2*(48+6)
                edge_faceInfo = torch.cat((face_z.flatten(-2, -1) * self.z_scaled, edge_face_bbox), dim=-1)
                e_0 = edge_z.flatten(-2, -1) * self.z_scaled       # b*ne*12

                total_count += 1

                for idx, step in enumerate([10, 50, 100, 200, 500]):
                    # Evaluate at timestep
                    timesteps = torch.randint(step - 1, step, (b, 1), device=self.device).long()  # [batch, 1]

                    # Add noise
                    noise_data = self.diffusion.add_noise(e_0, edge_mask, t=timesteps)
                    e_t, noise, t = noise_data['x_t'], noise_data['noise'], self.diffusion.normalize_t(
                        noise_data['t'])     # b*ne*12, b*ne*18, b*1

                    # Predict noise
                    pred_noise = self.model(e_t, edge_faceInfo, edge_vertex, edge_mask, t)  # b*ne*12

                    if torch.isnan(pred_noise).any() or torch.isinf(pred_noise).any():
                        print("Has nan!!!!")
                        torch.save(pred_noise.detach().cpu(), 'bad_noise.pt')
                        assert False

                    assert pred_noise.shape == noise.shape

                    # Loss
                    loss = torch.nn.functional.mse_loss(pred_noise[edge_mask], noise[edge_mask])

                    total_loss[idx] += loss

            progress_bar.update(1)
        progress_bar.close()

        mse = [loss / total_count for loss in total_loss]
        self.model.train()  # set to train
        wandb.log({"Val-010": mse[0], "Val-050": mse[1], "Val-100": mse[2], "Val-200": mse[3], "Val-500": mse[4]},
                  step=self.iters)

    def save_model(self):
        torch.save(self.model.module.state_dict(), os.path.join(self.save_dir, 'epoch_' + str(self.epoch) + '.pt'))
