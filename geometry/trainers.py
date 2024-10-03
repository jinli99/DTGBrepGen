from tqdm import tqdm
import wandb
import os
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDPMScheduler
from model import (AutoencoderKL1D, FaceBboxTransformer, AutoencoderKLFastEncode, AutoencoderKL1DFastEncode,
                   FaceGeomTransformer, VertGeomTransformer, EdgeGeomTransformer)
from geometry.diffusion import DDPM
from utils import xe_mask, make_mask
from geometry.dataFeature import GraphFeatures


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
        n_layers = 8
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
                    face_bbox, fef_adj, node_mask, class_label = data   # b*n*32*32*3, b*n*6, b*n*n, b*n, b*1
                else:
                    face_bbox, fef_adj, node_mask = data   # b*n*32*32*3, b*n*6, b*n*n, b*1
                    class_label = None
                num_faces = torch.nonzero(node_mask.int().sum(0), as_tuple=True)[0][-1].cpu().item()+1
                face_bbox = face_bbox[:, :num_faces, :]        # b*n*6
                fef_adj = fef_adj[:, :num_faces, :num_faces]   # b*n*n
                node_mask = node_mask[:, :num_faces]           # b*n
                e_0 = torch.nn.functional.one_hot(fef_adj, num_classes=self.edge_classes)   # b*n*n*m

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
                    face_bbox, fef_adj, node_mask, class_label = data   # b*n*32*32*3, b*n*6, b*n*n, b*1, b*1
                else:
                    face_bbox, fef_adj, node_mask = data   # b*n*32*32*3, b*n*6, b*n*n, b*1
                    class_label = None
                num_faces = torch.nonzero(node_mask.int().sum(0), as_tuple=True)[0][-1].cpu().item()+1
                face_bbox = face_bbox[:, :num_faces, :]        # b*n*6
                fef_adj = fef_adj[:, :num_faces, :num_faces]   # b*n*n
                node_mask = node_mask[:, :num_faces]           # b*n
                e_0 = torch.nn.functional.one_hot(fef_adj, num_classes=self.edge_classes)   # b*n*n*m
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


class VertGeomTrainer:
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
        n_layers = 8
        hidden_mlp_dims = {'x': 256, 'e': 128, 'y': 128}
        # The dimensions should satisfy dx % n_head == 0
        hidden_dims = {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128}
        input_dims, output_dims = dataset_info['input_dims'], dataset_info['output_dims']
        input_dims['x'] += 3
        output_dims['x'] += 3
        model = VertGeomTransformer(n_layers=n_layers, input_dims=input_dims, hidden_mlp_dims=hidden_mlp_dims,
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
                    # b*nv*3, b*nv*nv, b*nv*nf*6, b*nv, b*nv*nf, b*1
                    vert_geom, vv_adj, vertFace_bbox, vert_mask, vertFace_mask, class_label = data
                else:
                    # b*nv*3, b*nv*nv, b*nv*nf*6, b*nv, b*nv*nf, b*1
                    vert_geom, vv_adj, vertFace_bbox, vert_mask, vertFace_mask = data
                    class_label = None
                nv = torch.nonzero(vert_mask.int().sum(0), as_tuple=True)[0][-1].cpu().item()+1
                vf = torch.nonzero(vertFace_mask.flatten(0, 1).int().sum(0), as_tuple=True)[0][-1].cpu().item()+1
                vert_geom = vert_geom[:, :nv, ...]                # b*nv*3
                vv_adj = vv_adj[:, :nv, :nv]                          # b*nv*nv
                vertFace_bbox = vertFace_bbox[:, :nv, :vf, ...]   # b*nv*vf*6
                vert_mask = vert_mask[:, :nv]                     # b*nv
                vertFace_mask = vertFace_mask[:, :nv, :vf]                  # b*nv*vf
                vv_adj = torch.nn.functional.one_hot(vv_adj, num_classes=2)   # b*nv*nv*2
                b = vert_geom.shape[0]

                # # Augment the surface position (see https://arxiv.org/abs/2106.15282)
                # conditions = [surfPos]
                # aug_data = []
                # for data in conditions:
                #     aug_timesteps = torch.randint(0, 15, (bsz,), device=self.device).long()
                #     aug_noise = torch.randn(data.shape).to(self.device)
                #     aug_data.append(self.noise_scheduler.add_noise(data, aug_noise, aug_timesteps))
                # surfPos = aug_data[0]

                # rescaled the latent z,  b*nv*nf*6
                vertFace_info = vertFace_bbox.clone().detach()
                x_0, e_0 = xe_mask(x=vert_geom, e=vv_adj, node_mask=vert_mask)    # b*nv*3, b*nv*nv*2

                self.optimizer.zero_grad()  # zero gradient

                # Add noise
                noise_data = self.diffusion.add_noise(x_0, vert_mask)
                x_t, y = noise_data['x_t'], self.diffusion.normalize_t(noise_data['t'])  # b*nv*3, b*1

                # Extract features
                with torch.cuda.amp.autocast(enabled=False):
                    feat = self.extract_feat(e_0, vert_mask)
                x_t_feat = torch.cat((x_t, feat[0]), dim=-1).float()   # b*n*9
                e_t_feat = torch.cat((e_0, feat[1]), dim=-1).float()   # b*n*n*2
                y_feat = torch.cat((feat[2], y), dim=-1).float()       # b*12

                # Predict start
                pred_noise = self.model(x_t_feat, e_t_feat, vertFace_info, y_feat, vert_mask, vertFace_mask)   # b*n*3

                if torch.isnan(pred_noise).any() or torch.isinf(pred_noise).any():
                    print("Has nan!!!!")
                    torch.save(pred_noise.detach().cpu(), 'bad_noise.pt')
                    assert False

                # Loss
                vert_mse_loss = torch.nn.functional.mse_loss(pred_noise[vert_mask], noise_data['noise'][vert_mask])

                # Update model
                self.scaler.scale(vert_mse_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0)  # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 20 == 0:
                wandb.log({"Loss-noise": vert_mse_loss}, step=self.iters)

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
                    # b*nv*3, b*nv*nv, b*nv*nf*6, b*nv, b*nv*nf, b*1
                    vert_geom, vv_adj, vertFace_bbox, vert_mask, vertFace_mask, class_label = data
                else:
                    # b*nv*3, b*nv*nv, b*nv*nf*6, b*nv, b*nv*nf, b*1
                    vert_geom, vv_adj, vertFace_bbox, vert_mask, vertFace_mask = data
                    class_label = None
                nv = torch.nonzero(vert_mask.int().sum(0), as_tuple=True)[0][-1].cpu().item()+1
                vf = torch.nonzero(vertFace_mask.flatten(0, 1).int().sum(0), as_tuple=True)[0][-1].cpu().item()+1
                vert_geom = vert_geom[:, :nv, ...]                    # b*nv*3
                vv_adj = vv_adj[:, :nv, :nv]                          # b*nv*nv
                vertFace_bbox = vertFace_bbox[:, :nv, :vf, ...]       # b*nv*nf*6
                vert_mask = vert_mask[:, :nv]                         # b*nv
                vertFace_mask = vertFace_mask[:, :nv, :vf]                    # b*nv*nf
                vv_adj = torch.nn.functional.one_hot(vv_adj, num_classes=2)   # b*n*n*2
                b = vert_geom.shape[0]

                # rescaled the latent z,  b*nv*nf*6
                vertFace_info = vertFace_bbox.clone().detach()
                x_0, e_0 = xe_mask(x=vert_geom, e=vv_adj, node_mask=vert_mask)    # b*nv*3, b*nv*nv*2

                total_count += 1

                for idx, step in enumerate([10, 50, 100, 200, 500]):
                    # Evaluate at timestep
                    timesteps = torch.randint(step - 1, step, (b, 1), device=self.device).long()  # [batch, 1]

                    # Add noise
                    noise_data = self.diffusion.add_noise(x_0, vert_mask, t=timesteps)
                    x_t, y = noise_data['x_t'], self.diffusion.normalize_t(noise_data['t'])  # b*nv*3, b*1

                    # Extract features
                    feat = self.extract_feat(e_0, vert_mask)
                    x_t_feat = torch.cat((x_t, feat[0]), dim=-1).float()  # b*n*9
                    e_t_feat = torch.cat((e_0, feat[1]), dim=-1).float()  # b*n*n*2
                    y_feat = torch.cat((feat[2], y), dim=-1).float()  # b*12

                    # Predict start
                    pred_noise = self.model(x_t_feat, e_t_feat, vertFace_info, y_feat, vert_mask, vertFace_mask)   # b*n*3

                    if torch.isnan(pred_noise).any() or torch.isinf(pred_noise).any():
                        print("Has nan!!!!")
                        torch.save(pred_noise.detach().cpu(), 'bad_noise.pt')
                        assert False

                    # Loss
                    vert_mse_loss = torch.nn.functional.mse_loss(pred_noise[vert_mask], noise_data['noise'][vert_mask])

                    total_loss[idx] += vert_mse_loss

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

        # Initialize network
        model = EdgeGeomTransformer(n_layers=8, edge_geom_dim=12)
        model = nn.DataParallel(model)  # distributed training
        self.model = model.to(self.device).train()

        # Initialize diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='linear',
            prediction_type='epsilon',
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,
        )

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
                    edge_geom, edgeFace_bbox, edgeVert_geom, edge_mask, class_label = data
                else:
                    edge_geom, edgeFace_bbox, edgeVert_geom, edge_mask = data
                    class_label = None

                ne = edge_mask.max().cpu().item() + 1
                edge_geom = edge_geom[:, :ne, ...]             # b*ne*12
                edgeFace_bbox = edgeFace_bbox[:, :ne, ...]     # b*ne*2*6
                edgeVert_geom = edgeVert_geom[:, :ne, ...]     # b*ne*2*3
                edge_mask = make_mask(edge_mask, ne)           # b*ne

                # b*ne*2*6
                x_0 = edge_geom * self.z_scaled                # b*ne*12
                x_0, _ = xe_mask(x=x_0, node_mask=edge_mask)

                # Zero gradient
                self.optimizer.zero_grad()

                # Add noise
                t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (x_0.shape[0],),
                                  device=self.device).long()  # b
                noise = torch.randn(x_0.shape).to(self.device)
                x_t = self.noise_scheduler.add_noise(x_0, noise, t)

                # Predict noise
                pred_noise = self.model(x_t, edgeFace_bbox, edgeVert_geom, edge_mask, t.unsqueeze(-1))    # b*ne*12

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
                    edge_geom, edgeFace_bbox, edgeVert_geom, edge_mask, class_label = data
                else:
                    edge_geom, edgeFace_bbox, edgeVert_geom, edge_mask = data
                    class_label = None

                ne = edge_mask.max().cpu().item() + 1
                edge_geom = edge_geom[:, :ne, ...]             # b*ne*12
                edgeFace_bbox = edgeFace_bbox[:, :ne, ...]     # b*ne*2*6
                edgeVert_geom = edgeVert_geom[:, :ne, ...]     # b*ne*2*3
                edge_mask = make_mask(edge_mask, ne)           # b*ne

                # b*ne*2*6
                x_0 = edge_geom * self.z_scaled                # b*ne*12
                x_0, _ = xe_mask(x=x_0, node_mask=edge_mask)

                total_count += 1

                for idx, step in enumerate([10, 50, 100, 200, 500]):
                    # Evaluate at timestep
                    t = torch.randint(step - 1, step, (x_0.shape[0],), device=self.device).long()  # b
                    noise = torch.randn(x_0.shape).to(self.device)
                    x_t = self.noise_scheduler.add_noise(x_0, noise, t)

                    # Predict noise
                    pred_noise = self.model(x_t, edgeFace_bbox, edgeVert_geom, edge_mask, t.unsqueeze(-1))  # b*ne*12

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


class FaceGeomTrainer:
    def __init__(self, args, train_dataset, val_dataset, dataset_info):
        # Initialize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.use_cf = args.cf
        self.z_scaled = args.z_scaled
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.edge_classes = args.edge_classes

        # Initialize network
        model = FaceGeomTransformer(n_layers=8, face_geom_dim=48)
        model = nn.DataParallel(model)    # distributed training
        self.model = model.to(self.device).train()

        # Initialize diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='linear',
            prediction_type='epsilon',
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,
        )

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
                    # b*nf*48, b*nf*6, b*nf*fv*3, b*nf*fe*12, b*1, b*nf*1, b*fe*1, b*1
                    face_geom, face_bbox, faceVert_geom, faceEdge_geom, face_mask, faceVert_mask, faceEdge_mask, class_label = data
                else:
                    face_geom, face_bbox, faceVert_geom, faceEdge_geom, face_mask, faceVert_mask, faceEdge_mask = data
                    class_label = None
                nf = face_mask.max()
                fv = faceVert_mask.max()
                fe = faceEdge_mask.max()
                face_geom = face_geom[:, :nf, ...]                   # b*nf*48
                face_bbox = face_bbox[:, :nf, :]                     # b*nf*6
                faceVert_geom = faceVert_geom[:, :nf, :fv, ...]      # b*nf*fv*3
                faceVert_mask = faceVert_mask[:, :nf, ...]           # b*nf*1
                faceEdge_geom = faceEdge_geom[:, :nf, :fe, ...]      # b*nf*fe*12
                faceEdge_mask = faceEdge_mask[:, :nf, ...]           # b*nf*1

                face_mask = make_mask(face_mask, nf)                 # b*nf
                faceVert_mask = make_mask(faceVert_mask, fv)         # b*nf*fv
                faceEdge_mask = make_mask(faceEdge_mask, fe)         # b*nf*fe

                x_0 = face_geom * self.z_scaled                      # b*nf*48
                x_0, _ = xe_mask(x=x_0, node_mask=face_mask)

                # # Augment the surface position and latent (see https://arxiv.org/abs/2106.15282)
                # conditions = [face_bbox, faceVert_geom, faceEdge_geom]
                # aug_data = []
                # for cond in conditions:
                #     aug_timesteps = torch.randint(0, 15, (x_0.shape[0],), device=self.device).long()
                #     aug_noise = torch.randn(cond.shape).to(self.device)
                #     aug_data.append(self.noise_scheduler.add_noise(cond, aug_noise, aug_timesteps))
                # face_bbox, faceVert_geom, faceEdge_geom = aug_data[0], aug_data[1], aug_data[2]

                self.optimizer.zero_grad()  # zero gradient

                # Add noise
                t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (x_0.shape[0],),
                                  device=self.device).long()              # b
                noise = torch.randn(x_0.shape).to(self.device)            # b*nf*48
                x_t = self.noise_scheduler.add_noise(x_0, noise, t)

                # Predict noise
                pred_noise = self.model(x_t, face_bbox, faceVert_geom, faceEdge_geom, face_mask, faceVert_mask,
                                        faceEdge_mask, t.unsqueeze(-1))   # b*n*48

                if torch.isnan(pred_noise).any() or torch.isinf(pred_noise).any():
                    print("Has nan!!!!")
                    torch.save(pred_noise.detach().cpu(), 'bad_noise.pt')
                    assert False

                # Loss
                face_mse_loss = torch.nn.functional.mse_loss(pred_noise[face_mask], noise[face_mask])

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
                    # b*nf*48, b*nf*6, b*nf*fv*3, b*nf*fe*12, b*1, b*nf*1, b*fe*1, b*1
                    face_geom, face_bbox, faceVert_geom, faceEdge_geom, face_mask, faceVert_mask, faceEdge_mask, class_label = data
                else:
                    face_geom, face_bbox, faceVert_geom, faceEdge_geom, face_mask, faceVert_mask, faceEdge_mask = data
                    class_label = None
                nf = face_mask.max()
                fv = faceVert_mask.max()
                fe = faceEdge_mask.max()
                face_geom = face_geom[:, :nf, ...]                   # b*nf*48
                face_bbox = face_bbox[:, :nf, :]                     # b*nf*6
                faceVert_geom = faceVert_geom[:, :nf, :fv, ...]      # b*nf*fv*3
                faceVert_mask = faceVert_mask[:, :nf, ...]           # b*nf*1
                faceEdge_geom = faceEdge_geom[:, :nf, :fe, ...]      # b*nf*fe*12
                faceEdge_mask = faceEdge_mask[:, :nf, ...]           # b*nf*1

                face_mask = make_mask(face_mask, nf)                 # b*nf
                faceVert_mask = make_mask(faceVert_mask, fv)         # b*nf*fv
                faceEdge_mask = make_mask(faceEdge_mask, fe)         # b*nf*fe

                x_0 = face_geom * self.z_scaled                      # b*nf*48
                x_0, _ = xe_mask(x=x_0, node_mask=face_mask)

            total_count += 1

            for idx, step in enumerate([10, 50, 100, 200, 500]):
                # Evaluate at timestep
                t = torch.randint(step - 1, step, (x_0.shape[0], ), device=self.device).long()  # b
                noise = torch.randn(x_0.shape).to(self.device)
                x_t = self.noise_scheduler.add_noise(x_0, noise, t)

                # Predict noise
                pred_noise = self.model(x_t, face_bbox, faceVert_geom, faceEdge_geom, face_mask, faceVert_mask,
                                        faceEdge_mask, t.unsqueeze(-1))   # b*n*48

                if torch.isnan(pred_noise).any() or torch.isinf(pred_noise).any():
                    print("Has nan!!!!")
                    torch.save(pred_noise.detach().cpu(), 'bad_noise.pt')
                    assert False

                # Loss
                face_mse_loss = torch.nn.functional.mse_loss(pred_noise[face_mask], noise[face_mask])

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
