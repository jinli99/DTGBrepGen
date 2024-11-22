from tqdm import tqdm
import wandb
import os
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDPMScheduler
from model import (AutoencoderKL1D, FaceBboxTransformer, AutoencoderKLFastEncode, AutoencoderKL1DFastEncode,
                   FaceGeomTransformer, VertGeomTransformer, EdgeGeomTransformer)
from utils import xe_mask, make_mask


class EdgeGeomTrainer:

    def __init__(self, args, train_dataset, val_dataset):
        # Initialize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.use_cf = args.use_cf
        self.aug = args.data_aug
        self.use_pc = args.use_pc
        self.z_scaled = args.z_scaled
        self.max_edge = args.max_edge
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize network
        model = EdgeGeomTransformer(n_layers=args.EdgeGeomModel['n_layers'],
                                    edge_geom_dim=args.EdgeGeomModel['edge_geom_dim'],
                                    d_model=args.EdgeGeomModel['d_model'],
                                    nhead=args.EdgeGeomModel['nhead'],
                                    use_cf=self.use_cf)
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
                    edge_geom, edgeFace_bbox, edgeVert_geom, edge_mask, class_label = data
                else:
                    edge_geom, edgeFace_bbox, edgeVert_geom, edge_mask = data
                    class_label = None

                ne = edge_mask.max().cpu().item()
                edge_geom = edge_geom[:, :ne, ...]             # b*ne*32*3
                edgeFace_bbox = edgeFace_bbox[:, :ne, ...]     # b*ne*2*6
                edgeVert_geom = edgeVert_geom[:, :ne, ...]     # b*ne*2*3
                edge_mask = make_mask(edge_mask, ne)           # b*ne

                # Pass through edge VAE to sample latent z
                with torch.no_grad():
                    edge_u = edge_geom.flatten(0, 1).permute(0, 2, 1)
                    edge_z = self.edge_vae(edge_u)
                    edge_z = edge_z.unflatten(0, (edge_geom.shape[0], ne)).permute(0, 1, 3, 2)   # b*ne*4*3

                x_0 = edge_z.flatten(-2, -1) * self.z_scaled                # b*ne*12
                x_0, _ = xe_mask(x=x_0, node_mask=edge_mask)

                # Zero gradient
                self.optimizer.zero_grad()

                # Add noise
                t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (x_0.shape[0],),
                                  device=self.device).long()  # b
                noise = torch.randn(x_0.shape).to(self.device)
                x_t = self.noise_scheduler.add_noise(x_0, noise, t)

                # Predict noise
                pred_noise = self.model(x_t, edgeFace_bbox, edgeVert_geom, edge_mask, class_label, t.unsqueeze(-1))    # b*ne*12

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
                # print("*****", loss.item())

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

                ne = edge_mask.max().cpu().item()
                edge_geom = edge_geom[:, :ne, ...]             # b*ne*32*3
                edgeFace_bbox = edgeFace_bbox[:, :ne, ...]     # b*ne*2*6
                edgeVert_geom = edgeVert_geom[:, :ne, ...]     # b*ne*2*3
                edge_mask = make_mask(edge_mask, ne)           # b*ne

                with torch.no_grad():
                    edge_u = edge_geom.flatten(0, 1).permute(0, 2, 1)
                    edge_z = self.edge_vae(edge_u)
                    edge_z = edge_z.unflatten(0, (edge_geom.shape[0], ne)).permute(0, 1, 3, 2)   # b*ne*4*3

                # b*ne*2*6
                x_0 = edge_z.flatten(-2, -1) * self.z_scaled                                     # b*ne*12
                x_0, _ = xe_mask(x=x_0, node_mask=edge_mask)

                total_count += 1

                for idx, step in enumerate([10, 50, 100, 200, 500]):
                    # Evaluate at timestep
                    t = torch.randint(step - 1, step, (x_0.shape[0],), device=self.device).long()  # b
                    noise = torch.randn(x_0.shape).to(self.device)
                    x_t = self.noise_scheduler.add_noise(x_0, noise, t)

                    # Predict noise
                    pred_noise = self.model(x_t, edgeFace_bbox, edgeVert_geom, edge_mask, class_label, t.unsqueeze(-1))  # b*ne*12

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
    def __init__(self, args, train_dataset, val_dataset):
        # Initialize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.use_cf = args.use_cf
        self.aug = args.data_aug
        self.use_pc = args.use_pc
        self.z_scaled = args.z_scaled
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.edge_classes = args.edge_classes

        # Initialize network
        model = FaceGeomTransformer(n_layers=args.FaceGeomModel['n_layers'],
                                    edge_geom_dim=args.FaceGeomModel['face_geom_dim'],
                                    d_model=args.FaceGeomModel['d_model'],
                                    nhead=args.FaceGeomModel['nhead'],
                                    use_cf=self.use_cf)
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

                self.optimizer.zero_grad()  # zero gradient

                # Add noise
                t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (x_0.shape[0],),
                                  device=self.device).long()              # b
                noise = torch.randn(x_0.shape).to(self.device)            # b*nf*48
                x_t = self.noise_scheduler.add_noise(x_0, noise, t)

                # Predict noise
                pred_noise = self.model(x_t, face_bbox, faceVert_geom, faceEdge_geom, face_mask, faceVert_mask,
                                        faceEdge_mask, class_label, t.unsqueeze(-1))   # b*n*48

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
                print("******", face_mse_loss.item())


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
                                        faceEdge_mask, class_label, t.unsqueeze(-1))   # b*n*48

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
