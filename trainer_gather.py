from tqdm import tqdm
import wandb
import os
import torch
import torch.nn as nn
import pickle
from diffusers import AutoencoderKL
from model import AutoencoderKL1D, GraphTransformer, AutoencoderKLFastEncode, AutoencoderKL1DFastEncode, EdgeTransformer
from diffusion import GraphDiffusion, EdgeDiffusion
from utils import custom_collate_fn, pad_and_stack, edge_reshape_mask, assert_weak_one_hot
from dataFeature import GraphFeatures


class SurfVAETrainer:
    """ Surface VAE Trainer """

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

        # Load pretrained surface vae (fast encode version)
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
        return

    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()
        loss_fn = nn.MSELoss()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train
        for surf_uv in self.train_dataloader:
            with torch.cuda.amp.autocast():
                surf_uv = surf_uv.to(self.device).permute(0, 3, 1, 2)
                self.optimizer.zero_grad()  # zero gradient

                # Pass through VAE
                posterior = self.model.encode(surf_uv).latent_dist
                z = posterior.sample()
                dec = self.model.decode(z).sample

                # Loss functions
                kl_loss = posterior.kl().mean()
                mse_loss = loss_fn(dec, surf_uv)
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
            for surf_uv in self.val_dataloader:
                surf_uv = surf_uv.to(self.device).permute(0, 3, 1, 2)

                posterior = self.model.encode(surf_uv).latent_dist
                z = posterior.sample()
                dec = self.model.decode(z).sample

                loss = mse_loss(dec, surf_uv).mean((1, 2, 3)).sum().item()
                total_loss += loss
                total_count += len(surf_uv)

        mse = total_loss / total_count
        self.model.train()  # set to train
        wandb.log({"Val-mse": mse}, step=self.iters)
        return mse

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'epoch_' + str(self.epoch) + '.pt'))
        return


class EdgeVAETrainer:
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

        # Load pretrained surface vae (fast encode version)
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
        return

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


class SurfDiffTrainer:
    def __init__(self, args, train_dataset, val_dataset, dataset_info):
        # Initialize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.use_cf = args.cf
        self.z_scaled = args.z_scaled
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion = GraphDiffusion(edge_classes=args.edge_classes,
                                        edge_marginals=dataset_info['marginal'],
                                        device=self.device)
        self.extract_feat = GraphFeatures(args.extract_type, args.max_face)
        self.edge_classes = args.edge_classes
        self.edge_ce_lambda = 0.001

        # Load pretrained surface vae (fast encode version)
        surf_vae = AutoencoderKLFastEncode(in_channels=3,
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
        surf_vae.load_state_dict(torch.load(args.surfvae), strict=False)
        surf_vae = nn.DataParallel(surf_vae)    # distributed inference
        self.surf_vae = surf_vae.to(self.device).eval()

        # Initialize network
        n_layers = 5
        hidden_mlp_dims = {'X': 256, 'E': 128, 'Y': 128}
        # The dimensions should satisfy dx % n_head == 0
        hidden_dims = {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128}
        input_dims, output_dims = dataset_info['input_dims'], dataset_info['output_dims']
        example_data = train_dataset[0]
        surfPos, surfPnt = example_data[0], example_data[1]   # num_faces*6, num_faces*32*32*3
        surfPnt = surfPnt.unsqueeze(0).to(self.device)
        with torch.no_grad():
            surf_uv = surfPnt.flatten(0, 1).permute(0, 3, 1, 2)
            surf_z = self.surf_vae(surf_uv)
            surf_z = surf_z.unflatten(0, (1, -1)).flatten(-2, -1).permute(0, 1, 3, 2)
        input_dims['X'] += surfPos.shape[-1] + surf_z.shape[-1] * surf_z.shape[-2]
        output_dims['X'] += surfPos.shape[-1] + surf_z.shape[-1] * surf_z.shape[-2]
        model = GraphTransformer(n_layers=n_layers, input_dims=input_dims, hidden_mlp_dims=hidden_mlp_dims,
                                 hidden_dims=hidden_dims, output_dims=output_dims,
                                 act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU())
        model = nn.DataParallel(model)    # distributed training
        self.model = model.to(self.device).train()

        hyper_params = {'edge_classes': args.edge_classes, 'edge_marginals': dataset_info['marginal'],
                        'node_distribution': dataset_info['node_distribution'], 'extract_type': args.extract_type,
                        'bbox_scaled': train_dataset.bbox_scaled,
                        'diff_dim': surfPos.shape[-1] + surf_z.shape[-1] * surf_z.shape[-2],
                        'input_dims': input_dims, 'output_dims': output_dims, 'hidden_dims': hidden_dims,
                        'hidden_mlp_dims': hidden_mlp_dims, 'n_layers': n_layers}
        with open(os.path.join(args.save_dir, 'hyper_params.pkl'), 'wb') as f:
            pickle.dump(hyper_params, f)

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
                                                            collate_fn=custom_collate_fn,
                                                            num_workers=16)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                          shuffle=False,
                                                          batch_size=args.batch_size,
                                                          collate_fn=custom_collate_fn,
                                                          num_workers=16)

    def train_one_epoch(self):
        """ Train the model for one epoch """
        self.model.train()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train
        for data in self.train_dataloader:   # [(num_faces*6, num_faces*32*32*3, num_faces*num_faces, 1)...]
            with torch.cuda.amp.autocast():
                if self.use_cf:
                    surfPos, surfPnt, ff_edges, class_label = [list(tup) for tup in zip(*data)]
                    surfPos = [x.to(self.device) for x in surfPos]
                    surfPnt = [x.to(self.device) for x in surfPnt]
                    ff_edges = [x.to(self.device) for x in ff_edges]
                    class_label = torch.tensor(class_label).to(self.device)
                else:
                    surfPos, surfPnt, ff_edges = [list(tup) for tup in zip(*data)]
                    surfPos = [x.to(self.device) for x in surfPos]
                    surfPnt = [x.to(self.device) for x in surfPnt]
                    ff_edges = [x.to(self.device) for x in ff_edges]
                    class_label = None

                surfPos, node_mask = pad_and_stack(surfPos)   # b*n*6, b*n
                surfPnt, _ = pad_and_stack(surfPnt)   # b*n*32*32*3
                bsz, num_faces = surfPnt.shape[0], surfPnt.shape[1]
                ff_edges_pad = torch.zeros((bsz, num_faces, num_faces), device=self.device, dtype=ff_edges[0].dtype)   # b*n*n
                for i in range(bsz):
                    m = ff_edges[i].shape[0]
                    ff_edges_pad[i, :m, :m] = ff_edges[i]
                e_0 = torch.nn.functional.one_hot(ff_edges_pad, num_classes=self.edge_classes)   # b*n*n*m

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
                    surf_uv = surfPnt.flatten(0, 1).permute(0, 3, 1, 2)
                    surf_latent = self.surf_vae(surf_uv)
                    surf_latent = surf_latent.unflatten(0, (bsz, -1)).flatten(-2, -1).permute(0, 1, 3, 2)

                x = surf_latent.flatten(-2, -1) * self.z_scaled   # rescaled the latent z  # b*n*48
                x_0 = torch.cat((x, surfPos), dim=-1)  # b*n*54

                self.optimizer.zero_grad()  # zero gradient

                # Add noise
                noise_data = self.diffusion.add_noise(x_0, e_0, node_mask)
                x_t, e_t, y = noise_data['x_t'], noise_data['e_t'], noise_data['y']  # b*n*54, b*n*n*m, b*1

                # Extract features
                feat = self.extract_feat(e_t, node_mask)
                x_t_feat = torch.cat((x_t, feat[0]), dim=-1).float()   # b*n*60
                e_t_feat = torch.cat((e_t, feat[1]), dim=-1).float()   # b*n*n*m
                y_feat = torch.cat((feat[2], y), dim=-1).float()       # b*12

                # Predict start
                x_pred, e_pred, y_pred = self.model(x_t_feat, e_t_feat, y_feat, node_mask)   # b*n*54, b*n*n*m, b*0

                # Loss
                true_logits = self.diffusion.q_posterior_logits(e_0, e_t, noise_data['t'])     # b*n*n*m
                pred_logits = self.diffusion.q_posterior_logits(e_pred, e_t, noise_data['t'])  # b*n*n*m
                edge_vb_loss = self.vb_mask_loss(true_logits, pred_logits, mask=node_mask)
                edge_ce_loss = self.ce_mask_loss(e_0, e_pred, node_mask)
                face_mse_loss = torch.nn.functional.mse_loss(x_pred[node_mask], noise_data['noise'][node_mask])
                # [1e-4, 3e-4], [1, 1.3], [0.5, 1.1]
                total_loss = 1000*edge_vb_loss + 0.2*edge_ce_loss + 0.3*face_mse_loss
                # print("Total Loss:", total_loss.item(), "Edge Vb Loss:", edge_vb_loss.item(),  "Edge CE Loss:", edge_ce_loss.item(), "Face MSE Loss:", face_mse_loss.item())

                # Update model
                self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0)  # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 20 == 0:
                wandb.log({"Total Loss": total_loss,
                           "Edge Vb Loss": edge_vb_loss,
                           "Edge CE Loss": edge_ce_loss,
                           "Face MSE Loss": face_mse_loss}, step=self.iters)

            self.iters += 1
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1
        return

    def save_model(self):
        torch.save(self.model.module.state_dict(), os.path.join(self.save_dir, 'epoch_'+str(self.epoch)+'.pt'))
        return

    @staticmethod
    def vb_mask_loss(q, p, mask):

        p_filtered = edge_reshape_mask(p, mask)
        q_filtered = edge_reshape_mask(q, mask)

        # Compute KL divergence (KL(q || p))
        kl_div = (torch.softmax(q_filtered+1e-6, dim=-1) * (
            torch.log_softmax(q_filtered+1e-6, dim=-1)
            - torch.log_softmax(p_filtered+1e-6, dim=-1)
        )).sum(-1).mean()

        return kl_div

    @staticmethod
    def ce_mask_loss(q, p, mask):   # b*n*n*m, b*n*n*m, b*n

        p_filtered = edge_reshape_mask(p, mask)
        q_filtered = edge_reshape_mask(q, mask)

        assert_weak_one_hot(q_filtered)
        q_filtered = torch.argmax(q_filtered, dim=-1)   # (b, )

        return torch.nn.functional.cross_entropy(p_filtered, q_filtered)


class EdgeDiffTrainer:

    def __init__(self, args, train_dataset, val_dataset, dataset_info):
        # Initialize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.use_cf = args.cf
        self.z_scaled = args.z_scaled
        self.max_edge = args.max_edge
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.diffusion = EdgeDiffusion(self.device)

        # Initialize network

        model = EdgeTransformer(n_layers=12, surf_geom_dim=48, edge_geom_dim=12)
        model = nn.DataParallel(model)  # distributed training
        self.model = model.to(self.device).train()

        # Load pretrained surface vae (fast encode version)
        surf_vae = AutoencoderKLFastEncode(in_channels=3,
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
        surf_vae.load_state_dict(torch.load(args.surfvae), strict=False)
        surf_vae = nn.DataParallel(surf_vae)  # distributed inference
        self.surf_vae = surf_vae.to(self.device).eval()

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
        edge_vae.load_state_dict(torch.load(args.edgevae), strict=False)
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
                                                            collate_fn=custom_collate_fn,
                                                            num_workers=16)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                          shuffle=False,
                                                          batch_size=args.batch_size,
                                                          collate_fn=custom_collate_fn,
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
                if self.use_cf:
                    edgePnt, edgePos, edge_surfPnt, edge_surfPos, class_label = [list(tup) for tup in zip(*data)]
                    edgePnt = [x.to(self.device) for x in edgePnt]              # [ne*32*3,...]
                    edgePos = [x.to(self.device) for x in edgePos]              # [ne*6,...]
                    edge_surfPnt = [x.to(self.device) for x in edge_surfPnt]    # [ne*2*32*32*3,...]
                    edge_surfPos = [x.to(self.device) for x in edge_surfPos]    # [ne*2*6,...]
                    class_label = torch.tensor(class_label).to(self.device)
                else:
                    edgePnt, edgePos, edge_surfPnt, edge_surfPos = [list(tup) for tup in zip(*data)]
                    edgePnt = [x.to(self.device) for x in edgePnt]
                    edgePos = [x.to(self.device) for x in edgePos]
                    edge_surfPnt = [x.to(self.device) for x in edge_surfPnt]
                    edge_surfPos = [x.to(self.device) for x in edge_surfPos]
                    class_label = None

                edgePnt, edge_mask = pad_and_stack(edgePnt)      # b*ne*32*3, b*ne
                edgePos, _ = pad_and_stack(edgePos)              # b*ne*6
                edge_surfPnt, _ = pad_and_stack(edge_surfPnt)    # b*ne*2*32*32*3
                edge_surfPos, _ = pad_and_stack(edge_surfPos)    # b*ne*2*6

                b, ne = edgePnt.shape[0], edgePnt.shape[1]

                # Pass through surface/edge VAE to sample latent z
                with torch.no_grad():
                    surf_uv = edge_surfPnt.flatten(1, 2).flatten(0, 1).permute(0, 3, 1, 2)
                    surf_z = self.surf_vae(surf_uv)
                    surf_z = surf_z.unflatten(0, (b, -1)).flatten(-2, -1).permute(0, 1, 3, 2).unflatten(1, (ne, 2))   # b*ne*2*16*3

                    edge_u = edgePnt.flatten(0, 1).permute(0, 2, 1)
                    edge_z = self.edge_vae(edge_u)
                    edge_z = edge_z.unflatten(0, (b, ne)).permute(0, 1, 3, 2)   # b*ne*4*3

                edge_surfInfo = torch.cat((surf_z.flatten(-2, -1) * self.z_scaled, edge_surfPos), dim=-1)  # b*ne*2*(48+6)
                e_0 = torch.cat((edge_z.flatten(-2, -1) * self.z_scaled, edgePos), dim=-1)       # b*ne*(12+6)

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
                e_t, noise, t = noise_data['e_t'], noise_data['noise'], self.diffusion.normalize_t(noise_data['t'])   # b*ne*18, b*ne*18, b*1

                # Predict noise
                pred_noise = self.model(e_t, edge_surfInfo, edge_mask, t)    # b*ne*18

                if torch.isnan(pred_noise).any():
                    print("Has nan!!!!")

                assert pred_noise.shape == noise.shape

                # Loss
                loss_geom = torch.nn.functional.mse_loss(pred_noise[edge_mask][..., :-6], noise[edge_mask][..., :-6])
                loss_bbox = torch.nn.functional.mse_loss(pred_noise[edge_mask][..., -6:], noise[edge_mask][..., -6:])
                total_loss = (1 - 6/pred_noise.shape[-1]) * loss_geom + 6/pred_noise.shape[-1] * loss_bbox

                # Update model
                self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0)  # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 20 == 0:
                wandb.log({"Loss-noise": total_loss, "Loss-noise(geom)": loss_geom, "Loss-noise(bbox)": loss_bbox},
                          step=self.iters)

            self.iters += 1
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1
        return

    def save_model(self):
        torch.save(self.model.module.state_dict(), os.path.join(self.save_dir, 'epoch_' + str(self.epoch) + '.pt'))
        return
