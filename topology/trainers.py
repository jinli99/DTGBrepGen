import os
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
from model import TopoSeqModel, FaceEdgeModel


class TopoSeqTrainer:
    def __init__(self, args, train_dataset, val_dataset):
        self.iters = 0
        self.epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = args.save_dir
        model = TopoSeqModel(max_num_edge=train_dataset.max_num_edge, max_seq_length=train_dataset.max_seq_length)
        model = nn.DataParallel(model)  # distributed training
        self.model = model.to(self.device).train()
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                            shuffle=True,
                                                            batch_size=args.batch_size,
                                                            num_workers=16)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                          shuffle=False,
                                                          batch_size=args.batch_size,
                                                          num_workers=16)

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

    @staticmethod
    def train_loss(logits, topo_seq, seq_mask):
        """
        Args:
            logits: A tensor of shape [batch_size, ns, ne+2].
            topo_seq: A tensor of shape [batch_size, ns].
            seq_mask: A tensor of shape [batch_size, ns]."""

        topo_seq = topo_seq[:, 1:] + 2       # b*(ns-1)
        logits = logits[:, :-1, :]           # b*(ns-1)*(ne+2)
        seq_mask = seq_mask[:, 1:]           # b*(ns-1)
        pred_dist = torch.distributions.categorical.Categorical(logits=logits)
        loss = -torch.sum(pred_dist.log_prob(topo_seq) * seq_mask) / seq_mask.sum()
        return loss

    def train_one_epoch(self):

        self.model.train()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        for data in self.train_dataloader:
            with torch.cuda.amp.autocast():
                data = [x.to(self.device) for x in data]
                # b*ne*2, b*ne, b*ns, b*ns
                edgeFace_adj, edge_mask, topo_seq, seq_mask = data
                ne = torch.nonzero(edge_mask.int().sum(0), as_tuple=True)[0][-1].cpu().item() + 1
                ns = torch.nonzero(seq_mask.int().sum(0), as_tuple=True)[0][-1].cpu().item() + 1
                edgeFace_adj = edgeFace_adj[:, :ne, :]
                edge_mask = edge_mask[:, :ne]
                topo_seq = topo_seq[:, :ns]
                seq_mask = seq_mask[:, :ns]
                logits = self.model(edgeFace_adj, edge_mask, topo_seq, seq_mask)    # b*ns*(ne+2)

                # Zero gradient
                self.optimizer.zero_grad()

                # Loss
                loss = self.train_loss(logits, topo_seq, seq_mask)

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

        self.model.eval()

        progress_bar = tqdm(total=len(self.val_dataloader))
        progress_bar.set_description(f"Testing")
        total_loss = []

        for data in self.val_dataloader:
            with torch.no_grad():
                data = [x.to(self.device) for x in data]
                # b*ne*2, b*ne, b*ns, b*ns
                edgeFace_adj, edge_mask, topo_seq, seq_mask = data
                ne = torch.nonzero(edge_mask.int().sum(0), as_tuple=True)[0][-1].cpu().item() + 1
                ns = torch.nonzero(seq_mask.int().sum(0), as_tuple=True)[0][-1].cpu().item() + 1
                edgeFace_adj = edgeFace_adj[:, :ne, :]
                edge_mask = edge_mask[:, :ne]
                topo_seq = topo_seq[:, :ns]
                seq_mask = seq_mask[:, :ns]
                logits = self.model(edgeFace_adj, edge_mask, topo_seq, seq_mask)    # b*ns*(ne+2)

                # Loss
                loss = self.train_loss(logits, topo_seq, seq_mask)
                total_loss.append(loss.cpu().item())

            progress_bar.update(1)

        progress_bar.close()
        self.model.train()    # set to train

        # logging
        wandb.log({"Val": sum(total_loss) / len(total_loss)}, step=self.iters)

    def save_model(self):
        torch.save(self.model.module.state_dict(), os.path.join(self.save_dir, 'epoch_'+str(self.epoch)+'.pt'))
        return


"""MLP"""
class FaceEdgeTrainer:
    def __init__(self, args, train_dataset, val_dataset):
        self.iters = 0
        self.epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = args.save_dir
        model = FaceEdgeModel(nf=args.max_face, num_categories=args.edge_classes)
        model = nn.DataParallel(model)  # distributed training
        self.model = model.to(self.device).train()
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                            shuffle=True,
                                                            batch_size=args.batch_size,
                                                            num_workers=16)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                          shuffle=False,
                                                          batch_size=args.batch_size,
                                                          num_workers=16)

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

    def train_one_epoch(self):

        self.model.train()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        for data in self.train_dataloader:
            with torch.cuda.amp.autocast():
                data = [x.to(self.device) for x in data]
                fe_topo, _ = data         # b*nf*nf, b*nf
                upper_indices = torch.triu_indices(fe_topo.shape[1], fe_topo.shape[1], offset=1)
                fe_topo_upper = fe_topo[:, upper_indices[0], upper_indices[1]]     # b*seq_len

                # Zero gradient
                self.optimizer.zero_grad()

                # b*seq_len*m, b*latent_dim, b*latent_dim
                adj, mu, logvar = self.model(fe_topo_upper)
                # Loss
                assert not torch.isnan(adj).any()
                kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                recon_loss = torch.nn.functional.cross_entropy(adj.reshape(-1, adj.shape[-1]),
                                                               fe_topo_upper.reshape(-1),
                                                               reduction='mean')
                loss = recon_loss + 100*kl_divergence

                # Update model
                self.scaler.scale(loss).backward()

                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0)  # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 20 == 0:
                wandb.log({"Loss": loss}, step=self.iters)

            self.iters += 1
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1

    def test_val(self):

        self.model.eval()

        progress_bar = tqdm(total=len(self.val_dataloader))
        progress_bar.set_description(f"Testing")
        total_loss = []

        for data in self.val_dataloader:
            with torch.no_grad():
                data = [x.to(self.device) for x in data]
                fe_topo, _ = data         # b*nf*nf, b*nf
                upper_indices = torch.triu_indices(fe_topo.shape[1], fe_topo.shape[1], offset=1)
                fe_topo_upper = fe_topo[:, upper_indices[0], upper_indices[1]]     # b*seq_len

                # b*seq_len*m, b*latent_dim, b*latent_dim
                adj, mu, logvar = self.model(fe_topo_upper)
                # Loss
                assert not torch.isnan(adj).any()
                kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                recon_loss = torch.nn.functional.cross_entropy(adj.reshape(-1, adj.shape[-1]),
                                                               fe_topo_upper.reshape(-1),
                                                               reduction='mean')
                loss = recon_loss + 100*kl_divergence

                total_loss.append(loss.cpu().item())

            progress_bar.update(1)

        progress_bar.close()
        self.model.train()    # set to train

        # logging
        wandb.log({"Val": sum(total_loss) / len(total_loss)}, step=self.iters)

    def save_model(self):
        torch.save(self.model.module.state_dict(), os.path.join(self.save_dir, 'epoch_'+str(self.epoch)+'.pt'))
        return


"""Graph"""
# class FaceEdgeTrainer:
#     def __init__(self, args, train_dataset, val_dataset):
#         self.iters = 0
#         self.epoch = 0
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.save_dir = args.save_dir
#         model = FaceEdgeModel(max_face=args.max_face, edge_classes=args.edge_classes)
#         model = nn.DataParallel(model)  # distributed training
#         self.model = model.to(self.device).train()
#         self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
#                                                             shuffle=True,
#                                                             batch_size=args.batch_size,
#                                                             num_workers=16)
#         self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
#                                                           shuffle=False,
#                                                           batch_size=args.batch_size,
#                                                           num_workers=16)
#
#         # Initialize optimizer
#         self.network_params = list(self.model.parameters())
#
#         self.optimizer = torch.optim.AdamW(
#             self.network_params,
#             lr=5e-4,
#             betas=(0.95, 0.999),
#             weight_decay=1e-6,
#             eps=1e-08,
#         )
#
#         self.scaler = torch.cuda.amp.GradScaler()
#
#     def train_one_epoch(self):
#
#         self.model.train()
#
#         progress_bar = tqdm(total=len(self.train_dataloader))
#         progress_bar.set_description(f"Epoch {self.epoch}")
#
#         for data in self.train_dataloader:
#             with torch.cuda.amp.autocast():
#                 data = [x.to(self.device) for x in data]
#                 fe_topo, mask = data         # b*nf*nf, b*nf
#
#                 # Zero gradient
#                 self.optimizer.zero_grad()
#
#                 # b*nf*nf*m, b*nf, b*nf*z, b*nf*z
#                 adj, face_state, mu, logvar = self.model(fe_topo, mask)
#
#                 # Loss
#                 assert not torch.isnan(face_state).any()
#                 face_loss = torch.nn.functional.binary_cross_entropy_with_logits(face_state, mask.float())
#                 kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#                 tri_mask = torch.triu(torch.ones(adj.shape[1], adj.shape[1]), diagonal=1).bool().to(adj.device)
#                 adj_upper = adj[:, tri_mask]          # shape: (b, num_upper_elements, m)
#                 fe_topo_upper = fe_topo[:, tri_mask]  # shape: (b, num_upper_elements)
#                 edge_loss = torch.nn.functional.cross_entropy(adj_upper.view(-1, adj.shape[-1]),
#                                                               fe_topo_upper.view(-1),
#                                                               reduction='mean')
#                 loss = face_loss + kl_divergence + edge_loss
#
#                 # Update model
#                 self.scaler.scale(loss).backward()
#
#                 nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0)  # clip gradient
#                 self.scaler.step(self.optimizer)
#                 self.scaler.update()
#
#             # logging
#             if self.iters % 20 == 0:
#                 wandb.log({"Loss": loss}, step=self.iters)
#
#             self.iters += 1
#             progress_bar.update(1)
#
#         progress_bar.close()
#         self.epoch += 1
#
#     def test_val(self):
#
#         self.model.eval()
#
#         progress_bar = tqdm(total=len(self.val_dataloader))
#         progress_bar.set_description(f"Testing")
#         total_loss = []
#
#         for data in self.val_dataloader:
#             with torch.no_grad():
#                 data = [x.to(self.device) for x in data]
#                 fe_topo, mask = data         # b*nf*nf, b*nf
#
#                 # b*nf*nf*m, b*nf, b*nf*z, b*nf*z
#                 adj, face_state, mu, logvar = self.model(fe_topo, mask)
#
#                 # Loss
#                 assert not torch.isnan(face_state).any()
#                 face_loss = torch.nn.functional.binary_cross_entropy_with_logits(face_state, mask.float())
#                 kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#                 tri_mask = torch.triu(torch.ones(adj.shape[1], adj.shape[1]), diagonal=1).bool().to(adj.device)
#                 adj_upper = adj[:, tri_mask]          # shape: (b, num_upper_elements, m)
#                 fe_topo_upper = fe_topo[:, tri_mask]  # shape: (b, num_upper_elements)
#                 edge_loss = torch.nn.functional.cross_entropy(adj_upper.view(-1, adj.shape[-1]),
#                                                               fe_topo_upper.view(-1),
#                                                               reduction='mean')
#                 loss = face_loss + kl_divergence + edge_loss
#                 total_loss.append(loss.cpu().item())
#
#             progress_bar.update(1)
#
#         progress_bar.close()
#         self.model.train()    # set to train
#
#         # logging
#         wandb.log({"Val": sum(total_loss) / len(total_loss)}, step=self.iters)
#
#     def save_model(self):
#         torch.save(self.model.module.state_dict(), os.path.join(self.save_dir, 'epoch_'+str(self.epoch)+'.pt'))
#         return