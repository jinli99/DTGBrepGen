import os
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
from model import EdgeVertModel, FaceEdgeModel
from utils import make_mask


class FaceEdgeTrainer:
    def __init__(self, args, train_dataset, val_dataset):
        self.iters = 0
        self.epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = args.save_dir
        self.edge_classes = args.edge_classes
        self.use_cf = args.use_cf
        self.use_pc = args.use_pc
        model = FaceEdgeModel(nf=args.max_face,
                              d_model=args.FaceEdgeModel['d_model'],
                              nhead=args.FaceEdgeModel['nhead'],
                              n_layers=args.FaceEdgeModel['n_layers'],
                              num_categories=args.edge_classes,
                              use_cf=self.use_cf)
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
                if self.use_cf:
                    fef_adj, _, class_label = data                                 # b*nf*nf, b*nf, b*1
                else:
                    fef_adj, _ = data                                              # b*nf*nf, b*nf
                    class_label = None
                upper_indices = torch.triu_indices(fef_adj.shape[1], fef_adj.shape[1], offset=1)
                fef_adj_upper = fef_adj[:, upper_indices[0], upper_indices[1]]     # b*seq_len

                # Zero gradient
                self.optimizer.zero_grad()

                # b*seq_len*m, b*latent_dim, b*latent_dim
                adj, mu, logvar = self.model(fef_adj_upper, class_label)

                # Loss
                assert not torch.isnan(adj).any()
                kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                recon_loss = torch.nn.functional.cross_entropy(adj.reshape(-1, adj.shape[-1]),
                                                               fef_adj_upper.reshape(-1),
                                                               reduction='mean')
                loss = recon_loss + kl_divergence

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
                if self.use_cf:
                    fef_adj, _, class_label = data                                 # b*nf*nf, b*nf, b*1
                else:
                    fef_adj, _ = data                                              # b*nf*nf, b*nf
                    class_label = None
                upper_indices = torch.triu_indices(fef_adj.shape[1], fef_adj.shape[1], offset=1)
                fef_adj_upper = fef_adj[:, upper_indices[0], upper_indices[1]]     # b*seq_len

                # b*seq_len*m, b*latent_dim, b*latent_dim
                adj, mu, logvar = self.model(fef_adj_upper, class_label)
                # Loss
                assert not torch.isnan(adj).any()
                kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                recon_loss = torch.nn.functional.cross_entropy(adj.reshape(-1, adj.shape[-1]),
                                                               fef_adj_upper.reshape(-1),
                                                               reduction='mean')
                # recon_loss = self.class_loss(adj.reshape(-1, adj.shape[-1]), fef_adj_upper.reshape(-1))
                loss = recon_loss + kl_divergence

                total_loss.append(loss.cpu().item())

            progress_bar.update(1)

        progress_bar.close()
        self.model.train()    # set to train

        # logging
        wandb.log({"Val": sum(total_loss) / len(total_loss)}, step=self.iters)

    def save_model(self):
        torch.save(self.model.module.state_dict(), os.path.join(self.save_dir, 'epoch_'+str(self.epoch)+'.pt'))
        return


class EdgeVertTrainer:
    def __init__(self, args, train_dataset, val_dataset):
        self.iters = 0
        self.epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = args.save_dir
        self.use_cf = args.use_cf
        self.use_pc = args.use_pc

        model = EdgeVertModel(max_num_edge=args.max_num_edge_topo,
                              max_seq_length=args.max_seq_length,
                              edge_classes=args.edge_classes,
                              max_face=args.max_face,
                              max_edge=args.max_edge,
                              d_model=args.EdgeVertModel['d_model'],
                              n_layers=args.EdgeVertModel['n_layers'],
                              use_cf=self.use_cf)
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
                if self.use_cf:
                    edgeFace_adj, edge_mask, share_id, topo_seq, seq_mask, class_label = data    # b*ne*2, b*1, b*ne, b*ns, b*1, b*1
                else:
                    edgeFace_adj, edge_mask, share_id, topo_seq, seq_mask = data                 # b*ne*2, b*1, b*ne, b*ns, b*1
                    class_label = None
                ne = edge_mask.max()
                ns = seq_mask.max()
                edgeFace_adj = edgeFace_adj[:, :ne, :]
                share_id = share_id[:, :ne]
                topo_seq = topo_seq[:, :ns]

                edge_mask = make_mask(edge_mask, ne)      # b*ne
                seq_mask = make_mask(seq_mask, ns)        # b*ns

                logits = self.model(edgeFace_adj, edge_mask, topo_seq, seq_mask, share_id, class_label)       # b*ns*(ne+2)

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
                if self.use_cf:
                    edgeFace_adj, edge_mask, share_id, topo_seq, seq_mask, class_label = data    # b*ne*2, b*1, b*ne, b*ns, b*1, b*1
                else:
                    edgeFace_adj, edge_mask, share_id, topo_seq, seq_mask = data                 # b*ne*2, b*1, b*ne, b*ns, b*1
                    class_label = None
                ne = edge_mask.max()
                ns = seq_mask.max()
                edgeFace_adj = edgeFace_adj[:, :ne, :]
                share_id = share_id[:, :ne]
                topo_seq = topo_seq[:, :ns]

                edge_mask = make_mask(edge_mask, ne)      # b*ne
                seq_mask = make_mask(seq_mask, ns)        # b*ns

                logits = self.model(edgeFace_adj, edge_mask, topo_seq, seq_mask, share_id, class_label)       # b*ns*(ne+2)

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
