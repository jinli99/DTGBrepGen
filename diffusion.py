import argparse
import torch.nn.functional as ff
import torch.nn as nn
import torch
import numpy as np
from utils import xe_mask, assert_weak_one_hot, make_edge_symmetric


def get_diff_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edge_classes', type=int, default=9, help='Number of edge classes')
    args = parser.parse_args()
    return args


def face_noise_schedule(num_timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    alphas = 1. - torch.linspace(beta_start, beta_end, num_timesteps)
    return torch.cumprod(alphas, dim=0)


def edge_num_noise_schedule(num_timesteps):

    # Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ.
    s = 0.008
    steps = num_timesteps + 2
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = (1 - alphas).squeeze()
    alphas = 1 - torch.clamp(betas, min=0, max=0.9999)
    return alphas


def qt_transition(alpha, u_e, edge_classes):
    """ Returns t-step transition matrices for X and E, from step 0 to step t.
    Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) * K

    alpha: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
    returns:  qe (bs, de, de)
    """

    q_t = alpha * torch.eye(edge_classes, device=alpha.device) + (1 - alpha) * u_e.to(alpha.device)

    return q_t


class GraphDiffusion(nn.Module):
    """Diffusion Models for Face Feature and Topology"""

    def __init__(self, edge_classes, edge_marginals, device):
        super(GraphDiffusion, self).__init__()
        self.edge_classes = edge_classes
        self.T = 500
        self.face_alphas_bar = face_noise_schedule(self.T).to(device)
        alphas = edge_num_noise_schedule(self.T).to(device)
        self.edge_alphas_bar = torch.exp(torch.cumsum(torch.log(alphas), dim=0))
        self.u_e = edge_marginals.unsqueeze(0).expand(self.edge_classes, -1).to(device)   # m*m
        self.eps = 1e-6

        q_one_step_transposed = []
        q_mats = []   # these are cumulative

        for i in range(len(alphas)):
            q_one_step_transposed.append(
                qt_transition(alphas[i], self.u_e, self.edge_classes).type(torch.float32).transpose(0, 1))
            q_mats.append(qt_transition(self.edge_alphas_bar[i], self.u_e, self.edge_classes).type(torch.float32))
        q_one_step_transposed = torch.stack(q_one_step_transposed, dim=0)   # T*m*m
        q_mats = torch.stack(q_mats, dim=0)

        # register
        self.register_buffer("q_one_step_transposed", q_one_step_transposed)
        self.register_buffer("q_mats", q_mats)

    @staticmethod
    def _at(a, t, x):
        # t is 1-d, x is integer value of 0 to num_classes - 1
        bs = t.shape[0]
        t = t.reshape((bs, *[1] * (x.dim() - 1))).type(torch.int)   # b*1*1
        # out[i, j, k, l, m] = a[t[i, j, k, l], x[i, j, k, l], m]
        return a[t, x, :]

    def normalize_t(self, t):
        return t/self.T

    def add_noise(self, x, e, node_mask, t=None):
        """
        The diffusion process q(x_t, e_t|x_0, e_0).

        Args:
        - x (torch.Tensor): The original node features x_0 with shape (b, n, d),
                            where b is the batch size, n is the number of nodes,
                            and d is the feature dimension.
        - e (torch.Tensor): The original edge features e_0 with shape (b, n, n)
                            or one-hot encoded shape (b, n, n, m), where m is
                            the number of edge classes.
        - node_mask (torch.Tensor): A mask indicating valid nodes with shape (b, n).
        - t (torch.Tensor, optional): The diffusion timestep with shape (b, 1).
                                      If not provided, a random timestep will be used.

        Returns:
        - x_t (torch.Tensor): The diffused node features x_t with added noise.
        - e_t (torch.Tensor): The one-hot encoded edge features.
        """

        if len(e.shape) == 3:
            e = ff.one_hot(e, self.edge_classes)   # one-hot encoding, b*n*n*m

        if t is None:
            t = torch.randint(0, self.T, size=(x.size(0), 1), device=x.device)

        # apply noise to face features
        noise = torch.randn_like(x)
        alpha_t_bar = self.face_alphas_bar[t].view(-1, 1, 1)
        x_diffused = torch.sqrt(alpha_t_bar) * x + torch.sqrt(1 - alpha_t_bar) * noise
        x_diffused = x_diffused * node_mask.unsqueeze(-1)  # b*n*d

        # apply noise to edge types
        probE = e.type(torch.float32) @ self.q_mats[t]  # b*n*n*m

        # sample from edge distribution
        b, n, _, m = probE.shape
        probE_flat = probE.view(b * n * n, m)
        edge_diffused = torch.multinomial(probE_flat, 1).view(b, n, n)  # b*n*n
        diag_mask = torch.eye(n, dtype=torch.bool).unsqueeze(0).expand(b, -1, -1)
        edge_diffused[diag_mask] = 0
        edge_diffused = ff.one_hot(edge_diffused, self.edge_classes)    # b*n*n*m
        _, edge_diffused = xe_mask(e=edge_diffused, node_mask=node_mask, check_sym=False)
        edge_diffused = make_edge_symmetric(edge_diffused)
        assert_weak_one_hot(edge_diffused)

        return {'x_t': x_diffused, 'e_t': edge_diffused, 'y': self.normalize_t(t), 'noise': noise, 't': t}

    def q_posterior_logits(self, e_0, e_t, t):
        """Compute q(e_{t-1} | e_0, e_t).
           if t == 0, this means we return the L_0 loss, so directly try to e_0 logits.
           otherwise, we return the L_{t-1} loss."""

        assert e_0.shape[0] == e_t.shape[0] == t.shape[0] and len(t.shape) == 2

        # if x_0 is integer, we convert it to one-hot.
        if len(e_0.shape) == 3:
            e_0_logits = torch.nn.functional.one_hot(e_0, self.edge_classes).type(torch.float32)   # b*n*n*m
        else:
            e_0_logits = e_0.type(torch.float32)

        if len(e_t.shape) == 4:
            assert_weak_one_hot(e_t)
            e_t = torch.argmax(e_t, dim=-1)

        # fact1 is "guess of x_{t-1}" from x_t
        # fact2 is "guess of x_{t-1}" from x_0

        fact1 = self._at(self.q_one_step_transposed, t, e_t)  # e_t * Q_t， b*n*n*m

        soft_masked = torch.softmax(e_0_logits, dim=-1)  # b*n*n*m
        q_mats2 = self.q_mats[t.type(torch.int) - 1].to(dtype=soft_masked.dtype)   # b*1*m*m
        fact2 = soft_masked @ q_mats2  # e_0 * Q_{t-1}_bar， b*n*n*m

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        t_broadcast = t.reshape((t.shape[0], *[1] * (e_t.dim())))

        bc = torch.where(t_broadcast == 0, e_0_logits, out)

        return bc

    def x_p_sample(self, x_t, eps, t):
        """Sample x_{t-1} ~ p(x_{t-1}|x_t)"""

        assert len(t.shape) == 1

        # Get the parameters for the current time step t
        alpha_bar_t = self.face_alphas_bar[t]
        alpha_bar_t_prev = self.face_alphas_bar[t - 1] if t > 0 else self.face_alphas_bar[0]
        alpha_t = self.face_alphas_bar[t] / alpha_bar_t_prev if t > 0 else self.face_alphas_bar[0]

        # Calculate the mean and variance for the posterior distribution q(x_{t-1} | x_t, x_0)
        mean = 1/(alpha_t**0.5)*(x_t - (1-alpha_t)/((1-alpha_bar_t)**0.5)*eps)

        # Calculate the variance for the posterior distribution q(x_{t-1} | x_t, x_0)
        var = (1 - alpha_bar_t_prev) * (1 - alpha_t) / (1 - alpha_bar_t)
        std_dev = var ** 0.5

        # Sample x_{t-1} from the normal distribution with the calculated mean and standard deviation
        noise = torch.randn_like(x_t) if t > 0 else 0
        return mean + std_dev * noise

    def e_p_sample(self, e_0_logits, e_t, t):
        e_t_logits = self.q_posterior_logits(e_0=e_0_logits, e_t=e_t, t=t)  # b*n*n*m
        e_t_prob = torch.softmax(e_t_logits, dim=-1)   # b*n*n*m
        b, n, _, m = e_t_prob.shape
        e_t = torch.nn.functional.one_hot(
            torch.multinomial(e_t_prob.reshape(-1, m), 1).reshape(b, n, n), num_classes=self.edge_classes)  # b*n*n*m
        e_t = make_edge_symmetric(e_t)  # b*n*n*m
        assert_weak_one_hot(e_t)
        return e_t


class EdgeDiffusion:

    def __init__(self, device):
        self.T = 500
        self.face_alphas_bar = face_noise_schedule(self.T).to(device)

    def normalize_t(self, t):
        return t/self.T

    def add_noise(self, e, edge_mask, t=None):    # b*ne*18, b*ne

        if t is None:
            t = torch.randint(0, self.T, size=(e.size(0), 1), device=e.device)   # b*1

        # apply noise to face features
        noise = torch.randn_like(e)
        alpha_t_bar = self.face_alphas_bar[t].view(-1, 1, 1)
        e_diffused = torch.sqrt(alpha_t_bar) * e + torch.sqrt(1 - alpha_t_bar) * noise
        e_diffused = e_diffused * edge_mask.unsqueeze(-1)  # b*n*d

        return {'e_t': e_diffused, 'noise': noise, 't': t}

    def p_sample(self, x_t, eps, t):
        """Sample x_{t-1} ~ p(x_{t-1}|x_t)"""

        assert len(t.shape) == 1

        # Get the parameters for the current time step t
        alpha_bar_t = self.face_alphas_bar[t]
        alpha_bar_t_prev = self.face_alphas_bar[t - 1] if t > 0 else self.face_alphas_bar[0]
        alpha_t = self.face_alphas_bar[t] / alpha_bar_t_prev if t > 0 else self.face_alphas_bar[0]

        # Calculate the mean and variance for the posterior distribution q(x_{t-1} | x_t, x_0)
        mean = 1/(alpha_t**0.5)*(x_t - (1-alpha_t)/((1-alpha_bar_t)**0.5)*eps)

        # Calculate the variance for the posterior distribution q(x_{t-1} | x_t, x_0)
        var = (1 - alpha_bar_t_prev) * (1 - alpha_t) / (1 - alpha_bar_t)
        std_dev = var ** 0.5

        # Sample x_{t-1} from the normal distribution with the calculated mean and standard deviation
        noise = torch.randn_like(x_t) if t > 0 else 0
        return mean + std_dev * noise