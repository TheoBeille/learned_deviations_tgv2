import random

import torch.nn as nn

from algorithm.fbs_step import unroll
from models.deviation_net import DeviationNet


class UnrolledFBS(nn.Module):
    """
    Forward-backward splitting unrolled for T iterations, where the
    deviations are predicted by a network and rescaled by the safeguard.
    """

    def __init__(self, params, shapes, n_channels, T=10, alpha=0.99, random_horizon=False,
                 use_safeguard=True, read_out="d"):
        super().__init__()
        self.params = params
        self.shapes = shapes
        self.T = T
        self.alpha = alpha
        # train on T + randint(0, T) iterations (as in Banert et al.) instead of T;
        # off by default: the deployed scheme restarts after exactly T iterations
        self.random_horizon = random_horizon
        # False: the network output is applied directly (no convergence guarantee)
        self.use_safeguard = use_safeguard
        # "d": the loss and the outputs read the resolvent output d_n, "x": the iterate x_n
        self.read_out = read_out
        self.dev_net = DeviationNet(n_channels)

    def forward(self, functions, T=None, keep_history=False,
                progress=False):
        """Returns (kkt, x_hist), see algorithm.fbs_step.unroll.
        Outside training the deviations stop after T iterations and the
        classical scheme restarts from the read-out at T."""
        if T is None:
            T = self.T
            if self.training and self.random_horizon:
                T += random.randint(0, self.T)

        device = next(self.parameters()).device
        return unroll(functions, self.params, self.shapes, T, device,
                      direction=self.dev_net, alpha=self.alpha,
                      use_safeguard=self.use_safeguard, keep_history=keep_history,
                      progress=progress, restart=None if self.training else self.T,
                      read_out=self.read_out)
