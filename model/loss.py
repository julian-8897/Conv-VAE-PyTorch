import torch.nn.functional as F
from torch import nn
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)


def elbo_loss(recon_x, x, mu, logvar):
    """
    Loss function for VAE:
    reconstruction term + regularization term
    """
    reconstruction_function = nn.MSELoss(reduction='sum')
    MSE = reconstruction_function(recon_x, x)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return MSE + KLD
