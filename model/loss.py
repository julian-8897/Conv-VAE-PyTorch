import torch.nn.functional as F
from torch import nn
import torch


def elbo_loss(recon_x, x, mu, logvar, beta=1):
    """
    ELBO Optimization objective for gaussian posterior
    (reconstruction term + regularization term)
    """
    reconstruction_function = nn.MSELoss(reduction='sum')
    MSE = reconstruction_function(recon_x, x)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return MSE + beta*KLD


def elbo_loss_flow(recon_x, x, mu, logvar, log_det):
    """
    ELBO Optimization objective for gaussian posterior
    (reconstruction term + regularization term)
    """
    reconstruction_function = nn.MSELoss(reduction='sum')
    MSE = reconstruction_function(recon_x, x)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return (MSE + KLD - log_det).mean()
