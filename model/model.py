import torch
from base import BaseModel
from torch import nn
from torch.nn import functional as F
from .types_ import *


class VanillaVAE(BaseModel):

    def __init__(self,
                 in_channels: int,
                 latent_dims: int,
                 hidden_dims: List[int] = None,
                 **kwargs) -> None:
        """Instantiates the VAE model

        Params:
            in_channels (int): Number of input channels
            latent_dims (int): Size of latent dimensions
            hidden_dims (List[int]): List of hidden dimensions
        """
        super(VanillaVAE, self).__init__()
        self.latent_dim = latent_dims

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dims)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dims)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dims, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the convolutional network
        and outputs the latent variables.

        Params:
            input (Tensor): Input tensor [N x C x H x W]

        Returns:
            mu (Tensor) and log_var (Tensor) of latent variables
        """

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        z = self.reparameterize(mu, log_var)

        return mu, log_var, z

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent variables
        onto the image space.

        Params:
            z (Tensor): Latent variable [B x D]

        Returns:
            result (Tensor) [B x C x H x W]
        """

        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1)

        Params:
            mu (Tensor): Mean of Gaussian latent variables [B x D]
            logvar (Tensor): log-Variance of Gaussian latent variables [B x D]

        Returns: 
            z (Tensor) [B x D]
        """

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        return z

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var, z = self.encode(input)

        return self.decode(z), mu, log_var

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.

        Params:
            num_samples (Int): Number of samples
            current_device (Int): Device to run the model

        Returns:
            samples (Tensor)
        """

        z = torch.randn(num_samples,
                        self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)

        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image

        Params:
            x (Tensor): input image Tensor [B x C x H x W]

        Returns:
            (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
