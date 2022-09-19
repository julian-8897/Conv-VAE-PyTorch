import torch
from base import BaseModel
from torch import nn
from torch.nn import functional as F
from .types_ import *


class PlanarFlow(nn.Module):
    def __init__(self, dim):
        """Instantiates one step of planar flow.
        Reference:
        Variational Inference with Normalizing Flows
        Danilo Jimenez Rezende, Shakir Mohamed
        (https://arxiv.org/abs/1505.05770)
        Args:
            dim: input dimensionality.
        """
        super(PlanarFlow, self).__init__()

        self.u = nn.Parameter(torch.randn(1, dim))
        self.w = nn.Parameter(torch.randn(1, dim))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        """Forward pass.
        Args:
            x: input tensor (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        def m(x):
            return F.softplus(x) - 1.

        def h(x):
            return torch.tanh(x)

        def h_prime(x):
            return 1. - h(x)**2

        inner = (self.w * self.u).sum()
        u = self.u + (m(inner) - inner) * self.w / self.w.norm()**2
        activation = (self.w * x).sum(dim=1, keepdim=True) + self.b
        x = x + u * h(activation)
        psi = h_prime(activation) * self.w
        log_det = torch.log(torch.abs(1. + (u * psi).sum(dim=1, keepdim=True)))

        return x, log_det


class Flow(nn.Module):
    def __init__(self, dim, type, length):
        """Instantiates a chain of flows.
        Args:
            dim: input dimensionality.
            type: type of flow.
            length: length of flow.
        """
        super(Flow, self).__init__()

        if type == 'planar':
            self.flow = nn.ModuleList([PlanarFlow(dim) for _ in range(length)])

    def forward(self, x):
        """Forward pass.
        Args:
            x: input tensor (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        [B, _] = list(x.size())
        if torch.cuda.is_available():
            log_det = torch.zeros(B, 1).cuda()
        else:
            log_det = torch.zeros(B, 1)

        for i in range(len(self.flow)):
            x, inc = self.flow[i](x)
            log_det = log_det + inc

        return x, log_det


class VanillaVAE(BaseModel):

    def __init__(self,
                 in_channels: int,
                 latent_dims: int,
                 hidden_dims: List = None,
                 flow=None,
                 length=None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()
        self.latent_dim = latent_dims

        if flow is not None:
            self.length = length
            self.flow = Flow(latent_dims, flow, length)

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
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        if self.flow is not None:
            return self.flow(z)
        else:
            return z

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)

        if self.flow is not None:
            z, log_det = self.reparameterize(mu, log_var)
            return self.decode(z), mu, log_var, log_det
        else:
            z = self.reparameterize(mu, log_var)
            return self.decode(z), mu, log_var

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
