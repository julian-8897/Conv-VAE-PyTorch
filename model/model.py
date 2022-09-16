# from torch.autograd import Variable
# import torchvision.models as models
# from torch import nn
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from base import BaseModel


# class Encoder(BaseModel):
#     """
#     encoder architecture for VAE
#     """

#     def __init__(self, nc, ndf, latent_dims):
#         super(Encoder, self).__init__()

#         self.nc = nc
#         self.ndf = ndf
#         self.latent_dims = latent_dims
#         self.conv1 = nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False)
#         self.batch1 = nn.BatchNorm2d(ndf)
#         self.conv2 = nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=1, bias=False)
#         self.batch2 = nn.BatchNorm2d(ndf*2)
#         self.conv3 = nn.Conv2d(ndf*2, ndf*4, 4, stride=2,
#                                padding=1, bias=False)
#         self.batch3 = nn.BatchNorm2d(ndf*4)
#         self.conv4 = nn.Conv2d(ndf*4, ndf*8, 4, stride=2,
#                                padding=1, bias=False)
#         self.batch4 = nn.BatchNorm2d(ndf*8)
#         self.conv5 = nn.Conv2d(ndf*8, ndf*8, 4, stride=2,
#                                padding=1, bias=False)
#         self.batch5 = nn.BatchNorm2d(ndf*8)
#         self.linear2 = nn.Linear(ndf*8*4, latent_dims)
#         self.linear3 = nn.Linear(ndf*8*4, latent_dims)
#         self.N = torch.distributions.Normal(0, 1)
#         self.leakyrelu = nn.LeakyReLU(0.2)
#         self.relu = nn.ReLU()

#     def reparametrise(self, mu, logvar):
#         # std = logvar.mul(0.5).exp_()
#         # eps = torch.FloatTensor(std.size()).normal_()
#         # eps = Variable(eps)
#         # return eps.mul(std).add_(mu)

#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps * std + mu

#     def forward(self, x):
#         # x = x.to(device)
#         x = self.leakyrelu(self.batch1(self.conv1(x)))
#         x = self.leakyrelu(self.batch2(self.conv2(x)))
#         x = self.leakyrelu(self.batch3(self.conv3(x)))
#         x = self.leakyrelu(self.batch4(self.conv4(x)))
#         x = self.leakyrelu(self.batch5(self.conv5(x)))

#         # x = self.resnet(x)
#         x = torch.flatten(x, start_dim=1)
#         mu = self.linear2(x)
#         logvar = self.linear3(x)
#         z = self.reparametrise(mu, logvar)
#         return z, mu, logvar


# class Decoder(BaseModel):
#     """
#     decoder architecture for VAE
#     """

#     def __init__(self, nc, ngf, latent_dims):
#         super().__init__()

#         self.unflatten = nn.Unflatten(dim=1, unflattened_size=(ngf*8*2, 4, 4))
#         self.ngf = ngf
#         self.latent_dims = latent_dims
#         self.leakyrelu = nn.LeakyReLU(0.2, True)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#         self.d1 = nn.Linear(latent_dims, ngf*8*2*4*4)
#         self.up1 = nn.UpsamplingNearest2d(scale_factor=1)
#         self.pd1 = nn.ReplicationPad2d(1)
#         self.d2 = nn.Conv2d(ngf*8*2, ngf*8, 3, 1, bias=False)
#         self.bn6 = nn.BatchNorm2d(ngf*8, 1.e-5)

#         self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
#         self.pd2 = nn.ReplicationPad2d(1)
#         self.d3 = nn.Conv2d(ngf*8, ngf*4, 3, 1, bias=False)
#         self.bn7 = nn.BatchNorm2d(ngf*4, 1.e-5)

#         self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
#         self.pd3 = nn.ReplicationPad2d(1)
#         self.d4 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
#         self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-5)

#         self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
#         self.pd4 = nn.ReplicationPad2d(1)
#         self.d5 = nn.Conv2d(ngf*2, ngf, 3, 1, bias=False)
#         self.bn9 = nn.BatchNorm2d(ngf, 1.e-5)

#         self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
#         self.pd5 = nn.ReplicationPad2d(1)
#         self.d6 = nn.Conv2d(ngf, nc, 3, 1, bias=True)

#     def forward(self, x):
#         x = self.relu(self.d1(x))
#         x = self.unflatten(x)
#         x = self.leakyrelu(self.bn6(self.d2(self.pd1((x)))))
#         x = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(x)))))
#         x = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(x)))))
#         x = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(x)))))
#         x = (self.d6(self.pd5(self.up5(x))))

#         return x


import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


# class MnistModel(BaseModel):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, num_classes)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

# class VariationalAutoencoder(BaseModel):
#     """
#     Full VAE architecture incorporating encoder and decoder
#     Parameters:
#     nc: number of input channels
#     ndf: horizontal size of image, use 64
#     ngf: vertical size of image, use 64
#     latent_dims: dimension of latent space
#     """

#     def __init__(self, nc, ndf, ngf, latent_dims):
#         super(VariationalAutoencoder, self).__init__()
#         self.encoder = Encoder(nc, ndf, latent_dims)
#         self.decoder = Decoder(nc, ngf, latent_dims)

#     def forward(self, x):
#         z, mu, logvar = self.encoder(x)
#         final = self.decoder(z)
#         return final, mu, logvar


import torch
from base import BaseModel
from torch import nn
from torch.nn import functional as F
from .types_ import *


class VanillaVAE(BaseModel):

    def __init__(self,
                 in_channels: int,
                 latent_dims: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
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
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
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
