from abc import abstractmethod, ABC
from typing import Any, List, Tuple

import torch
from torch import nn, Tensor

from src.utils.torch.general import get_device
from torch_geometric.nn import GAE, GCNConv


class BaseAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.recon_loss_module = None
        self.model_base_type = "ae"

    def encode(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        pass

    def loss_function(self, inputs: Tensor, recons: Tensor) -> dict:
        recon_loss = self.recon_loss_module(inputs, recons)
        loss_dict = {"recon_loss": recon_loss}
        return loss_dict

class GCNEncoder(torch.nn.Module):
    def __init__(self, input_channels:int, hidden_dims:int, latent_dim:int):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dims = hidden_dims
        self.latent_dim=latent_dim
        self.gcn1 = GCNConv(in_channels=input_channels, out_channels=hidden_dims, cached=True)
        self.relu = nn.ReLU()
        # self.gcn2 = GCNConv(in_channels=hidden_dims, out_channels=hidden_dims, cached=True)
        # self.gcn3 = GCNConv(in_channels=hidden_dims, out_channels=hidden_dims, cached=True)
        # self.gcn4 = GCNConv(in_channels=hidden_dims, out_channels=hidden_dims, cached=True)
        self.gcn5 = GCNConv(in_channels=hidden_dims, out_channels=latent_dim, cached=True)


        # if len(hidden_dims) > 0:
        #     modules = [nn.Sequential(GCNConv(in_channels=input_channels, out_channels=hidden_dims[0], cached=True), nn.ReLU())]
        #     for i in range(1,len(hidden_dims)):
        #         modules.append(nn.Sequential(GCNConv(in_channels=self.hidden_dims[i-1], out_channels=self.hidden_dims[i], cached=True), nn.ReLU()))
        #     modules.append(GCNConv(in_channels=self.hidden_dims[-1], out_channels=self.latent_dim))
        # else:
        #     modules = [nn.Sequential(GCNConv(in_channels=self.input_channels, out_channels=self.latent_dim))]
        # self.model = nn.Sequential(*modules)

    def forward(self, x, edge_index):
        z = self.gcn1(x, edge_index)
        z = self.relu(z)
        # z = self.gcn2(z, edge_index)
        # z = self.relu(z)
        # z = self.gcn3(z, edge_index)
        # z = self.relu(z)
        # z = self.gcn4(z, edge_index)
        # z = self.relu(z)
        z = self.gcn5(z, edge_index)
        return z
        #return self.model(x, edge_index)


class GraphConvAE(nn.Module):
    def __init__(self, input_channels:int, hidden_dims: List[int], latent_dim:int):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.model_base_type = "ae"

        self.encoder = GCNEncoder(input_channels=self.input_channels, hidden_dims=self.hidden_dims, latent_dim=self.latent_dim)
        self.model = GAE(encoder=self.encoder)

    def encode(self, x, edge_index) -> Tensor:
        return self.model.encode(x, edge_index)

    def forward(self, x, edge_index) -> dict:
        latents = self.model.encode(x, edge_index)
        recons = self.model.decoder.forward_all(latents)
        return {"recons":recons, "latents":latents}

    def loss_function(self, latents: Tensor, edge_index: Tensor) -> dict:
        return self.model.recon_loss(latents, pos_edge_index=edge_index)

    def test(self, latents:Tensor, pos_edge_index:Tensor, neg_edge_index:Tensor) -> Tuple[Tensor]:
        return self.model.test(z=latents, pos_edge_index=pos_edge_index, neg_edge_index=neg_edge_index)


class VanillaConvAE(BaseAE, ABC):
    def __init__(
        self,
        input_channels: int = 1,
        hidden_dims: List[int] = [64, 128, 256, 512, 512],
        lrelu_slope: int = 0.2,
        batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = input_channels
        # self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.lrelu_slope = lrelu_slope
        self.batchnorm = batchnorm
        self.updated = False
        self.n_latent_spaces = 1

        # Build encoder
        encoder_modules = [
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.hidden_dims[0],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.PReLU(),
            )
        ]

        for i in range(1, len(self.hidden_dims)):
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.hidden_dims[i - 1],
                        out_channels=self.hidden_dims[i],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.hidden_dims[i]),
                    nn.PReLU(),
                )
            )
        self.encoder = nn.Sequential(*encoder_modules)

        # Output of encoder are of shape 1024x4x4
        self.device = get_device()

        # if self.batchnorm:
        #     self.latent_mapper = nn.Sequential(
        #         nn.Linear(hidden_dims[-1] * 3 * 3, self.latent_dim),
        #         nn.BatchNorm1d(self.latent_dim),
        #     )
        # else:
        #     self.latent_mapper = nn.Linear(hidden_dims[-1] * 3 * 3, self.latent_dim)
        #
        # self.inv_latent_mapper = nn.Sequential(
        #     nn.Linear(self.latent_dim, hidden_dims[-1] * 3 * 3), nn.ReLU(inplace=True)
        # )

        # decoder
        decoder_modules = []
        for i in range(len(hidden_dims) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=hidden_dims[-1 - i],
                        out_channels=hidden_dims[-2 - i],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dims[-2 - i]),
                    nn.PReLU(),
                )
            )
        decoder_modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=hidden_dims[0],
                    out_channels=self.in_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.Sigmoid(),
            )
        )

        self.decoder = nn.Sequential(*decoder_modules)

    def encode(self, input: Tensor, extra_features: Tensor = None) -> Tensor:
        features = self.encoder(input=input)
        features = features.view(features.size(0), -1)
        latents = features
        return latents

    def decode(self, input: Tensor) -> Any:
        latent_features = input.view(input.size(0), self.hidden_dims[-1], 2, 2)
        output = self.decoder(input=latent_features)
        return output

    def forward(self, inputs: Tensor) -> dict:
        latents = self.encode(input=inputs)
        recons = self.decode(latents)
        output = {"recons": recons, "latents": latents}
        return output
