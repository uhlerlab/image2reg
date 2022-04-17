from abc import abstractmethod, ABC
from typing import Any, List

import torch
from torch import nn, Tensor
from torch_geometric.nn import GCNConv, Sequential

from src.utils.torch.general import get_device


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
    def __init__(self, in_channels, out_channels, hidden_dim, random_state: int = 1234):
        super().__init__()
        self.model = Sequential(
            "x, edge_index, edge_weight",
            [
                (GCNConv(in_channels, hidden_dim), "x, edge_index, edge_weight -> x"),
                torch.nn.PReLU(),
                (GCNConv(hidden_dim, out_channels), "x, edge_index, edge_weight -> x"),
                torch.nn.PReLU(),
                # (GCNConv(hidden_dim, hidden_dim), "x, edge_index, edge_weight -> x"),
                # torch.nn.PReLU(),
                # (GCNConv(hidden_dim, out_channels), "x, edge_index, edge_weight -> x"),
                # torch.nn.PReLU(),
                torch.nn.Linear(out_channels, out_channels),
            ],
        )

    def forward(self, x, edge_index, edge_weight=None):
        x = self.model(x, edge_index, edge_weight)
        return x


# class CustomGAE(torch.nn.Module):
#     def __init__(
#         self,
#         encoder: nn.Module,
#         transformer: nn.Module,
#         adj_decoder: nn.Module,
#         feat_decoder: nn.Module,
#         feat_loss: nn.Module,
#         classifier: nn.Module = None,
#         class_loss: nn.Module = None,
#         alpha: float = 1.0,
#         beta: float = 1.0,
#     ):
#         super().__init__()
#         self.gae = GAE(encoder=encoder, decoder=adj_decoder)
#         self.transformer = transformer
#         self.feat_decoder = feat_decoder
#         self.feat_loss = feat_loss
#         self.alpha = alpha
#         self.beta = beta
#         self.l2_loss = nn.MSELoss()
#         self.classifier = classifier
#         self.class_loss = class_loss
#
#     def encode(self, x, edge_index, edge_weight=None):
#         z = self.gae.encode(x, edge_index=edge_index, edge_weight=edge_weight)
#         if self.transformer is not None:
#             z = self.transformer(z)
#         return z
#
#     def decode(self, z):
#         adj = self.gae.decode(z)
#         feat = self.feat_decoder(z)
#         return adj, feat
#
#     def recon_loss(self, x, z, pos_edge_index, neg_edge_index=None):
#         gae_loss = self.gae.recon_loss(
#             z, pos_edge_index=pos_edge_index, neg_edge_index=neg_edge_index
#         )
#         # gae_loss = self.gae_l2_loss(z=z, pos_edge_index=pos_edge_index, neg_edge_index=neg_edge_index)
#         feat_loss = self.feat_loss(x, self.feat_decoder(z))
#         return self.alpha * gae_loss + self.beta * feat_loss
#
#     def classification_loss(self, z, labels, label_mask):
#         class_loss = self.class_loss(self.classifier(z)[label_mask], labels)
#         return self.gamma * class_loss
#
#     def test(self, z, pos_edge_index, neg_edge_index):
#         return self.gae.test(z, pos_edge_index, neg_edge_index)
#
#     def gae_l2_loss(self, z, pos_edge_index, neg_edge_index=None):
#         pos_preds = self.gae.decoder(z, pos_edge_index, sigmoid=True)
#         pos_true = torch.ones_like(pos_preds)
#
#         # Do not include self-loops in negative samples
#         pos_edge_index, _ = remove_self_loops(pos_edge_index)
#         pos_edge_index, _ = add_self_loops(pos_edge_index)
#         if neg_edge_index is None:
#             neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
#         neg_preds = self.gae.decoder(z, neg_edge_index, sigmoid=True)
#         neg_true = torch.zeros_like(neg_preds)
#         preds = torch.cat((pos_preds, neg_preds))
#         true = torch.cat((pos_true, neg_true))
#
#         return self.l2_loss(preds, true)


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
