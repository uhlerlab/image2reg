from abc import abstractmethod, ABC
from typing import Any, List
from torch import nn, Tensor

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


class VanillaAE(BaseAE, ABC):
    def __init__(
        self,
        input_dim: int = 2613,
        latent_dim: int = 128,
        hidden_dims: List = None,
        batchnorm_latent: bool = False,
        lrelu_slope: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.batchnorm_latent = batchnorm_latent
        self.lrelu_slope = lrelu_slope
        self.n_latent_spaces = 1

        # Build encoder model
        if len(self.hidden_dims) > 0:
            encoder_modules = [
                nn.Sequential(
                    nn.Linear(self.input_dim, self.hidden_dims[0]),
                    nn.PReLU(),
                    nn.BatchNorm1d(self.hidden_dims[0]),
                )
            ]

        for i in range(1, len(hidden_dims)):
            encoder_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]),
                    nn.PReLU(),
                    nn.BatchNorm1d(self.hidden_dims[i]),
                )
            )
        if self.batchnorm_latent:
            encoder_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[-1], self.latent_dim),
                    nn.BatchNorm1d(self.latent_dim),
                )
            )
        else:
            encoder_modules.append(nn.Linear(self.hidden_dims[-1], self.latent_dim))

        self.encoder = nn.Sequential(*encoder_modules)

        # Build decoder model
        decoder_modules = [
            nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dims[-1]),
                nn.PReLU(),
                nn.BatchNorm1d(self.hidden_dims[-1]),
            )
        ]
        for i in range(len(hidden_dims) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[-1 - i], self.hidden_dims[-2 - i]),
                    nn.PReLU(),
                    nn.BatchNorm1d(self.hidden_dims[-2 - i]),
                )
            )

        decoder_modules.append(nn.Linear(self.hidden_dims[0], self.input_dim))

        self.decoder = nn.Sequential(*decoder_modules)

    def encode(self, input: Tensor) -> Tensor:
        latents = self.encoder(input=input)
        return latents

    def decode(self, input: Tensor) -> Any:
        output = self.decoder(input=input)
        return output

    def forward(self, inputs: Tensor) -> dict:
        latents = self.encode(input=inputs)
        recons = self.decode(latents)
        output = {"recons": recons, "latents": latents}
        return output


class VanillaConvAE(BaseAE, ABC):
    def __init__(
        self,
        input_channels: int = 1,
        latent_dim: int = 128,
        hidden_dims: List[int] = [128, 256, 512, 1024, 1024],
        lrelu_slope: int = 0.2,
        batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = input_channels
        self.latent_dim = latent_dim
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

        if self.batchnorm:
            self.latent_mapper = nn.Sequential(
                nn.Linear(hidden_dims[-1] * 3 * 3, self.latent_dim),
                nn.BatchNorm1d(self.latent_dim),
            )
        else:
            self.latent_mapper = nn.Linear(hidden_dims[-1] * 3 * 3, self.latent_dim)

        self.inv_latent_mapper = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dims[-1] * 3 * 3), nn.ReLU(inplace=True)
        )

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

    def encode(self, input: Tensor) -> Tensor:
        features = self.encoder(input=input)
        features = features.view(features.size(0), -1)
        latents = self.latent_mapper(input=features)
        return latents

    def decode(self, input: Tensor) -> Any:
        latent_features = self.inv_latent_mapper(input)
        latent_features = latent_features.view(
            latent_features.size(0), self.hidden_dims[-1], 3, 3
        )
        output = self.decoder(input=latent_features)
        return output

    def forward(self, inputs: Tensor) -> dict:
        latents = self.encode(input=inputs)
        recons = self.decode(latents)
        output = {"recons": recons, "latents": latents}
        return output
