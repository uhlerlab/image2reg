from abc import ABC

import torch
from torch.functional import Tensor
from torch.hub import load_state_dict_from_url
from torch.nn import Module
from torch import nn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, model_urls
from typing import Type, List, Union, Optional, Callable, Any

from src.utils.torch.general import get_device


class SimpleConvClassifier(Module):
    def __init__(
        self,
        n_output_nodes: int,
        input_channels: int = 1,
        hidden_dims: List[int] = [32, 64, 128, 256, 256],
        batchnorm: bool = True,
        dropout_rate: float = 0,
    ) -> None:
        super().__init__()
        self.in_channels = input_channels
        self.hidden_dims = hidden_dims
        self.batchnorm = batchnorm
        self.updated = False
        self.n_output_nodes = n_output_nodes
        self.model_base_type = "clf"
        self.dropout_rate = dropout_rate

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
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.output_layer = nn.Linear(self.hidden_dims[-1] * 2 * 2, self.n_output_nodes)
        self.device = get_device()

    def forward(self, inputs: Tensor, extra_features: Tensor = None) -> dict:
        latents = self.encoder(inputs.to(self.device))
        latents = torch.flatten(latents, 1)
        x = self.dropout(latents)
        if extra_features is not None:
            x = torch.cat([x, extra_features], dim=1)
        outputs = self.output_layer(x)
        output = {"outputs": outputs, "latents": latents}
        return output


class CustomResNet(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        dropout_rate: float = 0,
    ) -> None:
        super().__init__(
            block=block,
            layers=layers,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
        )
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        self.model_base_type = "clf"

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        return super()._make_layer(
            block=block, planes=planes, blocks=blocks, stride=stride,
        )

    def forward(self, x: Tensor, extra_features: Tensor = None) -> dict:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        latents = torch.flatten(x, 1)
        x = self.dropout(latents)
        if extra_features is not None:
            x = torch.cat([x, extra_features.view(latents.size(0), -1)], dim=1)
        outputs = self.fc(x)

        output = {"outputs": outputs, "latents": latents}
        return output


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = CustomResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 classifier from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a classifier pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 classifier from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a classifier pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 classifier from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a classifier pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 classifier from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a classifier pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 classifier from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a classifier pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
    )


class SimpleDiscriminator(nn.Module, ABC):
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dims: List = [1024, 1024, 1024],
        n_classes: int = 2,
        trainable: bool = True,
        extra_feature_dim: int = 0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes
        self.trainable = trainable
        self.extra_feature_dim = extra_feature_dim
        self.model_base_type = "clf"

        if hidden_dims is not None:
            model_modules = [
                nn.Sequential(
                    nn.Linear(self.latent_dim + self.extra_feature_dim, hidden_dims[0]),
                    nn.ReLU(),
                    nn.BatchNorm1d(self.hidden_dims[0]),
                )
            ]
            for i in range(0, len(self.hidden_dims) - 1):
                model_modules.append(
                    nn.Sequential(
                        nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]),
                        nn.ReLU(),
                        nn.BatchNorm1d(self.hidden_dims[i + 1]),
                    )
                )
            model_modules.append(nn.Linear(self.hidden_dims[-1], self.n_classes))
            self.feature_extractor = nn.Sequential(*model_modules)
        else:
            self.model = nn.Linear(self.latent_dim, self.n_classes)

    def forward(self, input: Tensor, extra_features: Tensor = None) -> dict:
        if extra_features is not None:
            input = torch.cat([input, extra_features], dim=1)
        outputs = self.model(input)
        return {"outputs": outputs, "latents": None}


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int, n_output_nodes, hidden_dims: List = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_output_nodes = n_output_nodes
        self.model_base_type = "clf"

        modules = []
        if self.hidden_dims is not None and len(self.hidden_dims) > 0:
            modules.append(
                nn.Sequential(
                    nn.Linear(self.input_dim, self.hidden_dims[0]),
                    nn.ReLU(),
                    nn.BatchNorm1d(self.hidden_dims[0]),
                )
            )
            for i in range(1, len(self.hidden_dims)):
                modules.append(
                    nn.Sequential(
                        nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]),
                        nn.ReLU(),
                        nn.BatchNorm1d(self.hidden_dims[i]),
                    )
                )
            modules.append(nn.Linear(self.hidden_dims[-1], self.n_output_nodes))
        else:
            modules.append(nn.Linear(self.input_dim, self.n_output_nodes))
        self.model = nn.Sequential(*modules)

    def forward(self, input: Tensor, extra_features: Tensor = None) -> dict:
        if extra_features is not None:
            input = torch.cat([input, extra_features], dim=1)
        outputs = self.model(input)
        return {"outputs": outputs, "latents": None}
