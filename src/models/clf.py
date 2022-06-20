from abc import ABC
from typing import Type, List, Union, Any

import numpy as np
import torch
from torch import nn
from torch.functional import Tensor
from torch.hub import load_state_dict_from_url
from torch.nn import Module
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, model_urls

from src.utils.torch.general import get_device


class SimpleConvClassifier(Module):
    def __init__(
        self,
        n_output_nodes: int,
        input_channels: int = 1,
        hidden_dims: List[int] = [32, 64, 128, 256, 256],
        batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = input_channels
        self.hidden_dims = hidden_dims
        self.batchnorm = batchnorm
        self.updated = False
        self.n_output_nodes = n_output_nodes
        self.model_base_type = "clf"

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
        self.output_layer = nn.Linear(self.hidden_dims[-1] * 2 * 2, self.n_output_nodes)
        self.device = get_device()

    def forward(self, inputs: Tensor, extra_features: Tensor = None) -> dict:
        latents = self.encoder(inputs.float().to(self.device))
        latents = torch.flatten(latents, 1)
        if extra_features is not None:
            latents = torch.cat([latents, extra_features], dim=1)
        outputs = self.output_layer(latents)
        output = {"outputs": outputs, "latents": latents}
        return output


class ModelEnsemble(nn.Module):
    def __init__(self, models, input_dim: int, latent_dim: int, n_output_nodes: int):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(latent_dim),
            nn.PReLU(),
            nn.Linear(latent_dim, n_output_nodes),
        )
        self.device = get_device()
        # self.models.to(self.device)
        self.model_base_type = "clf"

    def forward(self, inputs: List[Tensor], extra_features: Tensor = None) -> dict:
        outputs = []

        for i in range(len(self.models)):
            output = self.models[i](inputs[i].to(self.device))
            outputs.append(output["outputs"])
        # latents = torch.cat(outputs, dim=1)
        latents = self.encoder(torch.cat(outputs, dim=1))
        if extra_features is not None:
            latents = torch.cat([latents, extra_features], dim=1)
        outputs = self.output_layer(latents)
        output = {"outputs": outputs, "latents": latents}
        return output


class CustomResNet(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__(
            block=block,
            layers=layers,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
        )
        self.model_base_type = "clf"
        self.device = get_device()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        return super()._make_layer(
            block=block, planes=planes, blocks=blocks, stride=stride,
        )

    def forward(self, x: Tensor, extra_features: Tensor = None) -> dict:
        x = x.float().to(self.device)
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
        if extra_features is not None:
            latents = torch.cat(
                [latents, extra_features.view(latents.size(0), -1)], dim=1
            )
        outputs = self.fc(latents)

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


class LatentClassifier(nn.Module):
    def __init__(
        self,
        latent_dim: int = 1024,
        hidden_dims: List = None,
        n_classes: int = 10,
        loss_fct: nn.Module = None,
        class_weights:np.ndarray=None
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes
        if class_weights is not None:
            self.class_weights = torch.FloatTensor(np.array(class_weights))
        else:
            self.class_weights = None
        if loss_fct is None:
            self.loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.loss_fct = loss_fct

        modules = []
        if self.hidden_dims is None:
            modules.append(nn.Linear(self.latent_dim, self.n_classes))
        else:
            modules.append(nn.Linear(self.latent_dim, self.hidden_dims[0]))
            modules.append(nn.BatchNorm1d(self.hidden_dims[0]))
            modules.append(nn.PReLU())
            for i in range(1, len(self.hidden_dims)):
                modules.append(nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]))
                modules.append(nn.BatchNorm1d(self.hidden_dims[i]))
                modules.append(nn.PReLU())
            modules.append(nn.Linear(self.hidden_dims[-1], self.n_classes))
        self.model = nn.Sequential(*modules)

    def forward(self, latents: Tensor):
        return self.model(latents)

    def loss(self, latents: Tensor, label_mask: Tensor, labels: Tensor):
        preds = self.forward(latents)
        return self.loss_fct(preds[label_mask], labels[label_mask])

    def reset_parameters(self):
        for layer in self.model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


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
        self.device = get_device()
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
        outputs = self.model(input.float().to(self.device))
        return {"outputs": outputs, "latents": input}


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int, n_output_nodes, hidden_dims: List = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_output_nodes = n_output_nodes
        self.device = get_device()
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
        outputs = self.model(input.float().to(self.device))
        return {"outputs": outputs, "latents": input}
