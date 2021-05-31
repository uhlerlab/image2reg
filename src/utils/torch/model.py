import torch
from torch.nn import L1Loss, MSELoss, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam, RMSprop
from torch.nn import Module
from torch.optim import Optimizer
from torch import nn

from src.helper.models import DomainConfig
from src.models.ae import VanillaConvAE

from torchvision import transforms, models

import logging

from src.models.clf import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    SimpleConvClassifier,
)
from src.utils.torch.general import get_device


def get_optimizer_for_model(optimizer_dict: dict, model: Module) -> Optimizer:
    optimizer_type = optimizer_dict.pop("type")
    if optimizer_type == "adam":
        optimizer = Adam(model.parameters(), **optimizer_dict)
    elif optimizer_type == "rmsprop":
        optimizer = RMSprop(model.parameters(), **optimizer_dict)
    else:
        raise NotImplementedError('Unknown optimizer type "{}"'.format(optimizer_type))
    return optimizer


def get_domain_configuration(
    name: str,
    model_dict: dict,
    optimizer_dict: dict,
    loss_fct_dict: dict,
    data_loader_dict: dict,
    data_key: str,
    label_key: str,
    train_model: bool = True,
) -> DomainConfig:

    model_type = model_dict.pop("type")
    if model_type.lower() == "vanillaconvae":
        model = VanillaConvAE(**model_dict)
    elif "resnet" in model_type.lower():
        model = initialize_imagenet_model(model_name=model_type, **model_dict)
    elif model_type.lower() == "simpleconvclf":
        model = SimpleConvClassifier(**model_dict)
    else:
        raise NotImplementedError('Unknown model type "{}"'.format(model_type))

    optimizer = get_optimizer_for_model(optimizer_dict=optimizer_dict, model=model)

    loss_fct_type = loss_fct_dict.pop("type")
    if loss_fct_type == "mae":
        loss_function = L1Loss()
    elif loss_fct_type == "mse":
        loss_function = MSELoss()
    elif loss_fct_type == "bce":
        loss_function = BCELoss()
    elif loss_fct_type == "bce_ll":
        loss_function = BCEWithLogitsLoss()
    elif loss_fct_type == "ce":
        if "weight" in loss_fct_dict:
            weight = torch.FloatTensor(loss_fct_dict["weight"]).to(get_device())
        else:
            weight = None
        loss_function = CrossEntropyLoss(weight)
    else:
        raise NotImplementedError(
            'Unknown loss function type "{}"'.format(loss_fct_type)
        )

    domain_config = DomainConfig(
        name=name,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        data_loader_dict=data_loader_dict,
        data_key=data_key,
        label_key=label_key,
        train_model=train_model,
    )

    return domain_config


def get_imagenet_extended_transformations_dict(input_size):
    data_transforms = {
        # In the original paper the random permutation were used to make models more general to diverse pictures.
        "train": transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # transforms.RandomAffine(degrees=180),
                transforms.RandomCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                # transforms.RandomAffine(degrees=180),
                # transforms.RandomCrop(input_size),
                # transforms.RandomCrop(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                # transforms.RandomAffine(degrees=180),
                # transforms.RandomCrop(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    return data_transforms


def get_image_net_transformations_dict(input_size):
    r""" Method returning a transformation dictionary, defining the transformations was applied when training,
    validating and testing models from :py:mod:`torchvision.models` on the ImageNet dataset.

    Parameters
    ----------
    input_size : int
        The size of the input that is supposed to be processed by the transformation pipeline.

    Returns
    -------
    data_transforms : dict
        A dictionary with the transformation pipelines for the training, validation and testing phase of the model
        training procedure as the values to the respective keys ``train``, ``val`` and ``test``.

    """
    data_transforms = {
        # In the original paper the random permutation were used to make models more general to diverse pictures.
        "train": transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    return data_transforms


def get_image_net_nonrandom_transformations_dict(input_size):
    r""" Method returning a transformation dictionary, defining the transformations was applied when training,
    validating and testing models from :py:mod:`torchvision.models` on the ImageNet dataset without the random data
    augmentations applied during training to improve the generalizability of the trained models.

    Parameters
    ----------
    input_size : int
        The size of the input that is supposed to be processed by the transformation pipeline.

    Returns
    -------
    data_transforms : dict
        A dictionary with the transformation pipelines for the training, validation and testing phase of the model
        training procedure as the values to the respective keys ``train``, ``val`` and ``test``.

    """
    transform_dict = get_image_net_transformations_dict(input_size)
    transform_dict["train"] = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform_dict


def initialize_imagenet_model(
    model_name,
    n_output_nodes,
    fix_feature_extractor: bool = False,
    pretrained: bool = True,
    fix_first_k_layers=None,
):
    r""" Method to get an initialized a Imagenet CNN model.

    Parameters
    ----------
    model_name : str
        Identifier of the model that is supposed to be used. Supported options include: ``"alexnet"``, ``"resnet18"``,
        ``"resnet34"``, ``"resnet50"``, ``"resnet101"``, ``"resnet152"``, ``"vgg"``, ``"densenet"``, ``"inception"``,
        ``"squeezenet"``.

    n_output_nodes : int
        Number of output neurons in the final layer of the model.

    fix_feature_extractor : bool
        Indicator, if set to True the model is completely frozen, i.e. none of the parameters will be optimized
        during the training of the model.

    pretrained : bool
        Indicator, if the weights of the model are supposed to be initialized with the weights obtained from training
        the network on the Imagenet dataset

    fix_first_k_layers : int
        The number of layers of the model that are frozen, i.e. whose parameters will not be optimized during the
        training of the model.

    Returns
    -------
    model_ft : :py:class:`~torch.nn.Module`
        The initialized CNN model.

    input_size : int
        The dimensions of the input the model_ft expects.

    """
    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific.
    model_ft = None
    input_size = 0

    model_name = model_name.lower()

    if model_name == "resnet18":
        model_ft = resnet18(pretrained=pretrained)
        set_parameter_requires_grad(model_ft, fix_feature_extractor, fix_first_k_layers)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_output_nodes)
        input_size = 224

    elif model_name == "resnet34":
        model_ft = resnet34(pretrained=pretrained)
        set_parameter_requires_grad(model_ft, fix_feature_extractor, fix_first_k_layers)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_output_nodes)
        input_size = 224

    elif model_name == "resnet50":
        model_ft = resnet50(pretrained=pretrained)
        set_parameter_requires_grad(model_ft, fix_feature_extractor, fix_first_k_layers)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_output_nodes)
        input_size = 224

    elif model_name == "resnet101":
        model_ft = resnet101(pretrained=pretrained)
        set_parameter_requires_grad(model_ft, fix_feature_extractor, fix_first_k_layers)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_output_nodes)
        input_size = 224

    elif model_name == "resnet152":
        model_ft = resnet152(pretrained=pretrained)
        set_parameter_requires_grad(model_ft, fix_feature_extractor, fix_first_k_layers)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_output_nodes)
        input_size = 224

    elif model_name == "alexnet":
        model_ft = models.alexnet(pretrained=pretrained)
        set_parameter_requires_grad(model_ft, fix_feature_extractor, fix_first_k_layers)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, n_output_nodes)
        input_size = 224

    elif model_name == "vgg":
        model_ft = models.vgg11_bn(pretrained=pretrained)
        set_parameter_requires_grad(model_ft, fix_feature_extractor, fix_first_k_layers)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, n_output_nodes)
        input_size = 224

    elif model_name == "squeezenet":
        model_ft = models.squeezenet1_0(pretrained=pretrained)
        set_parameter_requires_grad(model_ft, fix_feature_extractor, fix_first_k_layers)
        model_ft.classifier[1] = nn.Conv2d(
            512, n_output_nodes, kernel_size=(1, 1), stride=(1, 1)
        )
        model_ft.num_classes = n_output_nodes
        input_size = 224

    elif model_name == "densenet":

        model_ft = models.densenet121(pretrained=pretrained)
        set_parameter_requires_grad(model_ft, fix_feature_extractor, fix_first_k_layers)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, n_output_nodes)
        input_size = 224

    elif model_name == "inception":
        model_ft = models.inception_v3(pretrained=pretrained)
        set_parameter_requires_grad(model_ft, fix_feature_extractor, fix_first_k_layers)
        # Handle the auxiliary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, n_output_nodes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_output_nodes)
        input_size = 299

    else:
        logging.debug("---" * 20)
        logging.error("Invalid model name, exiting...")
        exit()

    return model_ft


def set_parameter_requires_grad(
    model, fix_feature_extractor: bool = False, fix_first_k_layers=None
):
    r""" Method to set prevent the update of certain parameters in a given model.

    Parameters
    ----------
    model : :py:class:`~torch.nn.Module`
        The model of question.

    fix_feature_extractor : bool
        An indicator, if set to True the model is completely frozen, i.e. none of the parameters will be optimized
        during the training of the model.

    fix_first_k_layers : int
        The number of layers of the model that are frozen, i.e. whose parameters will not be optimized during the
        training of the model.

    """
    if fix_feature_extractor:
        for param in model.parameters():
            param.requires_grad = False
        fix_first_k_layers = None

    if fix_first_k_layers is not None:
        ct = 0
        for child in model.children():
            if ct < fix_first_k_layers:
                for param in child.parameters():
                    param.requires_grad = False
                ct += 1
