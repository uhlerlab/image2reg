from torch.nn import L1Loss, MSELoss, BCELoss, BCEWithLogitsLoss
from torch.optim import Adam, RMSprop
from torch.nn import Module
from torch.optim import Optimizer

from src.helper.models import DomainConfig
from src.models.ae import VanillaConvAE


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
    recon_loss_fct_dict: dict,
    data_loader_dict: dict,
    data_key: str,
    label_key: str,
    train_model: bool = True,
) -> DomainConfig:

    model_type = model_dict.pop("type")
    if model_type == "VanillaConvAE":
        model = VanillaConvAE(**model_dict)
    else:
        raise NotImplementedError('Unknown model type "{}"'.format(model_type))

    optimizer = get_optimizer_for_model(optimizer_dict=optimizer_dict, model=model)

    recon_loss_fct_type = recon_loss_fct_dict.pop("type")
    if recon_loss_fct_type == "mae":
        recon_loss_function = L1Loss()
    elif recon_loss_fct_type == "mse":
        recon_loss_function = MSELoss()
    elif recon_loss_fct_type == "bce":
        recon_loss_function = BCELoss()
    elif recon_loss_fct_type == "bce_ll":
        recon_loss_function = BCEWithLogitsLoss()
    else:
        raise NotImplementedError(
            'Unknown loss function type "{}"'.format(recon_loss_fct_type)
        )

    domain_config = DomainConfig(
        name=name,
        model=model,
        optimizer=optimizer,
        recon_loss_function=recon_loss_function,
        data_loader_dict=data_loader_dict,
        data_key=data_key,
        label_key=label_key,
        train_model=train_model,
    )

    return domain_config
