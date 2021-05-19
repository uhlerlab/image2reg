import copy
import logging
import os
import time
from typing import Tuple

import torch
import numpy as np
from tqdm import tqdm

from src.helper.models import DomainConfig, DomainModelConfig
from src.utils.torch.evaluation import (
    visualize_model_performance,
    visualize_image_ae_performance,
)
from src.utils.torch.general import get_device


def train_val_test_loop(
    output_dir: str,
    domain_config: DomainConfig,
    num_epochs: int = 500,
    lamb: float = 1e-7,
    early_stopping: int = 20,
    device: str = None,
    save_freq: int = -1,
) -> Tuple[dict, dict]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get available device, if cuda is available the GPU will be used
    if not device:
        device = get_device()

    # Store start time of the training
    start_time = time.time()

    # Initialize early stopping counter
    es_counter = 0
    if early_stopping < 0:
        early_stopping = num_epochs

    total_loss_dict = {"train": [], "val": []}

    # Reserve space for best model weights
    best_model_weights = domain_config.domain_model_config.model.cpu().state_dict()
    best_total_loss = np.infty

    # Iterate over the epochs
    for i in range(num_epochs):
        logging.debug("---" * 20)
        logging.debug("---" * 20)
        logging.debug("Started epoch {}/{}".format(i + 1, num_epochs))
        logging.debug("---" * 20)

        # Check if early stopping is triggered
        if es_counter > early_stopping:
            logging.debug(
                "Training was stopped early due to no improvement of the validation"
                " loss for {} epochs.".format(early_stopping)
            )
            break

        # Iterate over training and validation phase
        for phase in ["train", "val"]:
            epoch_statistics = process_epoch(
                domain_config=domain_config, lamb=lamb, phase=phase, device=device,
            )

            logging.debug(
                "{} LOSS STATISTICS FOR EPOCH {}: ".format(phase.upper(), i + 1)
            )

            logging.debug(
                "Reconstruction loss  for {} domain: {:.8f}".format(
                    domain_config.name, epoch_statistics["recon_loss"]
                )
            )
            if "kl_loss" in epoch_statistics:
                logging.debug(
                    "KL loss for {} domain: {:.8f}".format(
                        domain_config.name, epoch_statistics["kl_loss"]
                    )
                )

            epoch_total_loss = epoch_statistics["total_loss"]
            total_loss_dict[phase].append(epoch_total_loss)
            logging.debug("***" * 20)
            logging.debug(
                "Total {} loss for {} domain: {:.8f}".format(
                    phase, domain_config.name, epoch_total_loss
                )
            )
            logging.debug("***" * 20)

            if phase == "val":
                # Save model states if current parameters give the best validation loss
                if epoch_total_loss < best_total_loss:
                    es_counter = 0
                    best_total_loss = epoch_total_loss

                    best_model_weights = copy.deepcopy(
                        domain_config.domain_model_config.model.cpu().state_dict()
                    )
                    best_model_weights = best_model_weights

                    torch.save(
                        best_model_weights,
                        "{}/best_model_weights.pth".format(output_dir),
                    )
                else:
                    es_counter += 1

        # Save model at checkpoints and visualize performance
        if i % save_freq == 0:
            checkpoint_dir = "{}/epoch_{}".format(output_dir, i)

            visualize_image_ae_performance(
                domain_model_config=domain_config.domain_model_config,
                epoch=i,
                output_dir=checkpoint_dir,
                device=device,
                phase="train",
            )

            visualize_image_ae_performance(
                domain_model_config=domain_config.domain_model_config,
                epoch=i,
                output_dir=checkpoint_dir,
                device=device,
                phase="val",
            )

            visualize_model_performance(
                output_dir=checkpoint_dir,
                domain_config=domain_config,
                dataset_types=["train", "val"],
                device=device,
            )

            torch.save(
                domain_config.domain_model_config.model.state_dict(),
                "{}/model.pth".format(checkpoint_dir),
            )

    # Training complete
    time_elapsed = time.time() - start_time

    logging.debug("###" * 20)
    logging.debug(
        "Training completed in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, int(time_elapsed % 60)
        )
    )

    # Load best model
    domain_config.domain_model_config.model.load_state_dict(best_model_weights)

    if "test" in domain_config.data_loader_dict:
        epoch_statistics = process_epoch(
            domain_config=domain_config, lamb=lamb, phase="test", device=device,
        )

        logging.debug("TEST LOSS STATISTICS")

        logging.debug(
            "Reconstruction Loss for {} domain: {:.8f}".format(
                domain_config.name, epoch_statistics["recon_loss"]
            )
        )

        if "kl_loss" in epoch_statistics:
            logging.debug(
                "KL loss for {} domain: {:.8f}".format(
                    domain_config.name, epoch_statistics["kl_loss"]
                )
            )
        logging.debug("***" * 20)
        logging.debug(
            "Total test loss for {} domain: {:.8f}".format(
                domain_config.name, epoch_statistics["total_loss"]
            )
        )
        logging.debug("***" * 20)

        # Visualize model performance
        test_dir = "{}/test".format(output_dir)
        os.makedirs(test_dir, exist_ok=True)

        visualize_image_ae_performance(
            domain_model_config=domain_config.domain_model_config,
            epoch=i,
            output_dir=test_dir,
            device=device,
            phase="test",
        )

        visualize_model_performance(
            output_dir=test_dir,
            domain_config=domain_config,
            dataset_types=["train", "val", "test"],
            device=device,
        )

        torch.save(
            domain_config.domain_model_config.model.state_dict(),
            "{}/model.pth".format(test_dir),
        )

        # Visualize performance
    return domain_config.domain_model_config.model, total_loss_dict


def process_epoch(
    domain_config: DomainConfig,
    lamb: float = 1e-7,
    phase: str = "train",
    device: str = "cuda:0",
) -> dict:
    # Get domain configurations for the domain
    domain_model_config = domain_config.domain_model_config
    data_loader_dict = domain_config.data_loader_dict
    data_loader = data_loader_dict[phase]
    data_key = domain_config.data_key
    label_key = domain_config.label_key

    # Initialize epoch statistics
    recon_loss = 0
    kl_loss = 0
    total_loss = 0

    model_base_type = domain_model_config.model.model_base_type.lower()

    # Iterate over batches
    for index, samples in enumerate(tqdm(data_loader, desc="Epoch progress")):
        # Set model_configs
        domain_model_config.inputs = samples[data_key]
        domain_model_config.labels = samples[label_key]

        batch_statistics = train_autoencoder(
            domain_model_config=domain_model_config,
            lamb=lamb,
            phase=phase,
            device=device,
            model_base_type=model_base_type,
        )

        recon_loss += batch_statistics["recon_loss"]
        if model_base_type == "vae":
            kl_loss += batch_statistics["kl_loss"]
        total_loss += batch_statistics["total_loss"]

    # Get average over batches for statistics
    recon_loss /= len(data_loader.dataset)
    kl_loss /= len(data_loader.dataset)
    total_loss /= len(data_loader.dataset)

    epoch_statistics = {
        "recon_loss": recon_loss,
        "total_loss": total_loss,
    }
    if model_base_type == "vae":
        epoch_statistics["kl_loss"] = kl_loss
    return epoch_statistics


def train_autoencoder(
    domain_model_config: DomainModelConfig,
    lamb: float = 1e-8,
    phase: str = "train",
    device: str = "cuda:0",
    model_base_type: str = "ae",
) -> dict:
    # Get all parameters of the configuration for domain i
    model = domain_model_config.model
    optimizer = domain_model_config.optimizer
    inputs = domain_model_config.inputs
    train = domain_model_config.trainable

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

    # Set V/AE model to train if defined in respective configuration
    model.to(device)

    if phase == "train" and train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    # Forward pass of the V/AE
    inputs = torch.FloatTensor(inputs).to(device)

    outputs = model(inputs)
    recons = outputs["recons"]

    if model_base_type == "vae":
        mu = outputs["mu"]

        logvar = outputs["logvar"]

        loss_dict = model.loss_function(
            inputs=inputs, recons=recons, mu=mu, logvar=logvar
        )

        recon_loss = loss_dict["recon_loss"]
        kld_loss = loss_dict["kld_loss"]

        kl_loss = kld_loss
        total_loss = recon_loss + kl_loss * lamb

    elif model_base_type == "ae":
        recon_loss = domain_model_config.recon_loss_function(inputs, recons)
        total_loss = recon_loss
    else:
        raise RuntimeError("Unknown model type: {}".format(model_base_type))

    # Backpropagate loss and update parameters if we are in the training phase
    if phase == "train":
        total_loss.backward()
        if train:
            optimizer.step()
            model.updated = True
        scheduler.step(total_loss)

    # Get summary statistics
    batch_size = inputs.size(0)
    total_loss_item = recon_loss.item() * batch_size

    batch_statistics = {"recon_loss": recon_loss.item() * batch_size}

    if model_base_type == "vae":
        batch_statistics["kl_loss"] = kl_loss.item()
        total_loss_item += kl_loss.item() * lamb

    batch_statistics["total_loss"] = total_loss_item

    return batch_statistics
