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
    save_latents_from_model,
    visualize_image_ae_performance, get_confusion_matrices,
)
from src.utils.torch.general import get_device


def model_train_val_test_loop(
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

    model_base_type = domain_config.domain_model_config.model.model_base_type

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
        if i % save_freq == 0:
            checkpoint_dir = "{}/epoch_{}".format(output_dir, i)
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Iterate over training and validation phase
        for phase in ["train", "val"]:
            epoch_statistics = process_single_epoch(
                domain_config=domain_config,
                lamb=lamb,
                phase=phase,
                device=device,
                epoch=i,
            )

            logging.debug(
                "{} LOSS STATISTICS FOR EPOCH {}: ".format(phase.upper(), i + 1)
            )

            if "recon_loss" in epoch_statistics:
                logging.debug(
                    "Reconstruction loss  for {} domain: {:.8f}".format(
                        domain_config.name, epoch_statistics["recon_loss"]
                    )
                )

            if "clf_loss" in epoch_statistics:
                logging.debug(
                    "Classification loss for {} domain: {:.8f}".format(
                        domain_config.name, epoch_statistics["clf_loss"]
                    )
                )

            if "clf_accuracy" in epoch_statistics:
                logging.debug(
                    "Classification accuracy for domain {}: {:.8f}".format(
                        domain_config.name, epoch_statistics["clf_accuracy"]
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
                if model_base_type in ["ae", "vae"]:
                    if domain_config.name == "image":
                        visualize_image_ae_performance(
                        domain_model_config=domain_config.domain_model_config,
                        epoch=i,
                        output_dir=checkpoint_dir,
                        device=device,
                        phase=phase,
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
        epoch_statistics = process_single_epoch(
            domain_config=domain_config, lamb=lamb, phase="test", device=device
        )

        logging.debug("TEST LOSS STATISTICS")

        if "recon_loss" in epoch_statistics:
            logging.debug(
                "Test reconstruction loss  for {} domain: {:.8f}".format(
                    domain_config.name, epoch_statistics["recon_loss"]
                )
            )

        if "clf_loss" in epoch_statistics:
            logging.debug(
                "Test classification loss for {} domain: {:.8f}".format(
                    domain_config.name, epoch_statistics["clf_loss"]
                )
            )

        if "clf_accuracy" in epoch_statistics:
            logging.debug(
                "Test classification accuracy for domain {}: {:.8f}".format(
                    domain_config.name, epoch_statistics["clf_accuracy"]
                )
            )

        if "kl_loss" in epoch_statistics:
            logging.debug(
                "Test KL loss for {} domain: {:.8f}".format(
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

        if model_base_type in ["ae", "vae"] and domain_config.name == "image":
            visualize_image_ae_performance(
                domain_model_config=domain_config.domain_model_config,
                epoch=i,
                output_dir=test_dir,
                device=device,
                phase="test",
            )
        elif model_base_type == "clf":
            confusion_matrices = get_confusion_matrices(domain_config=domain_config,
                                                        dataset_types=["train", "val", "test"])
            logging.debug("Confusion matrices for classifier: %s", confusion_matrices)

        save_latents_from_model(
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


def process_single_epoch(
    domain_config: DomainConfig,
    lamb: float = 1e-7,
    phase: str = "train",
    device: str = "cuda:0",
    epoch: int = -1,
) -> dict:
    # Get domain configurations for the domain
    domain_model_config = domain_config.domain_model_config
    data_loader_dict = domain_config.data_loader_dict
    data_loader = data_loader_dict[phase]
    data_key = domain_config.data_key
    label_key = domain_config.label_key

    # Initialize epoch statistics
    recon_loss = 0
    clf_loss = 0
    n_correct = 0
    n_total = 0
    kl_loss = 0
    total_loss = 0

    model_base_type = domain_model_config.model.model_base_type.lower()

    # Iterate over batches
    for index, samples in enumerate(
        tqdm(data_loader, desc="Epoch {} progress for {} phase".format(epoch, phase))
    ):
        # Set model_configs
        domain_model_config.inputs = samples[data_key]
        domain_model_config.labels = samples[label_key]

        batch_statistics = process_single_batch(
            domain_model_config=domain_model_config,
            lamb=lamb,
            phase=phase,
            device=device,
            model_base_type=model_base_type,
        )
        if "recon_loss" in batch_statistics:
            recon_loss += batch_statistics["recon_loss"]

        if "clf_loss" in batch_statistics:
            clf_loss += batch_statistics["clf_loss"]

        if "kl_loss" in batch_statistics:
            kl_loss += batch_statistics["kl_loss"]

        if "n_correct" in batch_statistics:
            n_correct += batch_statistics["n_correct"]

        if "n_total" in batch_statistics:
            n_total += batch_statistics["n_total"]

        total_loss += batch_statistics["total_loss"]

    # Get average over batches for statistics
    recon_loss /= len(data_loader.dataset)
    clf_loss /= len(data_loader.dataset)
    if n_total != 0:
        clf_accuracy = n_correct / n_total
    else:
        clf_accuracy = -1
    kl_loss /= len(data_loader.dataset)
    total_loss /= len(data_loader.dataset)

    epoch_statistics = {
        "total_loss": total_loss,
    }
    if model_base_type in ["ae", "vae"]:
        epoch_statistics["recon_loss"] = recon_loss

    if model_base_type == "vae":
        epoch_statistics["kl_loss"] = kl_loss

    if model_base_type == "clf":
        epoch_statistics["clf_loss"] = clf_loss
        epoch_statistics["clf_accuracy"] = clf_accuracy

    return epoch_statistics


# Todo thin that function
def process_single_batch(
    domain_model_config: DomainModelConfig,
    lamb: float = 1e-8,
    phase: str = "train",
    device: str = "cuda:0",
    model_base_type: str = None,
) -> dict:
    # Get all parameters of the configuration for domain i
    model = domain_model_config.model
    optimizer = domain_model_config.optimizer
    inputs = domain_model_config.inputs
    labels = domain_model_config.labels
    train = domain_model_config.trainable

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

    # Set model to train if defined in respective configuration
    model.to(device)

    if phase == "train" and train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    # Forward pass of the model
    inputs = torch.FloatTensor(inputs).to(device)
    labels = torch.LongTensor(labels).to(device)

    outputs = model(inputs)

    if model_base_type is None or model_base_type not in ["ae", "vae", "clf"]:
        raise RuntimeError("Unknown model given. No base type defined.")

    if model_base_type == "vae":
        recons = outputs["recons"]
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
        recons = outputs["recons"]
        recon_loss = domain_model_config.loss_function(inputs, recons)
        total_loss = recon_loss
    elif model_base_type == "clf":
        clf_outputs = outputs["outputs"]
        clf_loss = domain_model_config.loss_function(clf_outputs, labels)
        _, preds = torch.max(clf_outputs, 1)
        n_total = preds.size(0)
        n_correct = torch.sum(torch.eq(labels, preds)).item()
        total_loss = clf_loss
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
    total_loss_item = total_loss.item() * batch_size

    batch_statistics = dict()

    if model_base_type in ["ae", "vae"]:
        batch_statistics["recon_loss"] = recon_loss.item() * batch_size
    if model_base_type == "vae":
        batch_statistics["kl_loss"] = kl_loss.item()
        total_loss_item += kl_loss.item() * lamb
    if model_base_type == "clf":
        batch_statistics["clf_loss"] = clf_loss.item() * batch_size
        batch_statistics["n_correct"] = n_correct
        batch_statistics["n_total"] = n_total

    batch_statistics["total_loss"] = total_loss_item

    return batch_statistics
