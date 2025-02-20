import copy
import logging
import os
import time
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from src.helper.models import DomainConfig, DomainModelConfig, BatchModelConfig
from src.utils.torch.evaluation import visualize_image_ae_performance
from src.utils.torch.general import get_device


def model_train_val_test_loop(
    output_dir: str,
    domain_config: DomainConfig,
    num_epochs: int = 500,
    early_stopping: int = 20,
    lamb: float = 0.01,
    device: str = None,
    save_freq: int = -1,
    batch_clf_config: BatchModelConfig = None,
) -> Tuple[dict, dict, dict]:
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

    total_loss_dict = {"train": [], "val": [], "test": None}

    # Reserve space for best classifier weights
    best_model_weights = domain_config.domain_model_config.model.cpu().state_dict()
    best_total_loss = np.infty
    best_accuracy = 0
    best_epoch = -1

    model_base_type = domain_config.domain_model_config.model.model_base_type

    logging.debug(
        "Start training of classifier {}".format(
            str(domain_config.domain_model_config.model)
        )
    )

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
                " balanced accuracy for {} epochs.".format(early_stopping)
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
                batch_clf_config=batch_clf_config,
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

            if "clf_balanced_accuracy" in epoch_statistics:
                logging.debug(
                    "Classification balanced accuracy for domain {}: {:.8f}".format(
                        domain_config.name, epoch_statistics["clf_balanced_accuracy"]
                    )
                )

            if "adv_batch_clf_loss" in epoch_statistics:
                logging.debug(
                    "Adversarial batch classification loss for {} domain: {:.8f}"
                    .format(domain_config.name, epoch_statistics["adv_batch_clf_loss"])
                )

            if "batch_clf_loss" in epoch_statistics:
                logging.debug(
                    "Batch classification loss for {} domain: {:.8f}".format(
                        domain_config.name, epoch_statistics["batch_clf_loss"]
                    )
                )

            if "batch_clf_accuracy" in epoch_statistics:
                logging.debug(
                    "Batch classification accuracy for domain {}: {:.8f}".format(
                        domain_config.name, epoch_statistics["batch_clf_accuracy"]
                    )
                )

            if "batch_clf_balanced_accuracy" in epoch_statistics:
                logging.debug(
                    "Batch classification balanced accuracy for domain {}: {:.8f}"
                    .format(
                        domain_config.name,
                        epoch_statistics["batch_clf_balanced_accuracy"],
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
                # Save classifier states if current parameters give the best validation loss

                # TODO undo changes that triggers early stopping based on accuracy not loss (for single target vs control desired)
                # if epoch_total_loss < best_total_loss:
                if (
                    batch_clf_config is None
                    and epoch_statistics["clf_balanced_accuracy"] > best_accuracy
                ) or (
                    batch_clf_config is not None and epoch_total_loss < best_total_loss
                ):
                    best_epoch = i
                    es_counter = 0
                    best_total_loss = epoch_total_loss
                    best_accuracy = epoch_statistics["clf_balanced_accuracy"]

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

            # Save classifier at checkpoints and visualize performance
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
                "{}/classifier.pth".format(checkpoint_dir),
            )

    # Training complete
    time_elapsed = time.time() - start_time

    logging.debug("###" * 20)
    logging.debug(
        "Training completed in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, int(time_elapsed % 60)
        )
    )

    logging.debug("Best model found at epoch {}".format(best_epoch + 1))
    logging.debug("***" * 20)

    # Load best classifier
    domain_config.domain_model_config.model.load_state_dict(best_model_weights)

    if "test" in domain_config.data_loader_dict:
        epoch_statistics = process_single_epoch(
            domain_config=domain_config,
            lamb=lamb,
            phase="test",
            device=device,
            batch_clf_config=batch_clf_config,
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

        if "clf_balanced_accuracy" in epoch_statistics:
            logging.debug(
                "Test classification balanced accuracy for domain {}: {:.8f}".format(
                    domain_config.name, epoch_statistics["clf_balanced_accuracy"]
                )
            )

        if "adv_batch_clf_loss" in epoch_statistics:
            logging.debug(
                "Adversarial batch classification loss for {} domain: {:.8f}".format(
                    domain_config.name, epoch_statistics["adv_batch_clf_loss"]
                )
            )

        if "batch_clf_loss" in epoch_statistics:
            logging.debug(
                "Batch classification loss for {} domain: {:.8f}".format(
                    domain_config.name, epoch_statistics["batch_clf_loss"]
                )
            )

        if "batch_clf_accuracy" in epoch_statistics:
            logging.debug(
                "Batch classification accuracy for domain {}: {:.8f}".format(
                    domain_config.name, epoch_statistics["batch_clf_accuracy"]
                )
            )
        if "batch_clf_balanced_accuracy" in epoch_statistics:
            logging.debug(
                "Batch classification balanced accuracy for domain {}: {:.8f}".format(
                    domain_config.name, epoch_statistics["batch_clf_balanced_accuracy"]
                )
            )

        total_loss_dict["test"] = epoch_statistics["total_loss"]
        best_total_loss_dict = {
            "train": total_loss_dict["train"][best_epoch],
            "val": total_loss_dict["val"][best_epoch],
            "test": total_loss_dict["test"],
        }
        logging.debug(
            "Total test loss for {} domain: {:.8f}".format(
                domain_config.name, epoch_statistics["total_loss"]
            )
        )
        logging.debug("***" * 20)

        logging.debug("***" * 20)

        # Visualize classifier performance
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
            pass
            # confusion_matrices = get_confusion_matrices(
            #     domain_config=domain_config, dataset_types=["train", "val", "test"]
            # )
            # # logging.debug("Confusion matrices for classifier: %s", confusion_matrices)
            # plot_confusion_matrices(confusion_matrices, output_dir=output_dir)

        torch.save(
            domain_config.domain_model_config.model.state_dict(),
            "{}/classifier.pth".format(test_dir),
        )

    return (
        domain_config.domain_model_config.model,
        total_loss_dict,
        best_total_loss_dict,
    )


def process_single_epoch(
    domain_config: DomainConfig,
    lamb: float = 1e-3,
    phase: str = "train",
    device: str = "cuda:0",
    epoch: int = -1,
    batch_clf_config=None,
) -> dict:
    # Get domain configurations for the domain
    domain_model_config = domain_config.domain_model_config
    data_loader_dict = domain_config.data_loader_dict
    data_loader = data_loader_dict[phase]
    data_key = domain_config.data_key
    label_key = domain_config.label_key
    extra_feature_key = domain_config.extra_feature_key
    batch_key = domain_config.batch_key

    # Initialize epoch statistics    recon_loss = 0
    recon_loss = 0
    clf_loss = 0
    n_correct = 0
    n_total = 0
    adv_batch_clf_loss = 0
    batch_clf_loss = 0
    batch_n_correct = 0
    batch_n_total = 0
    total_loss = 0
    labels = np.array([])
    preds = np.array([])
    batch_preds = np.array([])
    batch_labels = np.array([])

    model_base_type = domain_model_config.model.model_base_type.lower()

    # Iterate over batches
    for index, samples in enumerate(
        tqdm(data_loader, desc="Epoch {} progress for {} phase".format(epoch, phase))
    ):
        # Set model_configs
        domain_model_config.inputs = samples[data_key]
        domain_model_config.labels = samples[label_key]

        if extra_feature_key is not None:
            domain_model_config.extra_features = samples[extra_feature_key]

        if batch_key is not None:
            domain_model_config.batch_labels = samples[batch_key]

        batch_statistics = process_single_batch(
            domain_model_config=domain_model_config,
            lamb=lamb,
            phase=phase,
            device=device,
            model_base_type=model_base_type,
            batch_clf_config=batch_clf_config,
        )
        if "recon_loss" in batch_statistics:
            recon_loss += batch_statistics["recon_loss"]

        if "clf_loss" in batch_statistics:
            clf_loss += batch_statistics["clf_loss"]

        if "n_correct" in batch_statistics:
            n_correct += batch_statistics["n_correct"]

        if "n_total" in batch_statistics:
            n_total += batch_statistics["n_total"]

        if "preds" in batch_statistics:
            preds = np.append(preds, batch_statistics["preds"])

        if "labels" in batch_statistics:
            labels = np.append(labels, batch_statistics["labels"])

        if "adv_batch_clf_loss" in batch_statistics:
            adv_batch_clf_loss += batch_statistics["adv_batch_clf_loss"]

        if "batch_preds" in batch_statistics:
            batch_preds = np.append(batch_preds, batch_statistics["batch_preds"])

        if "batch_labels" in batch_statistics:
            batch_labels = np.append(batch_labels, batch_statistics["batch_labels"])

        if "batch_clf_loss" in batch_statistics:
            batch_clf_loss += batch_statistics["batch_clf_loss"]

        if "batch_n_correct" in batch_statistics:
            batch_n_correct += batch_statistics["batch_n_correct"]

        if "batch_n_total" in batch_statistics:
            batch_n_total += batch_statistics["batch_n_total"]

        total_loss += batch_statistics["total_loss"]

    # Get average over batches for statistics
    recon_loss /= len(data_loader.dataset)
    clf_loss /= len(data_loader.dataset)
    if n_total != 0:
        clf_accuracy = n_correct / n_total
    else:
        clf_accuracy = -1
    if len(preds) > 0:
        clf_bac = balanced_accuracy_score(labels, preds)
    else:
        clf_bac = -1

    if batch_n_total != 0:
        batch_clf_accuracy = batch_n_correct / batch_n_total
    else:
        batch_clf_accuracy = -1

    if len(batch_preds) > 0:
        batch_clf_bac = balanced_accuracy_score(batch_labels, batch_preds)
    else:
        batch_clf_bac = -1

    total_loss /= len(data_loader.dataset)

    epoch_statistics = {
        "total_loss": total_loss,
    }
    if model_base_type in ["ae", "vae"]:
        epoch_statistics["recon_loss"] = recon_loss

    if model_base_type == "clf":
        epoch_statistics["clf_loss"] = clf_loss
        epoch_statistics["clf_accuracy"] = clf_accuracy
        epoch_statistics["clf_balanced_accuracy"] = clf_bac

    if batch_clf_config is not None:
        epoch_statistics["adv_batch_clf_loss"] = adv_batch_clf_loss / len(
            data_loader.dataset
        )
        epoch_statistics["batch_clf_loss"] = batch_clf_loss / len(data_loader.dataset)
        epoch_statistics["batch_clf_accuracy"] = batch_clf_accuracy
        epoch_statistics["batch_clf_balanced_accuracy"] = batch_clf_bac

    return epoch_statistics


# Todo thin that function
def process_single_batch(
    domain_model_config: DomainModelConfig,
    lamb: float = 1e-3,
    phase: str = "train",
    device: str = "cuda:0",
    model_base_type: str = None,
    batch_clf_config: BatchModelConfig = None,
) -> dict:
    # Get all parameters of the configuration for domain i
    model = domain_model_config.model
    optimizer = domain_model_config.optimizer
    inputs = domain_model_config.inputs
    labels = domain_model_config.labels
    extra_features = domain_model_config.extra_features
    batch_labels = domain_model_config.batch_labels
    train = domain_model_config.trainable

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

    # Set classifier to train if defined in respective configuration
    model.to(device)

    if phase == "train" and train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    if batch_clf_config is not None:
        batch_clf = batch_clf_config.model
        batch_clf_loss_function = batch_clf_config.loss_function
        batch_clf_optim = batch_clf_config.optimizer
        batch_clf.to(device)

    # Forward pass of the classifier
    inputs = inputs
    labels = torch.LongTensor(labels).to(device)

    if extra_features is not None:
        extra_features = extra_features.float().to(device)

    if batch_labels is not None:
        batch_labels = batch_labels.float().to(device)
        outputs = model(inputs, extra_features, batch_labels)
    else:
        outputs = model(inputs, extra_features)

    total_loss = 0

    if model_base_type is None or model_base_type not in ["ae", "clf"]:
        raise RuntimeError("Unknown classifier given. No base type defined.")

    if model_base_type == "ae":
        recons = outputs["recons"]
        latents = outputs["latents"]
        recon_loss = domain_model_config.loss_function(inputs, recons)
        total_loss += recon_loss

    elif model_base_type == "clf":
        clf_outputs = outputs["outputs"]
        latents = outputs["latents"]
        clf_loss = domain_model_config.loss_function(clf_outputs, labels)
        _, preds = torch.max(clf_outputs, 1)
        n_total = preds.size(0)
        n_correct = torch.sum(torch.eq(labels, preds)).item()
        total_loss += clf_loss
    else:
        raise RuntimeError("Unknown classifier type: {}".format(model_base_type))

    if batch_clf_config is not None:
        batch_clf.eval()
        batch_clf_outputs = batch_clf(latents, extra_features)["outputs"]
        # Todo hard-coded for max batch which is encoded as label 0
        pseudo_batch_labels = torch.zeros_like(batch_labels.max(dim=1)[1])
        pseudo_batch_labels = torch.randint_like(batch_labels.max(dim=1)[1], low=0, high=batch_labels.max(dim=1)[1].max()+1)
        pseudo_batch_clf_loss = batch_clf_loss_function(
            batch_clf_outputs, pseudo_batch_labels
        )
        _, pseudo_batch_preds = torch.max(batch_clf_outputs, 1)
        # batch_n_total = batch_preds.size(0)
        pseudo_batch_n_correct = torch.sum(
            torch.eq(pseudo_batch_labels, pseudo_batch_preds)
        ).item()
        total_loss += lamb * pseudo_batch_clf_loss

    # Backpropagate loss and update parameters if we are in the training phase
    if phase == "train":
        total_loss.backward()
        if train:
            optimizer.step()
            model.updated = True
        scheduler.step(total_loss)

    # Get summary statistics
    batch_size = labels.size(0)
    total_loss_item = total_loss.item() * batch_size

    # Train batch classifier
    if batch_clf_config:
        model.eval()
        with torch.no_grad():
            latents = model(inputs, extra_features, batch_labels)["latents"]

        if phase == "train":
            batch_clf.train()
            batch_clf_optim.zero_grad()

        batch_clf_outputs = batch_clf(latents, extra_features)["outputs"]
        batch_clf_loss = batch_clf_loss_function(
            batch_clf_outputs, batch_labels.max(dim=1)[1]
        )
        _, batch_preds = torch.max(batch_clf_outputs, 1)
        batch_n_total = batch_preds.size(0)
        batch_n_correct = torch.sum(
            torch.eq(batch_labels.max(dim=1)[1], batch_preds)
        ).item()

        if phase == "train":
            batch_clf_loss.backward()
            batch_clf_optim.step()

    batch_statistics = dict()

    if model_base_type in ["ae", "vae"]:
        batch_statistics["recon_loss"] = recon_loss.item() * batch_size
    if model_base_type == "clf":
        batch_statistics["clf_loss"] = clf_loss.item() * batch_size
        batch_statistics["n_correct"] = n_correct
        batch_statistics["n_total"] = n_total
        batch_statistics["preds"] = preds.detach().cpu().numpy()
        batch_statistics["labels"] = labels.detach().cpu().numpy()
    if batch_clf_config is not None:
        batch_statistics["adv_batch_clf_loss"] = (
            pseudo_batch_clf_loss.item() * lamb * batch_size
        )
        batch_statistics["adv_batch_n_correct"] = pseudo_batch_n_correct
        batch_statistics["batch_clf_loss"] = batch_clf_loss.item() * batch_size
        batch_statistics["batch_n_correct"] = batch_n_correct
        batch_statistics["batch_n_total"] = batch_n_total
        batch_statistics["batch_preds"] = batch_preds.detach().cpu().numpy()
        batch_statistics["batch_labels"] = (
            batch_labels.max(dim=1)[1].detach().cpu().numpy()
        )

    batch_statistics["total_loss"] = total_loss_item

    return batch_statistics
