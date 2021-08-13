import logging
from typing import List
from random import seed
import os
import numpy as np
import torch

from src.utils.basic.visualization import plot_train_val_hist
from src.utils.torch.general import get_device


class BaseExperiment:
    def __init__(
        self,
        output_dir: str,
        train_val_test_split: List = [0.7, 0.2, 0.1],
        batch_size: int = 64,
        num_epochs: int = 500,
        early_stopping: int = 20,
        random_state: int = 42,
    ):
        # I/O attributes
        self.output_dir = output_dir

        # Training attributes
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split

        # Other attributes
        self.random_state = random_state
        self.loss_dict = None
        self.best_loss_dict = None
        self.device = get_device()

        # Fix random seeds for reproducibility and limit applicable algorithms to those believed to be deterministic
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        seed(self.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def visualize_loss_evolution(self):
        plot_train_val_hist(
            training_history=self.loss_dict["train"],
            validation_history=self.loss_dict["val"],
            output_dir=self.output_dir + "/",
            y_label="total_loss",
            title="Fitting history",
        )

    def evaluate_test_performance(self):
        logging.debug("***" * 20)
        logging.debug("Train loss: {:.8f} ".format(self.best_loss_dict["train"]))
        logging.debug("Val loss: {:.8f} ".format(self.best_loss_dict["val"]))
        logging.debug("Test loss: {:.8f} ".format(self.best_loss_dict["test"]))
        logging.debug("***" * 20)


class BaseExperimentCV:
    def __init__(
        self,
        output_dir: str,
        n_folds: int = 4,
        train_val_split: List = [0.8, 0.2],
        batch_size: int = 64,
        num_epochs: int = 500,
        early_stopping: int = 20,
        random_state: int = 42,
    ):

        # I/O attributes
        self.output_dir = output_dir
        self.fold_output_dir = None

        # Training attributes
        self.n_folds = n_folds
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.early_stopping = early_stopping

        # Other attributes
        self.random_state = random_state
        self.loss_dicts = None
        self.best_loss_dicts = None
        self.trained_models = None
        self.device = get_device()

        # Fix random seeds for reproducibility and limit applicable algorithms to those believed to be deterministic
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        seed(self.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def visualize_loss_evolution(self, idx: int = 0):
        plot_train_val_hist(
            training_history=self.loss_dicts[idx]["train"],
            validation_history=self.loss_dicts[idx]["val"],
            output_dir=self.fold_output_dir + "/",
            y_label="total_loss",
            title="Fitting history",
        )

    def evaluate_cv_performance(self):
        train_losses = []
        val_losses = []
        test_losses = []
        for loss_dict in self.best_loss_dicts:
            train_losses.append(loss_dict["train"])
            val_losses.append(loss_dict["val"])
            test_losses.append(loss_dict["test"])
        logging.debug("***" * 20)
        logging.debug(
            "CV train loss: {:.8f} (+/- {:.8f})".format(
                np.mean(train_losses), np.std(train_losses)
            )
        )
        logging.debug(
            "CV val loss: {:.8f} (+/- {:.8f})".format(
                np.mean(val_losses), np.std(val_losses)
            )
        )
        logging.debug(
            "CV test loss: {:.8f} (+/- {:.8f})".format(
                np.mean(val_losses), np.std(val_losses)
            )
        )
        logging.debug("***" * 20)
