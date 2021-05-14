from typing import List
from random import seed
import os
import numpy as np
import torch


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
