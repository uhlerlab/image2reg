import copy
import logging
import os
from typing import List
import torch

from src.experiments.base import BaseExperiment, BaseExperimentCV
from src.helper.data import DataHandler, DataHandlerCV
from src.utils.basic.visualization import plot_confusion_matrices
from src.utils.torch.data import init_profile_dataset

from src.utils.torch.exp import model_train_val_test_loop
from src.utils.torch.general import get_device
from src.utils.torch.model import get_domain_configuration


class BaseProfileEmbeddingExperiment(BaseExperiment):
    def __init__(
        self,
        output_dir,
        data_config: dict,
        model_config: dict,
        domain_name: str,
        save_freq: int = 50,
        train_val_test_split: List[float] = [0.7, 0.2, 0.1],
        batch_size: int = 64,
        num_epochs: int = 500,
        early_stopping: int = 20,
        random_state: int = 1234,
    ):
        super().__init__(
            output_dir=output_dir,
            train_val_test_split=train_val_test_split,
            batch_size=batch_size,
            num_epochs=num_epochs,
            early_stopping=early_stopping,
            random_state=random_state,
        )
        self.data_loader_dict = None
        self.data_set = None
        self.data_key = None
        self.label_key = None
        self.extra_feature_key = None
        self.index_key = None
        self.domain_config = None
        self.data_config = data_config
        self.model_config = model_config
        self.domain_name = domain_name
        self.save_freq = save_freq

    def initialize_profile_data_set(self):
        self.data_key = self.data_config.pop("data_key")
        self.label_key = self.data_config.pop("label_key")
        if "index_key" in self.data_config:
            self.index_key = self.data_config.pop("index_key")
        if "extra_feature_key" in self.data_config:
            self.extra_feature_key = self.data_config.pop("extra_feature_key")

        self.data_set = init_profile_dataset(**self.data_config)

    def initialize_domain_config(self):
        model_config = self.model_config["model_config"]
        optimizer_config = self.model_config["optimizer_config"]
        loss_config = self.model_config["loss_config"]
        if self.label_weights is not None:
            loss_config["weight"] = self.label_weights

        self.domain_config = get_domain_configuration(
            name=self.domain_name,
            model_dict=model_config,
            data_loader_dict=None,
            data_key=self.data_key,
            label_key=self.label_key,
            index_key=self.index_key,
            extra_feature_key=self.extra_feature_key,
            optimizer_dict=optimizer_config,
            loss_fct_dict=loss_config,
        )

    def load_model(self, weights_fname):
        weights = torch.load(weights_fname)
        self.domain_config.domain_model_config.model.load_state_dict(weights)

    def initialize_data_loader_dict(self, drop_last_batch: bool = True):

        dh = DataHandler(
            dataset=self.data_set,
            batch_size=self.batch_size,
            num_workers=0,
            random_state=self.random_state,
            transformation_dicts=None,
            drop_last_batch=drop_last_batch,
        )
        dh.stratified_train_val_test_split(splits=self.train_val_test_split)
        dh.get_data_loader_dict(shuffle=True)
        self.data_loader_dict = dh.data_loader_dict
        self.label_weights = dh.dataset.label_weights
        self.data_set = dh.dataset

    def train_models(self):
        self.domain_config.data_loader_dict = self.data_loader_dict
        (
            self.trained_model,
            self.loss_dict,
            self.best_loss_dict,
        ) = model_train_val_test_loop(
            output_dir=self.output_dir,
            domain_config=self.domain_config,
            num_epochs=self.num_epochs,
            early_stopping=self.early_stopping,
            device=self.device,
            save_freq=self.save_freq,
        )
