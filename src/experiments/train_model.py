import os
from typing import List
import torch

from src.experiments.base import BaseExperiment
from src.helper.data import DataHandler
from src.utils.torch.data import init_image_dataset, init_profile_dataset
from src.utils.torch.evaluation import (
    visualize_latent_space_pca_walk,
    save_latents_to_hdf,
)
from src.utils.torch.exp import model_train_val_test_loop
from src.utils.torch.general import get_device
from src.utils.torch.model import (
    get_domain_configuration,
    get_image_net_transformations_dict,
    get_image_net_nonrandom_transformations_dict,
    get_imagenet_extended_transformations_dict,
    get_randomflips_transformation_dict,
)


class TrainModelExperiment(BaseExperiment):
    def __init__(
        self,
        output_dir: str,
        data_config: dict,
        model_config: dict,
        domain_name: str,
        train_val_test_split: List[float] = [0.7, 0.2, 0.1],
        batch_size: int = 64,
        num_epochs: int = 64,
        early_stopping: int = -1,
        random_state: int = 42,
        save_freq: int = -1,
        pseudo_rgb: bool = False,
    ):
        super().__init__(
            output_dir=output_dir,
            train_val_test_split=train_val_test_split,
            batch_size=batch_size,
            num_epochs=num_epochs,
            early_stopping=early_stopping,
            random_state=random_state,
        )

        self.data_config = data_config
        self.model_config = model_config
        self.domain_name = domain_name
        self.save_freq = save_freq
        self.pseudo_rbg = pseudo_rgb

        self.data_set = None
        self.data_transform_pipeline_dict = None
        self.data_loader_dict = None
        self.data_set = None
        self.data_key = None
        self.label_key = None
        self.extra_feature_key = None
        self.index_key = None
        self.domain_config = None

        self.trained_model = None
        self.loss_dict = None
        self.label_weights = None

        self.device = get_device()

    def initialize_image_data_set(self):
        self.data_key = self.data_config.pop("data_key")
        self.label_key = self.data_config.pop("label_key")
        if (
            "extra_features" in self.data_config
            and len(self.data_config["extra_features"]) > 0
        ):
            self.extra_feature_key = "extra_features"
        if "index_key" in self.data_config:
            self.index_key = self.data_config.pop("index_key")

        self.data_set = init_image_dataset(**self.data_config)

    def initialize_profile_data_set(self):
        self.data_key = self.data_config.pop("data_key")
        self.label_key = self.data_config.pop("label_key")
        if "index_key" in self.data_config:
            self.index_key = self.data_config.pop("index_key")
        if "extra_feature_key" in self.data_config:
            self.extra_feature_key = self.data_config.pop("extra_feature_key")

        self.data_set = init_profile_dataset(**self.data_config)

    def initialize_data_loader_dict(
        self, drop_last_batch: bool = False, data_transform_pipeline: str = None
    ):
        if data_transform_pipeline is None:
            self.data_transform_pipeline_dict = None
        elif data_transform_pipeline == "imagenet_random":
            self.data_transform_pipeline_dict = get_image_net_transformations_dict(224)
        elif data_transform_pipeline == "imagenet_nonrandom":
            self.data_transform_pipeline_dict = get_image_net_nonrandom_transformations_dict(
                224
            )
        elif data_transform_pipeline == "imagenet_extended_random":
            self.data_transform_pipeline_dict = get_imagenet_extended_transformations_dict(
                224
            )
        elif data_transform_pipeline == "randomflips":
            self.data_transform_pipeline_dict = get_randomflips_transformation_dict()

        dh = DataHandler(
            dataset=self.data_set,
            batch_size=self.batch_size,
            num_workers=0,
            random_state=self.random_state,
            transformation_dict=self.data_transform_pipeline_dict,
            drop_last_batch=drop_last_batch,
        )
        dh.stratified_train_val_test_split(splits=self.train_val_test_split)
        dh.get_data_loader_dict(shuffle=True)
        self.data_loader_dict = dh.data_loader_dict
        self.label_weights = dh.dataset.label_weights
        self.dataset = dh.dataset

    def initialize_domain_config(self):
        model_config = self.model_config["model_config"]
        optimizer_config = self.model_config["optimizer_config"]
        loss_config = self.model_config["loss_config"]
        if self.label_weights is not None:
            loss_config["weight"] = self.label_weights

        self.domain_config = get_domain_configuration(
            name=self.domain_name,
            model_dict=model_config,
            data_loader_dict=self.data_loader_dict,
            data_key=self.data_key,
            label_key=self.label_key,
            index_key=self.index_key,
            extra_feature_key=self.extra_feature_key,
            optimizer_dict=optimizer_config,
            loss_fct_dict=loss_config,
        )

    def train_models(
        self, lamb: float = 0.01,
    ):
        self.trained_model, self.loss_dict = model_train_val_test_loop(
            output_dir=self.output_dir,
            domain_config=self.domain_config,
            num_epochs=self.num_epochs,
            lamb=lamb,
            early_stopping=self.early_stopping,
            device=self.device,
            save_freq=self.save_freq,
        )

    def load_models(self, weights_fname):
        weights = torch.load(weights_fname)
        self.domain_config.domain_model_config.model.load_state_dict(weights)

    def extract_and_save_latents(self):
        device = get_device()
        for dataset_type in ["val", "test"]:
            save_path = os.path.join(
                self.output_dir, "{}_latents.h5".format(str(dataset_type))
            )
            save_latents_to_hdf(
                domain_config=self.domain_config,
                save_path=save_path,
                dataset_type=dataset_type,
                device=device,
            )
        save_path = os.path.join(self.output_dir, "all_latents.h5")
        self.dataset.set_transform_pipeline(self.data_transform_pipeline_dict["val"])
        save_latents_to_hdf(
            domain_config=self.domain_config,
            save_path=save_path,
            dataset=self.dataset,
            device=device,
        )

    def visualize_loss_evolution(self):
        super().visualize_loss_evolution()

    def visualize_latent_space_pca_walk(
        self, dataset_type: str = "test", n_components: int = 2, n_steps: int = 11
    ):
        output_dir = os.path.join(self.output_dir, "pc_latent_walk")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.domain_config.domain_model_config.classifier.model_base_type not in [
            "ae",
            "vae",
        ]:
            raise RuntimeError("Only implemented for autoencoder models")
        else:
            visualize_latent_space_pca_walk(
                domain_config=self.domain_config,
                output_dir=output_dir,
                dataset_type=dataset_type,
                n_components=n_components,
                n_steps=n_steps,
            )
