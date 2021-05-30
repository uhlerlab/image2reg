from typing import List
import torch

from src.experiments.base import BaseExperiment
from src.helper.data import DataHandler
from src.utils.torch.data import init_nuclei_image_dataset
from src.utils.torch.exp import model_train_val_test_loop
from src.utils.torch.general import get_device
from src.utils.torch.model import (
    get_domain_configuration,
    get_image_net_transformations_dict,
    get_image_net_nonrandom_transformations_dict,
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
        self.data_key = None
        self.label_key = None
        self.domain_config = None

        self.trained_model = None
        self.loss_dict = None

        self.device = get_device()

    def initialize_image_data_set(self):
        image_dir = self.data_config["image_dir"]
        metadata_file = self.data_config["metadata_file"]
        if "target_list" in self.data_config:
            target_list = self.data_config["target_list"]
        else:
            target_list = None
        if "n_control_samples" in self.data_config:
            n_control_samples = self.data_config["n_control_samples"]
        else:
            n_control_samples = None
        if "pseudo_rgb" in self.data_config:
            pseudo_rgb = self.data_config["pseudo_rgb"]
        else:
            pseudo_rgb = False

        self.data_set = init_nuclei_image_dataset(
            image_dir=image_dir,
            metadata_file=metadata_file,
            target_list=target_list,
            n_control_samples=n_control_samples,
            pseudo_rgb=pseudo_rgb,
        )
        self.data_key = self.data_config["data_key"]
        self.label_key = self.data_config["label_key"]

    def initialize_data_loader_dict(
        self, drop_last_batch: bool = False, data_transform_pipeline: str = None
    ):
        if data_transform_pipeline == "imagenet_random":
            self.data_transform_pipeline_dict = get_image_net_transformations_dict(224)
        elif data_transform_pipeline == "imagenet_nonrandom":
            self.data_transform_pipeline_dict = get_image_net_nonrandom_transformations_dict(
                224
            )
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

    def initialize_domain_config(self):
        model_config = self.model_config["model_config"]
        optimizer_config = self.model_config["optimizer_config"]
        loss_config = self.model_config["loss_config"]

        self.domain_config = get_domain_configuration(
            name=self.domain_name,
            model_dict=model_config,
            data_loader_dict=self.data_loader_dict,
            data_key=self.data_key,
            label_key=self.label_key,
            optimizer_dict=optimizer_config,
            loss_fct_dict=loss_config,
        )

    def train_models(
        self, lamb: float = 0.00000001,
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

    def load_model(self, weights_fname):
        weights = torch.load(weights_fname)
        self.domain_config.domain_model_config.model.load_state_dict(weights)

    def visualize_loss_evolution(self):
        super().visualize_loss_evolution()

    def visualize_latent_space_pca_walk(self, dataset_type:str="test", n_components:int=2, n_steps:int=10):
        output_dir = os.path.join(self.output_dir, "pc_latent_walk")
        if not os.path.exist(output_dir):
            os.makedirs(output_dir)

        if self.domain_config.domain_model_config.model.model_base_type not in ["ae", "vae"]:
            raise RuntimeError("Only implemented for autoencoder models")
        else:
            visualize_latent_space_pca_walk(domain_config=self.domain_config, output_dir=self.output_dir,
                                            dataset_type=dataset_type, n_components=n_components, n_steps=n_steps)
