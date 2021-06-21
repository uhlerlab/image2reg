import copy
import logging
import os
from collections import Counter
from typing import List

import torch
import numpy as np
import pandas as pd
from PIL import Image
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import Tensor
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from skimage.io import imread

from src.utils.basic.general import combine_path
from src.utils.basic.io import get_file_list


class LabeledDataset(Dataset):
    def __init__(self):
        super(LabeledDataset, self).__init__()
        self.labels = None
        self.transform_pipeline = None


class TorchProfileDataset(LabeledDataset):
    def __init__(
        self,
        feature_label_file: str,
        label_col: str = "label",
        n_control_samples: int = None,
        target_list: List = None,
        exclude_features: List = None,
    ):
        super().__init__()
        self.feature_labels = pd.read_csv(feature_label_file, index_col=0)
        self.feature_labels = self.feature_labels.drop(columns=exclude_features)
        # logging.debug(list(self.feature_labels.columns))
        self.label_col = label_col
        self.n_control_samples = n_control_samples
        self.target_list = target_list

        if target_list is not None:
            self.feature_labels = self.feature_labels.loc[
                self.feature_labels[label_col].isin(target_list), :
            ]
        if n_control_samples is not None and "EMPTY" in target_list:
            idc = np.array(list(range(len(self.feature_labels)))).reshape(-1, 1)
            labels = self.feature_labels[self.label_col]
            target_n_samples = dict(Counter(labels))
            target_n_samples["EMPTY"] = n_control_samples
            idc, _ = RandomUnderSampler(
                sampling_strategy=target_n_samples, random_state=1234
            ).fit_resample(idc, labels)
            self.feature_labels = self.features.iloc[idc.flatten(), :]

        logging.debug(
            "Label counts: %s",
            dict(Counter(np.array(self.feature_labels[self.label_col]))),
        )
        self.labels = np.array(self.feature_labels.loc[:, label_col])
        le = LabelEncoder().fit(self.labels)
        self.labels = le.transform(self.labels)

        self.label_weights = (
            len(self.labels) / np.unique(self.labels, return_counts=True)[1]
        )
        self.label_weights /= np.sum(self.label_weights)
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        logging.debug("Classes are coded as follows: %s", le_name_mapping)

        self.features = np.array(self.feature_labels.drop(columns=label_col))
        sc = StandardScaler()
        self.features = sc.fit_transform(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = {
            "profile": torch.FloatTensor(self.features[idx]),
            "label": self.labels[idx],
        }
        return sample


class TorchImageDataset(LabeledDataset):
    def __init__(
        self,
        image_dir,
        metadata_file,
        image_file_col: str = "image_file",
        plate_col: str = "plate",
        label_col: str = "gene_symbol",
        extra_features: List = None,
        nuclei_density_col: str = "nuclei_count_image",
        elongation_ratio_col: str = "aspect_ratio_cluster_ratio",
        target_list: List = None,
        n_control_samples: int = None,
        transform_pipeline: transforms.Compose = None,
        pseudo_rgb: bool = False,
        nmco_feature_file: str = None,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.metadata_file = metadata_file
        self.image_file_col = image_file_col
        self.plate_col = plate_col
        self.label_col = label_col
        self.metadata = pd.read_csv(self.metadata_file, index_col=0)
        self.extra_features = extra_features

        if target_list is not None:
            self.metadata = self.metadata.loc[
                self.metadata[label_col].isin(target_list), :
            ]
        if n_control_samples is not None and "EMPTY" in target_list:
            idc = np.array(list(range(len(self.metadata)))).reshape(-1, 1)
            labels = self.metadata[self.label_col]
            target_n_samples = dict(Counter(labels))
            target_n_samples["EMPTY"] = n_control_samples
            idc, _ = RandomUnderSampler(
                sampling_strategy=target_n_samples, random_state=1234
            ).fit_resample(idc, labels)
            self.metadata = self.metadata.iloc[idc.flatten(), :]

        logging.debug(
            "Label counts: %s", dict(Counter(np.array(self.metadata[self.label_col]))),
        )

        # Numpy data type problem leads to strings being cutoff when applying along axis
        self.image_locs = np.apply_along_axis(
            combine_path,
            0,
            [
                np.repeat(image_dir, len(self.metadata)).astype(str),
                np.array(self.metadata.loc[:, self.plate_col], dtype=str),
                np.array(self.metadata.loc[:, self.image_file_col], dtype=str),
            ],
        ).astype(object)

        if len(self.metadata) != len(self.image_locs):
            raise RuntimeError(
                "Number of image samples does not match the given metadata."
            )

        if nmco_feature_file is not None:
            self.nmco_feature_file = nmco_feature_file
            self.nmco_features = pd.read_csv(self.nmco_feature_file, index_col=0)
            # Ensure that metadata and additional features are aligned
            self.nmco_features.index = np.array(
                self.nmco_features.loc[:, image_file_col]
            )
            self.nmco_features = self.nmco_features.loc[
                list(self.metadata.loc[:, image_file_col])
            ]
            self.nmco_features = self.nmco_features.drop(
                columns=[image_file_col, label_col]
            )

            self.nmco_features = StandardScaler().fit_transform(self.nmco_features)
        else:
            self.nmco_features = None

        self.labels = np.array(self.metadata.loc[:, label_col])
        le = LabelEncoder().fit(self.labels)
        self.labels = le.transform(self.labels)

        self.nuclei_densities = np.array(self.metadata.loc[:, nuclei_density_col]).astype(float)
        self.nuclei_densities -= self.nuclei_densities.mean()
        self.nuclei_densities /= self.nuclei_densities.std()

        self.elongation_ratios = np.array(self.metadata.loc[:, elongation_ratio_col])

        self.label_weights = (
            len(self.labels) / np.unique(self.labels, return_counts=True)[1]
        )
        self.label_weights /= np.sum(self.label_weights)
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        logging.debug("Classes are coded as follows: %s", le_name_mapping)
        self.set_transform_pipeline(transform_pipeline)
        self.pseudo_rgb = pseudo_rgb

    def __len__(self):
        return len(self.image_locs)

    def __getitem__(self, idx):
        image_loc = self.image_locs[idx]
        image = self.process_image(image_loc)
        gene_label = self.labels[idx]

        sample = {
            "id": image_loc,
            "image": image,
            "label": gene_label,
        }

        if self.extra_features is not None:
            extra_feature_vec = []
            if "nuclear_density" in self.extra_features:
                extra_feature_vec.append(self.nuclei_densities[idx])
            if "elongation_ratio" in self.extra_features:
                extra_feature_vec.append(self.elongation_ratios[idx])
            if "nmco" in self.extra_features:
                extra_feature_vec.extend(list(self.nmco_features[idx]))
            extra_feature_vec = torch.FloatTensor(np.array(extra_feature_vec).flatten())
            sample["extra_features"] = extra_feature_vec

        return sample

    def set_transform_pipeline(
        self, transform_pipeline: transforms.Compose = None
    ) -> None:
        if transform_pipeline is None:
            self.transform_pipeline = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform_pipeline = transform_pipeline

    def process_image(self, image_loc: str) -> Tensor:
        image = imread(image_loc)
        if (image > 255).any():
            image = image - image.min()
            image = image / image.max()
            image = np.array(np.clip(image, 0, 1) * 255, dtype=np.uint8)
        image = Image.fromarray(image)
        if self.pseudo_rgb:
            rgbimg = Image.new("RGB", image.size)
            rgbimg.paste(image)
            ##rgbimg = torch.stack([image]*3)
            image = rgbimg
        image = self.transform_pipeline(image)
        return image


class TorchTransformableSubset(Subset):
    def __init__(self, dataset: LabeledDataset, indices):
        super().__init__(dataset=dataset, indices=indices)
        # Hacky way to create a independent dataset instance such changes to the dataset of the subset instance are not
        # passed through --> might increase CPU/GPU memory usage linearly
        self.dataset = copy.deepcopy(self.dataset)
        self.transform_pipeline = None

    def set_transform_pipeline(self, transform_pipeline: transforms.Compose) -> None:
        try:
            self.dataset.set_transform_pipeline(transform_pipeline)
        except AttributeError as exception:
            logging.error(
                "Object must implement a subset of a dataset type that implements the "
                "set_transform_pipeline method."
            )
            raise exception
