import copy
import logging
import os
from abc import ABC
from collections import Counter
from typing import List, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from imblearn.under_sampling import RandomUnderSampler
from skimage.io import imread
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import Tensor
from torch.utils.data import Dataset, Subset, ConcatDataset
from torchvision import transforms

from src.utils.basic.general import combine_path


class LabeledSlideDataset(Dataset, ABC):
    def __init__(self):
        super(LabeledSlideDataset, self).__init__()
        self.labels = None
        self.transform_pipeline = None
        self.slide_image_labels = None
        self.slide_image_nuclei_dict = None
        self.slide_image_names = None

    def create_slide_image_nuclei_dict_labels(self):
        self.slide_image_nuclei_dict = dict()
        for i in range(len(self.slide_image_names)):
            slide_image_name = self.slide_image_names[i]
            if slide_image_name in self.slide_image_nuclei_dict:
                self.slide_image_nuclei_dict[slide_image_name].append(i)
            else:
                self.slide_image_nuclei_dict[slide_image_name] = [i]

        self.slide_image_labels = np.array(
            list(dict(zip(self.slide_image_names, self.labels)).values())
        )


class TorchProfileSlideDataset(LabeledSlideDataset):
    def __init__(
        self,
        feature_label_file: str,
        label_col: str = "label",
        n_control_samples: int = None,
        target_list: List = None,
        exclude_features: List = None,
        image_name_col: str = "image_name",
        slide_image_name_col: str = "slide_image_name",
    ):
        super().__init__()
        self.feature_labels = pd.read_csv(feature_label_file, index_col=0)
        self.feature_labels = self.feature_labels.drop(columns=exclude_features)
        self.label_col = label_col
        self.n_control_samples = n_control_samples
        self.target_list = target_list
        self.slide_image_name_col = slide_image_name_col

        if target_list is not None:
            self.feature_labels = self.feature_labels.loc[
                self.feature_labels[label_col].isin(target_list), :
            ]
        if "EMPTY" in target_list:
            idc = np.array(list(range(len(self.feature_labels)))).reshape(-1, 1)
            labels = self.feature_labels[self.label_col]
            idc, _ = RandomUnderSampler(
                sampling_strategy="majority", random_state=1234
            ).fit_resample(idc, labels)
            self.feature_labels = self.feature_labels.iloc[idc.flatten(), :]

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

        self.slide_image_names = np.array(
            self.feature_labels.loc[:, slide_image_name_col]
        )
        super().create_slide_image_nuclei_dict_labels()

        self.features = np.array(
            self.feature_labels.drop(
                columns=list(
                    set([label_col, slide_image_name_col, image_name_col]).intersection(
                        set(self.feature_labels.columns)
                    )
                )
            )
        )
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


class TorchImageSlideDataset(LabeledSlideDataset):
    def __init__(
        self,
        image_dir,
        metadata_file,
        image_file_col: str = "image_file",
        plate_col: str = "plate",
        label_col: str = "gene_symbol",
        slide_image_name_col: str = "slide_image_name",
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
        self.slide_image_name_col = slide_image_name_col
        self.target_list = target_list

        if self.target_list is not None:
            self.metadata = self.metadata.loc[
                self.metadata[label_col].isin(target_list), :
            ]
            self.target_list = sorted(self.target_list)
        else:
            self.target_list = sorted(list(set(self.metadata[label_col])))
        if n_control_samples is not None and "EMPTY_nan" in list(
            self.metadata[label_col]
        ):
            idc = np.array(list(range(len(self.metadata)))).reshape(-1, 1)
            labels = self.metadata[self.label_col]
            target_n_samples = dict(Counter(labels))
            target_n_samples["EMPTY_nan"] = n_control_samples
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
                columns=list(
                    set([image_file_col, label_col, slide_image_name_col]).intersection(
                        set(list(self.nmco_features.columns))
                    )
                )
            )

            self.nmco_features = StandardScaler().fit_transform(self.nmco_features)
        else:
            self.nmco_features = None

        self.labels = np.array(self.metadata.loc[:, label_col])
        le = LabelEncoder().fit(self.labels)
        self.labels = le.transform(self.labels)

        if nuclei_density_col in self.metadata.columns:
            self.nuclei_densities = np.array(
                self.metadata.loc[:, nuclei_density_col]
            ).astype(float)
            self.nuclei_densities -= self.nuclei_densities.mean()
            self.nuclei_densities /= self.nuclei_densities.std()

        if elongation_ratio_col in self.metadata.columns:
            self.elongation_ratios = np.array(
                self.metadata.loc[:, elongation_ratio_col]
            )

        self.label_weights = (
            len(self.labels) / np.unique(self.labels, return_counts=True)[1]
        )
        self.label_weights /= np.sum(self.label_weights)
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        logging.debug("Classes are coded as follows: %s", le_name_mapping)
        # self.set_transform_pipeline(transform_pipeline)
        self.pseudo_rgb = pseudo_rgb

        if slide_image_name_col in self.metadata.columns:
            self.slide_image_names = np.array(
                self.metadata.loc[:, slide_image_name_col]
            )
        else:
            self.slide_image_names = np.array(self.metadata.loc[:, image_file_col])
        super().create_slide_image_nuclei_dict_labels()

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
            "image_file": os.path.split(image_loc)[1],
            "slide_image_file": self.slide_image_names[idx],
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

    def process_image(
        self,
        image_loc: str,
        transform_pipeline: transforms.Compose = None,
        mask_loc=None,
        centroid=None,
    ) -> Tensor:
        image = imread(image_loc)
        if (image > 255).any():
            image_min = np.percentile(image, 0.1)
            image_max = np.percentile(image, 99.9)
            if image_max > 255:
                image = image - image_min
                image = image / image_max
            image = np.clip(image, 0, 1) * 255
            image = np.uint8(image)
        pil_image = Image.fromarray(image)
        if transform_pipeline is None:
            if centroid is not None:
                tensor_image = self.transform_pipeline(pil_image, centroid)
            else:
                tensor_image = self.transform_pipeline(pil_image)
        else:
            if centroid is not None:
                tensor_image = transform_pipeline(pil_image, centroid=centroid)
            else:
                tensor_image = transform_pipeline(pil_image)
        if not self.pseudo_rgb:
            tensor_image = image[0, :, :]
        return tensor_image


class TorchMultiImageSlideDataset(TorchImageSlideDataset):
    def __init__(
        self,
        nuclei_image_dir,
        nuclei_metadata_file,
        slide_image_dir,
        slide_mask_dir: str = None,
        image_file_col: str = "image_file",
        plate_col: str = "plate",
        label_col: str = "gene_symbol",
        slide_image_name_col: str = "slide_image_name",
        extra_features: List = None,
        nuclei_density_col: str = "nuclei_count_image",
        elongation_ratio_col: str = "aspect_ratio_cluster_ratio",
        target_list: List = None,
        n_control_samples: int = None,
        transform_pipeline: transforms.Compose = None,
        pseudo_rgb: bool = False,
        nmco_feature_file: str = None,
    ):
        super().__init__(
            image_dir=nuclei_image_dir,
            metadata_file=nuclei_metadata_file,
            image_file_col=image_file_col,
            plate_col=plate_col,
            label_col=label_col,
            slide_image_name_col=slide_image_name_col,
            extra_features=extra_features,
            nuclei_density_col=nuclei_density_col,
            elongation_ratio_col=elongation_ratio_col,
            target_list=target_list,
            n_control_samples=n_control_samples,
            transform_pipeline=transform_pipeline,
            pseudo_rgb=pseudo_rgb,
            nmco_feature_file=nmco_feature_file,
        )
        self.nuclei_image_dir = nuclei_image_dir
        self.nuclei_metadata = self.metadata
        self.slide_image_dir = slide_image_dir
        self.slide_mask_dir = slide_mask_dir

        self.nuclei_image_transform_pipeline = None
        self.slide_image_transform_pipeline = None

        self.nuclei_image_locs = self.image_locs
        self.slide_image_locs = np.apply_along_axis(
            combine_path,
            0,
            [
                np.repeat(self.slide_image_dir, len(self.metadata)).astype(str),
                np.array(self.metadata.loc[:, self.plate_col], dtype=str),
                np.array(self.metadata.loc[:, self.slide_image_name_col], dtype=str),
            ],
        ).astype(object)
        if self.slide_mask_dir is not None:
            self.slide_mask_locs = np.apply_along_axis(
                combine_path,
                0,
                [
                    np.repeat(self.slide_mask_dir, len(self.metadata)).astype(str),
                    np.array(self.metadata.loc[:, self.plate_col], dtype=str),
                    np.array(
                        self.metadata.loc[:, self.slide_image_name_col], dtype=str
                    ),
                ],
            ).astype(object)
        else:
            self.slide_mask_locs = None

        self.centroids_0 = np.array(self.nuclei_metadata.loc[:, "centroid_0"])
        self.centroids_1 = np.array(self.nuclei_metadata.loc[:, "centroid_1"])

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        nuclei_image_loc = self.nuclei_image_locs[idx]
        slide_image_loc = self.slide_image_locs[idx]
        centroid = [self.centroids_0[idx], self.centroids_1[idx]]
        if self.slide_mask_locs is not None:
            slide_mask_loc = self.slide_mask_locs[idx]
        else:
            slide_mask_loc = None
        nuclei_image = self.process_image(
            nuclei_image_loc, self.nuclei_image_transform_pipeline
        )
        slide_image = self.process_image(
            slide_image_loc,
            self.slide_image_transform_pipeline,
            mask_loc=slide_mask_loc,
            centroid=centroid,
        )
        gene_label = self.labels[idx]

        sample = {
            "id": nuclei_image_loc,
            "images": [nuclei_image, slide_image],
            "nuclei_image": nuclei_image,
            "slide_image": slide_image,
            "label": gene_label,
            "image_file": os.path.split(nuclei_image_loc)[1],
            "slide_image_file": os.path.split(slide_image_loc)[1],
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
        self, transform_pipelines: List[transforms.Compose] = None
    ) -> None:
        try:
            self.nuclei_image_transform_pipeline = transform_pipelines[0]
            self.slide_image_transform_pipeline = transform_pipelines[1]
        except AttributeError as exception:
            logging.error(
                "Object must implement a subset of a dataset type that implements the "
                "set_transform_pipeline method."
            )
            raise exception

    def process_image(
        self,
        image_loc: str,
        transform_pipeline: transforms.Compose = None,
        mask_loc=None,
        centroid=None,
    ) -> Tensor:
        return super().process_image(
            image_loc,
            transform_pipeline=transform_pipeline,
            mask_loc=mask_loc,
            centroid=centroid,
        )


class TorchTransformableSubset(Subset):
    def __init__(self, dataset: LabeledSlideDataset, indices):
        super().__init__(dataset=dataset, indices=indices)
        # Hacky way to create a independent dataset instance such changes to the dataset of the subset instance are not
        # passed through --> might increase CPU/GPU memory usage linearly
        self.dataset = copy.deepcopy(self.dataset)
        self.transform_pipeline = None

    def set_transform_pipeline(
        self, transform_pipelines: List[transforms.Compose]
    ) -> None:
        try:
            if len(transform_pipelines) == 1:
                self.dataset.set_transform_pipeline(transform_pipelines[0])
            else:
                self.dataset.set_transform_pipeline(transform_pipelines)
        except AttributeError as exception:
            logging.error(
                "Object must implement a subset of a dataset type that implements the "
                "set_transform_pipeline method."
            )
            raise exception


class TorchTransformableSuperset(ConcatDataset):
    def __init__(self, datasets: Iterable[LabeledSlideDataset]):
        super().__init__(datasets=datasets)
        self.tranform_pipeline = None
        self.datasets = copy.deepcopy(self.datasets)
        self.label_weights = self.datasets[0].label_weights
        self.target_list = self.datasets[0].target_list

    def set_transform_pipeline(self, transform_pipelines: List[transforms.Compose]):
        for dataset in self.datasets:
            try:
                if len(transform_pipelines) == 1:
                    dataset.set_transform_pipeline(transform_pipelines[0])
                else:
                    dataset.set_transform_pipeline(transform_pipelines)

            except AttributeError as exception:
                logging.error(
                    "Object must implement a subset of a dataset type that implements"
                    " the set_transform_pipeline method."
                )
            raise exception


class IndexedTensorDataset(Dataset):
    def __init__(self, data, labels, groups):
        super().__init__()
        self.data = data
        self.labels = labels
        self.groups = groups

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.groups[idx]

    def __len__(self):
        return len(self.groups)
