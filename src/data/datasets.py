import logging
import os

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
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
        self.transformation_pipeline = None


class TorchNucleiImageDataset(LabeledDataset):
    def __init__(
        self,
        image_dir,
        metadata_file,
        image_file_col: str = "image_file",
        plate_col: str = "plate",
        label_col: str = "gene_label",
    ):
        super().__init__()
        self.image_dir = image_dir
        self.metadata_file = metadata_file
        self.image_file_col = image_file_col
        self.plate_col = plate_col
        self.label_col = label_col
        self.metadata = pd.read_csv(self.metadata_file, index_col=0)

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
        # self.image_locs = []
        # for i in range(len(self.metadata)):
        #     plate = str(self.metadata.iloc[i,:][self.plate_col])
        #     image_file = self.metadata.iloc[i, :][self.image_file_col]
        #     self.image_locs.append(os.path.join(image_dir, plate, image_file))

        if len(self.metadata) != len(get_file_list(self.image_dir)):
            raise RuntimeError(
                "Number of image samples does not match the given metadata."
            )
        self.labels = np.array(self.metadata.loc[:,label_col])

    def __len__(self):
        return len(self.image_locs)

    def __getitem__(self, idx):
        image_loc = self.image_locs[idx]
        image = self.process_image(image_loc)
        gene_label = self.labels[idx]

        sample = {"id":image_loc, "image": image, "label": gene_label}
        return sample

    def set_transform_pipeline(
        self, transform_pipeline: transforms.Compose = None
    ) -> None:
        self.transform_pipeline = transform_pipeline

    def process_image(self, image_loc: str) -> Tensor:
        image = imread(image_loc)
        image = np.array(image, dtype=np.float32)
        image = (image - image.min())/(image.max()-image.min())
        image = np.clip(image, 0, 1)
        image = torch.from_numpy(image).unsqueeze(0)
        return image


class TorchTransformableSubset(Subset):
    def __init__(self, dataset: LabeledDataset, indices):
        super().__init__(dataset=dataset, indices=indices)

    def set_transform_pipeline(self, transform_pipeline: transforms.Compose) -> None:
        try:
            self.dataset.set_transform_pipeline(transform_pipeline)
        except AttributeError as exception:
            logging.error(
                "Object must implement a subset of a dataset type that implements the "
                "set_transform_pipeline method."
            )
            raise exception
