import logging

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from skimage.io import imread


class LabeledDataset(Dataset):
    def __init__(self):
        super(LabeledDataset, self).__init__()
        self.labels = None
        self.transformation_pipeline = None


class TorchNucleiImageDataset(LabeledDataset):

    def __init__(self, image_dir, metadata_file, image_file_col:str="image_file", label_col:str="gene"):
        self.image_dir = image_dir
        self.image_locs = get_file_list(sel.image_dir)
        self.metadata_file = metadata_file
        self.image_file_col = image_file_col
        self.label_col = label_col
        self.metadata = pd.read_csv(self.metadata_file)
        self.metadata["gene_label_num"] = LabelEncoder().fit_tramsform(np.array(list(self.metadata.loc[:,self.label_col])))

        if len(self.metadata) != self.image_locs:
            raise RuntimeError("Number of image samples does not match the given metadata.")

    def __len__(self):
        return len(self.image_locs)

    def __getitem__(self, idx):
        image_loc = self.image_locs[idx]
        image = self.process_image(image_loc)
        gene_label_num = self.metadata.loc[self.metadata[self.image_file_col] == os.path.split(image_loc)[1], "gene_label_num"]

        sample = {'img':image, 'label':gene_label_num}
        return sample

    def set_transform_pipeline(
        self, transform_pipeline: transforms.Compose = None
    ) -> None:
        self.transform_pipeline = transform_pipeline

    def process_image(self, image_loc: str) -> Tensor:
        image = imread(image_loc)
        image = np.array(image, dtype=np.float32)
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
