import logging
from typing import List

from src.data.datasets import TorchImageDataset


def init_image_dataset(
    image_dir: str,
    metadata_file: str,
    image_file_col: str = "image_file",
    label_col: str = "gene_symbol",
    plate_col: str = "plate",
    target_list: List = None,
    n_control_samples=None,
    pseudo_rgb: bool = False,
) -> TorchImageDataset:
    logging.debug(
        "Load image data set from {} and label information from {}.".format(
            image_dir, metadata_file
        )
    )
    image_dataset = TorchImageDataset(
        image_dir=image_dir,
        metadata_file=metadata_file,
        image_file_col=image_file_col,
        label_col=label_col,
        plate_col=plate_col,
        target_list=target_list,
        n_control_samples=n_control_samples,
        pseudo_rgb=pseudo_rgb,
    )
    logging.debug("Samples loaded: {}".format(len(image_dataset)))
    return image_dataset
