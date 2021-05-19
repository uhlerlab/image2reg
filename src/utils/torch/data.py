import logging
from torchvision import transforms

from src.data.datasets import TorchNucleiImageDataset


def init_nuclei_image_dataset(
    image_dir: str,
    metadata_file: str,
    image_file_col: str = "image_file",
    label_col: str = "gene_label",
) -> TorchNucleiImageDataset:
    logging.debug(
        "Load image data set from {} and label information from {}.".format(
            image_dir, metadata_file
        )
    )
    nuclei_dataset = TorchNucleiImageDataset(
        image_dir=image_dir,
        metadata_file=metadata_file,
        image_file_col=image_file_col,
        label_col=label_col,
    )
    logging.debug("Samples loaded: {}".format(len(nuclei_dataset)))
    return nuclei_dataset
