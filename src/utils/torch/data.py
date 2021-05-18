

def init_nuclei_image_dataset(
    image_dir:str, metadata_file,:str image_file_col:str="image_file", label_col:str="gene", transform_pipeline:transforms=None) -> TorchNucleiImageDataset:
    logging.debug(
        "Load images set from {} and label information from {}.".format(
            image_dir, label_fname
        )
    )
    nuclei_dataset = TorchNucleiImageDataset(
        image_dir=image_dir,
        metadata_file=metadata_file,
        image_file_col=image_file_col,
        label_col = label_col,
        transform_pipeline=transform_pipeline,
    )
    logging.debug("Samples loaded: {}".format(len(nuclei_dataset)))
    return nuclei_dataset