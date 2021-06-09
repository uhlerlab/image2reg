import os
from typing import List

import imageio
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch import FloatTensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.helper.models import DomainConfig, DomainModelConfig
from src.utils.basic.export import dict_to_csv_gz
from src.utils.basic.visualization import plot_image_seq
from src.utils.torch.general import get_device


def get_latent_representations_for_model(
    model: Module,
    dataset: Dataset,
    data_key: str = "seq_data",
    label_key: str = "label",
    index_key: str = "id",
    device: str = "cuda:0",
) -> dict:
    # create Dataloader
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)

    latent_representations = []
    labels = []
    index = []
    model.eval().to(device)

    for (idx, sample) in enumerate(
        tqdm(dataloader, desc="Compute latents for the evaluation")
    ):
        input = sample[data_key].to(device)
        if label_key is not None:
            labels.extend(sample[label_key].detach().cpu().numpy())
        if index_key in sample:
            index.extend(sample[index_key])

        output = model(input)
        latents = output["latents"]
        latent_representations.extend(latents.detach().cpu().numpy())

    latent_representations = np.array(latent_representations).squeeze()
    labels = np.array(labels).squeeze()

    latent_dict = {"latents": latent_representations}

    if len(labels) != 0:
        latent_dict["labels"] = labels
    if len(index) != 0:
        latent_dict["index"] = index

    return latent_dict


def save_latents_to_csv_gz(
    domain_config: DomainConfig,
    save_path: str,
    dataset_type: str = "val",
    device: str = "cuda:0",
):
    model = domain_config.domain_model_config.classifier
    try:
        dataset = domain_config.data_loader_dict[dataset_type].dataset
    except KeyError:
        raise RuntimeError(
            "Unknown dataset_type: {}, expected one of the following: train, val, test"
            .format(dataset_type)
        )
    save_latents_and_labels_to_csv_gz(
        model=model,
        dataset=dataset,
        save_path=save_path,
        data_key=domain_config.data_key,
        label_key=domain_config.label_key,
        device=device,
    )


def save_latents_and_labels_to_csv_gz(
    model: Module,
    dataset: Dataset,
    save_path: str,
    data_key: str = "image",
    label_key: str = "label",
    device: str = "cuda:0",
):
    data = get_latent_representations_for_model(
        model=model,
        dataset=dataset,
        data_key=data_key,
        label_key=label_key,
        device=device,
    )

    expanded_data = {}
    if "latents" in data:
        latents = data["latents"]
        for i in range(latents.shape[1]):
            expanded_data["zs_{}".format(i)] = latents[:, i]
    if "ids" in data:
        index = data["ids"]
    else:
        index = None
    if "labels" in data:
        expanded_data["labels"] = data["labels"]

    dict_to_csv_gz(data=expanded_data, save_path=save_path, index=index)


def save_latents_from_model(
    output_dir: str,
    domain_config: DomainConfig,
    dataset_types: List[str] = None,
    device: str = "cuda:0",
):
    os.makedirs(output_dir, exist_ok=True)
    if dataset_types is None:
        dataset_types = ["train", "val"]

    domain_name = domain_config.name
    for dataset_type in dataset_types:
        save_latents_to_csv_gz(
            domain_config=domain_config,
            save_path=output_dir
            + "/{}_latent_representations_{}.csv.gz".format(domain_name, dataset_type),
            dataset_type=dataset_type,
            device=device,
        )


def visualize_image_ae_performance(
    domain_model_config: DomainModelConfig,
    epoch: int,
    output_dir: str,
    phase: str,
    device: str = "cuda:0",
):
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    image_ae = domain_model_config.model.to(device)
    image_inputs = domain_model_config.inputs.to(device)
    size = image_inputs.size(2), image_inputs.size(3)

    recon_images = image_ae(image_inputs)["recons"]

    for i in range(image_inputs.size()[0]):
        imageio.imwrite(
            os.path.join(image_dir, "%s_epoch_%s_inputs_%s.jpg" % (phase, epoch, i)),
            np.uint8(image_inputs[i].cpu().data.view(size).numpy() * 255),
        )
        imageio.imwrite(
            os.path.join(image_dir, "%s_epoch_%s_recons_%s.jpg" % (phase, epoch, i)),
            np.uint8(recon_images[i].cpu().data.view(size).numpy() * 255),
        )


def get_confusion_matrices(domain_config: DomainConfig, dataset_types: List = ["test"]):
    confusion_matrices = {}
    for dataset_type in dataset_types:
        confusion_matrices[dataset_type] = get_confusion_matrix(
            domain_config, dataset_type
        )
    return confusion_matrices


def get_confusion_matrix(domain_config: DomainConfig, dataset_type: str = "test"):
    device = get_device()
    model = domain_config.domain_model_config.classifier.to(device).eval()
    dataloader = domain_config.data_loader_dict[dataset_type]
    all_labels = []
    all_preds = []

    for i, sample in enumerate(dataloader):
        inputs = sample[domain_config.data_key].to(device)
        labels = sample[domain_config.label_key]
        outputs = model(inputs)["outputs"]
        _, preds = torch.max(outputs, 1)

        all_labels.extend(list(labels.detach().cpu().numpy()))
        all_preds.extend(list(preds.detach().cpu().numpy()))

    return confusion_matrix(all_labels, all_preds)


def visualize_latent_space_pca_walk(
    domain_config: DomainConfig,
    output_dir: str,
    dataset_type: str = "test",
    n_components: int = 2,
    n_steps: int = 11,
):

    device = get_device()
    dataset = domain_config.data_loader_dict[dataset_type].dataset
    model = domain_config.domain_model_config.classifier.to(device)
    model.eval()
    data_key = domain_config.data_key
    label_key = domain_config.label_key

    latent_dict = get_latent_representations_for_model(
        dataset=dataset, model=model, data_key=data_key, label_key=label_key
    )
    latents = latent_dict["latents"]
    labels = latent_dict["labels"]

    sc = StandardScaler()
    norm_latents = sc.fit_transform(latents)
    pc = PCA(n_components=n_components)
    pc.fit(norm_latents)
    latent = np.mean(latents, axis=0)
    # latent = latents[np.random.randint(0, len(latents))]
    components = pc.components_

    latent_space_walk_results = {}
    for i in range(n_components):
        stepseq = np.linspace(-10, 10, num=n_steps)
        pc_latents = np.array([latent] * n_steps)
        for j in range(n_steps):
            pc_latents[j, :] += stepseq[j] * components[i]
        walk_latents = torch.FloatTensor(pc_latents).to(device)
        walk_recons = model.decode(walk_latents).detach().cpu().numpy()
        latent_space_walk_results["pc" + str(i)] = walk_recons

    for k in latent_space_walk_results.keys():
        plot_image_seq(
            output_dir=output_dir, prefix=k, image_seq=latent_space_walk_results[k]
        )
