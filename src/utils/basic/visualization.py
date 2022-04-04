import logging
import os
from typing import Iterable

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import ConfusionMatrixDisplay


def plot_train_val_hist(
    training_history: ndarray,
    validation_history: ndarray,
    output_dir: str,
    y_label: str,
    title=None,
    posfix: str = "",
):
    r""" A function to visualize the evolution of the training and validation loss during the training.
    Parameters
    ----------
    training_history : list, numpy.ndarray
        The training loss for the individual training epochs.
    validation_history : list, numpy.ndarray
        The validation lss for the individual training epochs.
    output_dir : str
        The path of the directory the visualization of the evolution of the loss is stored in.
    y_label : str
        The label of the y-axis of the visualization.
    title : None, str
        The title of the visualization. If ``None`` is given, it is `'Fitting History` by default.
    posfix : str
        An additional posfix for the file path.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)
    epochs = np.arange(len(training_history))
    lines = plt.plot(epochs, training_history, epochs, validation_history)
    plt.ylabel(y_label)
    plt.xlabel("Epoch")
    plt.legend(
        ("Training Loss", "Validation Loss", "Validation loss"),
        loc="upper right",
        markerscale=2.0,
    )
    if title is None:
        title = "Fitting History"
    plt.title(title)
    plt.savefig(output_dir + "plotted_fitting_hist{}.png".format(posfix))
    plt.close()


def show_plane(ax, plane, cmap="gray", title=None):
    ax.imshow(plane, cmap=cmap)
    ax.axis("off")

    if title:
        ax.set_title(title)


def explore_samples_of_dataset(datasets, cmap="gray", dataset_id=0):
    from ipywidgets import interact

    dataset_names = list(datasets.keys())

    n = len(datasets[dataset_names[dataset_id]])

    @interact(plane=(0, n - 1))
    def display_slice(plane=0):
        fig, ax = plt.subplots(figsize=(12, 12), nrows=1, ncols=1)

        show_plane(
            ax,
            datasets[dataset_names[dataset_id]][plane],
            cmap=cmap,
            title="{}_{}".format(dataset_names[dataset_id], str(plane)),
        )
        plt.show()

    return display_slice


def explore_segmentation_of_dataset(
    datasets, segmented_datasets, cmap="gray", dataset_id=0
):
    from ipywidgets import interact

    dataset_names = list(datasets.keys())

    n = len(datasets[dataset_names[dataset_id]])

    @interact(plane=(0, n - 1))
    def display_slice(plane=0):
        fig, ax = plt.subplots(figsize=(40, 20), nrows=1, ncols=2)

        show_plane(
            ax[0],
            datasets[dataset_names[dataset_id]][plane],
            cmap=cmap,
            title="{}_{}".format(dataset_names[dataset_id], str(plane)),
        )
        show_plane(
            ax[1],
            segmented_datasets[dataset_names[dataset_id]][plane],
            cmap=create_ade20k_label_colormap(),
            title="Segmented {}_{}".format(dataset_names[dataset_id], str(plane)),
        )
        plt.show()

    return display_slice


def create_ade20k_label_colormap():
    """Creates a label colormap used in ADE20K segmentation benchmark.
    Returns:
      A colormap for visualizing segmentation results.
    """
    colors = (
        np.asarray(
            [
                [0, 0, 0],
                [120, 120, 120],
                [180, 120, 120],
                [6, 230, 230],
                [80, 50, 50],
                [4, 200, 3],
                [120, 120, 80],
                [140, 140, 140],
                [204, 5, 255],
                [230, 230, 230],
                [4, 250, 7],
                [224, 5, 255],
                [235, 255, 7],
                [150, 5, 61],
                [120, 120, 70],
                [8, 255, 51],
                [255, 6, 82],
                [143, 255, 140],
                [204, 255, 4],
                [255, 51, 7],
                [204, 70, 3],
                [0, 102, 200],
                [61, 230, 250],
                [255, 6, 51],
                [11, 102, 255],
                [255, 7, 71],
                [255, 9, 224],
                [9, 7, 230],
                [220, 220, 220],
                [255, 9, 92],
                [112, 9, 255],
                [8, 255, 214],
                [7, 255, 224],
                [255, 184, 6],
                [10, 255, 71],
                [255, 41, 10],
                [7, 255, 255],
                [224, 255, 8],
                [102, 8, 255],
                [255, 61, 6],
                [255, 194, 7],
                [255, 122, 8],
                [0, 255, 20],
                [255, 8, 41],
                [255, 5, 153],
                [6, 51, 255],
                [235, 12, 255],
                [160, 150, 20],
                [0, 163, 255],
                [140, 140, 140],
                [250, 10, 15],
                [20, 255, 0],
                [31, 255, 0],
                [255, 31, 0],
                [255, 224, 0],
                [153, 255, 0],
                [0, 0, 255],
                [255, 71, 0],
                [0, 235, 255],
                [0, 173, 255],
                [31, 0, 255],
                [11, 200, 200],
                [255, 82, 0],
                [0, 255, 245],
                [0, 61, 255],
                [0, 255, 112],
                [0, 255, 133],
                [255, 0, 0],
                [255, 163, 0],
                [255, 102, 0],
                [194, 255, 0],
                [0, 143, 255],
                [51, 255, 0],
                [0, 82, 255],
                [0, 255, 41],
                [0, 255, 173],
                [10, 0, 255],
                [173, 255, 0],
                [0, 255, 153],
                [255, 92, 0],
                [255, 0, 255],
                [255, 0, 245],
                [255, 0, 102],
                [255, 173, 0],
                [255, 0, 20],
                [255, 184, 184],
                [0, 31, 255],
                [0, 255, 61],
                [0, 71, 255],
                [255, 0, 204],
                [0, 255, 194],
                [0, 255, 82],
                [0, 10, 255],
                [0, 112, 255],
                [51, 0, 255],
                [0, 194, 255],
                [0, 122, 255],
                [0, 255, 163],
                [255, 153, 0],
                [0, 255, 10],
                [255, 112, 0],
                [143, 255, 0],
                [82, 0, 255],
                [163, 255, 0],
                [255, 235, 0],
                [8, 184, 170],
                [133, 0, 255],
                [0, 255, 92],
                [184, 0, 255],
                [255, 0, 31],
                [0, 184, 255],
                [0, 214, 255],
                [255, 0, 112],
                [92, 255, 0],
                [0, 224, 255],
                [112, 224, 255],
                [70, 184, 160],
                [163, 0, 255],
                [153, 0, 255],
                [71, 255, 0],
                [255, 0, 163],
                [255, 204, 0],
                [255, 0, 143],
                [0, 255, 235],
                [133, 255, 0],
                [255, 0, 235],
                [245, 0, 255],
                [255, 0, 122],
                [255, 245, 0],
                [10, 190, 212],
                [214, 255, 0],
                [0, 204, 255],
                [20, 0, 255],
                [255, 255, 0],
                [0, 153, 255],
                [0, 41, 255],
                [0, 255, 204],
                [41, 0, 255],
                [41, 255, 0],
                [173, 0, 255],
                [0, 245, 255],
                [71, 0, 255],
                [122, 0, 255],
                [0, 255, 184],
                [0, 92, 255],
                [184, 255, 0],
                [0, 133, 255],
                [255, 214, 0],
                [25, 194, 194],
                [102, 255, 0],
                [92, 0, 255],
            ]
        )
        / 255
    )

    cmap = LinearSegmentedColormap.from_list("segmentation_cmap", colors, N=256)
    return cmap


def plot_image_seq(output_dir, image_seq, prefix: str = ""):
    for i in range(len(image_seq)):
        imageio.imwrite(
            os.path.join(output_dir, "{}_walk_recon_{}.jpg".format(prefix, str(i))),
            np.uint8(np.squeeze(image_seq[i] * 255)),
        )


def plot_confusion_matrices(
    confusion_matrices_dict: dict, output_dir: str, display_labels: np.ndarray = None
):
    for k, cmatrix in confusion_matrices_dict.items():
        c = cmatrix.shape[0] // 2
        fig, ax = plt.subplots(figsize=[6 + c, 4 + c])
        cmd = ConfusionMatrixDisplay(cmatrix, display_labels=display_labels)
        cmd.plot(ax=ax, values_format=".2f", xticks_rotation="vertical")
        plt.savefig(os.path.join(output_dir, "confusion_matrix_{}.png".format(str(k))))
        plt.close()


def save_confusion_matrices(
    confusion_matrices_dict: dict, output_dir: str, labels=Iterable
):
    for k, cmatrix in confusion_matrices_dict.items():
        cmatrix_df = pd.DataFrame(cmatrix, columns=labels, index=labels)
        cmatrix_df.to_csv(os.path.join(output_dir, "{}_cmatrix.csv".format(k)))
