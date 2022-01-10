from src.utils.basic.io import get_file_list
import pandas as pd
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.manifold import TSNE
import seaborn as sns


class LogAnalyzer(object):
    def __init__(self, logfile: str):
        self.logfile = logfile

        self.train_loss = None
        self.val_loss = None
        self.test_loss = None
        self.best_epoch = None
        self.train_acc = None
        self.val_acc = None
        self.test_acc = None
        self.train_bacc = None
        self.val_bacc = None
        self.test_bacc = None

    def analyze(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.train_bacc = []
        self.val_bacc = []

        with open(self.logfile) as f:
            lines = f.readlines()
            for line in lines:
                line = line.lower()
                words = line.split()
                if "train" in line:
                    mode = "train"
                elif "val" in line:
                    mode = "val"
                elif "test" in line:
                    mode = "test"
                if "classification loss" in line:
                    loss = float(words[-1])
                    if mode == "train":
                        self.train_loss.append(loss)
                    elif mode == "val":
                        self.val_loss.append(loss)
                    elif mode == "test":
                        self.test_loss = loss
                elif "classification accuracy" in line:
                    acc = float(words[-1])
                    if mode == "train":
                        self.train_acc.append(acc)
                    elif mode == "val":
                        self.val_acc.append(acc)
                    elif mode == "test":
                        self.test_acc = acc
                elif "classification balanced accuracy" in line:
                    bacc = float(words[-1])
                    if mode == "train":
                        self.train_bacc.append(bacc)
                    elif mode == "val":
                        self.val_bacc.append(bacc)
                    elif mode == "test":
                        self.test_bacc = bacc
                elif "best model" in line:
                    self.best_epoch = int(words[-1]) - 1

        if len(self.train_acc) > 0:
            self.best_train_acc = self.train_acc[self.best_epoch]
            self.best_val_acc = self.val_acc[self.best_epoch]
            self.best_train_bacc = self.train_bacc[self.best_epoch]
            self.best_val_bacc = self.val_bacc[self.best_epoch]
        self.best_train_loss = self.train_loss[self.best_epoch]
        self.best_val_loss = self.val_loss[self.best_epoch]

    def __str__(self):
        return (
            "LogAnalyzer(train_acc={}, val_acc={}, test_acc={}, train_bacc={},"
            " val_bacc={}, test_bacc={})".format(
                self.best_train_acc,
                self.best_val_acc,
                self.test_acc,
                self.best_train_bacc,
                self.best_val_bacc,
                self.test_bacc,
            )
        )


def analyze_screen_results(screen_dir: str):
    results = {
        "target": [],
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
        "train_bacc": [],
        "val_bacc": [],
        "test_bacc": [],
    }
    logfiles = get_file_list(screen_dir, file_type_filter=".log")
    for logfile in logfiles:
        results["target"].append(logfile.split("/")[-2])
        la = LogAnalyzer(logfile)
        la.analyze()
        results["train_loss"].append(la.best_train_loss)
        results["val_loss"].append(la.best_val_loss)
        results["test_loss"].append(la.test_loss)
        results["train_acc"].append(la.best_train_acc)
        results["val_acc"].append(la.best_val_acc)
        results["test_acc"].append(la.test_acc)
        results["train_bacc"].append(la.best_train_bacc)
        results["val_bacc"].append(la.best_val_bacc)
        results["test_bacc"].append(la.test_bacc)
    return pd.DataFrame.from_dict(results)


def plot_spec_screen_results(
    root_dir: str,
    dataset_types: List[str],
    score_df: pd.DataFrame,
    label_col: str = "labels",
    figsize=[12, 6],
    filter_targets=None,
):
    subdirs = sorted(os.listdir(root_dir))
    for subdir in tqdm(subdirs, desc="Visualize latent spaces"):
        if filter_targets is None or subdir in filter_targets:
            fig, ax = plt.subplots(ncols=len(dataset_types), figsize=figsize)
            ax = ax.flatten()
            all_latents = []
            all_labels = []
            dataset_labels = []
            for i in range(len(dataset_types)):
                dataset_type = dataset_types[i]
                latents = pd.read_hdf(
                    os.path.join(root_dir, subdir, dataset_type + "_latents.h5")
                )
                all_labels.append(
                    np.array(
                        latents.loc[:, "labels"].map(
                            dict(zip([0, 1], sorted(["EMPTY", subdir.upper()])))
                        )
                    )
                )
                all_latents.append(np.array(latents.drop(columns=[label_col])))
                dataset_labels.extend([dataset_type] * len(latents))

            all_latents = np.concatenate(all_latents)
            all_labels = np.concatenate(all_labels)
            all_embs = TSNE(random_state=1234).fit_transform(all_latents)
            all_embs = pd.DataFrame(all_embs, columns=["tsne_0", "tsne_1"])
            all_embs["label"] = all_labels
            all_embs["dataset_type"] = np.array(dataset_labels)

            for i in range(len(dataset_types)):
                dataset_type = dataset_types[i]
                ax[i] = sns.scatterplot(
                    data=all_embs.loc[all_embs["dataset_type"] == dataset_type],
                    x="tsne_0",
                    y="tsne_1",
                    hue="label",
                    ax=ax[i],
                    hue_order=["EMPTY", subdir.upper()],
                    s=4,
                )
                score = score_df.loc[subdir.upper(), "{}_acc".format(dataset_type)]
                ax[i].set_title("{} data, acc: {:.4f}".format(dataset_type, score))
                ax[i].set_xlim([all_embs.tsne_0.min() - 20, all_embs.tsne_0.max() + 20])
                ax[i].set_ylim([all_embs.tsne_1.min() - 20, all_embs.tsne_1.max() + 20])

            fig.suptitle("Latent embeddings for {}".format(subdir))
            plt.show()
            plt.close()
