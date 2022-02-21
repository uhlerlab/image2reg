from collections import Counter

import numpy as np
import pandas as pd
import torch
from imblearn.under_sampling import RandomUnderSampler
import copy

from matplotlib import pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit,
    GroupKFold,
    LeaveOneGroupOut,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
import seaborn as sns
from tqdm.notebook import tqdm
from umap import UMAP

from src.data.datasets import IndexedTensorDataset


def plot_conditions_for_datadict(data_dict, figsize=[15, 6]):
    labels = {}
    for mode in data_dict.keys():
        for _, _, label in data_dict[mode].dataset:
            if mode in labels:
                labels[mode].append(label)
            else:
                labels[mode] = [label]
    all_label_counts = []
    for mode in labels.keys():
        label_counts = dict(Counter(labels[mode]))
        label_counts = pd.DataFrame(label_counts, index=[1]).transpose()
        label_counts.columns = ["count"]
        label_counts["dataset"] = mode
        all_label_counts.append(label_counts)
    all_label_counts = pd.concat(all_label_counts)
    all_label_counts["target"] = np.array(all_label_counts.index)
    all_label_counts = all_label_counts.sort_values("count", ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.barplot(data=all_label_counts, x="target", y="count", hue="dataset")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    return fig, ax, all_label_counts


class CustomKNNClassifier(object):
    def __init__(self, clf, samples, ks=list(range(1, 11))):
        super().__init__()
        self.clf = clf
        self.ks = ks
        self.samples = samples


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def train_model_topk_acc(
    model,
    optimizer,
    criterion,
    data_dict,
    nn_clf=None,
    n_epochs=1000,
    early_stopping=50,
    log_epochs=False,
    device="cuda:0",
):

    model.to(device)
    criterion.to(device)
    print(device)

    best_val_loss = np.infty
    best_model_weights = None
    es_counter = 0
    model.apply(weight_reset)

    if log_epochs:
        epoch_nn_clf = nn_clf
    else:
        epoch_nn_clf = None

    for i in tqdm(range(n_epochs)):
        if es_counter > early_stopping:
            break
        for mode in ["train", "val"]:
            loss, topk_acc, mean_topk_acc = process_single_epoch(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                data=data_dict[mode],
                mode=mode,
                nn_clf=None,
                device=device,
            )
            if log_epochs:
                print("{} loss: {}".format(mode, loss))
                if topk_acc is not None:
                    print("{} top-k accuracies: {}".format(mode, topk_acc))
                if mean_topk_acc is not None:
                    print("{} mean top-k accuracies: {}".format(mode, mean_topk_acc))
                print("---" * 20)

            if mode == "train":
                if loss < best_val_loss:
                    best_val_loss = loss
                    best_model_weights = copy.deepcopy(model.state_dict())
                else:
                    es_counter += 1
        if log_epochs:
            print("---" * 20)

    model.load_state_dict(best_model_weights)

    for mode in ["train", "val", "test"]:
        loss, topk_acc, mean_topk_acc = process_single_epoch(
            model, optimizer, criterion, data_dict[mode], "test", nn_clf=nn_clf
        )
        print("{} loss: {}".format(mode, loss))

        if topk_acc is not None:
            print("{} top-k accuracies: {}".format(mode, topk_acc))

        if mean_topk_acc is not None:
            print("{} mean top-k accuracies: {}".format(mode, mean_topk_acc))

    return model, topk_acc, mean_topk_acc


def process_single_epoch(
    model, optimizer, criterion, data, mode, nn_clf=None, device="cuda:0"
):
    total_loss = 0
    all_outputs = []
    all_targets = []
    if mode == "train":
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    for inputs, labels, targets in data:
        inputs = inputs.type(torch.FloatTensor).to(device)
        labels = labels.type(torch.FloatTensor).to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(mode == "train"):
            outputs = model(inputs)
            if nn_clf is not None:
                all_outputs.extend(list(outputs.clone().detach().cpu().numpy()))
                all_targets.extend(list(targets))
            loss = criterion(outputs.to(device), labels.to(device))
            total_loss += loss.item() * outputs.size(0)

        if mode == "train":
            loss.backward()
            optimizer.step()

    if nn_clf is not None:
        topk_acc = {}
        mean_topk_acc = {}
        for k in nn_clf.ks:
            k_correct = 0
            neighbor_preds = nn_clf.clf.kneighbors(
                np.array(all_outputs), k, return_distance=False
            )
            for i in range(len(neighbor_preds)):
                if all_targets[i] in nn_clf.samples[neighbor_preds[i]]:
                    k_correct += 1
            topk_acc[k] = np.round(k_correct / len(all_targets), 5)

            k_correct=0
            mean_outputs = pd.DataFrame(all_outputs)
            mean_outputs["target"] = all_targets
            mean_outputs = mean_outputs.groupby("target").mean()
            mean_nb_preds = nn_clf.clf.kneighbors(
                np.array(mean_outputs), k, return_distance=False
            )
            targets = list(mean_outputs.index)
            for i in range(len(mean_nb_preds)):
                if k == 5 and len(targets) == 1:
                    print("Target:", targets[i], "5NN:", nn_clf.samples[mean_nb_preds[i]])
                if targets[i] in nn_clf.samples[mean_nb_preds[i]]:
                    k_correct += 1
            mean_topk_acc[k] = np.round(k_correct / len(targets), 5)
    else:
        topk_acc = None
        mean_topk_acc=None

    total_loss /= len(data.dataset)
    return total_loss, topk_acc, mean_topk_acc


def get_data_dict(
    data,
    labels,
    val_test_size=[0.2, 0.2],
    batch_size=32,
    group_labels=None,
    scale_x=True,
    scale_y=False,
    random_state=1234,
    balanced=False,
):

    idc = np.array(list(range(len(data))))
    if balanced:
        idc, _ = RandomUnderSampler(random_state=random_state).fit_resample(
            idc.reshape(-1, 1), list(labels.index)
        )
        idc = idc.ravel()

    if group_labels is None:
        train_val_idc, test_idc = train_test_split(
            idc, test_size=val_test_size[1], random_state=random_state
        )
        train_idc, val_idc = train_test_split(
            train_val_idc, test_size=(val_test_size[0] / (1 - val_test_size[1]))
        )
    else:
        group_labels = group_labels[idc]
        gss = GroupShuffleSplit(
            n_splits=2, test_size=val_test_size[1], random_state=random_state
        )
        train_val_idc, test_idc = next(gss.split(idc, groups=group_labels))
        gss = GroupShuffleSplit(
            n_splits=2,
            test_size=(val_test_size[0] / (1 - val_test_size[1])),
            random_state=random_state,
        )
        train_val_groups = group_labels[train_val_idc]
        train_val_idc = idc[train_val_idc]
        test_idc = idc[test_idc]
        train_idc, val_idc = next(gss.split(train_val_idc, groups=train_val_groups))
        train_idc = train_val_idc[train_idc]
        val_idc = train_val_idc[val_idc]

    train_data, train_labels = data.iloc[train_idc], labels.iloc[train_idc]
    val_data, val_labels = data.iloc[val_idc], labels.iloc[val_idc]
    test_data, test_labels = data.iloc[test_idc], labels.iloc[test_idc]

    if scale_x:
        sc = StandardScaler().fit(train_data)
        train_data = sc.transform(train_data)
        val_data = sc.transform(val_data)
        test_data = sc.transform(test_data)

    if scale_y:
        sc = StandardScaler().fit(train_labels)
        train_labels = sc.transform(train_labels)
        val_labels = sc.transform(val_labels)
        test_labels = sc.transform(test_labels)

    train_dataset = IndexedTensorDataset(
        torch.FloatTensor(np.array(train_data)),
        torch.FloatTensor(np.array(train_labels)),
        list(labels.iloc[train_idc].index),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = IndexedTensorDataset(
        torch.FloatTensor(np.array(val_data)),
        torch.FloatTensor(np.array(val_labels)),
        list(labels.iloc[val_idc].index),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = IndexedTensorDataset(
        torch.FloatTensor(np.array(test_data)),
        torch.FloatTensor(np.array(test_labels)),
        list(labels.iloc[test_idc].index),
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def get_logo_data_dicts(
    data,
    labels,
    group_labels,
    val_size=0.05,
    batch_size=128,
    random_state=1234,
    scale_x=True,
    scale_y=False,
    balanced=False,
):
    logo = LeaveOneGroupOut()
    data_dicts = []

    idc = np.array(list(range(len(data))))
    if balanced:
        idc, _ = RandomUnderSampler(random_state=random_state).fit_resample(
            idc.reshape(-1, 1), list(labels.index)
        )
        idc = list(idc.ravel())
        data = data.iloc[idc]
        labels = labels.iloc[idc]
        group_labels = np.array(labels.index)

    for train_val_idc, test_idc in logo.split(data, groups=group_labels):

        test_data, test_labels = data.iloc[test_idc], labels.iloc[test_idc]

        train_val_data, train_val_labels = (
            data.iloc[train_val_idc],
            labels.iloc[train_val_idc],
        )

        gss = GroupShuffleSplit(n_splits=100, test_size=val_size)
        train_val_group_labels = np.array(train_val_labels.index)
        train_idc, val_idc = next(
            gss.split(train_val_data, groups=train_val_group_labels)
        )

        train_data, train_labels = (
            train_val_data.iloc[train_idc],
            train_val_labels.iloc[train_idc],
        )
        val_data, val_labels = (
            train_val_data.iloc[val_idc],
            train_val_labels.iloc[val_idc],
        )

        if scale_x:
            sc = StandardScaler().fit(train_data)
            train_data = sc.transform(train_data)
            val_data = sc.transform(val_data)
            test_data = sc.transform(test_data)

        if scale_y:
            sc = StandardScaler().fit(train_labels)
            train_labels = sc.transform(train_labels)
            val_labels = sc.transform(val_labels)
            test_labels = sc.transform(test_labels)

        train_dataset = IndexedTensorDataset(
            torch.FloatTensor(np.array(train_data)),
            torch.FloatTensor(np.array(train_labels)),
            list(train_labels.index),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = IndexedTensorDataset(
            torch.FloatTensor(np.array(val_data)),
            torch.FloatTensor(np.array(val_labels)),
            list(val_labels.index),
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_dataset = IndexedTensorDataset(
            torch.FloatTensor(np.array(test_data)),
            torch.FloatTensor(np.array(test_labels)),
            list(test_labels.index),
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        data_dict = {"train": train_loader, "val": val_loader, "test": test_loader}
        data_dicts.append(data_dict)
    return data_dicts


def evaluate_top_k_accuracy(preds, labels, node_embs, k=5, random_state=1234):
    np.random.seed(random_state)
    k_correct = 0
    k_baseline_correct = 0
    neigh = NearestNeighbors(n_neighbors=k, p=2)
    targets = np.array(node_embs.index)
    fitted_neigh = neigh.fit(np.array(node_embs))

    neighbor_preds = fitted_neigh.kneighbors(preds, k, return_distance=False)
    for i in range(len(neighbor_preds)):
        if labels[i] in targets[neighbor_preds[i]]:
            k_correct += 1
        if labels[i] in np.random.choice(targets, size=k):
            k_baseline_correct += 1
    return k_correct / len(labels), k_baseline_correct / len(labels)


def get_preds_label_dict(model, data):
    preds = []
    labels = []
    model.eval()
    for batch_inputs, _, batch_labels in data:
        batch_preds = model(batch_inputs)
        preds.extend(list(batch_preds.clone().detach().numpy()))
        labels.extend(list(batch_labels))
    return np.array(preds), np.array(labels)


def label_point(x, y, val, ax, size=10, highlight=None):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    for i in range(len(x)):
        if highlight is not None and val[i] == highlight:
            c = "r"
            weight = "bold"
        else:
            c = "k"
            weight = "normal"
        if x[i] > xmin and x[i] < xmax and y[i] > ymin and y[i] < ymax:
            ax.text(x[i] + 0.02, y[i], val[i], {"size": size, "c": c, "weight": weight})


def plot_translation(
    model,
    dataset,
    node_embs,
    figsize=[10, 6],
    random_state=1234,
    text_size=10,
    crop=False,
    highlight_target=None,
    filter_targets=None,
):
    all_outputs = []
    all_targets = []
    model.eval()
    for inputs, labels, targets in dataset:
        outputs = model(inputs)
        all_outputs.append(list(outputs.clone().detach().cpu().numpy()))
        all_targets.append(targets)
    print(np.concatenate([np.array(node_embs), np.array(all_outputs)]).shape)
    umap = UMAP(random_state=random_state).fit(
        np.concatenate([np.array(node_embs), np.array(all_outputs)])
    )
    umap_node_embs = pd.DataFrame(
        umap.transform(node_embs), index=node_embs.index, columns=["umap_0", "umap_1"]
    )
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        np.array(umap_node_embs.loc[:, "umap_0"]),
        np.array(umap_node_embs.loc[:, "umap_1"]),
        c="k",
        s=20,
        label="regulatory embeddings",
    )
    umap_pred_embs = pd.DataFrame(
        umap.transform(np.array(all_outputs)),
        index=all_targets,
        columns=["umap_0", "umap_1"],
    )
    umap_pred_embs["target"] = np.array(umap_pred_embs.index)

    if filter_targets is not None:
        umap_pred_embs = umap_pred_embs.loc[
            umap_pred_embs.loc[:, "target"].isin(filter_targets)
        ]
    ax = sns.scatterplot(
        data=umap_pred_embs, x="umap_0", y="umap_1", hue="target", s=5, alpha=0.7
    )
    if crop:
        umap_0_pred_embs = np.array(umap_pred_embs.loc[:, "umap_0"])
        umap_1_pred_embs = np.array(umap_pred_embs.loc[:, "umap_1"])

        ax.set_xlim(umap_0_pred_embs.min(), umap_0_pred_embs.max())
        ax.set_ylim(umap_1_pred_embs.min(), umap_1_pred_embs.max())

    label_point(
        np.array(umap_node_embs.loc[:, "umap_0"]),
        np.array(umap_node_embs.loc[:, "umap_1"]),
        np.array(umap_node_embs.index).astype("str"),
        ax=ax,
        size=text_size,
        highlight=highlight_target,
    )
    return fig, ax


def evaluate_loto_cv(
    model,
    data_dicts,
    targets,
    optimizer,
    criterion,
    nn_clf,
    n_epochs=1000,
    early_stopping=20,
    device="cuda:0",
):
    topk_accs = []
    mean_topk_accs = []
    for i in tqdm(range(len(data_dicts)), desc="Run LoTo CV"):
        model.apply(weight_reset)
        data_dict = data_dicts[i]
        _, topk_acc, mean_topk_acc = train_model_topk_acc(
            model=model,
            data_dict=data_dict,
            optimizer=optimizer,
            criterion=criterion,
            n_epochs=n_epochs,
            early_stopping=early_stopping,
            nn_clf=nn_clf,
            log_epochs=False,
            device=device,
        )
        topk_accs.append(list(topk_acc.values()))
        mean_topk_accs.append(list(mean_topk_acc.values()))
    topk_accs = dict(zip(targets, topk_accs))
    topk_accs = pd.DataFrame(
        topk_accs, index=["top{}".format(i) for i in list(nn_clf.ks)]
    )
    mean_topk_accs = dict(zip(targets, mean_topk_accs))
    mean_topk_accs = pd.DataFrame(
        mean_topk_accs, index=["mean_top{}".format(i) for i in list(nn_clf.ks)]
    )
    return topk_accs.transpose(), mean_topk_accs.transpose()
