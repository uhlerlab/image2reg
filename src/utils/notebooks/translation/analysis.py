import copy
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit,
    LeaveOneGroupOut,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

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
    use_val_data=False,
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

    epoch_modes = ["train"]
    if use_val_data:
        epoch_modes = epoch_modes + ["val"]
        val_mode = "val"
    else:
        val_mode = "train"

    for i in tqdm(range(n_epochs)):
        if es_counter > early_stopping:
            break
        for mode in epoch_modes:
            loss, topk_acc, mean_topk_acc, mean_knns = process_single_epoch(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                data=data_dict[mode],
                mode=mode,
                nn_clf=epoch_nn_clf,
                device=device,
            )
            if log_epochs:
                print("{} loss: {}".format(mode, loss))
                if topk_acc is not None:
                    print("{} top-k accuracies: {}".format(mode, topk_acc))
                if mean_topk_acc is not None:
                    print("{} mean top-k accuracies: {}".format(mode, mean_topk_acc))
                print("---" * 20)

            if mode == val_mode:
                if loss < best_val_loss:
                    best_val_loss = loss
                    best_model_weights = copy.deepcopy(model.state_dict())
                else:
                    es_counter += 1
        if log_epochs:
            print("---" * 20)

    model.load_state_dict(best_model_weights)

    for mode in epoch_modes + ["test"]:
        loss, topk_acc, mean_topk_acc, mean_knns = process_single_epoch(
            model, optimizer, criterion, data_dict[mode], "test", nn_clf=nn_clf
        )
        print("{} loss: {}".format(mode, loss))

        if topk_acc is not None:
            print("{} top-k accuracies: {}".format(mode, topk_acc))

        if mean_topk_acc is not None:
            print("{} mean top-k accuracies: {}".format(mode, mean_topk_acc))

    return model, topk_acc, mean_topk_acc, mean_knns


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
        nn_hit_idc = []
        neighbor_preds = nn_clf.clf.kneighbors(
            np.array(all_outputs),
            n_neighbors=len(nn_clf.samples),
            return_distance=False,
        )
        for i in range(len(neighbor_preds)):
            nn_hit_idc.append(
                np.where(nn_clf.samples[neighbor_preds[i].flatten()] == all_targets[i])[
                    0
                ]
            )

        mean_hit_idc = []
        mean_outputs = pd.DataFrame(all_outputs)
        mean_outputs["target"] = all_targets
        mean_outputs = mean_outputs.groupby("target").mean()
        mean_nb_preds = nn_clf.clf.kneighbors(
            np.array(mean_outputs),
            n_neighbors=len(nn_clf.samples),
            return_distance=False,
        )
        targets = list(mean_outputs.index)
        mean_knns = []
        for i in range(len(mean_nb_preds)):
            mean_hit_idc.append(
                np.where(nn_clf.samples[mean_nb_preds[i]] == targets[i])
            )
            mean_knns.append(nn_clf.samples[mean_nb_preds[i]])
        topk_acc = {}
        mean_topk_acc = {}
        for k in nn_clf.ks:
            topk_acc[k] = np.round(np.mean(np.array(nn_hit_idc) < k), 6)
            mean_topk_acc[k] = np.round(np.mean(np.array(mean_hit_idc) < k), 6)
    else:
        topk_acc = None
        mean_topk_acc = None
        mean_knns = None

    total_loss /= len(data.dataset)
    return total_loss, topk_acc, mean_topk_acc, mean_knns


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
        train_labels = pd.DataFrame(
            sc.transform(train_labels),
            index=train_labels.index,
            columns=train_labels.columns,
        )
        val_labels = pd.DataFrame(
            sc.transform(val_labels), index=val_labels.index, columns=val_labels.columns
        )
        test_labels = pd.DataFrame(
            sc.transform(test_labels),
            index=test_labels.index,
            columns=test_labels.columns,
        )

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

        if val_size > 0:
            gss = GroupShuffleSplit(n_splits=100, test_size=val_size)
            train_val_group_labels = np.array(train_val_labels.index)
            train_idc, val_idc = next(
                gss.split(train_val_data, groups=train_val_group_labels)
            )
        else:
            train_idc = list(range(len(train_val_data)))
            val_idc = None

        train_data, train_labels = (
            train_val_data.iloc[train_idc],
            train_val_labels.iloc[train_idc],
        )
        if val_idc is not None:
            val_data, val_labels = (
                train_val_data.iloc[val_idc],
                train_val_labels.iloc[val_idc],
            )
        else:
            val_data = None
            val_labels = None

        if scale_x:
            sc = StandardScaler().fit(train_data)
            train_data = sc.transform(train_data)
            if val_data is not None:
                val_data = sc.transform(val_data)
            test_data = sc.transform(test_data)

        if scale_y:
            sc = StandardScaler().fit(train_labels)
            train_labels = pd.DataFrame(
                sc.transform(train_labels),
                index=train_labels.index,
                columns=train_labels.columns,
            )
            if val_labels is not None:
                val_labels = pd.DataFrame(
                    sc.transform(val_labels),
                    index=val_labels.index,
                    columns=val_labels.columns,
                )
            test_labels = pd.DataFrame(
                sc.transform(test_labels),
                index=test_labels.index,
                columns=test_labels.columns,
            )

        train_dataset = IndexedTensorDataset(
            torch.FloatTensor(np.array(train_data)),
            torch.FloatTensor(np.array(train_labels)),
            list(train_labels.index),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if val_data is not None:
            val_dataset = IndexedTensorDataset(
                torch.FloatTensor(np.array(val_data)),
                torch.FloatTensor(np.array(val_labels)),
                list(val_labels.index),
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None

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


def label_point(x, y, val, ax, size=10, highlight=None, highlight_other=None):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    for i in range(len(x)):
        if highlight is not None and val[i] == highlight:
            c = "r"
            weight = "bold"
        elif highlight_other is not None and val[i] in highlight_other:
            c = "b"
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
    highlight_nns=None,
    filter_targets=None,
    pred_size=10,
    reg_size=20,
):
    all_outputs = []
    all_targets = []
    model.eval()
    for inputs, labels, targets in dataset:
        outputs = model(inputs)
        all_outputs.append(list(outputs.clone().detach().cpu().numpy()))
        all_targets.append(targets)
    all_sample_size = np.concatenate(
        [np.array(node_embs), np.array(all_outputs)]
    ).shape[0]
    node_embs_size = len(node_embs)
    mds_embs = MDS(
        random_state=random_state,
        n_components=2,
        # perplexity=int(np.sqrt(all_sample_size)) + 1,
        # init="pca",
        # learning_rate=200,
    ).fit_transform(np.concatenate([np.array(node_embs), np.array(all_outputs)]))
    mds_node_embs = pd.DataFrame(
        mds_embs[:node_embs_size,], index=node_embs.index, columns=["mds_0", "mds_1"]
    )
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        np.array(mds_node_embs.loc[:, "mds_0"]),
        np.array(mds_node_embs.loc[:, "mds_1"]),
        c="k",
        s=reg_size,
        label="regulatory embeddings",
        alpha=0.7,
    )
    mds_pred_embs = pd.DataFrame(
        mds_embs[node_embs_size:], index=all_targets, columns=["mds_0", "mds_1"],
    )
    mds_pred_embs["target"] = np.array(mds_pred_embs.index)

    if filter_targets is not None:
        mds_pred_embs = mds_pred_embs.loc[
            mds_pred_embs.loc[:, "target"].isin(filter_targets)
        ]
    ax = sns.scatterplot(
        data=mds_pred_embs, x="mds_0", y="mds_1", hue="target", s=pred_size,
    )

    if crop:
        mds_0_pred_embs = np.array(mds_pred_embs.loc[:, "mds_0"])
        mds_1_pred_embs = np.array(mds_pred_embs.loc[:, "mds_1"])

        ax.set_xlim(mds_0_pred_embs.min(), mds_0_pred_embs.max())
        ax.set_ylim(mds_1_pred_embs.min(), mds_1_pred_embs.max())

    label_point(
        np.array(mds_node_embs.loc[:, "mds_0"]),
        np.array(mds_node_embs.loc[:, "mds_1"]),
        np.array(mds_node_embs.index).astype("str"),
        ax=ax,
        size=text_size,
        highlight=highlight_target,
        highlight_other=highlight_nns,
    )
    return fig, ax, mds_node_embs


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
        _, topk_acc, mean_topk_acc, _ = train_model_topk_acc(
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


def get_mean_knn_dict(embs, k=5):
    targets = np.array(list(embs.index))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(np.array(embs))
    nn_clf = CustomKNNClassifier(clf=nn, samples=targets, ks=k)
    knn_dict = {}
    for target in targets:
        preds = nn_clf.clf.kneighbors(
            np.array(embs.loc[target]).reshape(1, -1), return_distance=False
        )
        knn_dict[target] = list(targets[preds].flatten()[1:])
    knn_dict = {k: knn_dict[k] for k in sorted(knn_dict)}
    return knn_dict


def get_centroid_knn_dict(embs, labels, k=5):
    targets = sorted(np.unique(labels))
    knn_dict = {}

    for target in targets:
        filtered_embs = embs.loc[labels != target]
        filtered_labels = labels[labels != target]

        pred_input = embs.loc[labels == target]
        nn = NearestNeighbors(n_neighbors=k).fit(np.array(filtered_embs))
        nn_clf = CustomKNNClassifier(clf=nn, samples=filtered_labels, ks=k)
        preds = nn_clf.clf.kneighbors(np.array(pred_input), return_distance=False)
        nns = []
        for i in range(k):
            for pred in preds:
                nns.append(nn_clf.samples[pred[i]])
        indexes = np.unique(nns, return_index=True)[1]
        nns = [nns[index] for index in sorted(indexes)]
        knn_dict[target] = list(nns[:k])
    return knn_dict


def get_single_knn_dict(embs, labels, k=5):
    targets = np.unique(labels)
    knn_dict = {}

    for target in tqdm(targets):
        filtered_embs = embs.loc[labels != target]
        filtered_labels = labels[labels != target]

        pred_input = embs.loc[labels == target]
        nn = NearestNeighbors(n_neighbors=1).fit(np.array(filtered_embs))
        nn_clf = CustomKNNClassifier(clf=nn, samples=filtered_labels, ks=k)
        preds = nn_clf.clf.kneighbors(np.array(pred_input), return_distance=False)
        nns = nn_clf.samples[preds.flatten()]
        count_dict = dict(Counter(nns))
        count_dict = dict(
            sorted(count_dict.items(), key=lambda item: item[1], reverse=True)
        )
        knn_dict[target] = list(count_dict.keys())[:5]
    return knn_dict


def get_inv_gs_dict(targets, gs_dict):
    inv_gs_dict = {}
    for target in targets:
        inv_gs_dict[target] = []
        for k, v in gs_dict.items():
            if target in list(v):
                inv_gs_dict[target].append(k)
    return inv_gs_dict


def get_geneset_io(nn_dict, inv_geneset_dict, k=1):
    iou_dict = {}
    for target, nns in nn_dict.items():
        target_gs = set(inv_geneset_dict[target])
        nn_gs = []
        for i in range(k):
            nn_gs.extend(inv_geneset_dict[nns[i]])
        nn_gs = set(nn_gs)
        if len(target_gs) == 0:
            iou_dict[target] = np.nan
        else:
            iou_dict[target] = len(target_gs.intersection(nn_gs)) / len(
                target_gs.union(nn_gs)
            )
    return iou_dict


def scale_data(data, scaling=None):
    if scaling in ["minmax", "minmax_loto"]:
        scaler = MinMaxScaler()
        if scaling == "minmax":
            data = pd.DataFrame(
                scaler.fit_transform(data), columns=data.columns, index=data.index
            )
        elif scaling == "minmax_loto":
            scaled_data = data.copy()
            for idx in data.index:
                scaler.fit(np.array(data.loc[data.index != idx]))
                scaled_data.loc[idx] = scaler.transform(
                    np.array(scaled_data.loc[idx]).reshape(1, -1)
                )
            data = scaled_data
    elif scaling in ["znorm", "znorm_loto"]:
        scaler = StandardScaler()
        if scaling == "znorm":
            data = pd.DataFrame(
                scaler.fit_transform(data), columns=data.columns, index=data.index
            )
        elif scaling == "znorm_loto":
            scaled_data = data.copy()
            for idx in data.index:
                scaler.fit(np.array(data.loc[data.index != idx]))
                scaled_data.loc[idx] = scaler.transform(
                    np.array(scaled_data.loc[idx]).reshape(1, -1)
                )
            data = scaled_data
    return data


def get_embeddings(data, method="pca", scaling=None, seed=1234, selection=None):
    data = scale_data(data, scaling=scaling)
    if selection is not None:
        data = data.loc[data.index.isin(selection)]
    if method == "pca":
        mapper = PCA(n_components=2, random_state=seed)
    elif method == "mds":
        mapper = MDS(n_components=2, random_state=seed)
    elif method == "tsne":
        mapper = TSNE(
            n_components=2,
            random_state=seed,
            perplexity=int(np.sqrt(len(data))) + 1,
            init="pca",
            learning_rate="auto",
        )
    else:
        raise NotImplementedError
    embs = pd.DataFrame(
        mapper.fit_transform(data),
        columns=["{}_0".format(method), "{}_1".format(method)],
        index=data.index,
    )
    return embs


def plot_space(
    data,
    method="pca",
    scaling=None,
    seed=1234,
    figsize=[6, 4],
    label_points=True,
    text_size=10,
    selection=None,
):
    fig, ax = plt.subplots(figsize=figsize)
    embs = get_embeddings(
        data, method=method, scaling=scaling, seed=seed, selection=selection
    )
    ax = sns.scatterplot(data=embs, x="{}_0".format(method), y="{}_1".format(method))
    if label_points:
        label_point(
            np.array(embs.loc[:, "{}_0".format(method)]),
            np.array(embs.loc[:, "{}_1".format(method)]),
            np.array(embs.index).astype("str"),
            ax=ax,
            size=text_size,
        )
    return fig, ax


def get_sample_neighbor_dict(data, selection=None):
    if selection is not None:
        data = data.loc[data.index.isin(selection)]
    samples = np.array(list(data.index))
    nn = NearestNeighbors(n_neighbors=len(data))
    sample_neighbor_dict = {}
    nn.fit(np.array(data))
    for sample in samples:
        pred_idx = nn.kneighbors(
            np.array(data.loc[sample]).reshape(1, -1), return_distance=False
        )[0]
        sample_neighbor_dict[sample] = samples[pred_idx]
    return sample_neighbor_dict
