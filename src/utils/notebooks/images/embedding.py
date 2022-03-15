import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy as hc
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    mutual_info_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    rand_score,
    v_measure_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from umap import UMAP
from yellowbrick.cluster import KElbowVisualizer


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def plot_cc_score(
    cc_score,
    title,
    score_desc,
    figsize=[15, 6],
    fmt=".2f",
    cmap="RdYlGn",
    vmin=0,
    vmax=1,
    space_names=["space 1", "space 2"],
):
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(
        cc_score,
        cmap=cmap,
        annot=True,
        fmt=fmt,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": score_desc},
        ax=ax,
    )
    ax.set_xticklabels([i + 1 for i in range(len(cc_score[0]))])
    ax.set_yticklabels([i + 1 for i in range(len(cc_score[0]))])
    ax.set_xlabel("No. of clusters in {}".format(space_names[0]))
    ax.set_ylabel("No. of clusters in {}".format(space_names[1]))
    ax.set_title(title)


def compute_cc_score(sol1, sol2, score="ami"):
    if score == "mi":
        return mutual_info_score(sol1, sol2)
    elif score == "ami":
        return adjusted_mutual_info_score(sol1, sol2)
    elif score == "ari":
        return adjusted_rand_score(sol1, sol2)
    elif score == "ri":
        return rand_score(sol1, sol2)
    elif score == "v":
        return v_measure_score(sol1, sol2)
    elif score == "nmi":
        return normalized_mutual_info_score(sol1, sol2)
    else:
        raise NotImplementedError


def compute_cc_scores_perm_test(
    embs_1,
    embs_2,
    b=100,
    n_max_clusters=20,
    affinity="euclidean",
    linkages=["average", "average"],
    random_state=1234,
    score="ami",
):
    np.random.seed(random_state)
    perm_cc_scores = []
    cc_score = np.zeros((n_max_clusters, n_max_clusters))
    cluster_sols1 = []
    cluster_sols2 = []
    for i in range(1, n_max_clusters + 1):
        cluster_sol1 = AgglomerativeClustering(
            affinity=affinity, n_clusters=i, linkage=linkages[0]
        ).fit_predict(embs_1)
        cluster_sol2 = AgglomerativeClustering(
            affinity=affinity, n_clusters=i, linkage=linkages[1]
        ).fit_predict(embs_2)

        cluster_sols1.append(cluster_sol1)
        cluster_sols2.append(cluster_sol2)

        # print("Clusters", i)
        # print("Cluster sol 1", dict(zip(list(embs_1.index), cluster_sol1)))
        # print("Cluster sol 2", dict(zip(list(embs_1.index), cluster_sol2)))
        # print(" ")

    for i in range(n_max_clusters):
        for j in range(n_max_clusters):
            cc_score[i, j] = compute_cc_score(
                cluster_sols1[i], cluster_sols2[j], score=score
            )

    for i in tqdm(range(b)):
        perm_cc_score = np.zeros([n_max_clusters, n_max_clusters])
        for j in range(n_max_clusters):
            for k in range(n_max_clusters):
                cluster_sol1_perm = np.random.permutation(cluster_sols1[j])
                cluster_sol2_perm = np.random.permutation(cluster_sols2[k])
                perm_cc_score[j, k] = compute_cc_score(
                    cluster_sol1_perm, cluster_sol2_perm, score=score
                )
        perm_cc_scores.append(perm_cc_score)
    perm_cc_scores = np.array(perm_cc_scores)
    return {
        "cc_score": cc_score,
        "perm_cc_scores": perm_cc_scores,
        "pval": np.mean(perm_cc_scores > (cc_score - 1e-16), axis=0),
    }


def plot_geneset_embs(embs, geneset, title, figsize=[12, 6]):
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.scatterplot(
        data=embs.loc[embs.label.isin(geneset)],
        x="umap_0",
        y="umap_1",
        hue="label",
        ax=ax,
        s=8,
        alpha=0.7,
    )
    ax.scatter(
        np.array(embs.loc[~embs.label.isin(geneset), "umap_0"]),
        np.array(embs.loc[~embs.label.isin(geneset), "umap_1"]),
        c="tab:gray",
        alpha=0.03,
        label="REST",
        s=4,
    )
    ax.set_title(title)
    plt.legend()
    plt.show()
    plt.close()


def plot_struct_embs_cv(
    latents,
    random_state=1234,
    folds=["fold_0", "fold_1", "fold_2", "fold_3"],
    type_col="fold",
    label_col="labels",
    normalize_all=False,
):
    color_dict = {
        "fold_0": "tab:blue",
        "fold_1": "tab:green",
        "fold_2": "tab:red",
        "fold_3": "tab:orange",
    }

    tmp = latents.copy()
    latents = tmp.loc[:, ~latents.columns.isin([label_col, type_col])]
    labels = tmp.loc[:, label_col]
    types = tmp.loc[:, type_col]

    if normalize_all:
        zs = StandardScaler().fit_transform(latents)
    else:
        for t in folds:
            latents_fold = latents.loc[types == t]
            # ctrl_latents_fold = latents_fold.loc[labels == "EMPTY"]
            latents.loc[types == t] = StandardScaler().fit_transform(
                latents.loc[types == t]
            )
        latents = latents.loc[types.isin(folds)]
        labels = labels.loc[latents.index]
        types = types.loc[latents.index]
        zs = np.array(latents)

    embs = UMAP(random_state=random_state).fit_transform(zs)

    for target in tqdm(np.unique(labels)):
        fig, ax = plt.subplots(figsize=[10, 8])
        for t in folds:
            e = embs[types == t, :]
            l = labels[types == t]
            ax.scatter(
                e[l == target, 0],
                e[l == target, 1],
                c=color_dict[t],
                alpha=0.7,
                label="{}_{}".format(t.upper(), target.upper()),
                s=4,
            )
        ax.scatter(
            embs[labels != target, 0],
            embs[labels != target, 1],
            c="tab:gray",
            alpha=0.03,
            label="REST",
            s=4,
        )
        ax.set_xlabel("umap_0")
        ax.set_ylabel("umap_1")
        ax.set_title("Embeddings for {} vs REST".format(target.upper()))
        plt.legend()
        plt.show()
        plt.close()

    embs = pd.DataFrame(embs, columns=["umap_0", "umap_1"], index=latents.index)
    embs["label"] = labels

    return embs


def add_cluster_membership(model, latents, label_col):
    latents = latents.copy()
    latents["cluster"] = ""
    labels = latents.loc[:, label_col]
    for label in tqdm(np.unique(labels), desc="Add cluster memberships"):
        target_latents_idc = latents.loc[latents.loc[:, label_col] == label].index
        target_cluster_labels = model.fit_predict(
            latents.loc[target_latents_idc]._get_numeric_data()
        )
        target_cluster_labels = [
            "{}_{}".format(label, tcl) for tcl in target_cluster_labels
        ]
        latents.loc[target_latents_idc, "cluster"] = np.array(target_cluster_labels)
    return latents


def plot_clustertree(latents, title, metric="euclidean",method="average", figsize=[20, 4], text_size=10):
    plt.figure(figsize=figsize)
    corr_condensed = pdist(latents, metric=metric)
    z = hc.linkage(corr_condensed, method=method)
    dendrogram = hc.dendrogram(z, labels=latents.index, leaf_font_size=text_size)
    plt.title(title)
    plt.show()


def get_perm_test_results(
    fold_latents, node_embs, targets, score="mi", linkages=["average", "average"], b=200
):
    test_results = {}
    for i in range(len(fold_latents)):
        test_result = compute_cc_scores_perm_test(
            node_embs.loc[targets],
            fold_latents[i].loc[targets],
            score=score,
            linkages=linkages,
            b=b,
        )
        for k, v in test_result.items():
            if k not in test_results:
                test_results[k] = [v]
            else:
                test_results[k].append(v)
    return test_results


def plot_clustering(latents, label_col="labels", score="silhouette", random_state=1234):
    labels = sorted(np.unique(latents.loc[:, label_col]))
    model = KMeans(random_state=random_state)
    for label in tqdm(labels):
        print(label)
        visualizer = KElbowVisualizer(
            model, k=10, metric=score, timings=False, locate_elbow=True
        )
        target_latents = latents.loc[
            latents.loc[:, label_col] == label
        ]._get_numeric_data()
        visualizer.fit(target_latents)
        ax = visualizer.show()
