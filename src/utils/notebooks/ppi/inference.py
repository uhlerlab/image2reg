import community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from pcst_fast import pcst_fast
from tqdm import tqdm


def run_pcst_sensitivity_analyses(
    graph,
    bs,
    prize_key: str = "prize",
    weight_key: str = "cost",
    minmax_scale: bool = False,
):
    node_dict = dict(zip(list(graph.nodes()), list(range(len(graph.nodes())))))
    inv_edge_dict = dict(zip(list(range(len(graph.edges()))), list(graph.edges())))

    vertices = list(node_dict.values())
    edges = []
    prizes = []
    costs = []
    for node in graph.nodes(data=True):
        prizes.append(node[-1][prize_key])
    for edge in graph.edges(data=True):
        edges.append((node_dict[edge[0]], node_dict[edge[1]]))
        costs.append(edge[-1][weight_key])

    edges = np.array(edges)
    prizes = np.array(prizes)
    if minmax_scale:
        prizes = (prizes - prizes.min()) / (prizes.max() - prizes.min())
    costs = np.array(costs)

    pcs_tree_dict = {}
    augmented_pcs_tree_dict = {}

    for b in tqdm(bs, desc="Compute PCSTs"):
        v_idc, e_idc = pcst_fast(edges, prizes * b, costs, -1, 1, "strong", 0)
        selected_edges = [inv_edge_dict[e_idx] for e_idx in e_idc]
        # print(selected_edges)
        pcs_tree = graph.edge_subgraph(selected_edges)
        augmented_pcs_tree = graph.subgraph(pcs_tree.nodes())
        pcst_name = graph.name + "_b_{}".format(str(np.round(b, 3)))
        pcs_tree_dict[pcst_name] = pcs_tree
        augmented_pcs_tree_dict["augmented_" + pcst_name] = augmented_pcs_tree
    return pcs_tree_dict, augmented_pcs_tree_dict


def analyze_pcst_sensitivity_analyses_results(trees_dict, target_nodes, spec_targets):
    data = {
        "beta": [],
        "n_nodes": [],
        "n_edges": [],
        "n_connected_components": [],
        "n_louvain_clusters": [],
        "avg_node_degree": [],
        "std_node_degree": [],
        "n_leaf_nodes": [],
        "n_target_nodes": [],
        "n_spec_target_nodes": [],
        "n_spec_target_leafs": [],
        "avg_spec_target_degree": [],
        "std_spec_target_degree": [],
    }
    keys = []
    for key, tree in tqdm(trees_dict.items(), desc="Analyze tree"):
        keys.append(key)
        splitted = key.split("_")
        beta = splitted[-1]

        n_nodes = len(tree.nodes())
        n_edges = len(tree.edges())
        n_connected_components = nx.number_connected_components(tree)
        n_louvain_clusters = len(
            np.unique(list(community.best_partition(tree).values()))
        )
        node_degrees = []
        spec_target_degrees = []
        leaf_nodes = []
        for node in tree.nodes():
            node_degree = tree.degree(node)
            node_degrees.append(node_degree)
            if node_degree == 1:
                leaf_nodes.append(node)
            if node in spec_targets:
                spec_target_degrees.append(node_degree)
        avg_node_degree = np.mean(node_degrees)
        std_node_degree = np.std(node_degrees)
        n_leaf_nodes = len(leaf_nodes)
        n_targets = len(set(target_nodes).intersection(set(list(tree.nodes()))))
        n_spec_targets = len(set(spec_targets).intersection(set(list(tree.nodes()))))
        n_spec_target_leafs = len(set(spec_targets).intersection(set(leaf_nodes)))
        avg_spec_target_degree = np.mean(spec_target_degrees)
        std_spec_target_degree = np.std(spec_target_degrees)

        data["beta"].append(float(beta))
        data["n_nodes"].append(n_nodes)
        data["n_edges"].append(n_edges)
        data["n_connected_components"].append(n_connected_components)
        data["n_louvain_clusters"].append(n_louvain_clusters)
        data["avg_node_degree"].append(avg_node_degree)
        data["std_node_degree"].append(std_node_degree)
        data["n_leaf_nodes"].append(n_leaf_nodes)
        data["n_target_nodes"].append(n_targets)
        data["n_spec_target_nodes"].append(n_spec_targets)
        data["n_spec_target_leafs"].append(n_spec_target_leafs)
        data["avg_spec_target_degree"].append(avg_spec_target_degree)
        data["std_spec_target_degree"].append(std_spec_target_degree)

    data = pd.DataFrame.from_dict(data)
    data.index = keys
    return data


def plot_node_ious(pcst_dict):
    fig, ax = plt.subplots(figsize=[6, 4])
    n = len(pcst_dict)
    node_ious = np.zeros([n, n])
    trees = list(pcst_dict.values())
    for i in tqdm(range(n - 1)):
        for j in range(i + 1, n):
            nodes_i = set(list(trees[i].nodes()))
            nodes_j = set(list(trees[j].nodes()))

            if len(nodes_i.union(nodes_j)) == 0:
                node_ious[i, j] = 0
            else:
                node_ious[i, j] = len(nodes_i.intersection(nodes_j)) / len(
                    nodes_i.union(nodes_j)
                )
    node_ious = node_ious + np.transpose(node_ious) + np.diag(np.ones(n))
    ax = sns.heatmap(node_ious, ax=ax, cmap="viridis")
    ax.set_xlabel("trees")
    ax.set_ylabel("trees")
    plt.title("IoU of nodes in the tree solutions")
    plt.show()
    plt.close()


def plot_solution_node_ious(
    sol1, sol2, xlabel="Networks 1", ylabel="Networks 2", figsize=[12, 8]
):
    fig, ax = plt.subplots(figsize=figsize)
    n = len(sol1)
    m = len(sol2)
    node_ious = np.zeros([n, m])
    sol1_networks = list(sol1.values())
    sol2_networks = list(sol2.values())
    for i in tqdm(range(n)):
        for j in range(1, m):
            nodes_i = set(list(sol1_networks[i].nodes()))
            nodes_j = set(list(sol2_networks[j].nodes()))

            if len(nodes_i.union(nodes_j)) == 0:
                node_ious[i, j] = 0
            else:
                node_ious[i, j] = len(nodes_i.intersection(nodes_j)) / len(
                    nodes_i.union(nodes_j)
                )
    ax = sns.heatmap(node_ious, ax=ax, cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title("IoU of nodes in the network solutions")
    return fig, ax, node_ious


def plot_solution_edge_ious(
    sol1, sol2, xlabel="Networks 1", ylabel="Networks 2", figsize=[12, 8]
):
    fig, ax = plt.subplots(figsize=figsize)
    n = len(sol1)
    m = len(sol2)
    edges_ious = np.zeros([n, m])
    sol1_networks = list(sol1.values())
    sol2_networks = list(sol2.values())
    for i in tqdm(range(n)):
        for j in range(1, m):
            edges_i = set(list(sol1_networks[i].edges()))
            edges_j = set(list(sol2_networks[j].edges()))

            if len(edges_i.union(edges_j)) == 0:
                edges_ious[i, j] = 0
            else:
                edges_ious[i, j] = len(edges_i.intersection(edges_j)) / len(
                    edges_i.union(edges_j)
                )
    ax = sns.heatmap(edges_ious, ax=ax, cmap="viridis")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title("IoU of nodes in the network solutions")
    return fig, ax, edges_ious


def summarize_analyses_results_visually(results, figsize=[12, 10]):
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=figsize)
    ax = ax.flatten()
    ax[0] = sns.lineplot(data=results, x="beta", y="n_nodes", ax=ax[0])
    ax[1] = sns.lineplot(data=results, x="beta", y="n_edges", ax=ax[1])
    ax[2] = sns.lineplot(data=results, x="beta", y="avg_node_degree", ax=ax[2])
    ax[3] = sns.lineplot(data=results, x="beta", y="n_leaf_nodes", ax=ax[3])
    ax[4] = sns.lineplot(data=results, x="beta", y="n_target_nodes", ax=ax[4])
    ax[5] = sns.lineplot(data=results, x="beta", y="n_spec_target_nodes", ax=ax[5])

    plt.show()
    plt.close()
