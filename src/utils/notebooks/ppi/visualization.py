import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd


def plot_degree_dist(graph, figsize=[8, 6], title="", smoothing=1):
    degree_freq = nx.degree_histogram(graph)
    degrees = range(len(degree_freq))
    node_degrees = [graph.degree(n) for n in graph.nodes()]
    plt.figure(figsize=figsize)
    plt.loglog(degrees[::smoothing], degree_freq[::smoothing], "o-")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title(title)
    return node_degrees


def plot_degree_dist_for_nodes(
    graph, nodes, figsize=[8, 6], title="", smoothing=1, bins=50
):
    node_names = []
    node_degrees = []
    for node in graph.nodes():
        if node in nodes:
            node_degrees.append(graph.degree(node))
            node_names.append(node)
    plt.figure(figsize=figsize)
    plt.hist(node_degrees, bins=bins)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title(title)
    return node_degrees, node_names


def plot_eigenvector_centrality_dist(graph, figsize=[8, 6], title="", bins=50):
    eigen_cent = nx.eigenvector_centrality(graph)
    sns.displot(
        np.log10(list(eigen_cent.values())),
        height=figsize[1],
        aspect=figsize[0] / figsize[1],
        kind="hist",
        bins=bins,
    )
    plt.xlabel("Eigenvector centrality (log10)")
    plt.title(title)


def plot_eigenvector_centrality_dist_nodes(
    graph, nodes, figsize=[8, 6], title="", bins=50
):
    eigen_cent = []
    for node in nodes:
        if node in graph.nodes():
            eigen_cent.append(nx.eigenvector_centrality(graph, u=node))
    sns.displot(
        np.log10(list(eigen_cent)),
        height=figsize[1],
        aspect=figsize[0] / figsize[1],
        kind="hist",
        bins=bins,
    )
    plt.xlabel("Eigenvector centrality (log10)")
    plt.title(title)


def plot_closeness_centrality_dist_nodes(
    graph, nodes, figsize=[8, 6], title="", bins=50
):
    close_cent = []
    node_names = []
    for node in nodes:
        if node in graph.nodes():
            close_cent.append(nx.closeness_centrality(graph, u=node))
            node_names.append(node)
    sns.displot(
        list(close_cent),
        height=figsize[1],
        aspect=figsize[0] / figsize[1],
        kind="hist",
        bins=bins,
    )
    plt.xlabel("Closeness centrality")
    plt.title(title)
    return pd.DataFrame(close_cent, columns=["closeness_centrality"], index=node_names)


def plot_average_shortest_path_length_dist_nodes(
    graph, nodes, figsize=[8, 6], title="", bins=50
):
    avg_spls = []
    node_names = []
    for source_node in nodes:
        if source_node in graph.nodes():
            spls = []
            for target_node in nodes:
                if target_node in graph.nodes() and source_node != target_node:
                    spls.append(
                        nx.shortest_path_length(
                            graph, source=source_node, target=target_node
                        )
                    )
            avg_spls.append(np.mean(spls))
            node_names.append(source_node)
    sns.displot(
        list(avg_spls),
        height=figsize[1],
        aspect=figsize[0] / figsize[1],
        kind="hist",
        bins=bins,
    )
    plt.xlabel("Average closest path length between targets")
    plt.title(title)
    return pd.DataFrame(
        avg_spls, columns=["avg_closest_path_to_target"], index=node_names
    )


def plot_edge_cost_hist(graph, figsize=[8, 6], title="", bins=50):
    edges = graph.edges(data=True)
    costs = [e[-1]["cost"] for e in edges]
    g = sns.displot(
        costs,
        height=figsize[1],
        aspect=figsize[0] / figsize[1],
        kind="hist",
        bins=bins,
    )
    plt.yscale("log")
    plt.xlabel("Edge costs")
    plt.title(title)
    return costs


def plot_edge_weight_dist_nodes(graph, nodes, figsize=[8, 6], title="", bins=50):
    edges = graph.edges(data=True)
    edge_costs = []
    for edge in edges:
        source = edge[0]
        target = edge[1]
        cost = edge[-1]["cost"]
        if source in nodes or target in nodes:
            edge_costs.append(cost)
    g = sns.displot(
        edge_costs,
        height=figsize[1],
        aspect=figsize[0] / figsize[1],
        kind="hist",
        bins=bins,
    )
    plt.yscale("log")
    plt.xlabel("Edge costs")
    plt.title(title)
    return edge_costs


def plot_prize_array(prize_array, figsize=[8, 6], title="", bins=50):
    g = sns.displot(
        prize_array,
        height=figsize[1],
        aspect=figsize[0] / figsize[1],
        kind="hist",
        bins=bins,
    )
    plt.xlabel("Node prize")
    plt.title(title)
