import numpy as np
import pandas as pd
import networkx as nx
from networkx import NetworkXNoPath
from tqdm import tqdm
import logging as logger
import community
from pcst_fast import pcst_fast

from networkx.algorithms.approximation import steinertree


def get_node_id_dict(graph):
    node_id_dict = dict(zip(list(range(len(graph.nodes()))), list(graph.nodes())))
    return node_id_dict


def get_edge_array(graph, node_id_dict):
    inv_node_id_dict = {v: k for k, v in node_id_dict.items()}
    edge_array = []
    for edge in graph.edges(data=True):
        edge_array.append(
            [inv_node_id_dict[edge[0]], inv_node_id_dict[edge[1]], edge[-1]["cost"]]
        )
    return np.array(edge_array)


def compute_node_prize(graph, node, terminals, alpha, k=None):
    if k is None:
        k = np.infty
    min_shortest_path_length = np.infty
    for terminal in terminals:
        if node == terminal:
            terminal_shortest_path_length = 0
        else:
            try:
                terminal_shortest_path_length = nx.shortest_path_length(
                    graph, source=node, target=terminal
                )
            except NetworkXNoPath:
                terminal_shortest_path_length = np.infty
        min_shortest_path_length = min(
            min_shortest_path_length, terminal_shortest_path_length
        )
    if min_shortest_path_length > k:
        node_prize = 0
    else:
        node_prize = alpha ** (-min_shortest_path_length)
    return node_prize


def get_prize_array(graph, terminals, node_id_dict, alpha, beta, k=None):
    inv_node_id_dict = {v: k for k, v in node_id_dict.items()}
    if k is None:
        k = np.infty
    prize_array = np.zeros(len(inv_node_id_dict))
    for node in tqdm(graph.nodes(), desc="Compute node prizes"):
        prize_array[inv_node_id_dict[node]] = compute_node_prize(
            graph, node, terminals, alpha, k
        )
    return prize_array


def get_prizes_df(graph, terminals, alpha, k=None):
    prizes = []
    node_names = []
    if k is None:
        kstr = "None"
    else:
        kstr = str(k)
    for node in tqdm(
        graph.interactome_graph.nodes(),
        desc="Compute prizes for (alpha,k)=({},{})".format(alpha, kstr),
    ):
        node_names.append(node)
        prizes.append(
            compute_node_prize(graph.interactome_graph, node, terminals, alpha, k)
        )
    prizes_df = pd.DataFrame(
        np.column_stack([node_names, prizes]), columns=["name", "prize"]
    )
    prizes_df["prize"] = pd.to_numeric(prizes_df["prize"])
    return prizes_df


def output_tree_as_networkx(graph, vertex_indices, edge_indices):
    """
    Adapted from https://fraenkel-lab.github.io/OmicsIntegrator2/html/_modules/graph.html
    """

    if len(vertex_indices) == 0:
        logger.warning("The resulting Tree is empty. Try different parameters.")
        return nx.empty_graph(0), nx.empty_graph(0)

    # Replace the edge indices with the actual edges (protein1 name, protein2 name) by indexing into the interactome
    edges = graph.interactome_dataframe.loc[edge_indices]
    tree = nx.from_pandas_edgelist(edges, "protein1", "protein2", edge_attr=True)

    # Set all the attributes on graph
    nx.set_node_attributes(
        tree,
        graph.node_attributes.reindex(list(tree.nodes()))
        .dropna(how="all")
        .to_dict(orient="index"),
    )
    # Set a flag on all the edges which were selected by PCST (before augmenting the tree)
    nx.set_edge_attributes(tree, True, name="in_solution")

    # Create a new graph including all edges between all selected nodes, not just those edges selected by PCST.
    augmented_tree = nx.compose(graph.interactome_graph.subgraph(tree.nodes()), tree)

    # Post-processing
    # betweenness(augmented_tree)
    # louvain_clustering(augmented_tree)
    # annotate_graph_nodes(augmented_tree)

    return tree, augmented_tree


def compute_minimum_terminal_distance(graph, terminals):

    result = {}
    for node in graph.nodes():
        min_shortest_path_length = np.infty
        for terminal in terminals:
            if node == terminal:
                terminal_shortest_path_length = 0
            else:
                try:
                    terminal_shortest_path_length = nx.shortest_path_length(
                        graph, source=node, target=terminal
                    )
                except NetworkXNoPath:
                    terminal_shortest_path_length = np.infty
            min_shortest_path_length = min(
                min_shortest_path_length, terminal_shortest_path_length
            )
        result[node] = min_shortest_path_length
    return result


def get_prizes_df_from_dist_df(graph, min_terminals_dist_dict, alpha, k):
    prizes = []
    node_names = []
    if k is None:
        kstr = "None"
        k = np.infty
    else:
        kstr = str(k)
    for node in tqdm(
        graph.interactome_graph.nodes(),
        desc="Compute prizes for (alpha,k)=({},{})".format(alpha, kstr),
    ):
        node_names.append(node)
        if min_terminals_dist_dict[node] > k:
            prize = 0
        else:
            prize = alpha ** (-min_terminals_dist_dict[node])
        prizes.append(prize)
    prizes_df = pd.DataFrame(
        np.column_stack([node_names, prizes]), columns=["name", "prize"]
    )
    prizes_df["prize"] = pd.to_numeric(prizes_df["prize"])
    return prizes_df


def run_pcst_sensitivity_analyses(graph, terminals, alphas, betas, ks=None):
    hyperparameter_grid = []
    if ks is None:
        ks = [None]
    trees_dict = {}
    augmented_trees_dict = {}
    edges = graph.edges
    costs = graph.costs
    min_terminals_dist_dict = compute_minimum_terminal_distance(
        graph.interactome_graph, terminals
    )
    for alpha in alphas:
        for k in ks:
            prizes_df = get_prizes_df_from_dist_df(
                graph, min_terminals_dist_dict, alpha, k
            )
            graph.params.b = 1
            graph._prepare_prizes(prizes_df)
            graph.node_attributes.loc[terminals, "terminals"] = True
            graph.terminals = np.where(graph.node_attributes["terminal"] == True)[0]
            prizes = graph.prizes
            for beta in betas:
                prizes = beta * prizes
                vertex_indices, edge_indices = pcst_fast(
                    edges, prizes, costs, -1, 1, "strong", 0
                )
                tree, augmented_tree = output_tree_as_networkx(
                    graph, vertex_indices, edge_indices
                )
                if k is None:
                    kstr = "None"
                else:
                    kstr = str(k)
                tree_id = "tree_alpha_{}_beta_{}_k_{}".format(alpha, beta, kstr)
                trees_dict[tree_id] = tree
                augmented_trees_dict[tree_id] = augmented_tree
                hyperparameter_grid.append(np.array([alpha, beta, k]))

    return trees_dict, augmented_trees_dict, np.array(hyperparameter_grid), graph


def run_pcsf_sensitivity_analyses(
    graph, graph_params, terminals, ws, alphas, betas, ks=None
):
    if ks is None:
        ks = [None]
    trees_dict = {}
    hyperparameters_grid = []
    augmented_trees_dict = {}
    min_terminals_dist_dict = compute_minimum_terminal_distance(
        graph.interactome_graph, terminals
    )
    for alpha in alphas:
        for k in ks:
            prizes_df = get_prizes_df_from_dist_df(
                graph, min_terminals_dist_dict, alpha, k
            )
            for w in ws:
                for beta in betas:
                    graph_params["b"] = beta
                    graph_params["w"] = w
                    graph._reset_hyperparameters(graph_params)
                    graph._prepare_prizes(prizes_df)
                    graph.node_attributes.loc[terminals, "terminals"] = True
                    graph.terminals = np.where(
                        graph.node_attributes["terminal"] == True
                    )[0]
                    # graph.prizes = np.array(prizes_df.loc[:, "prize"])
                    vertex_indices, edge_indices = graph.pcsf()
                    tree, augmented_tree = output_tree_as_networkx(
                        graph, vertex_indices, edge_indices
                    )
                    if k is None:
                        kstr = "None"
                    else:
                        kstr = str(k)
                    tree_id = "tree_alpha_{}_beta_{}_k_{}_w_{}".format(
                        alpha, beta, kstr, w
                    )
                    trees_dict[tree_id] = tree
                    augmented_trees_dict[tree_id] = augmented_tree
                    hyperparameters_grid.append([alpha, beta, k, w])
    return trees_dict, augmented_trees_dict, np.array(hyperparameters_grid), graph


def run_st_analyses(
    interactome, terminals, weight_key="cost",
):
    steiner_tree = steinertree.steiner_tree(
        interactome, terminal_nodes=terminals, weight=weight_key
    )

    augmented_steiner_tree = nx.compose(
        interactome.subgraph(steiner_tree.nodes()), steiner_tree
    )
    return steiner_tree, augmented_steiner_tree


def knn_expansion(graph, steiner_tree, terminals, k=1):
    steiner_tree_edges = list(steiner_tree.edges(data=True))
    extended_edges = []
    for node in steiner_tree.nodes():
        if steiner_tree.degree(node) == 1 and node in terminals:
            leaf_neighbor_connections = graph.edges(node, data=True)
            edge_cost_dict = {}
            for i, edge in enumerate(leaf_neighbor_connections):
                if edge not in steiner_tree.edges():
                    edge_cost_dict[i] = edge[-1]["cost"]
            sorted_edge_cost_dict = {
                k: v
                for k, v in sorted(edge_cost_dict.items(), key=lambda item: item[1])
            }
            for i in range(min(k, len(sorted_edge_cost_dict))):
                extended_edges.append(
                    list(leaf_neighbor_connections)[
                        list(sorted_edge_cost_dict.keys())[i]
                    ]
                )
    extended_edges = extended_edges + steiner_tree_edges
    all_edges = []
    for edge in extended_edges:
        all_edges.append((edge[0], edge[1]))
    steiner_tree = graph.edge_subgraph(all_edges)
    return steiner_tree


def get_pcst_expansion_edge_list(graph, root, k=1):
    node_dict = dict(zip(list(graph.nodes()), list(range(len(graph.nodes())))))
    inv_edge_dict = dict(zip(list(range(len(graph.edges()))), list(graph.edges())))

    edges = []
    prizes = []
    costs = []
    for node in graph.nodes(data=True):
        prizes.append(node[-1]["prize"])
    for edge in graph.edges(data=True):
        edges.append((node_dict[edge[0]], node_dict[edge[1]]))
        costs.append(edge[-1]["cost"])
    prizes = np.array(prizes) * k
    root_idx = list(node_dict.keys()).index(root)
    v_idc, e_idc = pcst_fast(edges, prizes, costs, root_idx, 1, "strong", 0)
    selected_edges = [inv_edge_dict[e_idx] for e_idx in e_idc]
    return selected_edges


def pcst_expansion(graph, steiner_tree, k=1):
    st_nodes = list(steiner_tree.nodes())
    expanded_st_edge_list = list(steiner_tree.edges())
    for node in tqdm(st_nodes, desc="Expand nodes using PCST"):
        if steiner_tree.degree(node) == 1:
            selected_nodes = list(set(graph.nodes()) - set(st_nodes)) + [node]
            selected_subgraph = graph.subgraph(selected_nodes)
            expanded_st_edge_list += get_pcst_expansion_edge_list(
                selected_subgraph, root=node, k=k
            )
    expanded_steiner_tree = graph.edge_subgraph(expanded_st_edge_list)
    return expanded_steiner_tree


def expand_st_solution(
    interactome_graph, steiner_tree, terminals, expansion_mode=None, k=1
):
    if expansion_mode is None:
        pass
    elif expansion_mode == "knn":
        steiner_tree = knn_expansion(
            graph=interactome_graph, steiner_tree=steiner_tree, terminals=terminals, k=k
        )
    elif expansion_mode == "pcst":
        steiner_tree = pcst_expansion(interactome_graph, steiner_tree=steiner_tree, k=k)

    augmented_steiner_tree = nx.compose(
        interactome_graph.subgraph(steiner_tree.nodes()), steiner_tree
    )

    return steiner_tree, augmented_steiner_tree


def run_st_sensitivity_analyses(
    interactome_graph,
    steiner_tree,
    terminals,
    expansion_modes,
    alphas=[None],
    ks=[None],
):
    tree_dict = {}
    augmented_tree_dict = {}
    for expansion_mode in expansion_modes:
        for k in ks:
            for alpha in alphas:
                tree, augmented_tree = expand_st_solution(
                    interactome_graph, steiner_tree, terminals, expansion_mode, k
                )
                tree_id = "tree_expansion_{}_alpha_{}_k_{}".format(
                    expansion_mode, alpha, k
                )
                tree_dict[tree_id] = tree
                augmented_tree_dict[tree_id] = augmented_tree
    return tree_dict, augmented_tree_dict


def analyze_pcst_sensitivity_analyses_results(trees_dict, target_nodes):
    data = {
        "alpha": [],
        "beta": [],
        "k": [],
        "n_nodes": [],
        "n_edges": [],
        "n_connected_components": [],
        "n_louvain_clusters": [],
        "avg_node_degree": [],
        "std_node_degree": [],
        "n_leaf_nodes": [],
        "n_target_nodes": [],
        "n_target_leafs": [],
        "avg_target_degree": [],
        "std_target_degree": [],
    }
    keys = []
    for key, tree in tqdm(trees_dict.items(), desc="Analyze tree:"):
        keys.append(key)
        splitted = key.split("_")
        alpha = splitted[2]
        beta = splitted[4]
        k = splitted[6]

        n_nodes = len(tree.nodes())
        n_edges = len(tree.edges())
        n_connected_components = nx.number_connected_components(tree)
        n_louvain_clusters = len(
            np.unique(list(community.best_partition(tree).values()))
        )
        node_degrees = []
        target_degrees = []
        leaf_nodes = []
        for node in tree.nodes():
            node_degree = tree.degree(node)
            node_degrees.append(node_degree)
            if node_degree == 1:
                leaf_nodes.append(node)
            if node in target_nodes:
                target_degrees.append(node_degree)
        avg_node_degree = np.mean(node_degrees)
        std_node_degree = np.std(node_degrees)
        avg_target_degree = np.mean(target_degrees)
        std_target_degree = np.std(target_degrees)
        n_target_leafs = len(set(target_nodes).intersection(set(leaf_nodes)))
        n_leaf_nodes = len(leaf_nodes)
        n_target_nodes = len(set(target_nodes).intersection(set(list(tree.nodes()))))

        data["alpha"].append(float(alpha))
        data["beta"].append(float(beta))
        if k == "None":
            k = None
        else:
            k = float(k)
        data["k"].append(k)
        data["n_nodes"].append(n_nodes)
        data["n_edges"].append(n_edges)
        data["n_connected_components"].append(n_connected_components)
        data["n_louvain_clusters"].append(n_louvain_clusters)
        data["avg_node_degree"].append(avg_node_degree)
        data["std_node_degree"].append(std_node_degree)
        data["n_leaf_nodes"].append(n_leaf_nodes)
        data["n_target_nodes"].append(n_target_nodes)
        data["n_target_leafs"].append(n_target_leafs)
        data["avg_target_degree"].append(avg_target_degree)
        data["std_target_degree"].append(std_target_degree)

    data = pd.DataFrame.from_dict(data)
    data.index = keys
    return data


def analyze_pcsf_sensitivity_analyses_results(trees_dict, target_nodes):
    data = {
        "alpha": [],
        "beta": [],
        "k": [],
        "w": [],
        "n_nodes": [],
        "n_edges": [],
        "n_connected_components": [],
        "n_louvain_clusters": [],
        "avg_node_degree": [],
        "std_node_degree": [],
        "n_leaf_nodes": [],
        "n_target_nodes": [],
        "n_target_leafs": [],
        "avg_target_degree": [],
        "std_target_degree": [],
    }
    keys = []
    for key, tree in tqdm(trees_dict.items(), desc="Analyze tree:"):
        keys.append(key)
        splitted = key.split("_")
        alpha = splitted[2]
        beta = splitted[4]
        k = splitted[6]
        w = splitted[8]

        n_nodes = len(tree.nodes())
        n_edges = len(tree.edges())
        n_connected_components = nx.number_connected_components(tree)
        n_louvain_clusters = len(
            np.unique(list(community.best_partition(tree).values()))
        )
        node_degrees = []
        target_degrees = []
        leaf_nodes = []
        for node in tree.nodes():
            node_degree = tree.degree(node)
            node_degrees.append(node_degree)
            if node_degree == 1:
                leaf_nodes.append(node)
            if node in target_nodes:
                target_degrees.append(node_degree)
        avg_node_degree = np.mean(node_degrees)
        std_node_degree = np.std(node_degrees)
        avg_target_degree = np.mean(target_degrees)
        std_target_degree = np.std(target_degrees)
        n_target_leafs = len(set(target_nodes).intersection(set(leaf_nodes)))
        n_leaf_nodes = len(leaf_nodes)
        n_target_nodes = len(set(target_nodes).intersection(set(list(tree.nodes()))))

        data["alpha"].append(float(alpha))
        data["beta"].append(float(beta))
        if k == "None":
            k = None
        else:
            k = float(k)
        data["k"].append(k)
        data["w"].append(w)
        data["n_nodes"].append(n_nodes)
        data["n_edges"].append(n_edges)
        data["n_connected_components"].append(n_connected_components)
        data["n_louvain_clusters"].append(n_louvain_clusters)
        data["avg_node_degree"].append(avg_node_degree)
        data["std_node_degree"].append(std_node_degree)
        data["n_leaf_nodes"].append(n_leaf_nodes)
        data["n_target_nodes"].append(n_target_nodes)
        data["n_target_leafs"].append(n_target_leafs)
        data["avg_target_degree"].append(avg_target_degree)
        data["std_target_degree"].append(std_target_degree)

    data = pd.DataFrame.from_dict(data)
    data.index = keys
    return data


def analyze_st_sensitivity_analyses_results(trees_dict, target_nodes):
    data = {
        "expansion": [],
        "alpha": [],
        "k": [],
        "n_nodes": [],
        "n_edges": [],
        "n_connected_components": [],
        "n_louvain_clusters": [],
        "avg_node_degree": [],
        "std_node_degree": [],
        "n_leaf_nodes": [],
        "n_target_nodes": [],
        "n_target_leafs": [],
        "avg_target_degree": [],
        "std_target_degree": [],
    }
    keys = []
    for key, tree in tqdm(trees_dict.items(), desc="Analyze tree:"):
        keys.append(key)
        splitted = key.split("_")
        expansion = splitted[2]
        alpha = splitted[4]
        if alpha == "None":
            alpha = np.nan
        k = splitted[6]
        if k == None:
            k = np.nan
        n_nodes = len(tree.nodes())
        n_edges = len(tree.edges())
        n_connected_components = nx.number_connected_components(tree)
        n_louvain_clusters = len(
            np.unique(list(community.best_partition(tree).values()))
        )
        node_degrees = []
        target_degrees = []
        leaf_nodes = []
        for node in tree.nodes():
            node_degree = tree.degree(node)
            node_degrees.append(node_degree)
            if node_degree == 1:
                leaf_nodes.append(node)
            if node in target_nodes:
                target_degrees.append(node_degree)
        avg_node_degree = np.mean(node_degrees)
        std_node_degree = np.std(node_degrees)
        avg_target_degree = np.mean(target_degrees)
        std_target_degree = np.std(target_degrees)
        n_target_leafs = len(set(target_nodes).intersection(set(leaf_nodes)))
        n_leaf_nodes = len(leaf_nodes)
        n_target_nodes = len(set(target_nodes).intersection(set(list(tree.nodes()))))

        data["expansion"].append(expansion)
        data["alpha"].append(alpha)
        data["k"].append(k)
        data["n_nodes"].append(n_nodes)
        data["n_edges"].append(n_edges)
        data["n_connected_components"].append(n_connected_components)
        data["n_louvain_clusters"].append(n_louvain_clusters)
        data["avg_node_degree"].append(avg_node_degree)
        data["std_node_degree"].append(std_node_degree)
        data["n_leaf_nodes"].append(n_leaf_nodes)
        data["n_target_nodes"].append(n_target_nodes)
        data["n_target_leafs"].append(n_target_leafs)
        data["avg_target_degree"].append(avg_target_degree)
        data["std_target_degree"].append(std_target_degree)

    data = pd.DataFrame.from_dict(data)
    data.index = keys
    return data
