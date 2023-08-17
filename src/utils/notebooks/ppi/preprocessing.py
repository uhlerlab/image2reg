import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, spearmanr, pearsonr
from sklearn.metrics import mutual_info_score
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm


def find_markers(data, target, avg_log_fc=0.25, min_pct=0.1):
    target_data = data.loc[target]
    other_data = data.drop([target], axis=0)
    results = {
        "log_expr_target": [],
        "avg_log_expr_other": [],
        "log_fc": [],
        "abs_log_fc": [],
        "pval": [],
    }
    genes = list(data.columns)
    selected_genes = []
    for gene in genes:
        log_other_expr = np.array(other_data.loc[:, gene])
        avg_log_expr_other = log_other_expr.mean()
        log_expr_target = target_data.loc[gene]
        log_fc = log_expr_target - avg_log_expr_other
        if (
            avg_log_fc < np.abs(log_fc)
            and np.mean(log_other_expr == 0) <= (1 - min_pct)
            and np.mean(log_expr_target == 0) <= (1 - min_pct)
        ):
            results["log_expr_target"].append(log_expr_target)
            results["avg_log_expr_other"].append(avg_log_expr_other)
            results["log_fc"].append(log_fc)
            results["abs_log_fc"].append(np.abs(log_fc))
            results["pval"].append(ttest_1samp(log_other_expr, log_expr_target)[1])
            selected_genes.append(gene)
    results["fdr"] = fdrcorrection(results["pval"])[1]
    results = pd.DataFrame.from_dict(results)
    results.index = selected_genes
    return results


def search_for_confidence_cutoff(ppi, targets):
    n_edges = []
    avg_degrees = []
    n_nodes = []
    n_targets = []
    cutoffs = list(np.arange(0, 1, 0.001))
    ppi_edge_list = nx.to_pandas_edgelist(ppi)
    for cutoff in tqdm(cutoffs):
        graph = nx.from_pandas_edgelist(
            ppi_edge_list.loc[ppi_edge_list["cost"] < cutoff, :],
        )
        n_edges.append(len(graph.edges()))
        n_nodes.append(len(graph.nodes()))
        n_targets.append(len((targets).intersection(set(list((graph.nodes()))))))
        degrees = [val for (node, val) in graph.degree()]
        if len(graph.nodes()) > 0:
            avg_degrees.append(sum(degrees) / len(graph.nodes()))
        else:
            avg_degrees.append(0)
    edge_cost_overview = pd.DataFrame.from_dict(
        {
            "cutoff": cutoffs,
            "n_nodes": n_nodes,
            "n_targets": n_targets,
            "n_edges": n_edges,
            "avg_degree": avg_degrees,
        }
    )
    return edge_cost_overview


def compute_bootstrap_p(x, y, metric, b=1000, random_state=1234):
    np.random.seed(random_state)
    bootstrap_metrics = []
    if metric == "pearson":
        sample_metric = pearsonr(x, y)[0]
    elif metric == "spearman":
        sample_metric = spearmanr(x, y)[0]
    else:
        raise NotImplementedError("Unknown metric provided: {}".format(metric))
    for i in range(b):
        x_boot = np.random.choice(x, size=len(x), replace=True)
        y_boot = np.random.choice(y, size=len(y), replace=True)
        if metric == "pearson":
            bootstrap_metric = pearsonr(x_boot, y_boot)[0]
        elif metric == "spearman":
            bootstrap_metric = spearmanr(x_boot, y_boot)[0]
        else:
            raise NotImplementedError("Unknown metric provided: {}".format(metric))
        bootstrap_metrics.append(bootstrap_metric)
    p_boot = (1 / b) * np.sum(bootstrap_metrics <= sample_metric)
    return p_boot


def compute_mi_score(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def compute_edge_weights(
    ppi,
    data,
    metrics=["pearsonr", "spearmanr", "mi", "pearsonp", "spearmanp"],
    b=1000,
    random_state=1234,
    attr_name=None,
):
    for metric in metrics:
        if attr_name is None:
            edge_attribute_name = metric
        else:
            edge_attribute_name = "_".join([attr_name, metric])
        for (u, v) in tqdm(
            ppi.edges(), desc="Compute edge weights for {}".format(metric)
        ):
            x = np.array(data.loc[:, u])
            y = np.array(data.loc[:, v])
            if metric == "pearsonr":
                association = pearsonr(x, y)[0]
                cost = 1 - np.abs(association)
            elif metric == "spearmanr":
                association = spearmanr(x, y)[0]
                cost = 1 - np.abs(association)
            elif metric == "mi":
                cost = compute_mi_score(x, y, 100)
            elif metric == "pearsonp":
                # association = compute_bootstrap_p(
                #     x, y, metric="pearson", b=b, random_state=random_state
                # )
                association = pearsonr(x, y)[1]
                cost = association
            elif metric == "spearmanp":
                # association = compute_bootstrap_p(
                #     x, y, metric="spearman", b=b, random_state=random_state
                # )
                association = spearmanr(x, y)[1]
                cost = association
            else:
                raise NotImplementedError("Unknown metric provided: {}".format(metric))
            ppi.edges[u, v][edge_attribute_name] = cost
    return ppi


def run_hub_node_cutoff_analyses(
    ppi, cutoffs, targets, keep_targets=True, specific_targets=None
):
    degree_dict = dict(ppi.degree())
    nodes = np.array(list(degree_dict.keys()))
    degrees = np.array(list(degree_dict.values()))

    results = {
        "cutoff": [],
        "n_connected_components": [],
        "n_nodes_largest_comp": [],
        "n_targets_largest_comp": [],
        "n_edges_largest_comp": [],
        "avg_degree_largest_comp": [],
    }
    if specific_targets is not None:
        results["n_specific_targets_largest_comp"] = []
    for cutoff in tqdm(cutoffs, desc="Screen cutoffs"):
        results["cutoff"].append(cutoff)
        selected_nodes = set(list(nodes[degrees <= np.quantile(degrees, cutoff)]))
        if keep_targets:
            selected_nodes = selected_nodes.union(
                targets.intersection(set(list(nodes)))
            )
        s_ppi = ppi.subgraph(selected_nodes)
        ccomps = [s_ppi.subgraph(c).copy() for c in nx.connected_components(s_ppi)]
        largest_comp = max(ccomps, key=len)
        results["n_connected_components"].append(len(ccomps))
        results["n_nodes_largest_comp"].append(len(largest_comp.nodes()))
        results["n_edges_largest_comp"].append(len(largest_comp.edges()))
        results["avg_degree_largest_comp"].append(
            np.mean(list(dict(largest_comp.degree()).values()))
        )
        results["n_targets_largest_comp"].append(
            len(targets.intersection(largest_comp.nodes()))
        )
        if specific_targets is not None:
            results["n_specific_targets_largest_comp"].append(
                len(specific_targets.intersection(largest_comp.nodes()))
            )
    return pd.DataFrame(results)
