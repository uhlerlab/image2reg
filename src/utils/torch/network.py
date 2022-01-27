import logging

import networkx as nx
import torch
from sklearn.model_selection import train_test_split

from src.utils.torch.general import get_device
import numpy as np
from tqdm import tqdm


def process_single_epoch_n2v(model, optimizer, loader):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(model.device), neg_rw.to(model.device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_n2v_model(model, optimizer, loader, n_epochs=300):
    loss_hist = []
    for i in tqdm(range(n_epochs)):
        logging.debug("EPOCH {}/{}".format(i + 1, n_epochs))
        loss = process_single_epoch_n2v(model=model, optimizer=optimizer, loader=loader)
        logging.debug("TRAIN loss:", loss)
        logging.debug("---" * 30)
        loss_hist.append(loss)
    print("Final loss:", loss)
    return model, loss_hist


def process_single_epoch_gae(
    model,
    data,
    node_feature_key,
    mode,
    optimizer,
    edge_weight_key=None,
    reconstruct_features: bool = False,
):
    inputs = getattr(data, node_feature_key).float()
    if edge_weight_key is not None:
        edge_weight = getattr(data, edge_weight_key).float()
    else:
        edge_weight = None

    if hasattr(data, "pos_edge_index"):
        pos_edge_index = data.pos_edge_index
        neg_edge_index = data.neg_edge_index
    else:
        pos_edge_index = data.pos_edge_label_index
        neg_edge_index = None

    if mode == "train":
        model.train()
        optimizer.zero_grad()
        latents = model.encode(inputs, data.edge_index, edge_weight=edge_weight)
        # Negative edges created via negative sampling
        if reconstruct_features:
            loss = model.recon_loss(inputs, latents, pos_edge_index=pos_edge_index)
        else:
            loss = model.recon_loss(latents, pos_edge_index=pos_edge_index)
        loss.backward()
        optimizer.step()
    else:
        model.eval()

        with torch.no_grad():
            latents = model.encode(inputs, data.edge_index, edge_weight=edge_weight)
            if hasattr(data, "node_mask"):
                inputs = inputs[data.node_mask]
                latents = latents[data.node_mask]
            if reconstruct_features:
                loss = model.recon_loss(
                    inputs,
                    latents,
                    pos_edge_index=pos_edge_index,
                    neg_edge_index=neg_edge_index,
                )
            else:
                loss = model.recon_loss(
                    latents,
                    pos_edge_index=pos_edge_index,
                    neg_edge_index=neg_edge_index,
                )
    return loss.item()


def test_link_pred(model, data, node_feature_key, edge_weight_key=None):
    device = get_device()
    # device = torch.device("cpu")
    model.eval().to(device)
    inputs = getattr(data, node_feature_key).float()
    if edge_weight_key is not None:
        edge_weight = getattr(data, edge_weight_key).float()
    else:
        edge_weight = None

    latents = model.encode(inputs, data.edge_index, edge_weight=edge_weight,)
    auc, ap = model.test(latents, data.pos_edge_label_index, data.neg_edge_label_index,)
    return auc, ap


def train_gae(
    model,
    data_dict,
    node_feature_key,
    optimizer,
    n_epochs=500,
    early_stopping=20,
    edge_weight_key=None,
    link_pred=False,
    reconstruct_features: bool = False,
):
    device = get_device()
    # device = torch.device("cpu")
    model.to(device)
    model.device = device
    # print("Using {}".format(device))
    best_val_loss = np.infty
    loss_hist = {"train": [], "val": []}
    es_counter = 0

    best_model_weights = None
    best_epoch = -1

    for i in tqdm(range(n_epochs)):
        # print("---" * 20)
        # print("EPOCH {}/{}".format(i + 1, n_epochs))
        if es_counter < early_stopping:
            for mode in ["train", "val"]:
                data = data_dict[mode].to(device)
                loss = process_single_epoch_gae(
                    model=model,
                    data=data,
                    node_feature_key=node_feature_key,
                    mode=mode,
                    optimizer=optimizer,
                    edge_weight_key=edge_weight_key,
                    reconstruct_features=reconstruct_features,
                )
                # print("{} loss:".format(mode.upper()), loss)
                loss_hist[mode].append(loss)

                if mode == "val":
                    if loss < best_val_loss:
                        es_counter = 0
                        best_val_loss = loss
                        best_model_weights = model.state_dict()
                        best_epoch = i
                    else:
                        es_counter += 1
                    if link_pred:
                        auc, ap = test_link_pred(
                            model=model,
                            data=data,
                            node_feature_key=node_feature_key,
                            edge_weight_key=edge_weight_key,
                        )
                        logging.debug("VAL AUC: {} \t AP: {}".format(auc, ap))
        else:
            print("Training stopped after {} epochs".format(i + 1))
            print("Best model found at epoch {}".format(best_epoch))
            break

    print("---" * 20)
    model.load_state_dict(best_model_weights)
    data = data_dict["test"].to(device)
    test_loss = process_single_epoch_gae(
        model=model,
        data=data,
        node_feature_key=node_feature_key,
        mode="test",
        optimizer=optimizer,
        edge_weight_key=edge_weight_key,
        reconstruct_features=reconstruct_features,
    )
    print("TRAIN loss: {}".format(loss_hist["train"][best_epoch]))
    print("VAL loss: {}".format(loss_hist["val"][best_epoch]))
    print("TEST loss: {}".format(test_loss))
    if link_pred:
        auc, ap = test_link_pred(
            model=model,
            data=data,
            node_feature_key=node_feature_key,
            edge_weight_key=edge_weight_key,
        )
        print("TEST AUC: {} \t AP: {}".format(auc, ap))
    loss_hist["test"] = test_loss
    return model, loss_hist


def add_pos_negative_edge_indices(graph_data, add_pos_edges=True):
    selected_nodes = torch.LongTensor(list(range(graph_data.num_nodes)))[
        graph_data.node_mask
    ]
    adj = torch.zeros(graph_data.num_nodes, graph_data.num_nodes, dtype=torch.bool)
    adj[graph_data.edge_index[0], graph_data.edge_index[1]] = True
    adj = adj[selected_nodes]
    adj = adj[:, selected_nodes]
    pos_edge_index = adj.nonzero(as_tuple=False).t()
    neg_adj = torch.ones(len(adj), len(adj), dtype=torch.bool)
    neg_adj[pos_edge_index[0], pos_edge_index[1]] = False
    neg_edge_index = neg_adj.nonzero(as_tuple=False).t()
    if add_pos_edges:
        graph_data.pos_edge_index = torch.LongTensor(pos_edge_index)
    graph_data.neg_edge_index = torch.LongTensor(neg_edge_index)
    return graph_data


def network_train_val_test_split(network, train_val_test_size, random_state=1234):
    train_size, val_size, test_size = train_val_test_size
    nodes = list(network.nodes())
    train_val_nodes, test_nodes = train_test_split(
        nodes, test_size=test_size, random_state=random_state
    )
    train_nodes, val_nodes = train_test_split(
        train_val_nodes, test_size=val_size / (1 - test_size), random_state=random_state
    )

    train_network = nx.Graph(network.subgraph(train_nodes))
    for edge in train_network.edges(data=True):
        edge[-1]["edge_mask"] = True
    for node in train_network.nodes(data=True):
        node[-1]["node_mask"] = True
    train_network.name = "train_network"

    val_network = nx.Graph(network.subgraph(train_nodes + val_nodes))
    for edge in val_network.edges(data=True):
        edge[-1]["edge_mask"] = edge[0] in val_nodes and edge[1] in val_nodes
    for node in val_network.nodes(data=True):
        node[-1]["node_mask"] = node[0] in val_nodes
    val_network.name = "val_network"

    test_network = nx.Graph(network.subgraph(train_nodes + val_nodes + test_nodes))
    for edge in test_network.edges(data=True):
        edge[-1]["edge_mask"] = edge[0] in test_nodes and edge[1] in test_nodes
    for node in test_network.nodes(data=True):
        node[-1]["node_mask"] = node[0] in test_nodes
    test_network.name = "test_network"

    return train_network, val_network, test_network
