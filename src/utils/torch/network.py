import copy
import logging

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.utils import negative_sampling
from tqdm import tqdm

from src.utils.torch.general import get_device


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
    gae,
    data,
    node_feature_key,
    mode,
    optimizer,
    edge_weight_key=None,
    feature_decoder=None,
    latent_classifier=None,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
):
    inputs = getattr(data, node_feature_key).float()
    if latent_classifier is not None:
        labels = data.label.long()
        if hasattr(data, "label_mask"):
            label_mask = data.label_mask.bool()
        else:
            label_mask = torch.ones_like(labels).bool()

    if edge_weight_key is not None:
        edge_weight = getattr(data, edge_weight_key).float()
    else:
        edge_weight = None

    if hasattr(data, "pos_edge_index"):
        pos_edge_index = data.pos_edge_index
    else:
        pos_edge_index = data.pos_edge_label_index

    if hasattr(data, "neg_edge_index"):
        neg_edge_index = data.neg_edge_index
    elif hasattr(data, "neg_edge_label_index"):
        neg_edge_index = data.neg_edge_label_index
    else:
        neg_edge_index = None

    if mode == "train":
        gae.train()
        optimizer.zero_grad()
        latents = gae.encode(inputs, data.edge_index, edge_weight=edge_weight)

        # Negative edges created via negative sampling
        gae_recon_loss = gae.recon_loss(
            latents, pos_edge_index=pos_edge_index, neg_edge_index=None
        )

        loss = alpha * gae_recon_loss

        if feature_decoder is not None:
            feat_recon_loss = feature_decoder.loss(inputs, latents)
            loss += beta * feat_recon_loss
        else:
            feat_recon_loss = None

        if latent_classifier is not None:
            class_loss = latent_classifier.loss(latents, label_mask, labels)
            loss += gamma * class_loss
        else:
            class_loss = None

        loss.backward()
        optimizer.step()
    else:
        gae.eval()

        if feature_decoder is not None:
            feature_decoder.eval()
        if latent_classifier is not None:
            latent_classifier.eval()

        with torch.no_grad():
            latents = gae.encode(inputs, data.edge_index, edge_weight=edge_weight)
            if hasattr(data, "node_mask"):
                inputs = inputs[data.node_mask]
                latents = latents[data.node_mask]

                if latent_classifier is not None:
                    label_mask = label_mask[data.node_mask]
                    labels = labels[data.node_mask]

            gae_recon_loss = gae.recon_loss(
                latents, pos_edge_index=pos_edge_index, neg_edge_index=neg_edge_index,
            )

            loss = alpha * gae_recon_loss
            if feature_decoder is not None:
                feat_recon_loss = feature_decoder.loss(inputs, latents)
                loss += beta * feat_recon_loss
            else:
                feat_recon_loss = None

            if latent_classifier is not None:
                class_loss = latent_classifier.loss(latents, label_mask, labels)
                loss += gamma * class_loss
            else:
                class_loss = None

            if mode in ["test"]:
                gae.eval()
                aucs = []
                aps = []
                if neg_edge_index is None:
                    for i in range(100):
                        neg_edge_index = negative_sampling(
                            pos_edge_index, latents.size(0)
                        )
                        auc, ap = gae.test(
                            latents,
                            pos_edge_index=pos_edge_index,
                            neg_edge_index=neg_edge_index,
                        )
                        aucs.append(auc)
                        aps.append(ap)
                    auc = np.array(aucs).mean()
                    ap = np.array(aps).mean()
                else:
                    auc, ap = gae.test(
                        latents,
                        pos_edge_index=pos_edge_index,
                        neg_edge_index=neg_edge_index,
                    )
                print(mode.upper(), "AUC: {}".format(auc), "AP: {}".format(ap))
    epoch_loss_hist = {
        "total_loss": loss.item(),
        "gae_recon_loss": alpha * gae_recon_loss.item(),
        "feat_recon_loss": beta * feat_recon_loss.item(),
        "class_loss": gamma * class_loss.item(),
        "mode": mode,
    }
    return epoch_loss_hist


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
    gae,
    data_dict,
    node_feature_key,
    optimizer,
    n_epochs=500,
    early_stopping=20,
    edge_weight_key=None,
    link_pred=False,
    feature_decoder=None,
    latent_classifier=None,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
):
    device = get_device()
    # device = torch.device("cpu")
    gae.to(device)
    gae.device = device
    if feature_decoder is not None:
        feature_decoder.to(device)
    if latent_classifier is not None:
        latent_classifier.to(device)
    # print("Using {}".format(device))
    best_val_loss = np.infty
    loss_hist = []
    es_counter = 0

    best_gae_model_weights = None
    best_feature_decoder_weights = None
    best_latent_clf_weights = None
    best_epoch = -1

    for i in tqdm(range(n_epochs)):
        if es_counter < early_stopping:
            for mode in ["train", "val"]:
                data = data_dict[mode].to(device)
                epoch_loss_hist = process_single_epoch_gae(
                    gae=gae,
                    data=data,
                    node_feature_key=node_feature_key,
                    mode=mode,
                    optimizer=optimizer,
                    edge_weight_key=edge_weight_key,
                    feature_decoder=feature_decoder,
                    latent_classifier=latent_classifier,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                )
                # print("{} loss:".format(mode.upper()), loss)
                loss_hist.append(pd.DataFrame(epoch_loss_hist, index=[i]))

                if mode == "val":
                    if epoch_loss_hist["total_loss"] < best_val_loss:
                        es_counter = 0
                        best_val_loss = epoch_loss_hist["total_loss"]
                        best_gae_model_weights = copy.deepcopy(gae.state_dict())
                        if feature_decoder is not None:
                            best_feature_decoder_weights = copy.deepcopy(
                                feature_decoder.state_dict()
                            )
                        if latent_classifier is not None:
                            best_latent_clf_weights = copy.deepcopy(
                                latent_classifier.state_dict()
                            )
                        best_epoch = i
                    else:
                        es_counter += 1
                    if link_pred:
                        auc, ap = test_link_pred(
                            model=gae,
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
    gae.load_state_dict(best_gae_model_weights)
    if feature_decoder is not None:
        feature_decoder.load_state_dict(best_feature_decoder_weights)
    if latent_classifier is not None:
        latent_classifier.load_state_dict(best_latent_clf_weights)
    data = data_dict["test"].to(device)
    test_loss = process_single_epoch_gae(
        gae=gae,
        data=data,
        node_feature_key=node_feature_key,
        mode="test",
        optimizer=optimizer,
        edge_weight_key=edge_weight_key,
        feature_decoder=feature_decoder,
        latent_classifier=latent_classifier,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    loss_hist = pd.concat(loss_hist)
    print(
        "TRAIN loss: {}".format(
            np.array(loss_hist.loc[loss_hist.loc[:, "mode"] == "train", "total_loss"])[
                best_epoch
            ]
        )
    )
    print(
        "VAL loss: {}".format(
            np.array(loss_hist.loc[loss_hist.loc[:, "mode"] == "val", "total_loss"])[
                best_epoch
            ]
        )
    )
    print("TEST loss: {}".format(test_loss["total_loss"]))
    if link_pred:
        auc, ap = test_link_pred(
            model=gae,
            data=data,
            node_feature_key=node_feature_key,
            edge_weight_key=edge_weight_key,
        )
        print("TEST AUC: {} \t AP: {}".format(auc, ap))
    loss_hist = pd.concat([loss_hist, pd.DataFrame(test_loss, index=[-1])])
    loss_hist["epoch"] = np.array(loss_hist.index)
    return gae, feature_decoder, latent_classifier, loss_hist


def add_pos_negative_edge_indices(graph_data, add_pos_edges=True):
    selected_nodes = torch.LongTensor(list(range(graph_data.num_nodes)))
    if hasattr(graph_data, "node_mask"):
        selected_nodes = selected_nodes[graph_data.node_mask]
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
