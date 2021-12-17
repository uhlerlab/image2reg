import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import Node2Vec, InnerProductDecoder, GAE
from tqdm import tqdm
import seaborn as sns
import torch_geometric.transforms as T

from src.models.ae import FeatureDecoder, CustomGAE, GCNEncoder
from src.utils.torch.general import get_device
from src.utils.torch.network import train_n2v_model, train_gae


def get_gae_latents_for_seed(
        graph_data,
        seeds,
        input_dim,
        node_feature_key,
        link_pred=False,
        reconstruct_features=False,
        feature_decoder_params=None,
        feat_loss=None,
        alpha=1,
        beta=1,
        latent_dim=32,
        hidden_dim=128,
        lr=1e-3,
        n_epochs=100,
        early_stopping=50,
        plot_loss=False,
):
    latents_dict = {}
    for seed in seeds:

        # Ensure reproducibility
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        # No train-val-test split
        if not link_pred:
            modified_graph_data = graph_data
            modified_graph_data.pos_edge_label_index = modified_graph_data.edge_index

            data_dict = {
                "train": modified_graph_data,
                "val": modified_graph_data,
                "test": modified_graph_data,
            }
        else:
            random_link_splitter = T.RandomLinkSplit(
                is_undirected=True,
                add_negative_train_samples=False,
                num_val=0.1,
                num_test=0.2,
                split_labels=True,
            )
            train_link_data, val_link_data, test_link_data = random_link_splitter(
                graph_data
            )
            data_dict = {
                "train": train_link_data,
                "val": val_link_data,
                "test": test_link_data,
            }
        if reconstruct_features:
            feat_decoder = FeatureDecoder(**feature_decoder_params)
            gae = CustomGAE(
                encoder=GCNEncoder(
                    in_channels=input_dim,
                    hidden_dim=hidden_dim,
                    out_channels=latent_dim,
                ),
                adj_decoder=InnerProductDecoder(),
                feat_decoder=feat_decoder,
                alpha=alpha,
                beta=beta,
                feat_loss=feat_loss,
            )
            optimizer = torch.optim.Adam(gae.parameters(), lr=lr)
            gae, loss_hist = train_gae(
                model=gae,
                data_dict=data_dict,
                node_feature_key=node_feature_key,
                optimizer=optimizer,
                n_epochs=n_epochs,
                early_stopping=early_stopping,
                link_pred=link_pred,
                reconstruct_features=reconstruct_features,
            )

        else:
            gae = GAE(
                GCNEncoder(
                    in_channels=input_dim,
                    hidden_dim=hidden_dim,
                    out_channels=latent_dim,
                    random_state=seed,
                )
            )
            optimizer = torch.optim.Adam(gae.parameters(), lr=lr)
            gae, loss_hist = train_gae(
                model=gae,
                data_dict=data_dict,
                node_feature_key=node_feature_key,
                optimizer=optimizer,
                n_epochs=n_epochs,
                early_stopping=early_stopping,
                link_pred=link_pred,
            )

        gae.eval()
        inputs = getattr(graph_data, node_feature_key).float()
        latents = gae.encode(inputs, graph_data.edge_index)
        latents = latents.detach().cpu().numpy()

        latents_dict[seed] = latents

        if plot_loss:
            fig, ax = plt.subplots(figsize=[6, 4])
            ax.plot(np.arange(1, len(loss_hist["train"]) + 1), loss_hist["train"])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Minimal loss: {:.4f}".format(np.min(loss_hist["train"])))
            plt.show()

    return latents_dict


def get_n2v_latents_for_seed(
        graph_data,
        seeds,
        latent_dim=64,
        walk_length=30,
        context_size=10,
        walks_per_node=50,
        batch_size=128,
        num_workers=10,
        lr=0.01,
        n_epochs=100,
        plot_loss=False,
        device=None,
):
    if device is None:
        device = get_device()
        # device = torch.device("cpu")

    latents_dict = {}
    for seed in seeds:
        # Change initialization
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        n2v_model = Node2Vec(
            graph_data.edge_index,
            embedding_dim=latent_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=1,
            p=1,
            q=1,
            sparse=True,
        ).to(device)
        n2v_model.device = device

        n2v_loader = n2v_model.loader(
            batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        n2v_optimizer = torch.optim.SparseAdam(list(n2v_model.parameters()), lr=lr)

        fitted_n2v_model, loss_hist = train_n2v_model(
            model=n2v_model,
            optimizer=n2v_optimizer,
            loader=n2v_loader,
            n_epochs=n_epochs,
        )

        fitted_n2v_model.eval()
        latents = (
            fitted_n2v_model(torch.arange(graph_data.num_nodes, device=device))
                .cpu()
                .detach()
                .numpy()
        )

        latents_dict[seed] = latents

        if plot_loss:
            fig, ax = plt.subplots(figsize=[6, 4])
            ax.plot(np.arange(1, n_epochs + 1), loss_hist)
            fig.suptitle("Loss during training")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            plt.show()
    return latents_dict


def stability_cocluster_screen(latents_dict, affinity="euclidean", linkage="average"):
    ami_matrices = []
    ks = latents_dict.keys()
    latents = list(latents_dict.values())
    for i in range(len(latents)):
        for j in tqdm(range(len(latents))):
            ami_matrices.append(
                compute_ami_matrix(
                    latents[i], latents[j], affinity=affinity, linkage=linkage
                )
            )
    return ami_matrices


def compute_ami_matrix(latents_1, latents_2, affinity="euclidean", linkage="average"):
    ami = np.zeros([15, 15])
    if isinstance(affinity, str):
        affinity_1 = affinity_2 = affinity
    else:
        affinity_1, affinity_2 = affinity[0], affinity[1]
    if isinstance(linkage, str):
        linkage_1 = linkage_2 = linkage
    else:
        linkage_1, linkage_2 = linkage[0], linkage[1]

    for i in range(0, 15):
        cluster_sol1 = AgglomerativeClustering(
            affinity=affinity_1, n_clusters=i + 1, linkage=linkage_1
        ).fit_predict(latents_1)
        #             cluster_sol1 = KMeans(random_state=0, n_clusters=i+1).fit_predict(latents_2)
        for j in range(0, 15):
            #                 cluster_sol2 = KMeans(random_state=0, n_clusters=j+1).fit_predict(latents_2)
            cluster_sol2 = AgglomerativeClustering(
                affinity=affinity_2, n_clusters=j + 1, linkage=linkage_2
            ).fit_predict(latents_2)

            ami[i, j] = adjusted_mutual_info_score(cluster_sol1, cluster_sol2)
    return ami


def plot_amis_matrices(names, amis, figsize=[30, 30]):
    fig, ax = plt.subplots(figsize=figsize, ncols=len(names), nrows=len(names))
    ax = ax.flatten()
    for i in range(len(names)):
        for j in range(len(names)):
            if amis[i] is not None:
                ax[j + i * len(names)] = sns.heatmap(
                    amis[i * len(names) + j],
                    ax=ax[i * len(names) + j],
                    vmin=0,
                    vmax=1,
                    cbar=(j == len(names) - 1),
                    cmap="seismic",
                )
                ax[j + i * len(names)].set_title(
                    "max AMI: {:.2f}".format(np.max(amis[i * len(names) + j][1:, 1:])),
                )
            if i == j:
                ax[i * len(names) + j].set_title(
                    "Model: {}".format(names[j]), weight="bold", c="red"
                )
            ax[j + i * len(names)].set_xticklabels([k + 1 for k in range(len(amis[i * len(names) + j]))])
            ax[j + i * len(names)].set_yticklabels([k + 1 for k in range(len(amis[i * len(names) + j]))])
    plt.show()
