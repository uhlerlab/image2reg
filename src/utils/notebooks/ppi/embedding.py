import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score
from torch_geometric.nn import Node2Vec
from tqdm import tqdm
import seaborn as sns

from src.utils.torch.network import train_n2v_model


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
        # device = get_device()
        device = torch.device("cpu")

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
    for i in range(0, 15):
        cluster_sol1 = AgglomerativeClustering(
            affinity=affinity, n_clusters=i + 1, linkage=linkage
        ).fit_predict(latents_1)
        #             cluster_sol1 = KMeans(random_state=0, n_clusters=i+1).fit_predict(latents_2)
        for j in range(0, 15):
            #                 cluster_sol2 = KMeans(random_state=0, n_clusters=j+1).fit_predict(latents_2)
            cluster_sol2 = AgglomerativeClustering(
                affinity=affinity, n_clusters=j + 1, linkage=linkage
            ).fit_predict(latents_2)

            ami[i, j] = adjusted_mutual_info_score(cluster_sol1, cluster_sol2)
    return ami


def plot_amis_matrices(seeds, amis, figsize=[16, 16]):
    fig, ax = plt.subplots(figsize=figsize, ncols=len(seeds), nrows=len(seeds))
    ax = ax.flatten()
    j = 0
    for i in range(len(ax)):
        if amis[i] is not None:
            ax[i] = sns.heatmap(
                amis[i], ax=ax[i], vmin=0, vmax=1, cbar=((i + 1) % (len(seeds)) == 0)
            )
        else:
            ax[i].text(0.5, 0.5, seeds[j])
            j += 1
    plt.show()
