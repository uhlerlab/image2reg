from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import rff
import os
import torch
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import solve
import traceback
import sys
import argparse

def get_prediction_for_model(
    X_train,
    Y_train,
    x_test,
    model_type="ntk",
    model_params_dict=None,
    model=None,
    use_y_train_mean=False,
    n_neighbors_weights_features=0,
    n_weight_encodings=10,
    neighbor_weight_encoding=None,
    append_neighbor_embs=None,
    rff_sigma=1,
    normalize_prediction=False,
):

    # Get potentially transformed input and target features
    X_train, Y_train, x_test = get_input_and_target_features(
        X_train=X_train,
        Y_train=Y_train,
        x_test=x_test,
        use_y_train_mean=use_y_train_mean,
        n_neighbors_weights_features=n_neighbors_weights_features,
        n_weight_encodings=n_weight_encodings,
        neighbor_weight_encoding=neighbor_weight_encoding,
        append_neighbor_embs=append_neighbor_embs,
        rff_sigma=rff_sigma,
    )


    # Initialize model
    if model_type == "ntk":
        if model_params_dict is not None:
            model = NTK(**model_params_dict)
        else:
            model = NTK()
    else:
        if model is not None:
            model = model.fit(X_train, Y_train)
        else:
            raise NotImplementedError

    # Fit model and predict
    model = model.fit(X_train, Y_train)
    y_test = model.predict(x_test)

    if normalize_prediction:
        y_test = normalize(y_test)
    return y_test

def get_input_and_target_features(
    X_train,
    Y_train,
    x_test,
    use_y_train_mean=False,
    n_neighbors_weights_features=0,
    n_weight_encodings=10,
    neighbor_weight_encoding="positional",
    append_neighbor_embs=None,
    rff_sigma=1,
    cache_dir=None,
):
    pert_train_samples = list(X_train.index)
    test_target = list(x_test.index)[0]
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    x_test = np.array(x_test).reshape(1, -1)

    if use_y_train_mean:
        Y_train_mean = np.repeat(
            Y_train.mean(axis=0).reshape(1, -1), len(X_train), axis=0
        )
        X_train = np.concatenate([X_train, Y_train_mean], axis=-1)
        x_test = np.concatenate([x_test, Y_train_mean[0].reshape(1, -1)], axis=-1)

    ## Use Random Fourier Features of nearest neighbors of the training image embeddings
    if n_neighbors_weights_features > 0:
        nn = NearestNeighbors(n_neighbors=n_neighbors_weights_features)
        nn = nn.fit(X_train)
        X_train_and_test = np.concatenate([X_train, x_test], axis=0)
        neighbors_idcs = nn.kneighbors(X_train_and_test, return_distance=False)


        neighbor_weights = []
        for i in range(len(X_train_and_test)):
            neighbors_idc = neighbors_idcs[i]
            neighbors = X_train[neighbors_idc]
            neighbor_weight = np.matmul(
                    X_train_and_test[i], np.linalg.pinv(neighbors)
                )
            neighbor_weights.append(neighbor_weight)
        neighbors_weights = np.array(neighbor_weights)

        neighbor_weights = torch.FloatTensor(neighbor_weights)
        if neighbor_weight_encoding == "positional":
            encoding = rff.layers.PositionalEncoding(
                sigma=rff_sigma, m=n_weight_encodings
            )
        elif neighbor_weight_encoding == "gauss":
            encoding = rff.layers.GaussianEncoding(
                sigma=rff_sigma,
                input_size=neighbor_weights.size(-1),
                encoded_size=n_weight_encodings,
            )
        neighbor_weights = encoding(neighbor_weights).cpu().detach().numpy()

        X_train = np.concatenate([X_train, neighbor_weights[:-1]], axis=-1)
        x_test = np.concatenate([x_test, neighbor_weights[-1].reshape(1, -1)], axis=-1)

        # If desired append embeddings of the nearest neighbors (imaging or regulatory) as additional features
        if append_neighbor_embs is not None:
            neighbor_embs = []
            for i in range(len(X_train_and_test)):
                neighbors_idc = neighbors_idcs[i]
                if append_neighbor_embs == "images":
                    neighbors = X_train[neighbors_idc]
                elif append_neighbor_embs == "genes":
                    neighbors = Y_train[neighbors_idc]
                else:
                    raise NotImplementedError
                neighbor_embs.append(neighbors.reshape(-1))
            neighbor_embs = np.array(neighbor_embs)
            X_train = np.concatenate([X_train, neighbor_embs[:-1]], axis=-1)
            x_test = np.concatenate([x_test, neighbor_embs[-1].reshape(1, -1)], axis=-1)
    return X_train, Y_train, x_test

def get_nn_predictions(
    regulatory_embs,
    gene_perturbation_embs,
    target,
    model_type="ntk",
    model_params_dict=None,
    use_y_train_mean=False,
    n_neighbors_weights_features=0,
    n_weight_encodings=10,
    neighbor_weight_encoding=None,
    append_neighbor_embs=None,
    rff_sigma=1,
    model=None,
    permute=False,
    debug=False,
    return_predicted_embeddings=False,
    metric="euclidean",
):


    # Randomly permute both embeddings to compute a random baseline
    if permute:
        regulatory_embs = pd.DataFrame(
                np.random.permutation(np.array(regulatory_embs)),
                index=regulatory_embs.index,
                columns=regulatory_embs.columns,
            )
        gene_perturbation_embs = pd.DataFrame(
                np.random.permutation(np.array(gene_perturbation_embs)),
                index=gene_perturbation_embs.index,
                columns=gene_perturbation_embs.columns,
            )

   

    # Estimate the NearestNeighbor graph of the regulatory gene embeddings using euclidean distances
    reg_nn = NearestNeighbors(n_neighbors=len(regulatory_embs), metric=metric)
    reg_samples = np.array(list(regulatory_embs.index))
    reg_nn.fit(np.array(regulatory_embs))

    # Extract perturbation gene names
    pert_samples = np.array(list(gene_perturbation_embs.index))

    # Identify the test and train samples as the perturbation gene embeddings of the hold-out gene target
    # and all others respectively
    x_test = gene_perturbation_embs.loc[pert_samples == target]
    
     # Identify genes with both corresponding perturbation and regulatory gene embeddings
    shared_targets = set(regulatory_embs.index).intersection(gene_perturbation_embs.index)
    # Focus on the perturbation gene embeddings with corresponding regulatory gene embeddings
    gene_perturbation_embs = gene_perturbation_embs.loc[list(shared_targets)]
    
    
    pert_samples = np.array(list(gene_perturbation_embs.index))
    X_train = gene_perturbation_embs.loc[pert_samples != target]

    # Get corresponding regulatory gene embeddings of training samples

    X_train = X_train.sort_index()
    Y_train = regulatory_embs.loc[list(X_train.index)]

        # Get the predicted regulatory gene embedding for the hold-out gene target
    y_test = get_prediction_for_model(
            X_train=X_train,
            Y_train=Y_train,
            x_test=x_test,
            model_type=model_type,
            model_params_dict=model_params_dict,
            use_y_train_mean=use_y_train_mean,
            n_neighbors_weights_features=n_neighbors_weights_features,
            n_weight_encodings=n_weight_encodings,
            neighbor_weight_encoding=neighbor_weight_encoding,
            append_neighbor_embs=append_neighbor_embs,
            rff_sigma=rff_sigma,
            model=model,
        )
    regulatory_emb_prediction = y_test
    # Identify the sorted nearest neighbor list of the prediction among all regulatory gene embeddings
    regulatory_nns = reg_samples[reg_nn.kneighbors(y_test, return_distance=False)][0]

    # Return a dictionary with the key identifying the hold-out gene target and the corresponding value the
    # names of the regulatory gene embeddings closest to the prediction in increasing euclidean distance
    if not return_predicted_embeddings:
        return regulatory_nns
    else:
        return regulatory_nns, regulatory_emb_prediction


class NTK:
    def __init__(self, reg=1):
        super().__init__()
        self.reg = reg
        self.sol = None
        self.Xtrain = None

    def kernel(self, pair1, pair2):

        out = pair1 @ pair2.T + 1
        N1 = np.sum(np.power(pair1, 2), axis=-1).reshape(-1, 1) + 1
        N2 = np.sum(np.power(pair2, 2), axis=-1).reshape(-1, 1) + 1

        XX = np.sqrt(N1 @ N2.T)
        out = out / XX

        out = np.clip(out, -1, 1)

        first = (
            1
            / np.pi
            * (out * (np.pi - np.arccos(out)) + np.sqrt(1.0 - np.power(out, 2)))
            * XX
        )
        sec = 1 / np.pi * out * (np.pi - np.arccos(out)) * XX
        out = first + sec

        C = 1
        return out / C

    def fit(self, Xtrain, ytrain):
        K = self.kernel(Xtrain, Xtrain)
        sol = solve(K + self.reg * np.eye(len(K)), ytrain).T
        self.sol = sol
        self.Xtrain = Xtrain
        return self

    def predict(self, X):
        K = self.kernel(self.Xtrain, X)
        return (self.sol @ K).T
	

def run_translation_demo(embedding_dir, permute=False):
	target = os.path.split(embedding_dir)[0]
	target = os.path.split(target)[1]
	
	print("Read in image embeddings...")
	print(" ")
	train_pert_embs = pd.read_hdf(os.path.join(embedding_dir, "train_image_embeddings.h5"))
	test_pert_embs = pd.read_hdf(os.path.join(embedding_dir, "test_latents.h5"))
	test_pert_embs["labels"] = target.upper()
								 
	print("Successfully read in the image embeddings of {} training conditions and the one test condition, which is {}.".format(train_pert_embs.labels.nunique(), target))
	print("---"*30)

	pert_embs = pd.concat([train_pert_embs, test_pert_embs])
	pert_embs = pert_embs.loc[pert_embs.labels != "EMPTY"]
	
	print("Obtain gene perturbation embeddings for the training and test conditions as the mean of the corresponding image embeddings")
	mean_pert_embs = pert_embs.groupby("labels").mean()
	print("---"*30)
	
	print("Read in regulatory gene embeddings...")
	regulatory_embs = pd.read_csv(os.path.join(embedding_dir, "regulatory_embeddings.csv"), index_col=0)
	regulatory_embs = pd.DataFrame(normalize(regulatory_embs), index=regulatory_embs.index, columns=regulatory_embs.columns)
	print("")
	
	print("Successfully read in the regulatory embeddings of {} genes".format(len(regulatory_embs)))
	print("---"*30)
	
	print("Predict gene targeted in the held-out test condition via linking the gene perturbation and the regulatory gene embeddings...")
	
	if permute:
		print("")
		print("---"*30)
		print("Permute option was selected to assess the performance of a random baseline the regulatory and gene perturbation embeddings are randomly permuted before the kernel regression model is fit to link the two embeddings.")
		print("---"*30)
		  
	nn_preds = get_nn_predictions(
    regulatory_embs,
    mean_pert_embs,
    model_type="ntk",
    model_params_dict={"reg":1},
    use_y_train_mean=False,
    n_neighbors_weights_features=0,
    n_weight_encodings=5,
    neighbor_weight_encoding="positional",
    append_neighbor_embs="images",
    rff_sigma=10,
    target=target, 
	permute=permute)
	
	hit_idx = np.nan
	for i in range(len(nn_preds)):
		if nn_preds[i] == target:
			hit_idx = i+1
			break
	
	print(" ")
	print("The 10 genes closest to the predicted regulatory gene embeddings for the held-out condition ({}) are:".format(target))
	print(list(nn_preds[:10]))
	print("")
	print("The true overexpression target gene {} is the {} (out of {}) closest gene from the prediction.".format(target, hit_idx, len(regulatory_embs)))
	print("---"*30)
	print("")
	

	

if __name__ == "__main__":
	sys.path.append("..")
	
	arg_parser = argparse.ArgumentParser(description="Translation Demo")
	arg_parser.add_argument("--embedding_dir", metavar="EMBEDDING_DIR", help="Directory containing the regulatory embeddings (regulatory_embeddings.csv) and the image embeddings of the train condition (train_image_embeddings.h5) as well as the derived image embeddings of the held-out test condition (test_latents.h5)", required=True)
	arg_parser.add_argument("--random_mode", help="Enables permutation of the regulatory and gene perturbation embeddings which corresponds to a random prediction pipeline.", dest="random_mode", action="store_true", required=False)
	
	args = arg_parser.parse_args()
	if os.path.isdir(args.embedding_dir):
		embedding_dir = os.path.abspath(args.embedding_dir)
	else:
		raise RuntimeError("There is no directory with the name {}.".format(args.embedding_dir))
	if args.random_mode:
		  permute = True
	else:
		  permute = False
	try:
		run_translation_demo(embedding_dir, permute=permute)
	except:
		traceback.print_exc()
		sys.exit(1)
		
	sys.exit()
	
	
