# Reproducing the paper's figures

## Installation

### Clone the repository
First, clone our repository via
```
git clone https://github.com/uhlerlab/image2reg.git
cd image2reg
```
This also sets the working directory to the cloned repository. All further steps assume the working directory is set to ``image2reg`` and all file paths given relative to the ``image2reg`` directory.

### Create the conda environment
> [!Note]
> This protocol was tested and the respective complexity estimates are given using our installation for a system with a GPU.
> It is assumed that conda is installed on the system used to run the code and the shell is configured to a bash shell.

The required conda environment is created and all required packages are installed via
```
conda create --name image2reg python==3.8.10
conda activate image2reg
bash scripts/installation/setup_environment_cuda.sh
```

The above script will install the environment assuming that a GPU is present.
If you do not have a GPU, please instead install the environment via
```
conda create --name image2reg python==3.8.10
conda activate image2reg
bash scripts/installation/setup_environment_cpu.sh
```

## Download our data repository

We have deposited the data used by our analyses and required to reproduce the figures under the DOI [10.16907/c8d5790d-4f6a-47ef-8d8a-f7a712df8dfc](https://doi.org/10.16907/c8d5790d-4f6a-47ef-8d8a-f7a712df8dfc). Please follow the instructions on the website to download the data set. The data set contains of three individual tar archives (``experiments.tar``, ``resources.tar`` and ``screen.tar``). In total the data set is about 1 TB in size. 

Once the archives are download please navigate to the location of these tar files and unpack them via
```
tar -xvf experiments.tar
tar -xvf resources.tar
tar -xvf screen.tar
```

This will create a directory ``home/paysan_d/PycharmProjects/image2reg/data`` located in the location where the tar files are located. Please move this directory to the ``image2reg`` directory, i.e. such that it the new directory ``image2reg/data``.

Finally, please download the zipped directory ``extract_latents_from_rohban_trained.zip`` from the DOI [10.5281/zenodo.8414736](https://doi.org/10.5281/zenodo.8414736).
Navigate to the directory that contains the downloaded file and unzip it.
Replace the content of the directory ``image2reg/data/experiments/jump/images/embedding/extract_latents_from_rohban_trained`` with the unzipped directory.

## Generating the paper's figures

The table below lists all notebooks used to generate the figures of our manuscript.
To reproduce a specific figure, please start the jupyter server in the conda environment via
```
conda activate image2reg
jupyter notebook
```

Navigate to the respective notebook (see below) and run all cells to reproduce the figure.

### Figure 1
- The panels A and B were created without the use of the code.
### Figure 2
- Panel A and B were created without the use of the code.
- Panel C was created using the notebook ``notebooks/rohban/image/screen/screen_analyses_cv_final.ipynb``
- Panel D was created using the notebook ``notebooks/rohban/image/embedding/image_embeddings_analysis.ipynb``
- Panel E was created using the notebook ``notebooks/rohban/image/embedding/gene_perturbation_cluster_analysis.ipynb``
### Figure 3
- Panel A was created using the notebook ``notebooks/rohban/ppi/preprocesssing/inference_preparation_full_pruning.ipynb`` and visualizing the inferred Prize Collecting Steiner tree that is derived as an output of the notebook ``notebooks/rohban/ppi/inference/interactome_inference_final.ipynb`` and is saved as the file ``spearman_sol_cv.graphml`` via opening it and visualizing it using [Cytoscape](https://cytoscape.org/).
- Panel B is created using the notebook ``notebooks/rohban/ppi/embeddings/gene_embedding_clustering.ipynb``.
- Panel C is created using the R notebook ``notebooks/rohban/ppi/embeddings/gene_embedding_cluster_analyses.Rmd``; a compiled version is available as the file ``notebooks/rohban/ppi/embeddings/gene_embedding_cluster_analyses.html``.
### Figure 4
- Panel A was created without the use of the code.
- Panel B was created using the notebook ``notebooks/rohban/translation/mapping/translational_mapping_loto_gridsearch_final.ipynb``
- Panel C was created using the notebook ``notebooks/jump/translation/jump_translation_prediction_final.ipynb``




