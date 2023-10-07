# Step-by-step Protocol for our Analyses

---

> [!WARNING]
> The total computation time of all analyses described in this protocol can exceed 1000 hours of computation time on a CPU (depending on the used infrastructure) and generates approximately 2 TB of data.

This step-by-step protocol details how all presented analyses can be reproduced from scratch. Given the time and space complexity of the analyses and the amount of additional software packages this requires that this is only run by experienced users. To quickly test our Image2Reg pipeline or use it to perform inference on your own data set, please refer to respective documentation described in our [ReadMe file]()


---


## Installation
- *Time: 1 minute*
- *Size: 10 GB*

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

# 

## Data retrieval

### Image data from [Rohban et al. (2017)]()

#### Downloading the data set
- *Time: 15 hours*
- *Size: 270 GB*

To download the data set from Rohban et al. (2017) the Aspera CLI needs to be installed on your system.
Follow the instructions provided [here](https://www.biostars.org/p/9528910/) to install it.
Verify that it is installed via typing in
```
ascp
```
If the command is found the installation was successful and you can download the dataset from Rohban et al via
```
bash scripts/data/download_rohban_data.sh
```

#### Preparing the data set
- *Time: 1 hour*
- *Size: 50 GB*

The Rohban data set contains a number of metadata information in form of SQL databases.
To create the databases and prepare the respective data for the consecutive analyses, make sure that you have ``mysql`` installed it.
If an ``mysql`` server is installed and and instance is running, you can prepare the data from the Rohban data set via

```
conda activate image2reg
bash scripts/data/prepare_rohban_data.sh
```

### Gene expression data

#### Download the CMap gene signatures
- *Time: 10 minutes*
- *Size: 3 GB*

Create the directory to store the CMap data via
```
mkdir -p data/resources/gex/cmap
```

Next, download the follwing files from [clue.io](https://clue.io/data/CMap2020#LINCS2020):

- cellinfo_beta.txt
- geneinfo_beta.txt
- siginfo_beta.txt
- Level5_beta_trt_oe_n34171x12328.gctx

and place them in the created ``cmap`` directory.


#### Download the scRNA-seq data from the GEO database
- *Time: 1 minute*
- *Size: 1 GB*

Create the directory to store the scRNA-seq data via
```
mkdir -p data/resources/gex/scrnaseq
```

Next, download the file [GSE146773_Counts.csv.gz](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE146773&format=file&file=GSE146773%5FCounts%2Ecsv%2Egz) and place it the created ``scrnaseq`` directory.
Unzip the file via
```
find . -name '*.csv.gz' -print0 | xargs -0 -n1 gzip -d
```

#### Download the bulk RNA-seq from the CCLE database
- *Time: 1 minute*
- *Size: 1 GB*

Create the directory to store the bulk RNA-seq data via
```
mkdir -p data/resources/gex/ccle
```

Next, download the following files from [DepMap](https://depmap.org/portal/download/all/) and thereby make sure to select the DepMap version 21Q2:
- CCLE_expression.csv
- sample_info.csv
- 
Place the two files in the created ``ccle`` directory.

Rename the sample_info.csv file to CCLE_expression_sample_info.csv for better association via
```
cd data/resources/gex/ccle
mv sample_info.csv CCLE_expression_sample_info.csv
```

#### Gene set information
- *Time: 1 minute*
- *Size: 1 GB*

The gene set information were obtained from multiple sources as described in the manuscript. Since these are subject to change, we have provided the respective lists in this repository.
Prepare them for the consecutive analyses via:
```
mkdir -p data/resources/genesets
mv other/genesets data/resources/genesets
```

# 

### Protein-protein interaction data

#### Download the iRefIndexDB v14 data from the [OmicsIntegrator2 repository](https://github.com/fraenkel-lab/OmicsIntegrator2/tree/master)
- *Time: 1 minute*
- *Size: 1 GB*

Create the directory to store the data of the human protein-protein interactome as provided by the iRefIndexDB v14 via
```
mkdir -p data/resources/ppi
```

Download the preprocessed interactome from the OmicsIntegrator repository via
```
cd data/resources/ppi
wget "https://raw.githubusercontent.com/fraenkel-lab/OmicsIntegrator2/master/example/OI2_pipeline_data/iRefIndex_v14_MIScore_interactome_C9.costs.txt"
```

# 

## Data preprocessing

### Rohban imaging data

#### Setup the environment for the nuclear segmentation using a pretrained UNet model
- *Time: 5 minutes*
- *Size: 5 GB*

To segment the nuclei in the images, clone the a fork of the repository from volkerh/unet via
```
git clone "https://github.com/dpaysan/unet-nuclei.git"
cd unet-nuclei
```

Next, create the conda environment containing the dependencies to run the UNet segmentation via
```
conda create --name unet python=3.8.10
```

Activate the conda environment and install the required software libraries
```
conda activate unet
bash setup_unet_environment.sh
```

#### Run the nuclear segmentation
- *Time: 5 hours*
- *Size: 100 GB*

Start the jupyter server via
```
conda activate unet
jupyter notebook
```

Open and run the jupyter notebook located in ``unet/notebooks/rohban_segmentation.ipynb``.
The output, i.e. the segmentation masks will be stored in ``image2reg/data/resources/images/rohban/unet_masks``.

#### Preprocess the imaging data
- *Time: 5 hours*
- *Size: 80 GB*

> [!WARNING]
> The following will generate over 2.1 million image files as a result of the nuclear segmentation. It is expected that accessing your file system is impaired during the process due to the permanen I/O operations required to store the images.

Run the preprocessing script for the image data via
```
conda activate image2reg
python run.py --config config/preprocessing/full_image_pipeline.yml
```

The output is saved in a "timestamp" directory in ``data/experiments/rohban/images/preprocessing/full_pipeline`` to avoid overwriting data by accident.
Please move all content of "timestamp" directory such that it is located directly in the directory ``data/experiments/rohban/images/preprocessing/full_pipeline``.

# 

### Gene expression data

#### Preprocess the scRNA-seq data
- *Time: 5 minutes*
- *Size: 1 GB*

> [!IMPORTANT]
> The preprocessing is using the ``mygene`` package which uses the most recent available annotation file of the human genome from Encode by default. As a consequence you might observe slight differences of your results compared to ours. However, the differences would not affect the results qualitatively.

Start the jupyter server the conda environment via

```
conda activate image2reg
jupyter notebook
```

Start the jupyter notebook ``notebooks/rohban/ppi/gex_analyses/scgex_preprocessing.ipynb`` and run all cells. The notebook generates a file that contains the preprocessed scRNA-seq data namely ``data/experiments/rohban/gex/scrnaseq/fucci_adata.h5``.

#### Preprocess the CMap gene sigantures
- *Time: 2 minutes*
- *Size: < 1 GB*

Start the jupyter server in the conda environment via
```
conda activate image2reg
jupyter notebook
```

Start the jupyter notebook ``notebooks/rohban/ppi/gex_analyses/cmap_preprocessing.ipynb`` and run all cell in the notebook.
The final cell generates a file that contains the processed CMap gene signatures namely ``data/experiments/rohban/gex/cmap/mean_l5_signatures_tmp.csv``.

# 

## Identify impactful overexpression conditions

#### Generate the required data split files
- *Time: 5 minutes*
- *Size: 4 GB*

Download the ``target_list`` directory from the DOI [10.5281/zenodo.8415537](https://doi.org/10.5281/zenodo.8415537) and place it under ``data/resources/target_list``.

Start the jupyter server in the conda environment via
```
conda activate image2reg
jupyter notebook
```

Start the notebook ``notebooks/rohban/other/cv_screen_data_split.ipynb`` and run all cells in the notebook.
This generates a number of files located in ``data/experiments/rohban/images/preprocessing/screen_splits``.

#### Run the specificity screen
- *Time: 200 hours*
- *Size: 150 GB*

Run the specificity screen to identify impactul overexpression conditions via
```
conda activate image2reg
bash run_screen.sh
```

Finally, rename the output of the screen located in ``data/experiments/rohban/images/screen/nuclei_region`` via
```
conda activate image2reg
python scripts/experiments/rename_screen_dirs –root_dir data/experiments/rohban/images/screen/nuclei_region
```

#### Analyze the results
- *Time: 1 hour*
- *Size: 1GB*

Start the jupyter server in the conda environment
```
conda activate image2reg
jupyter notebook
```

Start the notebook ``notebooks/rohban/image/screen/screen_analyses_cv_final.ipynb`` and run all cells.
This creates a summary of the screen results and saves it as ``data/experiments/rohban/images/screen/specificity_screen_results_cv.csv``.

# 

## Gene perturbation embeddings

### General setup using all overexpression conditions

#### Generate data splits
- *Time: 2 minutes*
- *Size: 1 GB*

Start the jupyter server in the conda environment via
```
conda activate image2reg
jupyter notebook
```

Start the notebook notebooks/rohban/other/cv_specific_targets_data_split.ipynb and run all cells.
This creates the required metadata csv-files for the individual splits of the stratified four-fold group cross-validation in ``data/experiments/images/preprocessing/specific_targets_cv_stratified``.


#### Run four-fold 41 overexpression + 1 control classification
- *Time: 30 hours* 
- *Size: 4 GB*

Run the bash script performing the four-fold stratified grouped cross-validation approach via
```
conda activate image2reg
bash scripts/experiments/run_selected_targets.sh
```

This will perform all four folds and save all the outputs in "timestamp" directory located in ``data/experiments/rohban/images/embeddings/four_fold_cv/fold{0,1,2,3}``.
Copy all contents of all "timestamp" directories to their corresponding parent (i.e. ``fold``) directory and remove the then empty "timestamp" directory.


#### Analyze the classification results
- *Time: 30 minutes*
- *Size: 1 GB*


Start the jupyter server in the conda environment via
```
conda activate image2reg
jupyter notebook
```

Start the notebook ``notebooks/rohban/image/embedding/image_embeddings_analysis.ipynb`` and run all cells
This produces the e.g. the Fig. 2C of the manuscript.

Next, start the notebook ``notebooks/rohban/image/embedding/gene_perturbation_cluster_analysis.ipynb`` and run all cells to e.g. reproduce the Fig. 2e of the manuscript.

# 

### Leave-one-target-out cross-validation setup

#### Generate the data splits

- *Time: 10 minutes*
- *Size: 3 GB*

Start the jupyter server in the conda environment via
```
conda activate image2reg
jupyter notebook
```

Start the notebook ``notebooks/rohban/other/loto_data_splits.ipynb`` and run all cells.
This produces a number of csv files that describe the data splits for the four-fold stratified grouped cross-validation for the leave-one-target-out inference stored in ``data/experiments/rohban/images/preprocessing/loto_cv_stratified``.

#### Run the 41 multi-class classification
- *Time: 100 hours*
- *Size: 40 GB*

Start the leave-one-target-out classification experiment via
```
conda activate image2reg
bash scripts/experiments/run_loto_selected_targets.sh 
```

The results are stored in ``data/experiments/rohban/images/embeddings/leave_one_target_out/training``.
Place all contents in the timestamp directories located in ``training/<target>``directly into the ``training/<target>`` directory directly instead of in a timestamp subdirectory.

#### Analyze results
- *Time: 40 minutes*
- *Size: 1 GB*

Start jupyter server in conda environment via
```
conda activate image2reg
jupyter notebook
```

Start the notebook ``notebooks/rohban/image/embedding/image_embeddings_analysis_loto.ipynb`` and run all cells.
This creates a number of files located in the output directory ``data/experiments/rohban/images/embeddings/leave_one_target_out_embeddings``.

# 

## Identify the cell-type specific gene-gene interactome

#### Prepare human PPI for Steiner tree analysis
- *Time: 5 minutes*
- *Size: > 1 GB*

Start the jupyter server in the conda environment
```
conda activate image2reg
jupyter notebook
```

Run the notebook ``notebooks/rohban/ppi/preprocesssing/inference_preparation_full_pruning.ipynb`` and run all cells.
This saves preprocessed protein-protein interactome as a pickle file in ``data/experiments/rohban/interactome/preprocessing`` and also produces components of the e.g. the Fig. 3a of the manuscript.


#### Identify U2OS-specific GGI via the PCST algorithm
- *Time: 10 minutes*
- *Size: 1 GB*

Start the jupyter server in the conda environment via
```
conda activate image2reg
jupyter notebook
```

Start the jupyter notebook ``notebooks/rohban/ppi/inference/interactome_inference_final.ipynb`` and run all cells.
This i.a. saves the inferred gene-gene interactome as a pickle and .graphml file in ``data/experiments/rohban/interactome/inference_results``.
To visualize the inferred network and reproduce the visualization of the gene-gene interactome in Fig. 3a please open the .graphml file in [Cytoscape](https://cytoscape.org/).


#### Analyze the results
- *Time: 5 minutes*
- *Size: 1 GB*

> [!IMPORTANT]
> To reproduce our results please make sure that you use the same version of RStudio (v.1.3.959) and R (v.4.0.3) as well as of all additional packages listed in e.g. ``notebooks/rohban/image/embedding/gene_perturbation_go_analyses_rsessioninfo.txt`` and ``notebooks/rohban/ppi/embeddings/gene_embedding_cluster_analyses_rsessioninfo.txt``.


Start RStudio and open the Rmd Notebook ``notebooks/rohban/ppi/other/go_analysis_pcst_solution.Rmd``.
Set the working directory in the first cell to the location of the directory ``image2reg`` and run all cells to reproduce the GO results for the PCST solution shown in Fig. S12 of the manuscript.

# 

## Regulatory gene embeddings

#### Compute leave-one-target out gene embeddings
- *Time: 10 hours*
- *Size: 25 GB*


Start the jupyter server in the conda environment via
```
conda activate image2reg
jupyter notebook
```

Start the notebook ``notebooks/rohban/ppi/gex_analyses/cmap_full_clustering.ipynb`` and run all cells to generate the file ``data/experiments/rohban/other/mean_cmap_sig_clusters_all_covered_nodes.csv``.


Next, start the notebook ``notebooks/notebooks/rohban/ppi/embeddings/gae_gene_embs.ipynb`` and run all cells.
This trains for different choices of the hyperparameters weighing the different loss components for the GCAE the graph autoencoder.
The generated regulatory gene embeddings are saved in ``data/experiments/rohban/images/embeddings/leave_one_target_out/embeddings/<condition>/spearman_sol``, where condition is each of the 41 impactful OE conditions.
This also generates the output of all regulatory embeddings and the tSNE plot shown in Fig. 3b as well as the clustering that is assessed in Fig. 3c of the manuscript.


## Analyze results
- *Time: 10 minutes*
- *Size: 1 GB*

Start the jupyter server in the conda environment
```
conda activate image2reg
jupyter notebook
```

Open the notebook ``notebooks/rohban/ppi/embeddings/gene_embedding_clustering.ipynb`` and run all cells.
This saves the clustering solution of the inferred regulatory gene embeddings in ``data/experiments/rohban/cluster_infos/all_gene_embeddings_clusters.csv``.


Next, start RStudio and open the notebook ``notebooks/rohban/ppi/embeddings/gene_embedding_cluster_analyses.Rmd`` and run all chunks to reproduce e.g. Fig. 3c.


## Mapping gene perturbation to regulatory gene embeddings

#### Run gridsearch and analyze results
- *Time: 40 hours*
- *Size: 4 GB*

Start the jupyter server in the conda environment via
```
conda activate image2reg
jupyter notebook
```

Start the notebook ``notebooks/rohban/translation/mapping/translational_mapping_loto_gridsearch_final.ipynb`` and run all cells to rerun the gridsearch approach for the NTK regression to map from the gene perturbation to regulatory gene embeddings.
This creates a number of files located at ``data/experiments/rohban/translation``and e.g. plot the Fig. 4b of the manuscript.

# 

## Additional validation using JUMP-CP

###Obtain and prepare data

#####Download JUMP-CP data
- *Time: 4 hours*
*Size: 105 GB*

Start the jupyter server in the conda environment via
```
conda activate image2reg
jupyter notebook
```

Start the jupyter notebook ``notebooks/jump/eda/data_extraction.ipynb`` and run all cells to download the image data from the JUMP-CP data set from [Chandrasekaran et al.(2023)](https://www.biorxiv.org/content/10.1101/2023.03.23.534023v1) for the selected OE conditions including the illumination corrected images.
All generated data gets downloaded to ``data/resources/images/jump``.


#### Run nuclear segmentation using the corresponding jupyter notebook
- *Time: 3 hours*
- *Size: 100 GB*

Start the jupyter server in the unet conda environment via
```
conda activate unet
jupyter notebook
```

Open and run the jupyter notebook located in ``unet/notebook/jump_segmentation.ipynb``.
> [!IMPORTANT]
> Please be aware that this is not a path in the image2reg directory but the unet-nuclei directory you have cloned earlier. Please refer to the respective section for the Rohban data set in this protocol for more information.

Run all cells to generate the segmentation masks for all images and stores those in ``image2reg/data/resources/images/jump/unet_masks``.


#### Preprocess Rohban imaging data via script
- *Time: 100 hours*
- *Size:  300 GB*

> [!WARNING]
> This following steps generate roughly 5 million of image files as a result of the nuclear segmentation. It is expected that accessing your file system during the run time is substantially slower than usual due to the permanent I/O operations required to store the images and update the index file of your file system.


Run the preprocessing script via

```
conda activate image2reg
python run.py --config config/preprocessing/full_image_pipeline_jump.yml
```


This runs all preprocessing steps stores the outputs in a "timestamp" output directory in the directory ``data/experiments/jump/images/preprocessing/full_pipeline``.
By default all output directories created as a result of running the run.py will be named after the time point when the script was started.
For the consecutive analyses please copy the content of the timestamp output directory directly to the ``full_pipeline``  directory  and delete the then empty timestamp directory.


### Image and gene perturbation embeddings

#### Prepare the 4-fold cross-validated classification
- *Time: 5 minutes*
- *Size: 1 GB*

Start the jupyter server in the conda environment via
```
conda activate image2reg
jupyter notebook
```

Start the notebook ``notebooks/rohban/other/cv_specific_targets_data_split_jump.ipynb`` and run all cells to create the the metadata files that define the split of the data for the four-fold cross-validation.
The created files are stored in ``data/experiments/jump/images/preprocessing/specific_targets_cv_stratified``.


#### Run four-fold 31+1 classification
- *Time: 12 hours*
- *Size: 8 GB*


Run the training of the CNN ensemble on the image data from the JUMP data set in the conda environment via
```
conda activate image2reg
python run.py --config  config/image_embedding/specific_targets/cv_jump/nuclei_region/fold_0.yml
python run.py --config  config/image_embedding/specific_targets/cv_jump/nuclei_region/fold_1.yml
python run.py --config  config/image_embedding/specific_targets/cv_jump/nuclei_region/fold_2.yml
python run.py --config  config/image_embedding/specific_targets/cv_jump/nuclei_region/fold_3.yml
```


The results of the analyses are saved in the directory ``data/experiments/jump/images/embedding/specificity_target_emb_cv_strat/fold_#`` where # is 0,1,2 or 3 respectively.
By default the results are saved in a timestamp subdiretory.
Rename the timestamp directory to ``nuclei_regions``.


#### Compute single-cell image embeddings
- *Time: 9 hours*
- *Size:  10 GB*


Run the script to infer the image embeddings for all 175 train and potential test conditions in the conda environment via
```
conda activate image2reg
python run.py --config config/image_embedding/specific_targets/extract_latents/extract_latents_jump_data_resnet_ensemble_specific_targets.yml
```

The script saves all generated outputs in a timestamp directory in the directory ``data/experiments/jump/images/embedding/extract_latents_from_rohban_trained``.
Copy the content of the timestamp directory into the directory ``extract_latents_from_rohban_trained`` and then delete the timestamp directory.



#### Compute gene perturbation embeddings
- *Time: 2 hours*
- *Size: 15 GB*


Start the jupyter server in the conda environment via
```
conda activate image2reg
jupyter notebook
```

Start the notebook ``notebooks/jump/eda/eda_jump_image_representations.ipynb`` and run all cells to create e.g the Supplemental Figures S22 and generate the gene perturbation embeddings.
The latter are saved alongside other embeddings in ``data/experiments/jump/images/embedding/embeddings``.
The cells also download the morphological profiles for the JUMP-CP data set which will be saved in ``data/resources/images/jump/profiles``.


Next, start the jupyter notebook ``notebooks/jump/embeddings/analyses_jump_embedding_candidates.ipynb`` in the same jupyter session.
Run all cells to generate the input data for the translation analyses.
All generated data will be located in ``data/experiments/jump/images/embedding/all_embeddings``.



### Performance evaluation

#### Run gridsearch and analyze the results
- *Time: 3 hours*
- *Size: 3 GB*


Start the jupyter server in the conda environment via
```
conda activate image2reg
jupyter notebook
```

Start the notebook ``notebooks/jump/translation/jump_translation_prediction_final.ipynb`` and run all cells to perform the complete translation analysis and e.g.generate Fig. 4C.
This concludes the reproduction of all results presented in our study from scratch.

---


## Troubleshooting and common mistakes

1. Please always check that you have activated the correct conda environment, in particular if you encounter errors that indicate missing packages.
2. Since we use a number of external software packages from pypi which are not managed by us, please consult the respective documentation in case you encounter any problem e.g. during the installation of these.
3. Always make sure that your working directory is ``image2reg`` unless specified otherwise.
4. Ensure that you have a stable internet connection as in particular the download scripts might fail if it is interrupted.

If you encounter any problems and you cannot identify a solution, please open an issue in the GitHub repository and we will try our best to assist you.
