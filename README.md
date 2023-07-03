# Image2Reg: Linking Chromatin Images to Gene Regulation using Genetic Perturbation Screens

**by Daniel Paysan (#), Adityanarayanan Radhakrishnan (#), G.V. Shivashankar (^) and Caroline Uhler (^)**

The repository contains the code for the main methodology and analyses described in our paper: 
 >[*Image2Reg: Linking Chromatin Images to Gene Regulation using Genetic Perturbation Screens (Under Review)*](https://github.com/uhlerlab/Image2Reg).
 
 ![](https://github.com/dpaysan/image2reg/blob/389a275421f9d5508685ba0feb30f051085c54b2/imag2reg_pipeline.png)

---

## System requirements

The code has been developed on a system running Ubuntu 20.04. LTS with Python v3.8 installed using a Intel(R) Xeon(R) W-2255 CPU with 3.70GHz, 128GB RAM and a Nvidia RTX 4000 GPU with CUDA v.11.1.74 installed. Note that for setups with less available RAM and/or GPU, parameters like the batch size for the training of the neural networks might have to be adjusted.

## Installation/Environmental setup

To install the code please first clone this repository using
```
git clone https://github.com/uhlerlab/image2reg.git
cd image2reg
```

The software was built and tested using Python v3.8. Thus, please next install Python v3.8. While it is theoretically not required, we have used and thus recommend the package manager [miniconda](https://docs.conda.io/en/latest/miniconda.html) to setup and manage the computational environment. To install miniconda please follow he official installation instructions, which can be found [here](https://conda.io/projects/conda/en/stable/user-guide/install/linux.html).

Once miniconda is installed, you can create a conda environment running Python v3.8 in which the required software packages will be installed via:
```
conda create --name image2reg python==3.8
```

The final step of the installation consists of the installation of additional required packages which can be efficiently done via:
```
conda activate image2reg
bash scripts/installation/setup_environment_cuda.sh
```
Note that this installs the requried [Pytorch](https://pytorch.org/) associated packages with GPU accelaration using CUDA v11.1, which we had used to develop and run the code in this repository. Further, please note that some packages are not available from the anaconda cloud, which is why we use pip to install all packages. Using only pip for the installation avoids potentially broken environments. We have tested the installation on multiple hardware systems to ensure that it works as expected.

If no GPU is available on your system, please install the required packages without GPU support via: 
```
conda activate image2reg
bash scripts/installation/setup_environment_cpu.sh
```
Note that without GPU accelaration the run time of the code with respect to the training and evaluation of the neural networks is significantly longer.
Please also refer to the official Pytorch installation guide, which can be found [here](https://pytorch.org/get-started/locally/), in case you encounter any problems regarding the installation of packages such as ``torch, torchvision and torchaudio``. Similarly, please also consult the official documentation of Pytorch Geometric, which can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/) in case that you encounter any problems with the installation of the packages ``torch-geometric, torch-scatter, torch-sparse, torch-spline-conv``.

**Note that installing all additional packages as described before is highly recommended to recreate the environment the code was developed in.**

In total the estimated installation time is 10-20 minutes depending on the speed of the available internet connection to download the required software packages.

---

## Data resources

Note that the following data resources are only required if you want to reproduce the results presented in the paper. In case you want to apply the presented methodology to your own data, please skip this section.

The raw data including the images of the perturbation screen by [Rohban et. al, 2017](https://doi.org/10.7554/eLife.24060), the images from the JUMP data set by [Chandrasekaran et al, 2023](https://doi.org/10.1101/2023.03.23.534023) and the gene expression and protein-protein interaction data are publicly available from the sources described in the paper.

To facilitate the download of the two image data sets, we have provided some additional code. 

### Data from Rohban et al. (2017)
The raw images of the Rohban data set and the associated morphological profiles can be downloaded using 

```
bash scripts/data/download_rohban_data.sh
```
The script requires the [Aspera Client](https://www.ibm.com/products/aspera?utm_content=SRCWW&p1=Search&p4=43700074866463662&p5=p&gclid=CjwKCAjwm4ukBhAuEiwA0zQxk2sPzQlK4wH4MJJdL1Jwiw9QnYncuvghaJrocgIILEMFAHfDHRGJNBoC9wwQAvD_BwE&gclsrc=aw.ds) to be installed in the system. This is because the server which contains the imaging data (and which is not maintained by us) only provides access to the data using the Aspera client.
If you encounter an error saying ascp command not found while running the download script, please verify that the Aspera client is installed.

To install you the Aspera client on Linux you can run the following code (Taken from the tutorial [here](https://www.biostars.org/p/9528910/)).
```
wget https://ak-delivery04-mul.dhe.ibm.com/sar/CMA/OSA/0adrj/0/ibm-aspera-connect_4.1.3.93_linux.tar.gz
tar zxvf ibm-aspera-connect_4.1.3.93_linux.tar.gz
bash ibm-aspera-connect_4.1.3.93_linux.sh
```
By default the Aspera client will be installed in ``$HOME/.aspera/connect/bin/``, please add the respective directory to your ``$PATH`` environment variable, e.g. via

```
export PATH=$PATH:$HOME/.aspera/connect/bin
```
To verify the installation try running the ```ascp``` command in the terminal, if you see usage instructions of the client as an output it is correctly installed and you should be able to run the data retrieval script ``scripts/data/download_rohban_data.sh`` as described above. Please note if the script seems to be stuck after outputting ``Download imaging data...``, there might be an issue with the connection to the IDR server, where the data is located. In that case likely your firewall configuration block the access to the server. In that case please try a different network connection and/or use our intermediate data respository described below instead. 

### JUMP-CP imaging data set
The raw images and profiles of the JUMP data set can be downloaded using the notebook ``notebooks/jump/eda/data_extraction.ipynb``. 
To run the code please start the jupyter server via
```
jupyter notebook
```
and navigate to the respective notebook. Executing the cells and following the descriptions in the notebook will download the raw and respective metadata. If you encounter an error saying ascp command not found while running the download script, please verify that the Aspera client is installed. To install it on linux follow e.g. the tutorial found [here](https://www.biostars.org/p/9528910/).


### Our data repository
Additional intermediate outputs of the presented analyses including i.a. trained neural network models, the inferred gene-gene interactome, computed image, perturbation gene and regulatory gene embeddings can be downloaded from the PSI Data Catalog using the DOI: [10.16907/febfd200-8b72-48ba-8704-01e842314697](https://doi.psi.ch/detail/10.16907%2Ffebfd200-8b72-48ba-8704-01e842314697). Please note that the data is currently stored to the linked archive tape system and will be available to download from the above DOI by July 12th, 2023 the latest. In the meantime the same data can be retrieved from our [Google Drive](https://drive.google.com/drive/folders/18ITp40Hz1ZcXXKlCJz21_ujqZkzliz_b?usp=sharing). Please note that the intermediate data repository is 795GB in size and the time it the procurement of the data takes to prepare it for the download from the PSI Data Catalog depends on the load of the system. Please expect longer procurement and download times.

In the following, we will refer to this data set as "our data repository". Note that the intermediate results and the respective data in our data repository are optional and were generated as a results of the steps described in the following.

**To rerun all experiments using the intermediate results in our data repository, please place it as the ``data`` directory in this repository after you have cloned it.**
*Please note that you will have to depack (untar) the two .tar archives called ``experiments`` and ``resources``, that are contained in the downloaded folder before running the code.*

---

## Reproducing the paper results
The following description summarizes the steps to reproduce the results presented in the paper. To avoid long-run times (due to e.g. the large-scale screen to identify impactful gene perturbations), intermediate results can be downloaded from the referenced data resources mentioned above.

*Note that, solely running all experiments and analyses described in the paper took more than 200 hours of pure computation time on the used hardware setup due to the complexity of the computations and the size of the data sets. The different notebooks and scripts reference the locations of the required files according to the structure provided in our intermediate data repository. Thus, our code can be run without changing any file paths if the intermediate data repository is used. However, if you prefer to store the data differently than the structure used in our data repository, additional changes of the file locations in several scripts will be required. We are happy to assist in these cases.*

### 1. Data preprocessing

#### 1.1. Imaging data from Rohban et al.

The raw image data of the perturbation screen from Rohban et al. (2017) is preprocessed including several filtering steps and nuclear segmentation.
The full preprocessing pipeline can be run via
```
python run.py --config config/preprocessing/full_image_pipeline.yml
```
**Please skip that step if you are using our intermediate data repository and have not download the raw image data from Rohban et al. (2017).**  This is because the script requires the raw image data from Rohban et al. (2017) to be downloaded and the respective content of the ``ilum_corrected`` image files to be located at ``image2reg/data/resources/images/rohban/illum_corrected``. By default the download script ``scripts/data/download_rohban_data.sh will download all data including the ``illum_corrected`` directory to ``image2reg/data/resources/images/rohban/raw`` if run from within the``image2reg`` directory. Please simply copy the downloaded ``illum_corrected`` directory to the ``image2reg/data/resources/images/rohban/`` directory. Assuming you have downloaded in the intermediate data, you can then rerun the preprocessing as defined above. Please again note that since the raw data from Rohban et al. (2017) is not managed by us, we cannot provide all raw image files directly as part of our intermediate data repository.**

*A version of the output of the preprocessing pipeline, which e.g. contains the segmented single-nuclei images is available in our data repository at ``experiments/rohban/images/preprocessing/full_pipeline``.*

#### 1.2. Imaging data from JUMP

The raw image data of the perturbation screen of the JUMP consortium by Chandrasekaran et al. (2023) is preprocessed similar to the data from Rohban et al. (2017).
The full preprocessing pipeline can be run via
```
python run.py --config config/preprocessing/full_image_pipeline_jump.yml
```

*A version of the output of the preprocessing pipeline for the JUMP data set is available in our data repository at ``data/experiments/jump/images/preprocessing/full_pipeline``.*

#### 1.3. Gene expression data

Single-cell gene expression data from [Mahdessian et al, 2021](https://www.nature.com/articles/s41586-021-03232-9) was preprocessed as described in the paper using the notebook available in ```notebooks/ppi/gex_analyses/scgex_preprocessing.ipynb```. A version of the output, preprocessed gene expression data is available in our data repository at ```preprocessing/gex/fucci_adata.h5```.

CMap gene signature data from [DepMap, 2021](https://depmap.org/portal/) was preprocessed using the notebook available in ```notebooks/ppi/gex_analyses/cmap_preprocessing.ipynb```. The input data is also available on our data repository (```preprocessing/gex/CCLE_expression.csv```) but please make sure to reference the data source mentioned above and in the paper appropriately.

Note that this notebooks assumes that the gene-gene interactome (GGI) had already been inferred. Please see 3. on how to infer the GGI. 

In general in order to run any of the provided jupyter notebooks, please start the jupyter server in the setup computational environment via:
```
conda activate image2reg
jupyter notebook
```

---

### 2. Inference and analysis of the image and gene perturbation embeddings
#### 2.1. Identification of impactful gene perturbation

To identify the impact gene perturbations we ran a large-scale screen assessing the performance of proposed convolutional neural network architecture on the task of distinguishing between the perturbed and control cells using the chromatin image inputs obtained as a result of the preprocessing (see 1.1.).

The screen can be automatically run by calling 
```
bash scripts/experiments/run_screen.sh
```

Additionally, the script assumes that the config files specifying the individual training tasks of the model for the different perturbation targets are available. The notebooks ```notebooks/rohban/other/cv_screen_data_splits.ipynb``` and ```notebooks/rohban/other/create_screen_configs.ipynb``` provide functions to efficiently generate the resources required by the script to complete the screen.

*The results of the screen which include e.g. the trained convolutional neural networks and the log files describing the performance of the network on the individual binary classification tasks are available from our data repository at ``data/experiments/rohban/images/screen/nuclei_region``.*

Once the screen has been run the notebook ```notebooks/rohban/image/screen/screen_analyses_cv_final.ipynb``` can be used to analyze those results and identify the impact gene perturbations. Gene set information data can be obtained as described in the paper or directly from our data repository at ```resources/genesets```.


#### 2.2. Inference of image embeddings

To infer the image embeddings the convolutional neural network is trained on a multi-class classification task to distinguish between the different as impactful identified gene perturbation settings. Thereby, the embeddings of the images in the test sets provide corresponding image embeddings.

The related experiment can be run by calling
```
bash scripts/experiments/run_selected_targets.sh
```

The script uses data files and config files used by the script need to be available. The corresponding data is part of the available optional data resources but can also be efficiently generated from the output of the output of image preprocessing step (see step 1.1.) using the notebook ```notebooks/rohban/other/cv_specific_targets_data_split.ipynb```.

*A version of the output generated during this experiment are available in our data repository at ```experiments/rohban/images/embeddings/four_fold_cv```.*

To obtain the image embeddings in the leave-one-target-out evaluation scheme the related experiments can be performed by calling
```
bash scripts/experiments/run_loto_selected_targets.sh
```

The required metadata files for the experiments are again available as part of the optional data resources or can be efficiently generated using the notebooks ```notebooks/rohban/other/create_loto_configs.ipynb``` and ```notebooks/rohban/other/loto_data_splits.ipynb```.

*A version of the thereby obtained image embeddings are available in our data repository at ```experiments/rohban/images/embeddings/leave_one_target_out.csv```.*


#### 2.3. Analyses of the image embeddings

The analyses of the image embeddings and visualization of their representation can be assessed using the notebook ```notebooks/rohban/image/embedding/image_embeddings_analysis.ipynb```. The cluster analysis and visualization of the inferred image embeddings in the leave-one-target-out evaluation setup can be rerun using the code in ```notebooks/image/embedding/image_embeddings_analysis_loto.ipynb```.


#### 2.4. Analyses of the gene perturbation embeddings

The cluster analyses of the inferred gene perturbation embeddings are performed using the notebook ``notebooks/image/embedding/gene_perturbation_cluster_analysis.ipynb`` and ``notebooks/rohban/image/embedding/gene_perturbation_cluster_analysis.ipynb``. Gene ontology analyses were performed using the R notebook ``notebooks/rohban/image/embedding/gene_perturbations_go_analyses.Rmd``. Note that the preprocessed morphological profiles are available from the optional data resources but can be obtained by simply removing all features associated to channels other than the DNA channel from profiles available by Rohban et al. (2017).

---

### 3. Inference and analyses of the gene-gene interactome

#### 3.1. Gene-gene interactome inference

The gene-gene interactome is inferred using a prize-collecting Steiner tree (PCST) algorithm. The required inputs of the algorithm can be obtained using the notebook ```notebooks/rohban/ppi/preprocesssing/inference_preparation_full_pruning.ipynb```. In addition to the described single-cell gene expression data set the notebook requires asn estimate of the human protein-protein interactome and bulk gene expression data from the Cancer Cell Line Encyclopedia as input. The resources of those inputs are listed in the paper.

After the preprocessing of the inputs, a the gene-gene interactome can be inferred using the code available in the notebook ```notebooks/rohban/ppi/inference/interactome_inference_final.ipynb```. The input network to the PCST analyses and the output gene-gene interactome is also directly available as part of the optional data resources.

*Our data repository contains a version of both the preprocessed protein-protein interactome that is input to the PCST analysis (at ```experiments/rohban/interactome/preprocessing/cv/ppi_confidence_0594_hub_999_pruned_ccle_abslogfc_orf_maxp_spearmanr_cv.pkl```) as well as the finally output gene-gene interactome (at ```experiments/rohban/interactome/inference_results/spearman_sol_cv.pkl```).*

#### 3.2. Analysis of the inferred gene-gene interactome

The R notebook ```notebooks/rohban/ppi/other/go_analysis_pcst_solution.Rmd``` provides the code to evaluate the enrichment of biological processes associated to the mechanotransduction in cells in the inferred gene-gene interactome.

---

### 4. Inference and analyses of the regulatory gene embeddings

Given the previously computed inputs the proposed graph-convolutional autoencoder model can be trained to infer the regulatory gene embeddings within and outside of the leave-target-out evaluation setup described in the paper. The code required to run those experiments is available in ```notebooks/rohban/ppi/embeddings/gae_gene_embs.ipynb```. All required inputs for the analyses are outputs of the previously described steps.

*Additionally, they are also directly available in our data repository. In particular, the gene expression data in ```experiments/rohban/gex```, the gene set information in ```resources/genesets``` and additional cluster inputs in ```experiments/rohban/interactome/cluster_infos```. The results of all conducted experiments including the inferred regulatory gene embeddings that are input to the translation analysis in the leave-one-target-out evaluation setting (see section 5) are available at ```experiments/rohban/images/embeddings/leave_one_target_out```.*

The analysis of the clustering of the inferred gene-gene embedding are also included in that notebook. The R notebook in ```notebooks/rohban/ppi/embeddings/gene_embedding_cluster_analyses.Rmd```.

---

### 5. Mapping gene perturbation to regulatory gene embeddings

Finally, the code required to assess the alignment of the inferred regulatory gene and perturbation gene embeddings in the described leave-one-targe-out evaluation setup including the associated meta-analyses are available in the notebook ```notebooks/rohban/translation/mapping/translational_mapping_loto_gridsearch_final.ipynb```.
The results of the different translation models in the leave-one-target-out evaluation setting can also be obtained from our data repository in the ```data/experiments/rohban/translation``` directory.

---

### 6. Additional validation using the JUMP data set
While the above instructions explain how to reproduce the results using the imaging data set from Rohban et al. (2017), we here describe briefly how to reproduce the results for the additional validation using the JUMP data set. To this end, we assume the previous steps have already been completed.

#### 6.1. Inference of the image embeddings

First, we obtain image embeddings for the JUMP data set as described in the paper by retraining our CNN ensemble model on the data of the 31 selected overexpression conditions contained in the JUMP data set via:

```
python run.py --config  config/image_embedding/specific_targets/cv_jump/nuclei_region/fold_0.yml
```

*The metadata files referenced in the config files were obtained by applying a four-fold stratified cross-validation split to the JUMP data set using the functionalities in the notebook ```notebooks/rohban/other/cv_specific_targets_data_split_jump.ipynb``` but can also be found in our shared data repository at ```experiments/jump/images/preprocessing/specific_targets_cv_stratified``` in addition to the output of that step which can be found in ```data/experiments/jump/images/embedding/specificity_target_emb_cv_strat/fold_0```.*

Once trained, we obtain the image embeddings for all considered 175 overexpression conditions via:

```
python run.py --config config/image_embedding/specific_targets/extract_latents/extract_latents_jump_data_resnet_ensemble_specific_targets.yml
```

*The output of that step is located in ```data/experiments/jump/images/embedding/extract_latents_from_rohban_trained``` in our shared data repository.*

Finally, gene perturbation embeddings are obtained as for the Rohban data by averaging the embeddings across conditions. This is done using the output of the executing first the notebook ```notebooks/jump/eda/eda_jump_image_representations.ipynb``` followed by the notebook ```notebooks/jump/embeddings/analyses_jump_embedding_candidates.ipynb```.
*The final gene perturbation embeddings used for the additional validation that are output of the previous step are also available from our shared data repository in ```experiments/jump/images/embedding/all_embeddings```.

#### 6.2. Evaluation of the effectiveness of our pipeline

Using the computed gene perturbation embeddings, we evaluate our pipeline on predicting the targets of over novel/unseen 75 overexpression condition in the JUMP data set using the functionalities in the notebook ```notebooks/jump/translation/jump_translation_prediction_final.ipynb```.

---

## Application of the pipeline to other imaging-perturbation data

Our method is broadly applicable and provides a general framework to link cell images and gene regulation in genetic perturbation screens. However, the pipeline must be adjusted depending on the input data. The previously described set up as well as the selection of all hyperparameters including sizes of the cell images, architecture of the neural networks and so on should be tuned to the specific use case. The code provided in this repository including the notebooks in addition to the above description of the reproduction of the results presented in our paper should provide a general construct for such analyses, despite the fact that individual hyperparamters need to be selected by the user in dependence on the data. Please feel free to open an issue in the Github repository if you experience any issues in that context or need assistance.

---

## Questions/Issues

If you encounter any problems with setting up the software and/or need assistance with adapting the code to run it for your own data set, feel very free to open a respective issue. We will do our very best to assist you as quickly as possible.

---

## Credits

If you use the code provided in the directory please also reference our work as follows:

**TO BE ADDED**

If you use the provided data please make sure to also reference the the corresponding raw data resources described in the paper in addition to our work.


