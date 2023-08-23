# Image2Reg: Linking Chromatin Images to Gene Regulation using Genetic Perturbation Screens

**by Daniel Paysan (#), Adityanarayanan Radhakrishnan (#), G.V. Shivashankar (^) and Caroline Uhler (^)**

The repository contains the code for the main methodology and analyses described in our paper: 
 >[*Image2Reg: Linking Chromatin Images to Gene Regulation using Genetic Perturbation Screens (Under Review)*](https://github.com/uhlerlab/Image2Reg).
 
 ![](https://github.com/dpaysan/image2reg/blob/389a275421f9d5508685ba0feb30f051085c54b2/imag2reg_pipeline.png)

---
## Changelog

### August 18th, 2023
We have added a novel demonstration of our pipeline that can be easily run without the need of even previously installing the coding environment and/or downloading any data. The demo can be used to run our pipeline in the inference mode, i.e. we provide a pretrained version of the pipeline but show how given images of five selected OE conditions it predicts the corresponding target genes out-of-sample (no information regarding these were used to setup the pipeline as described in the paper).

### August 2nd, 2023
On **July 17th 2023** the ``hdbscan`` package broke due to number of changes of the name resolution. As a consequence the installation of any version of the package including the version 0.8.27 used in our software package was no longer able to be installed, leading to our installation script to no longer be able to run completely ([see here for more information](https://github.com/scikit-learn-contrib/hdbscan/issues/600)). We have updated the requirements file of our package to install the hotfix implemented in version hdbscan v.0.8.33. While we could not have anticipated such an issue suddenly occuring, we apologize for the inconvenience this may have caused. We have tested the updated installation script but please let us know if you encounter any issue with the installation on your end and/or running our code.

---

## System requirements

The code has been developed on a system running Ubuntu 20.04. LTS using a Intel(R) Xeon(R) W-2255 CPU with 3.70GHz, 128GB RAM and a Nvidia RTX 4000 GPU with CUDA v.11.1.74 installed. Note that for setups with less available RAM and/or GPU, parameters like the batch size for the training of the neural networks might have to be adjusted.

---

## Demonstration of Image2Reg

### Overview
To facilitate the use and testing of our pipeline, we have implemented an easy demonstration of how our pipeline can be used to predict novel, unseen overexpression conditions from chromatin images once trained.In particular, the demonstration will:
1. Install a minimal software environment containing the required python version 3.8.10 and a few additional python packages. Note that these packages are only a subset of all packages used to create the code contained in this repository. If you would like to install all packages, please refer to the next section in this documentation.
2. Download the required data to run the inference demonstration of our pipeline which in particular includes the chromatin images for five overexpression conditions from the dataset from Rohban et al. (2017) as well as e.g. the pretrained image encoder model used to obtain image embeddings from the chromatin images.
3. Preprocess the chromatin images for the inference of the image embeddings eventually yielding the gene perturbation embeddings via e.g. segmenting individual nuclei.
4. Obtain the image and consequently the gene perturbation embedding for the test condition by encoding the images using the pretrained convolutional neural network ensemble image encoder model.
5. Link the gene perturbation embeddings of all but the held-out test condition to their corresponding regulatory gene embeddings by training the kernel regression model.
6. Obtain the prediction of the regulatory embedding for the held-out test condition and use it to identify an ordered prediction set of for the gene overexpressed in the held-out test condition.

*Note that we have built the demo to run without a GPU to maximize its compatability.*

### Step-by-step guide

**A linux system is reqired to run the demo. 
The run time is approximately 10-60 minutes depending on the specifications of the system it is run on.**

#### 1. Perequisites: Anaconda installation
The only perequisite the demo application has is that the package manager [``Anaconda``](https://docs.anaconda.com/free/) or [``miniconda``](https://docs.conda.io/en/latest/miniconda.html) is installed on your system.

To test if it is install please open a terminal and type in: conda. If you see an error message saying the command was not found, it is not yet installed.
If it is not installed, you can install it via:
```
cd ~/
wget https://repo.anaconda.com/miniconda/Miniconda3-py311_23.5.2-0-Linux-x86_64.sh
bash Miniconda3-py311_23.5.2-0-Linux-x86_64.sh
```

This will start the installer, which will guide you through the installation of miniconda. If you encounter any issues, please refer to the official installation guide which can be found [here](https://docs.conda.io/en/latest/miniconda.html#installing).
Please note that after the installation you will have to close the terminal and open a new one before continuing with the next steps.

#### 2. Clone the repository

Next please clone this repository using
```
git clone https://github.com/uhlerlab/image2reg.git
cd image2reg
```

#### 3. Run the demo

You are now ready to run the demo. The demo can be run via
```
source scripts/demo/image2reg_demo.sh
```

This command will run the demo using the default parameters which will apply our pipeline to predict that *BRAF* is the gene targeted for overexpression in cells. To this end, it uses chromatin images from the perturbation data set from [Rohban et al. (2017)](https://elifesciences.org/articles/24060). The pipeline was set up without using any images of cells in the *BRAF*, respectively any other test condition you choose, and thus performs out of sample prediction.

#### 4. Specifying the held-out overexpression condition
This demo application can be used to run our Image2Reg inference pipeline for five different overexpression conditions namely: *BRAF, JUN, RAF1, SMAD4 and SREBF1*. The ``--condition`` argument can be used to specify for which of these conditions our Image2Reg pipeline should be run and predict the overexpression target gene from the corresponding chromatin images.
For instance, to run our pipeline for the *JUN* overexpression condition, simply run
```
source scripts/demo/image2reg_demo.sh --condition JUN
```

#### 5. Advanced run settings/developer options
In addition to specifying for which overexpression condition our pipeline should be run, there are three additional arguments that one can be used for the demo application:
1. ``--random``: If this argument is provided, the Image2Reg pipeline is run such that the inferred gene perturbation and regulatory gene embeddings are permuted prior the kernel regression is fit which eventually predicts the overexpression target. This recreates the random baseline described in our manuscript. Using this argument, you will observe a deteriated prediction performance of our pipeline which is expected.
2. ``--environment``: This argument can be used if one would like to specify a pre-existing conda environment that is supposed to be used to run the demo application. By default, if the argument is not provided a new conda environment will be setup as part of the demo application called ``image2reg_demo`` in which all required python packages will be installed and that is used to run our code.
3. ``--help``: This argument can be used to obtain help on the usage of our demo and in particular summarizes the meaning of the different arguments (i.e. ``--condition``, ``--random``, ``--environment``) described before.


**If you would like to reproduce all results of the paper from scratch please continue to the following section of the documentation. If not we appreciate you testing our code and look forward to the amazing applications we hope our solution will help to create.**

---

## Full installation and environmental setup

To install the code please first clone this repository using
```
git clone https://github.com/uhlerlab/image2reg.git
cd image2reg
```

The software was built and tested using Python v3.8.10. Thus, please next install Python v3.8.10. While it is theoretically not required, we have used and thus recommend the package manager [miniconda](https://docs.conda.io/en/latest/miniconda.html) to setup and manage the computational environment. To install miniconda please follow he official installation instructions, which can be found [here](https://conda.io/projects/conda/en/stable/user-guide/install/linux.html).

Once miniconda is installed, you can create a conda environment running Python v3.8 in which the required software packages will be installed via:
```
conda create --name image2reg python==3.8.10
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

**Note that installing all packages with the GPU accelaration of PyTorch using CUDA as described before is highly recommended to recreate the environment the code was developed in.**


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

**Please note that this step should only be run after the intermediate data repository was downloaded or the results of the Rohban analyses was reproduced. This is required as the download script will only download image data for the selected 175 OE conditions to save disk storage and as only these can be used for the additional validation of our pipeline on the JUMP-CP data set.**

The raw images and profiles of the JUMP data set can be downloaded using the notebook ``notebooks/jump/eda/data_extraction.ipynb``. 
To run the code please start the jupyter server via
```
jupyter notebook
```
and navigate to the respective notebook. Executing the cells and following the descriptions in the notebook will download the raw and respective metadata. If you encounter an error saying ascp command not found while running the download script, please verify that the Aspera client is installed. To install it on linux follow e.g. the tutorial found [here](https://www.biostars.org/p/9528910/).


### Our data repository
Additional intermediate outputs of the presented analyses including i.a. trained neural network models, the inferred gene-gene interactome, computed image, perturbation gene and regulatory gene embeddings can be downloaded from the PSI Data Catalog using the DOI: [10.16907/febfd200-8b72-48ba-8704-01e842314697](https://doi.psi.ch/detail/10.16907%2Ffebfd200-8b72-48ba-8704-01e842314697) since **July 17th, 2023**.
Please note that the intermediate data repository is **795GB** in size and the time it takes to prepare the data donwload from the PSI Data Catalog depends on the workload and the assoiciated waiting queue length of the Swiss Data Center that hosts the PSI Data Catalog taoe archive system.

The archiving job on the data was completed on July 17th, 2023. In the meantime we had also made the same data available on our [Google Drive](https://drive.google.com/drive/folders/18ITp40Hz1ZcXXKlCJz21_ujqZkzliz_b?usp=sharing), where we will keep the data available until the paper review is complete. 
For your convenience, you can either download the two tar archives (``experiments`` and ``resources``) as two large files or download the files individually from the respective folders (``experiments`` and ``resources``) if you choose to download the data from the the Google Drive.


In the following, we will refer to this data set as "our data repository". Note that the intermediate results and the respective data in our data repository are optional and were generated as a results of the steps described in the following.

**To rerun all experiments using the intermediate results in our data repository, please place it as the ``data`` directory in this repository after you have cloned it.**
*Please note that you will have to depack (untar) the two .tar archives called ``experiments`` and ``resources``, that are contained in the downloaded folder before running the code.*

---

## Reproducing the paper results
The following description summarizes the steps to reproduce the results presented in the paper. To avoid long-run times (due to e.g. the large-scale screen to identify impactful gene perturbations), intermediate results can be downloaded from the referenced data resources mentioned above.

**Please be aware that the following steps will partially overwrite the content in the ``data`` repository, i.e. the notebooks are set up such that their output is saved to same location as in the intermediate data repository. If you would like to simply reproduce the figures in our manuscript using the intermediate data repository, please skip to the respective *Reproducing the paper's figures* section.**

*Note that, solely running all experiments and analyses described in the paper took more than 200 hours of pure computation time on the used hardware setup due to the complexity of the computations and the size of the data sets. The different notebooks and scripts reference the locations of the required files according to the structure provided in our intermediate data repository. Thus, our code can be run without changing any file paths if the intermediate data repository is used. However, if you prefer to store the data differently than the structure used in our data repository, additional changes of the file locations in several scripts will be required. We are happy to assist in these cases.*

### 1. Data preprocessing

#### 1.1. Imaging data from Rohban et al.

The raw image data of the perturbation screen from Rohban et al. (2017) is preprocessed including several filtering steps and nuclear segmentation.
The full preprocessing pipeline can be run via
```
python run.py --config config/preprocessing/full_image_pipeline.yml
```

**Please skip that step if you are using our intermediate data repository and have not download the raw image data from Rohban et al. (2017).**  This is because the script requires the raw image data from Rohban et al. (2017) to be downloaded and the respective content of the ``ilum_corrected`` image files to be located at ``data/resources/images/rohban/illum_corrected``. By default the download script ``scripts/data/download_rohban_data.sh`` will download all data including the ``illum_corrected`` directory to ``image2reg/data/resources/images/rohban/raw`` if run from within the``image2reg`` directory. Please simply copy the downloaded ``illum_corrected`` directory to the ``data/resources/images/rohban/`` directory. Assuming you have downloaded in the intermediate data, you can then rerun the preprocessing as defined above. Please again note that since the raw data from Rohban et al. (2017) is not managed by us, we cannot provide all raw image files directly as part of our intermediate data repository.**

*A version of the output of the preprocessing pipeline, which e.g. contains the segmented single-nuclei images is available in our data repository at ``experiments/rohban/images/preprocessing/full_pipeline``.

<details>
  <summary>Additional remarks</summary>

 If you have run the above mentioned command. Please find the output of the pipeline in a new directory in the same directory. By default everytime ``python run.py`` is run an output directory named according to the timestep of the execution of the command will be created in the outpud directory specificied in the respetive config file provided as the --config argument. All outputs generated as part of the execution of the command will be stored in this directory. This was done to prevent the scripts from overwriting existing data.*

The UNet segmentation masks required for the preprocessing of the two imaging data sets are contained in the intermediate data repository located in the directory defined in the configuration file. If you are not using the intermediate data repository, please specifify the location of the segmentation masks for the images as the ``label_image_input_dir`` attribute in the ``config/preprocessing/full_image_pipeline.yml``. For each raw image the referenced directory must contained a segmentation mask with the same file name.

</details>

----

#### 1.2. Imaging data from JUMP

The raw image data of the perturbation screen of the JUMP consortium by Chandrasekaran et al. (2023) is preprocessed similar to the data from Rohban et al. (2017).
The full preprocessing pipeline can be run via
```
python run.py --config config/preprocessing/full_image_pipeline_jump.yml
```

*A version of the output of the preprocessing pipeline for the JUMP data set is available in our data repository at ``data/experiments/jump/images/preprocessing/full_pipeline``.*

#### 1.3. Gene expression data

Single-cell gene expression data from [Mahdessian et al, 2021](https://www.nature.com/articles/s41586-021-03232-9) was preprocessed as described in the paper using the notebook available in ```notebooks/rohban/ppi/gex_analyses/scgex_preprocessing.ipynb```. A version of the output, preprocessed gene expression data is available in our data repository at ```data/experiments/rohban/gex/scrnaseq/fucci_adata.h5```.

CMap gene signature data from [DepMap, 2021](https://depmap.org/portal/) was preprocessed using the notebook available in ```notebooks/rohban/ppi/gex_analyses/cmap_preprocessing.ipynb```. The input data is also available on our data repository (```data/resources/gex/ccle/CCLE_expression.csv```) but please make sure to reference the data source mentioned above and in the paper appropriately.

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

Once the screen has been run the notebook ```notebooks/rohban/image/screen/screen_analyses_cv_final.ipynb``` can be used to analyze those results and identify the impact gene perturbations. Gene set information data can be obtained as described in the paper or directly from our data repository at ```data/resources/genesets```.


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

*A version of the thereby obtained image embeddings are available in our data repository at ```data/experiments/rohban/images/embeddings/leave_one_target_out```.*


#### 2.3. Analyses of the image embeddings

The analyses of the image embeddings and visualization of their representation can be assessed using the notebook ```notebooks/rohban/image/embedding/image_embeddings_analysis.ipynb```. The cluster analysis and visualization of the inferred image embeddings in the leave-one-target-out evaluation setup can be rerun using the code in ```notebooks/image/embedding/image_embeddings_analysis_loto.ipynb```.


#### 2.4. Analyses of the gene perturbation embeddings

The cluster analyses of the inferred gene perturbation embeddings are performed using the notebook ``notebooks/rohban/image/embedding/gene_perturbation_cluster_analysis.ipynb`` and ``notebooks/rohban/image/embedding/image_embedding_analysis.ipynb``. Gene ontology analyses were performed using the R notebook ``notebooks/rohban/image/embedding/gene_perturbations_go_analyses.Rmd``. Note that the preprocessed morphological profiles are available from the optional data resources but can be obtained by simply removing all features associated to channels other than the DNA channel from profiles available by Rohban et al. (2017). 

---
*To run the referenced .Rmd files, please install R and RStudio following the official installation instructions provided [here](https://posit.co/download/rstudio-desktop/).
The setup used by us to run the experiments described in our paper used RStudio v.1.3.959 and R version 4.0.3. The R notebook ``notebooks/rohban/image/embedding/gene_perturbations_go_analyses.Rmd`` contain all the code to install the required additional R packages.
If you encounter any problems while installing the ``topGO`` package in the R notebook, please make sure that the required binaries are installed as defined in the output e.g. install ``libpng-dev`` via 
```
sudo apt install libpng-dev
```
if the R output identifies that package as missing during the installation.
Please further note that the results were produced using the previously described version of R and RStudio as well as the additional R packages as defined in the notebook ``notebooks/rohban/image/embedding/gene_perturbation_go_analyses_rsessioninfo.txt`` and ``notebooks/rohban/ppi/embeddings/gene_embedding_cluster_analyses_rsessioninfo.txt`` respectively.


*Since the gene sets used in the GO analysis are subject to change, there might be slight differences if other versions of in particular the packages ``clusterProfiler`` (should be version 3.18.1), ``topGO`` (should be version 2.42.0), ``org.Hs.db.eg`` (should be version 3.12.0) are installed.

To facilitate reproducing the results, we also provide the whole R environment in the Github repository as the file ``other/gene_pert_go_analyses.Rdata`` for the R notebook ``notebooks/rohban/image/embedding/gene_perturbations_go_analyses.Rmd`` and ``other/gene_embedding_cluster_data.Rdata`` for the R notebook ``notebooks/rohban/ppi/embeddings/gene_embedding_cluster_analyses.Rmd``, which is refered to in section 4.*
 
---

### 3. Inference and analyses of the gene-gene interactome

#### 3.1. Gene-gene interactome inference

The gene-gene interactome is inferred using a prize-collecting Steiner tree (PCST) algorithm. The required inputs of the algorithm can be obtained using the notebook ```notebooks/rohban/ppi/preprocesssing/inference_preparation_full_pruning.ipynb```. In addition to the described single-cell gene expression data set the notebook requires asn estimate of the human protein-protein interactome and bulk gene expression data from the Cancer Cell Line Encyclopedia as input. The resources of those inputs are listed in the paper.

After the preprocessing of the inputs, a the gene-gene interactome can be inferred using the code available in the notebook ```notebooks/rohban/ppi/inference/interactome_inference_final.ipynb```. The input network to the PCST analyses and the output gene-gene interactome is also directly available as part of the optional data resources.

*Our data repository contains a version of both the preprocessed protein-protein interactome that is input to the PCST analysis (at ```data/experiments/rohban/interactome/preprocessing/cv/ppi_confidence_0594_hub_999_pruned_ccle_abslogfc_orf_maxp_spearmanr_cv.pkl```) as well as the finally output gene-gene interactome (at ```data/experiments/rohban/interactome/inference_results/spearman_sol_cv.pkl```).*

#### 3.2. Analysis of the inferred gene-gene interactome

The R notebook ```notebooks/rohban/ppi/other/go_analysis_pcst_solution.Rmd``` provides the code to evaluate the enrichment of biological processes associated to the mechanotransduction in cells in the inferred gene-gene interactome.

---

### 4. Inference and analyses of the regulatory gene embeddings

Given the previously computed inputs the proposed graph-convolutional autoencoder model can be trained to infer the regulatory gene embeddings within and outside of the leave-target-out evaluation setup described in the paper. The code required to run those experiments is available in ```notebooks/rohban/ppi/embeddings/gae_gene_embs.ipynb```. All required inputs for the analyses are outputs of the previously described steps.

*Additionally, they are also directly available in our data repository. In particular, the gene expression data in ```data/experiments/rohban/gex```, the gene set information in ```data/resources/genesets``` and additional cluster inputs in ```data/experiments/rohban/interactome/cluster_infos```. The results of all conducted experiments including the inferred regulatory gene embeddings that are input to the translation analysis in the leave-one-target-out evaluation setting (see section 5) are available at ```data/experiments/rohban/images/embeddings/leave_one_target_out```.*

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

*The metadata files referenced in the config files were obtained by applying a four-fold stratified cross-validation split to the JUMP data set using the functionalities in the notebook ```notebooks/rohban/other/cv_specific_targets_data_split_jump.ipynb``` but can also be found in our shared data repository at ```data/experiments/jump/images/preprocessing/specific_targets_cv_stratified``` in addition to the output of that step which can be found in ```data/experiments/jump/images/embedding/specificity_target_emb_cv_strat/fold_0```.*

Once trained, we obtain the image embeddings for all considered 175 overexpression conditions via:

```
python run.py --config config/image_embedding/specific_targets/extract_latents/extract_latents_jump_data_resnet_ensemble_specific_targets.yml
```

*The output of that step is located in ```data/experiments/jump/images/embedding/extract_latents_from_rohban_trained``` in our shared data repository.*

Finally, gene perturbation embeddings are obtained as for the Rohban data by averaging the embeddings across conditions. This is done using the output of the executing first the notebook ```notebooks/jump/eda/eda_jump_image_representations.ipynb``` followed by the notebook ```notebooks/jump/embeddings/analyses_jump_embedding_candidates.ipynb```.
*The final gene perturbation embeddings used for the additional validation that are output of the previous step are also available from our shared data repository in ```data/experiments/jump/images/embedding/all_embeddings```.

#### 6.2. Evaluation of the effectiveness of our pipeline

Using the computed gene perturbation embeddings, we evaluate our pipeline on predicting the targets of over novel/unseen 75 overexpression condition in the JUMP data set using the functionalities in the notebook ```notebooks/jump/translation/jump_translation_prediction_final.ipynb```.

---

## Reproducing the paper's figures

To further facilitate the running of our code to reproduce the main figures, we here provide a list of the notebooks used to generate the corresponding panels. If our intermediate data repository was downloaded all notebooks can be run without the need of changing any file locations. Please note that to run the notebooks the required software packages as described in the **Installation/Environmental setup** section need to be installed. Additionally, the R and RStudio is required to run the .Rmd files (R notebooks). Please refer to the final part of section 2.4. of the section **Reproducing the paper results** for details on how to install these.

### Figure 1
- The panels A and B were created without the use of the code.
### Figure 2
- Panel A and B were created without the use of the code.
- Panel C was created using the notebook ``notebooks/rohban/image/screen/screen_analysis_cv_final.ipynb``
- Panel D was created using the notebook ``notebooks/rohban/image/embedding/image_embedding_analysis.ipynb``
- Panel E was created using the notebook ``notebooks/rohban/image/embedding/gene_perturbation_cluster_analysis.ipynb``
### Figure 3
- Panel A was created using the notebook ``notebooks/rohban/ppi/preprocessing/inference_preparation_full_pruning.ipynb`` and visualizing the inferred Prize Collecting Steiner tree that is derived by the notebook ``notebooks/rohban/ppi/inference/interactome_inference/final.ipynb`` and is saved as the file ``spearman_sol_cv.graphml`` via opening it and visualizing it using [Cytoscape]().
- Panel B is created using the notebook ``notebooks/rohban/ppi/embeddings/gene_embedding_clustering.ipynb``.
- Panel C is created using the R notebook ``notebooks/rohban/ppi/embeddings/gene_embedding_cluster_analyses.Rmd``; a compiled version is available as the file ``notebooks/rohban/ppi/embeddings/gene_embedding_cluster_analyses.html``.
### Figure 4
- Panel A was created without the use of the code.
- Panel B was created using the notebook ``notebooks/rohban/translation/mapping/translational_mapping_loto_gridsearch_final.ipynb``
- Panel C was created using the notebook ``notebooks/jump/translation/jump_translation_prediction_final.ipynb``


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


