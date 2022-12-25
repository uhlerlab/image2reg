# Image2Reg: Linking Chromatin Images to Gene Regulation using Genetic Perturbation Screens

**by Daniel Paysan (#), Adityanarayanan Radhakrishnan (#), G.V. Shivashankar (^) and Caroline Uhler (^)**

The repository contains the code for the main methodology and analyses described in our paper: 
 >[*Image2Reg: Linking Chromatin Images to Gene Regulation using Genetic Perturbation Screens (Under Review)*]().
 
 ![](https://github.com/dpaysan/image2reg/blob/389a275421f9d5508685ba0feb30f051085c54b2/imag2reg_pipeline.png)

---

## System requirements

The code has been developed on a system running Ubuntu 20.04. LTS with Python v3.8 installed using a Intel(R) Xeon(R) W-2255 CPU with 3.70GHz, 128GB RAM and a Nvidia RTX 4000 GPU with CUDA v.11.1.74 installed. Note that for setups with less available RAM and/or GPU, parameters like the batch size for the training of the neural networks might have to be adjusted.

## Installation/Environmental setup

To install the code please first clone this repository using
```
git clone https://github.com/uhlerlab/image2reg.git
```

The software was built and tested using Python v3.8. Thus, please next install Python v3.8. While it is theoretically not required, we have used and thus recommend the package manager [miniconda](https://docs.conda.io/en/latest/miniconda.html) to setup and manage the computational environment. To install miniconda please follow he official installation instructions, which can be found [here](https://conda.io/projects/conda/en/stable/user-guide/install/linux.html).

Once miniconda is installed, you can create a conda environment running Python version 3.8 in which the required software packages will be installed via:
```
conda create --name image2reg python==3.8
```

The final step of the installation consists of the installation of additional required packages which can be efficiently done via:
```
conda activate image2reg
pip install -r requirements.txt
```
Note that this installs the requried [Pytorch](https://pytorch.org/) associated packages with GPU accelaration using CUDA v11.1, which we had used to develop and run the code in this repository. 

In case that no GPU is available on the system you are planning to instal the software on please install the required packages without GPU support via: 
```
conda activate image2reg
pip install -r requirements_cpu.txt
```
*Note that without GPU accelaration the run time of the code with respect to the training and evaluation of the neural networks is significantly longer.
Please also refer to the official Pytorch installation guide, which can be found [here](https://pytorch.org/get-started/locally/), in case you encounter any problems regarding the installation of packages such as ``torch, torchvision and torchaudio``.*

If you encounter any problems with the installation of the extended list of the site packages please refer to ``requirements_minimal.txt`` for a list of minimally required software packages and corresponding version numbers that you will need to install either one-by-one or all-at-once via

```
conda activate image2reg
pip install -r requirements_minimal.txt
```
in order to setup a computational environment that can run the provided code. However, note that installing all additional packages as described before is highly recommended to recreate the environment the code was developed in.

In total the estimated installation time is 10-20 minutes depending on the speed of the available internet connection to download the required software packages.

---

## Data resources (Optional)

Note that the following data resources are only required if you want to reproduce the results presented in the paper. In case you want to apply the presented methodology to your own data, please skip this section.

The raw data including the images of the perturbation screen by [Rohban et. al, 2017](https://doi.org/10.7554/eLife.24060), the gene expression and protein-protein interaction data are publicly available from the sources described in the paper. Additional intermediate outputs of the presented analyses including i.a. trained neural network models, the inferred gene-gene interactome, computed image, perturbation gene and regulatory gene embeddings can be downloaded from our [Google Drive here](https://drive.google.com/drive/folders/1bl6YfG8GBpVjgHRIGZW_ycOvJ6r--a5Y?usp=sharing). Note that those intermediate results are optional and were generated as a results of the steps described in the following.

---

## Reproducing the paper results
The following description summarizes the steps to reproduce the results presented in the paper.

### 1. Data preprocessing

#### 1.1. Image data

The raw image data of the perturbation screen is preprocessed including several filtering steps and nuclear segmentation.
The full preprocessing pipeline can be run via
```
python run.py --config config/preprocessing/full_image_pipeline.yml
```
Please edit the config file to specify the location of the raw imaging data of your system.

#### 1.2. Gene expression data

Single-cell gene expression data from [Mahdessian et al, 2021](https://www.nature.com/articles/s41586-021-03232-9) was preprocessed as described in the paper using the notebook available in ```notebooks/ppi/gex_analyses/scgex_preprocessing.ipynb```.

CMap gene signature data from [DepMap, 2021](https://depmap.org/portal/) was preprocessed using the notebook available in ```notebooks/ppi/gex_analyses/cmap_preprocessing.ipynb```. Note that this notebooks assumes that the gene-gene interactome (GGI) had already been inferred. Please see 3. on how to infer the GGI.

*Note that in general in order to run any of the provided jupyter notebooks, please start the jupyter server in the setup computational environment via:*
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
bash scripts/run_screen.sh
```

Note that the path of the first line of the script needs to be adjusted to reflect the home directory of the code base. Additionally, the script assumes that the config files specifying the individual training tasks of the model for the different perturbation targets are available. The notebooks ```notebooks/other/cv_screen_data_splits.ipynb``` and ```notebooks/other/create_screen_configs.ipynb``` provide function efficiently generate the resources required by the script to complete the screen.

Once the screen has been run the notebook ```notebooks/screen/screen_analyses_cv.ipynb``` can be used to analyze those results and identify the impact gene perturbations. Gene set information data can be obtained as described in the paper or directly from the optional data resources.


#### 2.2. Inference of image embeddings

To infer the image embeddings the convolutional neural network is trained on a multi-class classification task to distinguish between the different as impactful identified gene perturbation settings. Thereby, the embeddings of the images in the test sets provide corresponding image embeddings.

The related experiment can be run by calling
```
bash scripts/run_selected_targets.sh
```

As before the path in the first line needs to be adjusted and the required config files as well as data resource are assumed to be available. The data corresponding data is part of the available optional data resources but can also be efficiently generated from the output of the output of image preprocessing step (see step 1.1.) using the notebook ```notebooks/other/cv_specific_targets_data_split.ipynb```.

To obtain the image embeddings in the leave-one-target-out evaluation scheme the related experiments can be performed by calling
```
bash scripts/run_loto_selected_targets.sh
bash scripts/run_extract_loto_latents.sh
```

The required for the experiments are again available as part of the optional data resources or can be efficiently generated using the notebooks ```notebooks/other/create_loto_configs.ipynb``` and ```notebooks/other/loto_data_splits.ipynb```.


#### 2.3. Analyses of the image embeddings

The analyses of the image embeddings and visualization of their representation can be assessed using the notebook ```notebooks/image/embedding/image_embedding_analysis.ipynb```. The cluster analysis and visualization of the inferred image embeddings in the leave-one-target-out evaluation setup can be rerun using the code in ```notebooks/image/embedding/image_embeddings_analysis_loto.ipynb```.


#### 2.4. Analyses of the gene perturbation embeddings

The cluster analyses of the inferred gene perturbation embeddings are performed using the notebook ``notebooks/image/embedding/gene_perturbation_cluster_analysis.ipynb``. Gene ontology analyses were performed using the R notebook ``notebooks/image/embedding/gene_perturbations_go_analyses``. Note that the preprocessed morphological profiles are available from the optional data resources but can be obtained by simply removing all features associated to channels other than the DNA channel from profiles available by Rohban et al. (2017).

---

### 3. Inference and analyses of the gene-gene interactome

#### 3.1. Gene-gene interactome inference

The gene-gene interactome is inferred using a prize-collecting Steiner tree (PCST) algorithm. The required inputs of the algorithm can be obtained using the notebook ```notebooks/ppi/preprocessing/inference_preparation_full_pruning.ipynb```. In addition to the described single-cell gene expression data set the notebook requires asn estimate of the human protein-protein interactome and bulk gene expression data from the Cancer Cell Line Encyclopedia as input. The resources of those inputs are listed in the paper.

After the preprocessing of the inputs, a the gene-gene interactome can be inferred using the code available in the notebook ```notebooks/ppi/inference/interactome_inference_final.ipynb```. The input network to the PCST analyses and the output gene-gene interactome is also directly available as part of the optional data resources.

#### 3.2. Analysis of the inferred gene-gene interactome

The R notebook ```notebooks/ppi/other/go_analysis_pcst_solution.Rmd``` provides the code to evaluate the enrichment of biological processes associated to the mechanotransduction in cells in the inferred gene-gene interactome.

---

### 4. Inference and analyses of the regulatory gene embeddings

Given the previously computed inputs the proposed graph-convolutional autoencoder model can be trained to infer the regulatory gene embeddings within and outside of the leave-target-out evaluation setup described in the paper. The code required to run those experiments is available in ```notebooks/ppi/embedding/gae_gene_embs.ipynb```. All required inputs for the analyses are outputs of the previously described steps but also directly available as part of the optional data resources.

The analysis of the clustering of the inferred gene-gene embedding are also included in that notebook. The R notebook in ```notebooks/ppi/embeddings/gene_embedding_cluster_analyses.Rmd```.

---

### 5. Mapping gene perturbation to regulatory gene embeddings

Finally, the code required to assess the alignment of the inferred regulatory gene and perturbation gene embeddings in the described leave-one-targe-out evaluation setup including the associated meta-analyses are available in the notebook ```notebooks/translation/mapping/translational_mapping_loto_gridsearch.ipynb```.

---

## Application of the pipeline to other imaging-perturbation data

Our method is broadly applicable and provides a general framework to link cell images and gene regulation in genetic perturbation screens. However, the pipeline must be adjusted depending on the input data. The previously described set up as well as the selection of all hyperparameters including sizes of the cell images, architecture of the neural networks and so on should be tuned to the specific use case. The code provided in this repository including the notebooks in addition to the above description of the reproduction of the results presented in our paper should provide a general construct for such analyses, despite the fact that individual hyperparamters need to be selected by the user in dependence on the data.

---

## Credits

If you use the code provided in the directory please also reference our work as follows:

**TO BE ADDED**

If you use the provided data please make sure to also reference the the corresponding raw data resources described in the paper in addition to our work.


