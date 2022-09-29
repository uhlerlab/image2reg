# Image2Reg: Linking Chromatin Images to Gene Regulation using Genetic Perturbation Screens

The repository contains the code for the main methodology and analyses described in our paper: 
 >[*Image2Reg: Linking Chromatin Images to Gene Regulation using Genetic Perturbation Screens*]().

---

## System requirements

The code has been developed on a system running Ubuntu 20.04. LTS running a Intel(R) Xeon(R) W-2255 CPU with 3.70GHz, 128GB RAM and a Nvidia RTX 4000 GPU. Note that for setups with less available RAM and/or GPU parameters like the batch size for the training of the neural networks might have to be adjusted.

## Installation

To install the code clone the repository and install the required software libraries and packages listed in the *requirements.txt* file:
```
git clone https://github.com/dpaysan/image2reg.git
conda create --name image2reg --file requirements.txt
conda activate image2reg
```
---

## Data resources (Optional)

Note that the following data requirements only apply if you want to reproduce the results presented in the paper. If you want to apply the presented methodology to your own data, please skip this section.

The raw data including the images of the perturbation screen by [Rohban et. al, 2017](), the gene expression and protein-protein interaction data are publicly available from the sources described in the paper. Additional intermediate outputs of the presented analyses including i.a. trained neural network models, the inferred gene-gene interactome, computed image, perturbation gene and regulatory gene embeddings can be downloaded from [](). Note that those intermediate results are optional and were generated as a results of the steps described in the following.

---

## Reproducing the paper results
The following description summarizes the steps to reproduce the results presented in the paper. The presented steps can similarly run using your own data.

### 1. Data preprocessing

#### 1.1. Image data

The raw image data of the perturbation screen is preprocessed including several filtering steps and nuclear segmentation.
The full preprocessing pipeline can be run via
```
python run.py --config config/preprocessing/full_image_pipeline.yml
```
Please edit the config file to specify the location of the raw imaging data of your system.

#### 1.2. Gene expression data

Single-cell gene expression data from [Mahdessian et al, 2021]() was preprocessed as described in the paper using the notebook available in ```notebooks/ppi/gex_analyses/scgex_preprocessing.ipynb```.

CMap gene signature data from [DepMap, 2021]() was preprocessed using the notebook available in ```notebooks/ppi/gex_analyses/cmap_preprocessing.ipynb```. Note that this notebooks assumes that the gene-gene interactome (GGI) had already been inferred. Please see 3. on how to infer the GGI.

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

The analyses of the image embeddings and visualization of their representation can be assessed using the notebook ```notebooks/image/embedding/image_embedding_analysis.ipynb```.


#### 2.4. Analyses of the gene perturbation embeddings

The cluster analyses of the inferred gene perturbation embeddings are performed using the notebook ``notebooks/image/embedding/gene_perturbation_cluster_analysis.ipynb``. Gene ontology analyses were performed using the R notebook ``notebooks/image/embedding/gene_perturbations_go_analyses``. Note that the preprocessed morphological profiles are available from the optional data resources but can be obtained by simply removing all features associated to channels other than the DNA channel from profiles available by Rohban et al. (2017).

---

### 3. Inference and analyses of the gene-gene interactome

#### 3.1. Gene-gene interactome inference

#### 3.2. Analysis of the inferred gene-gene interactome

---

### 4. Inference and analyses of the regulatory gene embeddings

---

### 5. Mapping gene perturbation to regulatory gene embeddings

---

## Credits

If you use provided data please make sure to reference the required papers for any external resources and our work. If you use the code provided in the directory please also reference our work as follows:


