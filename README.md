# Image2Reg: Linking Chromatin Images to Gene Regulation using Genetic Perturbation Screens

---

The repository contains the code for the main methodology and analyses described in our paper: *Image2Reg: Linking Chromatin Images to Gene Regulation using Genetic Perturbation Screens*[]().

---

## System requirements

The code has been developed on a system running Ubuntu 20.04. LTS running a Intel(R) Xeon(r) W-2255 CPU with 3.70GHz, 128GB RAM and a Nvidia RTX 4000 GPU. Note that for configurations with lower RAM and GPU configurations parameters like the batch size for the training of the neural networks might have to be adjusted.

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

## Reproducing the results
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

### 2. Inference of image and gene perturbation gene

#### 2.1. Identification of impactful gene perturbation

To identify the impact gene perturbations we ran a large-scale screen assessing the performance of proposed convolutional neural network architecture on the task of distinguishing between the perturbed and control cells using the chromatin image inputs obtained as a result of the preprocessing (see 1.1.).

The screen can be automatically run by running 
```
bash scripts/run_screen.sh
```

Note that the path of the first line of the script needs to be adjusted to reflect the home directory of the code base. Additionally, the script assumes that the config files specifying the individual training tasks of the model for the different perturbation targets are available. The notebooks ```notebooks/other/cv_screen_data_splits.ipynb``` and ```notebooks/other/create_screen_configs.ipynb``` provide function efficiently generate the resources required by the script to complete the screen.


#### 2.2. Inference of image embeddings



#### 2.3. Inference of gene perturbation embeddings


---

### 3. Inference of the gene-gene interactome

---

### 4. Inference of the regulatory gene embeddings

---

### 3. Mapping gene perturbation to regulatory gene embeddings

---

## Credits
