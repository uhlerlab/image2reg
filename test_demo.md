# Demo application using test inputs

## Overview

This variant of our demo application uses our Image2Reg pipeline to perform to predict the genes overexpressed in the cells based on two test images from the data set from [Rohban et al. (2017)](https://doi.org/10.7554/eLife.24060). The test images contain cells were BRAF was overexpressed. This variant of the demo is intended to quickly test the functionality and to familiarize oneself with the outputs of the application. We recommend running this variant first, before following the instructions of the [guide here](user_demo.md) that show how you can use our application to perform inference on your imaging data.

---

## Step-by-step guide

### 1. Activating the bash shell
Please open a new terminal and activate the bash shell via
```
bash
```

### 2. Set image2reg as your working directory
Please navigate to the location where you cloned the github repository to.
If you have followed our previous instructions this can be done via typing

```
cd ~/image2reg
```
 in the terminal.

 ### 3. Running the demo application 

 Run the demo application via typing in
 ```
source scripts/demo/image2reg_demo_new_data.sh --image_dir test_data/sample_data/TEST/raw --mask_dir test_data/sample_data/TEST/unet_masks
```

This will run the complete demo. No further user-interaction is required.
Please refer to the following section for more information on the triggered processes.

> [!IMPORTANT]
> To run the code please ensure that your working directory is ``image2reg``. The working directory can be changed via the ``cd`` command.

---

## Processes triggered by the demo application

### Demo preparation
First, a new conda environment called ``image2reg_demo`` is installed that contains all software packages required to run the code.
Second, a directory called ``test_data`` is downloaded from the [DOI 10.5281/zenodo.8354979](https://doi.org/10.5281/zenodo.8354979) and extracted within the image2reg repository.
The directory also contains the test input images (i.e. the raw chromatin images and the corresponding nuclear segmentation) mask our pipeline is applied to.
These test images are located in ``test_data/sample_data/TEST`` and are taken from the data set from Rohban et al. (2017) and contain cells from the overexpression of **BRAF**.

> [!WARNING]
> Please do not interrupt the download of the data directories and their extraction, i.e. unzipping. This could result in corrupt files which will break the demo application. If you encounter any problems indicating that some files were not found, please remove the ``test_data`` or the ``demo`` directory and restart the as described in above.

#

### Application of our Image2Reg
 
The demo will then perform all further the following inference steps for the two test input images and output the 10 genes that were most likely overexpressed in the cells captured in the data set (in decreasing order).

1. Preprocess the chromatin images provided by the user for which the pipeline should infer the perturbed gene.
2. Obtain the image and consequently the gene perturbation embedding for the test input images by encoding the images using the pretrained convolutional neural network ensemble image encoder model.
3. Link the gene perturbation embeddings of their corresponding regulatory gene embeddings by training the kernel regression model.
4. Output the 10 genes most likely overexpressed (in decreasing order) in the cells in the test input images.

> [!NOTE]
> The true gene target that was overexpressed in the test images was BRAF

---

## Troubleshooting

In case you run into any errors please also consult the following section and the [Troubleshooting section in our README file](README.md#Troubleshooting/Support).


