# Image2Reg: Linking Chromatin Images to Gene Regulation using Genetic Perturbation Screens

**by Daniel Paysan (#), Adityanarayanan Radhakrishnan (#), G.V. Shivashankar (^) and Caroline Uhler (^)**

The repository contains the code for the main methodology and analyses described in our paper: 
 >[*Image2Reg: Linking Chromatin Images to Gene Regulation using Genetic Perturbation Screens (Under Review)*](https://github.com/uhlerlab/Image2Reg).
 
 ![](https://github.com/dpaysan/image2reg/blob/389a275421f9d5508685ba0feb30f051085c54b2/imag2reg_pipeline.png)

---

## System requirements

The code has been developed on a system running Ubuntu 20.04. LTS using a Intel(R) Xeon(R) W-2255 CPU with 3.70GHz, 128GB RAM and a Nvidia RTX 4000 GPU with CUDA v.11.1.74 installed.
However, the demo application of our pipeline described in the following only requires a Linux system with at least 10 GB storage and a internet connection and thus a significantly less powerful system to run.

---

## Demonstration of Image2Reg

### Overview
To facilitate the use and testing of our pipeline, we have implemented a demo application that can be used to predict novel, unseen overexpression conditions from chromatin images and is easy to use with minimal software and storage requirements. In particular, our demo application runs depending on the number of input images in as little as 5 minutes and requires only roughly 10GB of storage.

When run, the demo application will perform all required to steps to run our pipeline, i.e. it will
1. Install a minimal software environment containing the required python version 3.8.10 and a few additional python packages.
2. Download the required data to run the inference demonstration of our pipeline like e.g. the pretrained image encoder model used to obtain image embeddings from the chromatin images, as well as example imaging data from [Rohban et al. (2017)]().
3. Preprocess the chromatin images provided by the user for which the pipeline should infer the perturbed gene.
4. Obtain the image and consequently the gene perturbation embedding for the test condition by encoding the images using the pretrained convolutional neural network ensemble image encoder model.
5. Link the gene perturbation embeddings of their corresponding regulatory gene embeddings by training the kernel regression model.
6. Output the 10 genes most likely overexpressed (in decreasing order) in the cells in the user-provided input images.

*Note that we have built the demo to run without a GPU to maximize its compatability.*

#

### Step-by-step guide




#### 1. Perequisites
A linux system is reqired to run the demo. The run time is approximately 10-60 minutes depending on the specifications of the system it is run on.
Further perequisities are described below.

<details>
<summary><b>Bash shell (1 minute)</b></summary>
The software is expected to be run in a bash shell. If your default shell is not a bash shell, please switch to the bash shell in your terminal via
```
bash
```

*Make sure that throughout the step by step guide, whenever you restart a terminal it is running the bash shell, i.e. if the bash shell is not your default shell, always run
``bash`` when you open a new terminal first.*

</details>

#

<details>
<summary><b>Anaconda installation (2 minutes)</b></summary>
 
The only perequisite the demo application has is that the package manager [``Anaconda``](https://docs.anaconda.com/free/) or [``miniconda``](https://docs.conda.io/en/latest/miniconda.html) is installed on your system.
To test if it is install please open a terminal and type in: ``conda``.
If you see an error message saying the command was not found, it is not yet installed.

##### Option 1: Anaconda/Miniconda is not installed

Please open a new terminal on your system (e.g. via the short-cut Ctrl+Alt+T if you are running Ubuntu or by typing in ``terminal`` in the application search of your system).
Then install miniconda via:
```
cd ~/
wget https://repo.anaconda.com/miniconda/Miniconda3-py311_23.5.2-0-Linux-x86_64.sh
bash Miniconda3-py311_23.5.2-0-Linux-x86_64.sh
```

This will start the installer, which will guide you through the installation of miniconda. 
To run the installation using the default setup:
- Press ``enter``, **until** you are asked to agree to the license agreement;
- Type in ``yes`` when asked to accept the license agreement;
- Press enter to use the default installation location;
- Finally type in ``yes`` when asked to run conda init.

If you encounter any issues, please refer to the official installation guide which can be found [here](https://docs.conda.io/en/latest/miniconda.html#installing).

*Please note that after the installation you will have to close the terminal and open a **new** one before continuing with the next steps.
Please open a new terminal on your system as described above.*



##### Option 2: Anaconda/Miniconda is installed</b></summary>
 
Please that conda was initialized properly via running
```conda init```
</details>

#

<details>
<summary><b>Input images (1 minute)</b></summary>
 
To run the demo application to apply our Image2Reg pipeline the user needs to provide two different imaging inputs:
- **Raw chromatin images**: Single-channel (i.e. black-white) images of cells stained for the chromatin similar to those from Rohban et al. (2017). The demo will automatically download a number of example images for 10 perturbation conditions from the Rohban et al. (2017) data set.
- **Nuclear segmentation masks**: Single-channel (i.e. black-white) images of the same size as the raw chromatin images that contain the nuclear segmentation masks. The format of the segmentation images follows the standard convention where all pixels that mark the mask of one nucleus are given the same but unique positive integer value. The background is assigned a pixel value of 0. Together with the raw chromatin images, the demo will also download the corresponding nuclear segmentation masks for these images. To associate a nuclear mask with its corresponding raw chromatin image, the demo requires the two image files to be named exactly the same.

For an initial try of our pipeline, we recommend proceeding with the Step-by-Step guide and use the example images our pipeline will download.

</details>

#

#### 2. Clone the repository (3 minutes)

If the perequisites are satisfied, please clone this repository by running the following command in a **new** terminal.
```
git clone https://github.com/uhlerlab/image2reg.git
```

#

#### 3. Run the demo application on user-provided imaging data
To finally run our demo application, run

```
cd image2reg
source scripts/demo/image2reg_demo_new_data.sh
```
in the same terminal.

This will trigger the following processes in the following.

<details>
<summary><b>Installation of the conda environmen and download of the required and sample data (no user interaction required)</b></summary>
 
First, a new conda environment called ``image2reg_demo`` is installed that contains all software packages required to run the code.
Second, a directory called ``test_data`` is downloaded from the DOI *.....* and extracted within the image2reg repository.
In addition to required e.g. pretrained model files, the directory ``test_data/sample_data`` contains raw chromatin images and corresponding nuclear segmentation masks for 10 perturbation conditions from Rohban et al. (2017), that can be used to test our demo.

</details>

#

<details>
<summary><b>Deposition of the image inputs</b></summary>
 
After the download, the demo application will stop and ask the user to confirm that the images input to our pipeline where deposited in the appropriate directories, namely all raw chromatin images in ``test_data/UNKNOWN/images/raw/plate`` and the respective nuclear segmentation masks in ``test_data/UNKNOWN/images/unet_masks/plate``.

*To test our demo application, you can for instance copy some images from the any condition in the ``test_data/sample_data`` to the respective directories. Please ensure that you copy for each selected raw chromatin image e.g. taken from ``test_data/sample_data/JUN/raw`` the associated (i.e. equally named) nuclear segmentation masks that can be found ``test_data/sample_data/JUN/unet_masks``.*

Please deposit the raw chromatin images and associated segmentation masks in the respective directories and confirm it via typing in 
```yes``` 
and hitting enter.

</details>

#

<details>
<summary><b>Prediction output (no user interaction required) </b></summary>
The demo will then perform all further inference steps described in the *Overview* section for the user-specified image data set and output the 10 genes that were most likely overexpressed in the cells captured in the data set (in decreasing order).

This completes the demo application.

</details>

#

#### *4. Advanced run settings/developer options (Optional)*
Our demo application can also be run with two additional arguments.
1. ``--environment``:    This argument can be used if one would like to specify a pre-existing conda environment that is supposed to be used to run the demo application. By default, if the argument is not provided a new conda environment will be setup as part of the demo application called ``image2reg_demo`` in which all required python packages will be installed and that is used to run our code.
2. ``--help``:    This argument can be used to obtain help on the usage of our demo

#

#### *5. Reproducing the study results using the demo application (Optional)*
In addition to above described demo application that applies our pipeline to user-provided image inputs, we have also developed a second application that reproduces the results of the leave-one-target-out cross-validation evaluation described in our manuscript for five selected perturbation condtions, namely *BRAF, JUN, RAF1, SMAD4 and SREBF1*. 

To run our second demo application, simply run
```
source scripts/demo/image2reg_demo.sh --condition <CONDITION>
```
where you replace ``<CONDITION>`` with either ``BRAF``, ``JUN``, ``RAF1``, ``SMAD4`` or ``SREBF1``.

We also provide some additional functionalities which are described in more detail in the following.

#

<details>
   <summary><b>Advanced run settings/developer options</b></summary>
 In addition to specifying for which overexpression condition our pipeline should be run, there are three additional arguments that one can be used for the demo application:

 1. ``--random``:    If this argument is provided, the Image2Reg pipeline is run such that the inferred gene perturbation and regulatory gene embeddings are permuted prior the kernel regression is fit which eventually predicts the overexpression target. This recreates the random baseline described in our manuscript. Using this argument, you will observe a deteriated prediction performance of our pipeline which is expected.
 2.  ``--environment``:    This argument can be used if one would like to specify a pre-existing conda environment that is supposed to be used to run the demo application. By default, if the argument is not provided a new conda environment will be setup as part of the demo application called ``image2reg_demo`` in which all required python packages will be installed and that is used to run our code.
 3.   `--help``:    This argument can be used to obtain help on the usage of our demo and in particular summarizes the meaning of the different arguments (i.e. ``--condition``, ``--random``, ``--environment``) described before.


Note that any of these arguments except for the ``--help`` command can be combined to select the setup for the demo application that you like.
As an example, if you would like to use a pre-existing conda environment e.g. ``imag2reg_demo`` and reproduce a *random* baseline prediction for our pipeline for the overexpression condition *SREBF1* run
```
source scripts/demo/image2reg_demo.sh --environment image2reg_demo --condition SREBF1 --random
```

</details>

#

#### 6. Final remarks

We appreciate you testing our code and look forward to the amazing applications we hope our solution will help to create.
If you would like to reproduce all results of the paper from scratch please refer to [this guide](reproducibility_guide.md). Please note that this will substantially larger computing resources and can take up to 300 hours of computation time. 


#

### Troubleshooting


---

## Changelog

- **September 6th, 2023.**&emsp;We have expanded the demo to enable running our pipeline on image data provided by the user using the models pretrained on the imaging data from Rohban et al. (2017) to facilitate the adaption of our pipeline to new imaging data sets.
- **August 18th, 2023.**&emsp;We have added a novel demonstration of our pipeline that can be easily run without the need of even previously installing the coding environment and/or downloading any data. The demo can be used to run our pipeline in the inference mode, i.e. we provide a pretrained version of the pipeline but show how given images of five selected OE conditions it predicts the corresponding target genes out-of-sample (no information regarding these were used to setup the pipeline as described in the paper).
- **August 2nd, 2023.**&emsp;On *July 17th 2023* the external ``hdbscan`` package broke due to number of changes of the name resolution. As a consequence the installation of any version of the package including the version 0.8.27 used in our software package was no longer able to be installed, leading to our installation script to no longer be able to run completely ([see here for more information](https://github.com/scikit-learn-contrib/hdbscan/issues/600)). We have updated the requirements file of our package to install the hotfix implemented in version hdbscan v.0.8.33. While we could not have anticipated such an issue suddenly occuring, we apologize for the inconvenience this may have caused. We have tested the updated installation script but please let us know if you encounter any issue with the installation on your end and/or running our code.

---


## Questions/Issues

If you encounter any problems with setting up the software and/or need assistance with adapting the code to run it for your own data set, feel very free to open a respective issue. We will do our very best to assist you as quickly as possible.

---

## Credits

If you use the code provided in the directory please also reference our work as follows:

**TO BE ADDED**

If you use the provided data please make sure to also reference the the corresponding raw data resources described in the paper in addition to our work.


