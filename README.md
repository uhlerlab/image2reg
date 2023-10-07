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
To facilitate the use and testing of our pipeline, we have implemented a demo application that can be used to predict novel, unseen overexpression conditions from chromatin images and is easy to use with minimal software and storage requirements. In particular, our demo application runs (depending on the number of input images) in as little as 5 minutes and requires only roughly 10GB of storage.

When run, the demo application will perform all required to steps to run our pipeline, i.e. it will
1. Install a minimal software environment containing the required python version 3.8.10 and a few additional python packages.
2. Download the required data to run the inference demonstration of our pipeline.
3. Preprocess the chromatin images provided by the user for which the pipeline should infer the perturbed gene.
4. Obtain the image and consequently the gene perturbation embedding for the test condition by encoding the images using the pretrained convolutional neural network ensemble image encoder model.
5. Link the gene perturbation embeddings of their corresponding regulatory gene embeddings by training the kernel regression model.
6. Output the 10 genes most likely overexpressed (in decreasing order) in the cells in the user-provided input images.

#

### Step-by-step guide

#### 1. Perequisites
A Linux system is required to run the demo.

##### Bash shell
To run the commands described in this guide, you need a bash shell.
To activate a bash shell after opening a terminal (e.g. via the short-cut Ctrl+Alt+T if you are running Ubuntu or by typing in ``terminal`` in the application search of your system), type in
```
bash
```

<details>
          <summary><b>
           Click here if you see the output: "command "bash" not found".
          </b></summary>
 
 Please install ``bash`` as described in the output of your system e.g. via
 ```
 sudo apt-get update
 sudo apt-get install bash
 ```
</details>

#

##### Anaconda installation
The package manager [``Anaconda``](https://docs.anaconda.com/free/) or [``miniconda``](https://docs.conda.io/en/latest/miniconda.html) needs to be installed on your system.
To test if it is installed, open a terminal on your system and type in
```
conda
```

<details>
 <summary><b>Click here if the command "conda" not found</b></summary>

If the command ``conda`` was not found, Anaconda or Miniconda is not installed on your system.
Please open a **new** terminal on your system.
Then install miniconda via:
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```

<!--This will start the installer, which will guide you through the installation of miniconda. 
To run the installation using the default setup:
- Press ``enter``, **until** you are asked to agree to the license agreement by typing in yes;
- Type in ``yes`` when asked to accept the license agreement;
- Press enter to use the default installation location;
- Finally type in ``yes`` when asked to run conda init.
-->

If you encounter any issues, please refer to the official installation guide which can be found [here](https://docs.conda.io/en/latest/miniconda.html#installing).

> [!WARNING]
> You need to close the terminal and open a **new** one to complete the installation

</details>

Make sure conda is initialized appropriately in your shell via typing
```
bash
conda init bash
source ~/.bashrc
```

#

#### 2. Clone the repository

If the perequisites are satisfied, please clone this repository by running the following command in a **new** terminal.
```
git clone https://github.com/uhlerlab/image2reg.git
```

#

#### 3. Running the demo application

There are three versions of our Image2Reg demo application we have developed. 
- [**Image2Reg for test inputs**](test_demo.md):   This variant runs our demo with default parameters and example inputs to quickly verify its functionality.
- [**Image2Reg for user-provided inputs**](user_demo.md):  This variant enables the application of our pipeline to user-provided input images.
- [**Image2Reg for reproducibility**](reproducibility_demo.md): This variant reproduces the results of the leave-one-target-out cross-validation for five selected perturbation conditions described in our paper.

**Please click on the name of the version you would like to run and follow the instructions.**

> [!NOTE]
> We recommend to first run the variant of our Image2Reg pipeline using test inputs before running it with user-provided inputs.


<!--
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

> [!NOTE]
> For an initial try of our pipeline, we recommend proceeding with the Step-by-Step guide and use the example images our pipeline will download.

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
Second, a directory called ``test_data`` is downloaded from the [DOI 10.5281/zenodo.8354979](https://doi.org/10.5281/zenodo.8354979) and extracted within the image2reg repository.
In addition to required e.g. pretrained model files, the directory ``test_data/sample_data`` contains raw chromatin images and corresponding nuclear segmentation masks for 10 perturbation conditions from Rohban et al. (2017), that can be used to test our demo.

> [!WARNING]
> Please do not interrupt the download of the data directories and their extraction, i.e. unzipping. This could result in corrupt files which will break the demo application. If you encounter any problems indicating that some files were not found, please remove the ``test_data`` or the ``demo`` directory and restart the demo to redownload the required data.

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

> [!IMPORTANT]
> To run the code please ensure that your working directory is ``image2reg``. The working directory can be changed via the ``cd`` command.

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
   <summary><b>Advanced run settings/developer options for the demo to partially reproduce the study results</b></summary>
 In addition to specifying for which overexpression condition our pipeline should be run, there are three additional arguments that one can be used for the demo application that is used to reproduce our results of our study for the selected perturbation conditions:

 1. ``--random``:    If this argument is provided, the Image2Reg pipeline is run such that the inferred gene perturbation and regulatory gene embeddings are permuted prior the kernel regression is fit which eventually predicts the overexpression target. This recreates the random baseline described in our manuscript. Using this argument, you will observe a deteriated prediction performance of our pipeline which is expected.
 2.  ``--environment``:    This argument can be used if one would like to specify a pre-existing conda environment that is supposed to be used to run the demo application. By default, if the argument is not provided a new conda environment will be setup as part of the demo application called ``image2reg_demo`` in which all required python packages will be installed and that is used to run our code.
 3.   ``--help``:    This argument can be used to obtain help on the usage of our demo and in particular summarizes the meaning of the different arguments (i.e. ``--condition``, ``--random``, ``--environment``) described before.


Note that any of these arguments except for the ``--help`` command can be combined to select the setup for the demo application that you like.
As an example, if you would like to use a pre-existing conda environment e.g. ``imag2reg_demo`` and reproduce a *random* baseline prediction for our pipeline for the overexpression condition *SREBF1* run
```
source scripts/demo/image2reg_demo.sh --environment image2reg_demo --condition SREBF1 --random
```

</details>

#

#### 6. Concluding remarks

We appreciate you testing our code and look forward to the amazing applications we hope our solution will help to create.
If you would like to reproduce all results of the paper from scratch, please refer to [this guide](test_protocol.md). Please note that this will substantially larger computing resources and can take over 1000 hours of computation time and generates roughly 2TB of data!
If you would like to reproduce the figures of our manuscript, please refer to [this guide](figure_reproc_guide.md) which also contains instruction to download all the data we have generated during all analyses from DOI-assigned data archive.


#

-->

#

### Troubleshooting/Support

In the enclosed table we summarize any error messages output by the demo if it is not used as intended and their meaning respectively how these can be resolved.
If you encounter any other errors, please open an issue in this repository and we will extend the list accordingly.

<details>
 <summary>
  <b>Table: Common Errors and Solutions</b>
 </summary>


<font size=8>
 
| Problem | Error Message(s) | Cause | Solution |
| --- | --- | --- | --- |
| **Empty input directory** | *The directory test_data/UNKNOWN/ images/raw/plate is empty.* | The demo requires the raw chromatin images for which the perturbed gene is supposed to be predicted to be located in the specified directory. | Please deposit the raw chromatin images in the directory ``test_data/UNKNOWN/images/raw/plate`` and restart the demo |
| **Empty nuclear mask directory** | *The directory test_data/UNKNOWN/ images/unet_masks/plate is empty.* | The demo requires the nuclear segmentation masks corresponding to the input raw chromatin images (i.e. the images located in ``test_data/UNKNOWN/images/raw/plate) to be located in the specified directory. | Please deposit the segmentation mask images in the directory ``test_data/UNKNOWN/images/unet_masks/plate`` and restart the demo.|
| **Missing/Wrong segmentation mask** | *FileNotFoundError: [Errno 2] No such file or directory* | The demo application requires for each raw chromatin image located in ``test_data/UNKNOWN/images/raw/plate`` a respective nuclear segmentation mask to be located in ``test_data/UNKNOWN/images/unet_masks/plate`` which has the same file name as the corresponding raw chromatin image and satifies the criteria described in the Perequisites section.  The error message occurs if for any raw image the corresponding mask was not found. | Please make sure that all mask images are deposited in the before mentioned directory and restart the demo |
| **Malformed mask image** | *`Cannot access <...>: No such file or directory* or *TypeError: Non-integer label_image types are ambiguous.* | The provided mask images need to satisfy the following criteria: a) a nuclear mask image is single-channel (black-white) image of the same dimensions as the corresponding raw chromatin image and b) each pixel is assigned an integer value where the background is assigned the value 0 and all other pixels get the value equal to the unique integer ID of the nucleus for which they mark the respective mask. Such nuclear mask images are e.g. the output of the function ``[skimage.measure.label](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label) | Please make sure you provide appropriate nuclear mask images in the ``test_data/UNKNOWN/images/unet_masks/plate directory and restart the demo. |
| **Missing directory or files** | *Cannot access <...>: No such file or directory.* | This error is most likely caused due to an malformed ``test_data`` directory likely due to an incomplete extraction or download of the data when the demo is run for the first time. | Please delete the ``test_data`` directory completely and restart the demo which will redownload the directory. Please make sure to not interrupt the download or extraction process but run the demo until it asks you to confirm that the input data has been deposited in the correct directories to avoid this error. |
| **Missing conda environment** | *Provided conda environment not found.* | This error only occurs if the demo is run with the ``--environment`` argument and a non-existing conda environment is provided. | Please make sure that the conda environment you provide exists on your system or simply run the demo without the ``--environment`` argument to safely install a new conda environment that contains all required software packages.|
| **Python module not found** | *ModuleNotFoundError: No module named '<module>'.* | This error occurs if the conda environment used to run the demo does not contain all the required python packages. If you have run the demo by specifying the environment via the ``--environment`` argument, please make sure that the provided conda environment contains all package listed in the file ``requirements/demo/requirements_demo.txt``. If you ran the demo without the ``--environment`` the newly installed conda environment is ensured to contain all packages, if the installation was successful and conda was appropriately initiliazed as described in the Perequisites section. |  Please run ``conda init`` in the terminal. Next run ``pip cache purge`` to remove any potentially malformed cached python packages and then restart our demo **without** providing the ``--environment`` argument to perform a fresh install of the conda environment used to run our demo. |
| **No or just one nuclei is found** | *ValueError: Empty data passed with indices specified.* or *ValueError: Found array with 1 sample(s)[...] while aminimum of 2 is required.* | Your provided input images were found to contain less than two nuclei. Please note that might be due to the used filter settings in our image preprocessing. | Please ensure that your input images contain at least two nuclei and the filters for the cell size and shape defined in the file ``config/demo/preprocessing/full_image_pipeline_new_target.yml`` are appropriate for the resolution and cell size of the images. Our choices are selected for the 20x images of U2OS cells from the Rohban et al. (2017) or the JUMP-CP data set. If your images/nuclei are of different resolution or size, you might want to adjust in particular the minimal/maximum nuclear area (``min_area`` and ``max_area``), the maximal area of the bounding box (``max_bbarea``), the maximum eccentricity (``max_eccentricitiy``), minimal solidity (``min_solidity``) and the minimal aspect ratio (``min_aspect_ratio``). All quantities are given in terms of pixels. For larger images of higher resolution and/or larger cells increase e.g. the maximum values for the area and the area of the bounding box. |

</font>
</details>

<!--
- **Empty input directory**:  The demo exits with the error message ``The directory test_data/UNKNOWN/images/raw/plate is empty.`` The demo requires the raw chromatin images for which the perturbed gene is supposed to be predicted to be located in the specified directory. Please deposit the raw chromatin images in the directory ``test_data/UNKNOWN/images/raw/plate`` and restart the demo;
- **Empty nuclear mask directory**:  The demo exits with the error message ``The directory test_data/UNKNOWN/images/unet_masks/plate is empty.`` The demo requires the nuclear segmentation masks corresponding to the input raw chromatin images (i.e. the images located in ``test_data/UNKNOWN/images/raw/plate) to be located in the specified directory. Please deposit the segmentation mask images in the directory ``test_data/UNKNOWN/images/unet_masks/plate`` and restart the demo;
- **Missing/Wrong segmentation mask**:  The demo exits with the error message: ``FileNotFoundError: [Errno 2] No such file or directory``. The demo application requires for each raw chromatin image located in ``test_data/UNKNOWN/images/raw/plate`` a respective nuclear segmentation mask to be located in ``test_data/UNKNOWN/images/unet_masks/plate`` which has the same file name as the corresponding raw chromatin image and satifies the criteria described in the Perequisites section. The error message occurs if for any raw image the corresponding mask was not found. Please make sure that all mask images are deposited in the before mentioned directory and restart the demo;

- **Malformed mask image**:  The demo exits with error messages such as ``ValueError: Label and intensity image shapes must match, except for channel (last) axis.`` or ``TypeError: Non-integer label_image types are ambiguous
``. The provided mask images need to satisfy the following criteria: a) a nuclear mask image is single-channel (black-white) image of the same dimensions as the corresponding raw chromatin image and b) each pixel is assigned an integer value where the background is assigned the value 0 and all other pixels get the value equal to the unique integer ID of the nucleus for which they mark the respective mask. Such nuclear mask images are e.g. the output of the function ``[skimage.measure.label](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label)``.
- **Missing directory or files**: The demo exits with an error message: ``Cannot access <...>: No such file or directory``. This error is most likely caused due to an malformed ``test_data`` directory likely due to an incomplete extraction or download of the data when the demo is run for the first time. Please delete the ``test_data`` directory completely and restart the demo which will redownload the directory. Please make sure to not interrupt the download or extraction process but run the demo until it asks you to confirm that the input data has been deposited in the correct directories to avoid this error.
- **Missing conda environment**:  The demo exits with the error message ``Provided conda environment not found.`` This error only occurs if the demo is run with the ``--environment`` argument and a non-existing conda environment is provided. Please make sure that the conda environment you provide exists on your system or simply run the demo without the ``--environment`` argument to safely install a new conda environment that contains all required software packages.

- **Python module not found**:  The demo exits with the error message ``ModuleNotFoundError: No module named '<module>'``. This error occurs if the conda environment used to run the demo does not contain all the required python packages. If you have run the demo by specifying the environment via the ``--environment`` argument, please make sure that the provided conda environment contains all package listed in the file ``requirements/demo/requirements_demo.txt``. If you ran the demo without the ``--environment`` the newly installed conda environment is ensured to contain all packages, if the installation was successful and conda was appropriately initiliazed as described in the Perequisites section. Please run ``conda init`` in the terminal. Next run ``pip cache purge`` to remove any potentially malformed cached python packages and then restart our demo **without** providing the ``--environment`` argument to perform a fresh install of the conda environment used to run our demo.
- **No or just one nuclei is found**: The demo exists with an error message such as ``ValueError: Empty data passed with indices specified`` or ``ValueError: Found array with 1 sample(s)[...] while aminimum of 2 is required.``. The demo requires at least two nuclei to be contained in the input images. Please make sure that your input images contain at least two nuclei. Please note that you might also have to adjust the filters for the cell size and shape defined in the file ``config/demo/preprocessing/full_image_pipeline_new_target.yml``. In particular you might want to adjust the minimal/maximum nuclear area (``min_area`` and ``max_area``), the maximal area of the bounding box (``max_bbarea``), the maximum eccentricity (``max_eccentricitiy``), minimal solidity (``min_solidity``) and the minimal aspect ratio (``min_aspect_ratio``). All quantities are given in terms of pixels. For larger images of higher resolution and/or larger cells increase e.g. the maximum values for the area and the area of the bounding box.
-->

---

## Changelog

<details>
 <summary><b>September 6th, 2023.</b></summary>
 
 We have expanded the demo to enable running our pipeline on image data provided by the user using the models pretrained on the imaging data from Rohban et al. (2017) to facilitate the adaption of our pipeline to new imaging data sets.
</details>

<details>
 <summary><b>August 18th, 2023.</b></summary>
 
We have added a novel demonstration of our pipeline that can be easily run without the need of even previously installing the coding environment and/or downloading any data. The demo can be used to run our pipeline in the inference mode, i.e. we provide a pretrained version of the pipeline but show how given images of five selected OE conditions it predicts the corresponding target genes out-of-sample (no information regarding these were used to setup the pipeline as described in the paper).
</details>

<details>
 <summary><b>August 2nd, 2023.</b></summary>
 
 On *July 17th 2023* the external ``hdbscan`` package broke due to number of changes of the name resolution. As a consequence the installation of any version of the package including the version 0.8.27 used in our software package was no longer able to be installed, leading to our installation script to no longer be able to run completely ([see here for more information](https://github.com/scikit-learn-contrib/hdbscan/issues/600)). We have updated the requirements file of our package to install the hotfix implemented in version hdbscan v.0.8.33. While we could not have anticipated such an issue suddenly occuring, we apologize for the inconvenience this may have caused. We have tested the updated installation script but please let us know if you encounter any issue with the installation on your end and/or running our code.
</details>

---

## Reproducing the paper's results (Advanced options) 

If you would like to reproduce all results of the paper from scratch, please refer to [this guide](test_protocol.md). Please note that this will require substantially larger computing resources and the described analyses can take over 1000 hours of computation time while generating roughly 2TB of data!
If you would like to reproduce the figures of our manuscript, please refer to [this guide](figure_reproc_guide.md) which also contains instruction to download all the data we have generated during all analyses from DOI-assigned data archive.

---

## Questions/Issues

If you encounter any problems with setting up the software and/or need assistance with adapting the code to run it for your own data set, feel very free to open a respective issue. We will do our very best to assist you as quickly as possible.

---

## Credits

If you use the code provided in the directory please also reference our work as follows:

**TO BE ADDED**

If you use the provided data please make sure to also reference the the corresponding raw data resources described in the paper in addition to our work.


