# Image2Reg: Linking Chromatin Images to Gene Regulation using Genetic Perturbation Screens

**by Daniel Paysan (#), Adityanarayanan Radhakrishnan (#), G.V. Shivashankar (^) and Caroline Uhler (^)**

The repository contains the code for the main methodology and analyses described in our paper: 
 >[*Image2Reg: Linking Chromatin Images to Gene Regulation using Genetic Perturbation Screens (Under Review)*](https://github.com/uhlerlab/Image2Reg).
 
 ![](https://github.com/dpaysan/image2reg/blob/389a275421f9d5508685ba0feb30f051085c54b2/imag2reg_pipeline.png)

---

## System requirements

The code has been developed on a system running Ubuntu 20.04. LTS using a Intel(R) Xeon(R) W-2255 CPU with 3.70GHz, 128GB RAM and a Nvidia RTX 4000 GPU with CUDA v.11.1.74 installed. Note that for setups with less available RAM and/or GPU, parameters like the batch size for the training of the neural networks might have to be adjusted.

---

## Demonstration of Image2Reg

### Overview
To facilitate the use and testing of our pipeline, we have implemented an easy demonstration of how our pipeline can be used to predict novel, unseen overexpression conditions from chromatin images once trained.
We here provide a brief overview of the functionality of the demo application for which we provide a detailed step-by-step guide below. 

In particular, it will:
1. Install a minimal software environment containing the required python version 3.8.10 and a few additional python packages. Note that these packages are only a subset of all packages used to create the code contained in this repository. If you would like to install all packages, please refer to the next section in this documentation.
2. Download the required data to run the inference demonstration of our pipeline like e.g. the pretrained image encoder model used to obtain image embeddings from the chromatin images, as well as the any imaging data from Rohban et al. (2017) if required.
3. Preprocess the chromatin images for the inference of the image embeddings eventually yielding the gene perturbation embeddings via e.g. segmenting individual nuclei.
4. Obtain the image and consequently the gene perturbation embedding for the test condition by encoding the images using the pretrained convolutional neural network ensemble image encoder model.
5. Link the gene perturbation embeddings of all but the held-out test condition to their corresponding regulatory gene embeddings by training the kernel regression model.
6. Obtain the prediction of the regulatory embedding for the held-out test condition and use it to identify an ordered prediction set of for the gene overexpressed in the held-out test condition.

*Note that we have built the demo to run without a GPU to maximize its compatability.*

#

### Step-by-step guide

*A linux system is reqired to run the demo. The run time is approximately 10-60 minutes depending on the specifications of the system it is run on.*


#### 1. Perequisites: Anaconda installation (2 minutes)
The only perequisite the demo application has is that the package manager [``Anaconda``](https://docs.anaconda.com/free/) or [``miniconda``](https://docs.conda.io/en/latest/miniconda.html) is installed on your system.

To test if it is install please open a terminal and type in: conda. If you see an error message saying the command was not found, it is not yet installed.
If it is not installed, please install it as follows:

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

#

#### 2. Clone the repository (3 minutes)

Next please clone this repository by running the following command in a **new** terminal.
```
git clone https://github.com/uhlerlab/image2reg.git
cd image2reg
```

#

#### 3. Run the demo (5-50 minutes)

You are now ready to run the demo. The demo can be run in the terminal using the command
```
source scripts/demo/image2reg_demo.sh
```

This command will run the demo using the default parameters which will apply our pipeline to predict that *BRAF* is the gene targeted for overexpression in cells. 
<!---To this end, it uses chromatin images from the perturbation data set from [Rohban et al. (2017)](https://elifesciences.org/articles/24060). The pipeline was set up without using any images of cells in the *BRAF*, respectively any other test condition you choose, and thus performs out of sample prediction. Note that to run the command your working directory needs to be image2reg. If you have followed the previous steps, this is ensured for by the ``cd image2reg`` command, if you run the code after having opened a new terminal please simply navigate to the location of the image2eg directory on your system.-->
Note that to run the command your working directory needs to be ``image2reg``. If you have followed the above instructions this is automatically ensured.

#

#### 4. Specifying the held-out overexpression condition
This demo application can be used to run our Image2Reg inference pipeline for five different overexpression conditions namely: *BRAF, JUN, RAF1, SMAD4 and SREBF1*. The ``--condition`` argument can be used to specify for which of these conditions our Image2Reg pipeline should be run and predict the overexpression target gene from the corresponding chromatin images.

For instance, to run our pipeline for the *JUN* overexpression condition, simply run in a terminal
```
source scripts/demo/image2reg_demo.sh --condition JUN
```

#

#### *5. Advanced run settings/developer options (Optional)*
In addition to specifying for which overexpression condition our pipeline should be run, there are three additional arguments that one can be used for the demo application:
1. ``--random``:    If this argument is provided, the Image2Reg pipeline is run such that the inferred gene perturbation and regulatory gene embeddings are permuted prior the kernel regression is fit which eventually predicts the overexpression target. This recreates the random baseline described in our manuscript. Using this argument, you will observe a deteriated prediction performance of our pipeline which is expected.
2. ``--environment``:    This argument can be used if one would like to specify a pre-existing conda environment that is supposed to be used to run the demo application. By default, if the argument is not provided a new conda environment will be setup as part of the demo application called ``image2reg_demo`` in which all required python packages will be installed and that is used to run our code.
3. ``--help``:    This argument can be used to obtain help on the usage of our demo and in particular summarizes the meaning of the different arguments (i.e. ``--condition``, ``--random``, ``--environment``) described before.


Note that any of these arguments except for the ``--help`` command can be combined to select the setup for the demo application that you like.
As an example, if you would like to use a pre-existing conda environment e.g. ``imag2reg_demo`` and reproduce a *random* baseline prediction for our pipeline for the overexpression condition *SREBF1* run
```
source scripts/demo/image2reg_demo.sh --environment image2reg_demo --condition SREBF1 --random
```

#

#### *6. Run the demo application on user-provided imaging data (Optional)*
The above demo application applies our Image2Reg pipeline to perform out-of-sample prediction for one of five selected overexpression conditions and the corresponding imaging data from Rohban et al. (2017).
However, our pipeline can also be applied to imaging data (i.e. chromatin images as well as corresponding nuclear segmentation masks) to predict which gene/s were most likely to be overexpressed in the captured cells.
To this end, the user only needs to provide the raw chromatin images and corresponding nuclear segmentation masks. Thereby, each segmentation mask image should be a image, where each background pixel is marked by a value of 0 and each pixel corresponding to the area of the same nucleus is assigned the same integer value. The segmentation mask and the corresponding raw chromatin images are expected to have matching file names.

If these inputs are available, simply run the following command in a terminal
```
cd image2reg
source scripts/demo/image2reg_demo_new_data.sh
```
to run our demo application to perform such inference. Note that the download is skipped if the directory already exists because you have run the demo application applied to user-specified data input before. Note that to run the command your working directory needs to be image2reg. If you have followed the previous steps, this is ensured for by the ``cd image2reg`` command.

This will trigger the following processes:
1. The conda environment used to run the code is set up
2. The data repository called ``test_data`` where the image data set should be deposited in is downloaded.
3. The application will ask you to confirm that you have deposited the imaging data you would like to apply our pipeline to in the appropriate directories, namely all raw chromatin images in ``test_data/UNKNOWN/images/raw/plate`` and the respective nuclear segmentation masks in ``test_data/UNKNOWN/images/unet_masks/plate``.
4. Please put your image data in the respective directories and type in ``yes`` to confirm it.
5. The demo will then perform all further inference steps described in the *Overview* section for the user-specified image data set and output the 10 genes that were most likely overexpressed in the cells captured in the data set (in decreasing order).

#

*Please note that the demo application makes use of models trained on the image data from Rohban et al. (2017). Just like any machine learning application if your imaging data differs vastly in terms of e.g. resolution, size of the cells imaged from those used in the Rohban data set, the models, in particular the image encoder model, should be retrained. The descriptions in the following section detailing how to reproduce all of our analysis from scratch together with the detailed explanations in our manuscript should provide sufficient input to perform this task. However, we are also more than happy to help you with your specific use case. Please simply open an issue in this repository and we will assist you as soon as possible.*

#

**If you would like to reproduce all results of the paper from scratch please refer to [this guide](reproducibility_guide.md). 
If not we appreciate you testing our code and look forward to the amazing applications we hope our solution will help to create.**

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


