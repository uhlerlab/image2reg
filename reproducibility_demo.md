# Demo application using test inputs

## Overview

This variant of our demo application can be used to run the Image2Reg pipeline to perform out-of-sample prediction for five selected overexpression conditions (*BRAF, JUN, RAF1, SMAD4, SREBF1*) from the dataset from [Rohban et al. (2017)](https://doi.org/10.7554/eLife.24060) thereby reproducing the results of the leave-one-target-out cross-validation approach for these conditions as reported in our manuscript.

In particular, this demo
1. Preprocesses the chromatin images of the selected test condition.
2. Obtains the image and consequently the gene perturbation embedding for the images of the selected test condition encoding the images using the convolutional neural network ensemble image encoder model that was trained on the imaging data from all **but** the selected test condition.
3. Links the gene perturbation embeddings of their corresponding regulatory gene embeddings by training the kernel regression model.
4. Outputs the 10 genes most likely overexpressed (in decreasing order) in the cells in the images corresponding to the selected test condition.

---

## Step-by-step guide

> [!WARNING]
> Before you proceed make sure that you have followed the instructions and in particular run the steps 1 and 2 of the [Step-by-Step guide in the README file](https://github.com/uhlerlab/image2reg/blob/master/README.md#step-by-step-guide).

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

 Run this variant of demo application via typing in
```
source scripts/demo/image2reg_demo.sh --condition <CONDITION>
```
where you replace ``<CONDITION>`` with either ``BRAF``, ``JUN``, ``RAF1``, ``SMAD4`` or ``SREBF1``.

#

### *Advanced run settings/developer options (Optional)*
 In addition to specifying for which overexpression condition our pipeline should be run, there are three additional arguments that one can be used for the demo application that is used to reproduce our results of our study for the selected perturbation conditions:

 1. ``--random``:    If this argument is provided, the Image2Reg pipeline is run such that the inferred gene perturbation and regulatory gene embeddings are permuted prior the kernel regression is fit which eventually predicts the overexpression target. This recreates the random baseline described in our manuscript. Using this argument, you will observe a deteriated prediction performance of our pipeline which is expected.
 2.  ``--environment``:    This argument can be used if one would like to specify a pre-existing conda environment that is supposed to be used to run the demo application. By default, if the argument is not provided a new conda environment will be setup as part of the demo application called ``image2reg_demo`` in which all required python packages will be installed and that is used to run our code.
 3.   ``--help``:    This argument can be used to obtain help on the usage of our demo and in particular summarizes the meaning of the different arguments (i.e. ``--condition``, ``--random``, ``--environment``) described before.


Note that any of these arguments except for the ``--help`` command can be combined to select the setup for the demo application that you like.
As an example, if you would like to use a pre-existing conda environment e.g. ``imag2reg_demo`` and reproduce a *random* baseline prediction for our pipeline for the overexpression condition *SREBF1* run
```
source scripts/demo/image2reg_demo.sh --environment image2reg_demo --condition SREBF1 --random
```
