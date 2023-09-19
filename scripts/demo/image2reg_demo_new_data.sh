#!/bin/bash

echo 

############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo
   echo "Our method Image2Reg pipeline predicts the perturbed (overexpressed) gene in cells from chromatin images." | fold -sw 80
   echo "We here provide a demo application that runs our pipeline for a user-defined imaging data set." | fold -sw 80
   echo 

   echo "The demo will preprocess the corresponding imaging data, i.e. segment individual nuclei, filter out segmentation artifacts and prepare these for the inference of the image embeddings that for each cell provide a high-dimensional description (or fingerprint) of its chromatin state, i.e. its chromatin organization and nuclear morphology." | fold -sw 80
   echo "Please follow the instructions output as text of the demo as well as those in the accompanying documentation." | fold -sw 80
   echo 
   echo "Syntax: source image2reg_demo.sh --help [-h]  | --environment [-e]" | fold -sw 80
   echo "Options:"
   echo "--------"
   echo "--help | -h     Print this Help."
   echo
   echo 
   
   echo "Arguments:"
   echo "----------"
   echo "--image_dir | -e	File path to the directory that contains the single-channel (black-white) chromatin images of the cells the Image2Reg pipeline is supposed to perform inference on." | fold -sw 80
      echo "--mask_dir | -e	File path to the directory that contains the single-channel (black-white) nuclear mask images corresponding to the images in the directory provided as the image_dir argument. Note that each mask image must have the exact same file name as the corresponding chromatin image." | fold -sw 80
   echo "--environment | -e	Name of the conda environment used to run the demo. If no argument is given a new conda enviornment will be setup called: image2reg_demo and all python packages required to run the demo will be installed." | fold -sw 80
   echo
}

############################################################
############################################################
# Main program                                             #
############################################################
############################################################

env=""
target="UNKNOWN"
random="no"
image_dir=""
mask_dir=""

while [ True ]; do
if [ "$1" = "--help" -o "$1" = "-h" ]
then
    Help
    return
elif [ "$1" = "--environment" -o "$1" = "-e" ]
then
    env=$2
    shift 2
elif [ "$1" = "--image_dir" -o "$1" = "-i" ]
then
    image_dir=$2
    shift 2
elif [ "$1" = "--mask_dir" -o "$1" = "-m" ]
then
    mask_dir=$2
    shift 2
else
    break
fi
done

if [ -z $image_dir ] || [ -z $mask_dir ]
then
  echo "Required --image_dir argument was not provided. Please ensure that you provide the an existing directory that contains the chromatin images as the --image_dir argument." | fold -sw 80
  echo "Similarly ensure that you provide an existing directory that contains for all the images in the image directory corresponding segmentation masks as the --mask_dir argument". | fold -sw 80
  echo "Please restart the demo and provide appropriate file paths." | fold -sw 80
  return 1
fi




echo "Starting Image2Reg demo..."
echo 

echo "Selected conda environment (if empty, a new environment called image2reg_demo will be installed):  $env." | fold -sw 80 
echo ""

echo "Selected inference for new image data set."
echo "Raw images are expected to be located in: $image_dir" | fold -sw 80
echo "Nuclear masks are expected to be located in: $mask_dir" | fold -sw 80
echo ""


sleep 5

echo "" 
if [ -z $env ]
then
	echo "Install new conda environment: image2reg_demo" | fold -sw 80
	sleep 1
	echo 
	conda create --name image2reg_demo python=3.8.10 -y
	conda activate image2reg_demo
	pip install setuptools==49.6.0
	pip install -r requirements/demo/requirements_demo.txt
	exit_code=$?
	if [ $exit_code != 0 ]
	then
	  echo
  	  echo "Error encountered during the installing the conda environment: $exit_code" | fold -sw 80
  	  echo "Please consult our running instructions for further guidance of using the code and restart the demo" | fold -sw 80
  	  return 1
  	fi
  	
	echo 
	echo "New conda environment: image2reg_demo successfully set up." | fold -sw 80
	sleep 1
	echo
else
	if { conda env list | grep -q "\b$env\b"; }
	then
		conda activate $env
		echo "Conda environment: $env successfully loaded." | fold -sw 80
	else
	  echo
  	  echo "Provided conda environment $env was not found." | fold -sw 80
    	  echo "Please make sure the provided conda environment exist and restart the demo." | fold -sw 80
    	  echo "Alternatively, restart the demo without the --environment argument to let the demo (re-)install an appropriate conda environment." | fold -sw 80
  	  return 1
	fi
	exit_code=$?
	if [ $exit_code != 0 ]
	then
	  echo
  	  echo "Error encountered during the activation of the conda environment: $exit_code" | fold -sw 80
    	  echo "Please consult our running instructions for further guidance of using the code and restart the demo" | fold -sw 80
  	  return 1
  	fi
	sleep 1
	echo ""
fi


echo ""
echo "------------------------------" | fold -sw 80
echo ""

echo "The demo also requires some intermediate data, e.g. the trained convolutional neural network image encoder to be available." | fold -sw 80
echo "All resources for the demo applied to perform inference on a new data image data set are located in the directory called test_data located within the image2reg repository."
echo "If the repository is missing, it will be automatically downloaded." | fold -sw 80
echo ""
sleep 2

trap "" SIGINT
if ! [ -d "test_data" ]
then
	echo "Data directory for the demo applied to perform inference on a new image data set not found. Downloading..." | fold -sw 80
	
	echo ""
	wget -O "test_data.zip" "https://zenodo.org/record/8354979/files/test_data.zip?download=1"
#	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10tlljqIDOzyFkSooDn-ee1byITt3dour" -O test_data.zip && rm -rf /tmp/cookies.txt
	echo ""
	echo "Unzipping the directory..." | fold -sw 80
	trap "" SIGINT
	unzip -q test_data.zip
	rm test_data.zip
	
	exit_code=$?
	if [ $exit_code != 0 ]
	then
	  echo
  	  echo "Error encountered during the download and extraction of the data repository: $exit_code" | fold -sw 80
  	  echo "Please consult our running instructions for further guidance of using the code, check your internet connection and restart the demo." | fold -sw 80
  	  return 1
  	fi
else
	echo "Demo data directory found. Skipping download..." | fold -sw 80
fi
trap SIGINT
sleep 1
echo ""
echo "------------------------------" | fold -sw 80


if [ -z "$(ls -A $image_dir)" ]; then
   echo
   echo "The directory $image_dir is empty."
   echo "Please restart the application after making sure that the directory given as the --image_dir argument contains your raw chromatin image inputs." | fold -sw 80
   return 1
fi

if [ -z "$(ls -A $mask_dir)" ]; then
   echo
   echo "The directory $image_dir is empty."
   echo "Please restart the application after making sure that the directory given as the --mask_dir argument contains your the nuclear segmentation masks for your images in $image_dir." | fold -sw 80
   return 1
fi


echo ""
echo "Clearing the cache..."
rm -rf test_data/UNKNOWN/images/raw/plate/*
rm -rf test_data/UNKNOWN/images/unet_masks/plate/*
rm -rf test_data/UNKNOWN/images/preprocessed/*
rm -rf test_data/UNKNOWN/images/metadata/*

echo "Copying the input images from $image_dir to test_data/UNKNOWN/images/raw/plate" | fold -sw 80
echo "Copying the mask images from $mask_dir to test_data/UNKNOWN/images/unet_masks/plate" | fold -sw 80
cp -r $image_dir/* test_data/UNKNOWN/images/raw/plate
cp -r $mask_dir/* test_data/UNKNOWN/images/unet_masks/plate
echo ""

echo "Demo preparation complete..." | fold -sw 80

echo ""

sleep 2

echo "Starting Image22Reg pipeline to predict the gene targeted for overexpression ($target) from the corresponding chromatin images from Rohban et al. (2017)...." | fold -sw 80
echo
sleep 1

if [ $target == "UNKNOWN" ]
then
	source scripts/demo/run_demo_new_target_arg.sh $random
else
	echo "Invalid target selected. Please have a look at the help of this demo by running source image2reg_demo_for_new_data.sh --help." | fold -sw 80
fi

echo
sleep 1
echo "Demo complete..." | fold -sw 80
echo "Thanks for using our Image2Reg pipeline." | fold -sw 80
