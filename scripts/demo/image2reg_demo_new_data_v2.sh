#! /bin/bash

echo 

############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo
   echo "Our method Image2Reg pipeline predicts the perturbed (overexpressed) gene in cells from chromatin images." | fold -sw 80
   echo " We here provide a demo application that runs our pipeline for a user-defined imaging data set." | fold -sw 80
   echo 

   echo "The demo will preprocess the corresponding imaging data, i.e. segment individual nuclei, filter out segmentation artifacts and prepare these for the inference of the image embeddings that for each cell provide a high-dimensional description (or fingerprint) of its chromatin state, i.e. its chromatin organization and nuclear morphology." | fold -sw 80
   echo 
   echo "Syntax: source image2reg_demo.sh --help [-h]  | --environment [-e]" | fold -sw 80
   echo "Options:"
   echo "--------"
   echo "--help | -h     Print this Help."
   echo
   echo 
   
   echo "Arguments:"
   echo "----------"
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

while [ True ]; do
if [ "$1" = "--help" ] || [ "$1" = "-h" ]
then
    Help
    return
elif [ "$1" = "--environment" ] || [ "$1" = "-e" ]
then
    env=$2
    shift 2
else
    break
fi
done



echo "Starting Image2Reg demo..."
echo 

echo "Selected conda environment (if empty, a new environment called image2reg_demo will be installed):  ""$env""." | fold -sw 80 
echo ""

echo "Selected inference for new image data set."
echo "Raw images are expected to be located in: image2reg/test_data/UNKNOWN/images/raw/plate." | fold -sw 80
echo "Nuclear masks are expected to be located in: image2reg/test_data/UNKNOWN/images/unet_masks/plate." | fold -sw 80
echo ""


sleep 5

echo "" 
if [ -z "$env" ]
then
	echo "Install new conda environment: image2reg_demo" | fold -sw 80
	sleep 1
	echo 
	conda create --name image2reg_demo python=3.8.10 -y
	conda activate image2reg_demo
	pip install setuptools==49.6.0
	pip install -r requirements/demo/requirements_demo.txt
	echo 
	echo "New conda environment: image2reg_demo successfully set up." | fold -sw 80
	sleep 1
	echo
else
	conda activate "$env"
	echo "Conda environment: $env successfully loaded." | fold -sw 80
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

if ! [ -d "test_data" ]
then
	echo "Data directory for the demo applied to perform inference on a new image data set not found. Downloading..." | fold -sw 80
	echo ""
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10tlljqIDOzyFkSooDn-ee1byITt3dour" -O test_data.zip && rm -rf /tmp/cookies.txt
	echo ""
	echo "Unzipping the directory..." | fold -sw 80
	unzip -q test_data.zip
	rm test_data.zip
else
	echo "Demo data directory found. Skipping download..." | fold -sw 80
fi

sleep 1
echo ""

echo "------------------------------" | fold -sw 80

echo ""
echo "If you have not done so already, please place now the image data which you would like to apply our Image2Reg pipeline to in the following directories." | fold -sw 80
echo ""
echo "Raw chromatin images: test_data/UNKNOWN/images/raw/plate" | fold -sw 80
echo "Nuclear segmentation masks: test_data/UNKNOWN/images/unet_masks/plate" | fold -sw 80
echo ""
echo "Please note that the raw chromatin images and their corresponding nuclear segmentation masks are required to be named exactly the same such that the pipeline can associate these two different image inputs to one another." | fold -sw 80

echo ""

read -p "Has the imaging data been placed in the above mentioned directories? [no]/yes: " proceed
proceed=${proceed:-no}

if [ "$proceed" -ne "yes" ]
then
  echo "Image data needs to be positioned in the directories: test_data/UNKNOWN/images/raw/plate and test_data/UNKNOWN/images/unet_masks/plate to apply our pipeline to the new imaging data set." |fold -sw 80
  echo "Please restart the demo application, once the data was placed in the respective directories." | fold -sw 80
  echo "Exiting demo..."
  exit 1
fi

echo ""

echo "Demo preparation complete..." | fold -sw 80

echo ""

sleep 2

echo "Starting Image22Reg pipeline to predict the gene targeted for overexpression (""$target"") from the corresponding chromatin images from Rohban et al. (2017)...." | fold -sw 80
echo
sleep 1

if [ "$target" = "UNKNOWN" ]
then
	bash scripts/demo/run_demo_new_target_arg.sh "$random"
else
	echo "Invalid target selected. Please have a look at the help of this demo by running source image2reg_demo_for_new_data.sh --help." | fold -sw 80
fi

echo
sleep 1
echo "Demo complete..." | fold -sw 80
echo "Thanks for running our code." | fold -sw 80
