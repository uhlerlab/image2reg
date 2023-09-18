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
   echo " We here provide a demo application that runs our pipeline one of the following overexpression conditions: BRAF, JUN, RAF1, SMAD4 or SREBF1." | fold -sw 80
   echo 

   echo "The demo will preprocess the corresponding imaging data, i.e. segment individual nuclei, filter out segmentation artifacts and prepare these for the inference of the image embeddings that for each cell provide a high-dimensional description (or fingerprint) of its chromatin state, i.e. its chromatin organization and nuclear morphology." | fold -sw 80
   echo 
   echo "Syntax: source image2reg_demo.sh --help [-h]  | --environment [-e] --condition [-c] --random [-r]" | fold -sw 80
   echo "Options:"
   echo "--------"
   echo "--help | -h     Print this Help."
   echo
   echo 
   
   echo "Arguments:"
   echo "----------"
   echo "--environment | -e	Name of the conda environment used to run the demo. If no argument is given a new conda enviornment will be setup called: image2reg_demo and all python packages required to run the demo will be installed." | fold -sw 80
   echo
   echo "--condition | -c	Name of the overexpression condition for which our Image2Reg pipeline is applied to predict the gene targeted for overexpression; must be one of the following: BRAF, JUN, RAF1, SMAD4, SREBF1. Independent of the choice the pipeline was trained without any use of the image information from the selected condition as our pipeline is used for out-of-sample prediction of unknown gene perturbations from chromatin images. If no argument is given, BRAF is selected." | fold -sw 80
   echo
   echo "--random | -r		Flag to run Image2Reg pipeline such that the random baseline model described in our manuscript is recreated via randomly permuting the gene perturbation and regulatory gene embeddings prior training the kernel regression model."	| fold -sw 80
}

############################################################
############################################################
# Main program                                             #
############################################################
############################################################

env=""
target="BRAF"
random="no"

while [ True ]; do
if [ "$1" = "--help" -o "$1" = "-h" ]
then
    Help
    return
elif [ "$1" = "--environment" -o "$1" = "-e" ]
then
    env=$2
    shift 2
elif [ "$1" = "--condition" -o "$1" = "-c" ]
then
    target=$2
    shift 2
elif [ "$1" = "--random" -o "$1" = "-r" ]
then
    random="yes"
    shift 1
else
    break
fi
done



echo "Starting Image2Reg demo"
echo 

echo "Selected conda environment (if empty, a new environment called image2reg_demo will be installed):  $env." | fold -sw 80
echo "Selected test condition:  $target." | fold -sw 80
if [ $random == "yes" ]
then
	echo "Selected random inference to recreate the random baseline." | fold -sw 80
fi

sleep 5

echo "" 
if [ -z $env ]
then
	echo "Install new conda environment: image2reg_demo" | fold -sw 80
	echo 
	conda create --name image2reg_demo python=3.8.10 -y
	conda activate image2reg_demo
	pip install setuptools==49.6.0
	pip install -r requirements/demo/requirements_demo.txt
	echo 
	echo "New conda environment: image2reg_demo successfully set up." | fold -sw 80
	echo
else
	conda activate $env
	echo "Conda environment: $env successfully loaded." | fold -sw 80
	echo ""
fi


echo ""
echo "------------------------------" | fold -sw 80
echo ""

echo "The demo also requires some intermediate data, e.g. the trained convolutional neural network image encoder to be available." | fold -sw 80
echo "All resources for the demo are located in the directory called demo located within the image2reg repository."
echo "If the repository is missing, it will be automatically downloaded." | fold -sw 80
echo ""

if ! [ -d "demo" ]
then
	echo "Data directory for the demo not found. Downloading..." | fold -sw 80
	echo ""
	wget -O "demo.zip" "https://zenodo.org/record/8354979/files/demo.zip?download=1"
#	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1H89cywD1kCP0MNnYEJJ2_sQDe73EL1Y2" -O demo.zip && rm -rf /tmp/cookies.txt
	echo ""
	echo "Unzipping the directory..." | fold -sw 80
	unzip -q demo.zip
	rm demo.zip
else
	echo "Demo data directory found. Skipping download..." | fold -sw 80
fi

echo ""

echo "------------------------------" | fold -sw 80

echo "Demo preparation complete..." | fold -sw 80
echo "Starting Image22Reg pipeline to predict the gene targeted for overexpression ($target) from the corresponding chromatin images from Rohban et al. (2017)...." | fold -sw 80
echo


if [ $target == "BRAF" ]
then
	bash scripts/demo/run_demo_braf_arg.sh $random
elif [ $target == "JUN" ]
then
	bash scripts/demo/run_demo_jun_arg.sh $random
elif [ $target == "RAF1" ]
then
	bash scripts/demo/run_demo_raf1_arg.sh $random
elif [ $target == "SMAD4" ]
then
	bash scripts/demo/run_demo_smad4_arg.sh $random
elif [ $target == "SREBF1" ]
then
	bash scripts/demo/run_demo_srebf1_arg.sh $random
else
	echo "Invalid target selected. Please have a look at the help of this demo by running source image2reg_demo.sh --help." | fold -sw 80
fi

echo
echo "Demo complete..." | fold -sw 80
echo "Thanks for testing our code." | fold -sw 80
