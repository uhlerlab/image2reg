#! /bin/bash


echo "Starting Image2Reg demo"
echo ""

echo "This shows the function of our method on predicting the perturbed (overexpressed) gene in a cell from chromatin images for the example of one of the following overexpression conditions: BRAF, JUN, RAF1, SMAD4 or SREBF1"
echo ""

echo "The demo will preprocess the corresponding imaging data, i.e. segment individual nuclei, filter out segmentation artifacts and prepare these for the inference of the image embeddings that for each cell provide a high-dimensional description (or fingerprint) of its chromatin state, i.e. its chromatin organization and nuclear morphology."
echo ""

echo "The demo requires a number of python packages to be installed that is used by our code."
echo "If you have already set up an anaconda environment and installed the required software packages, e.g. via our installation scripts for our image2reg repository, please provide the name of the conda environment that should be used. Alternatively, please simply hit enter to setup a new conda environment called image2reg demo, which will contain only the packages required for the demo."

read -p "Name of the conda environment to run the demo [create new environment]: " conda_env
conda_env=${conda_env:-scratch}

echo "" 
if [ $conda_env == "scratch" ]
then
	conda create --name image2reg_demo python=3.8.10 -y
	conda activate image2reg_demo
	pip install setuptools==49.6.0
	pip install -r requirements/demo/requirements_demo.txt
	echo ""
	echo "New conda environment image2reg_demo successfully set up."
	echo ""
else
	conda activate $conda_env
	echo "Conda environment successfully loaded."
	echo ""
fi


echo ""
echo "------------------------------"

echo "The demo also requires some intermediate data, e.g. the trained convolutional neural network image encoder to be available."
echo "All resources for the demo are located in the directory called demo located within the image2reg repository."
echo "If the repository is missing, it will be automatically downloaded."

if ! [ -d "demo" ]
then
	echo "Data directory for the demo not found. Downloading..."
	echo ""
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1H89cywD1kCP0MNnYEJJ2_sQDe73EL1Y2" -O demo.zip && rm -rf /tmp/cookies.txt
	echo ""
	echo "Unzipping the directory..."
	unzip -q demo.zip
	rm demo.zip
fi

echo "Please select for which of the five example conditions you would like to run our pipeline."
echo "Note that independent of the choice the pipeline was trained without any use of the image information from the selected condition as our pipeline is used for out-of-sample prediction of unknown gene perturbations from chromatin images."
echo ""

echo "To select a condition, please now type in the name of the corresponding gene targeted for overexpression (BRAF, JUN, RAF1, SMAD4 or SREBF1)"
read -p "Selected condition: " target
echo "$target was selected."
echo ""

echo "Starting Image22Reg pipeline to predict the gene targeted for overexpression ($target) from the corresponding chromatin images from Rohban et al. (2017)...."


if [ $target == "BRAF" ]
then
	bash scripts/demo/run_demo_braf.sh
elif [ $target == "JUN" ]
then
	bash scripts/demo/run_demo_jun.sh
elif [ $target == "RAF1" ]
then
	bash scripts/demo/run_demo_raf1.sh
elif [ $target == "SMAD4" ]
then
	bash scripts/demo/run_demo_smad4.sh
elif [ $target == "SREBF1" ]
then
	bash scripts/demo/run_demo_srebf1.sh
else
	echo "Invalid target selected. Please restart the demo and select one of the following targets: BRAF, JUN, RAF1, SMAD4, SREBF1"
fi

echo "Demo complete..."
echo "Thanks for testing our code."