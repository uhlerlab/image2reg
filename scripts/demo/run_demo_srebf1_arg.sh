#! /bin/bash

echo "Running Image2Reg inference pipeline for SREBF1..." | fold -sw 80
echo "---------------------------------------------------" | fold -sw 80
echo ""

echo "Preprocess chromatin images..."
echo "The output will be saved in the directory demo/SREBF1/images/preprocessed." | fold -sw 80
echo

python run_demo.py --config config/demo/preprocessing/full_image_pipeline_srebf1.yml --debug

echo
echo "Image preprocessing complete."
echo "------------------------------"
echo ""

echo "Obtain image embeddings using our convolutional neural network ensemble image encoder model..." | fold -sw 80
echo "The image embeddings for the chromatin images of the nuclei from the SREBF1 condition will be saved as test_latents.h5 in the directory demo/SREBF1/embeddings." | fold -sw 80
echo ""
python run_demo.py --config config/demo/image_embeddings/extract_loto_latents_srebf1.yml --debug

echo "Inference of the image embeddings complete." | fold -sw 80
echo ""

echo "Translation of the image embeddings to predict the overexpression target out of sample." | fold -sw 80
random_mode=$1
echo ""
if [ $random_mode == "yes" ]
then
	echo ""
	python scripts/demo/run_translation_demo.py --embedding_dir "demo/SREBF1/embeddings" --random_mode
else
	echo ""
	python scripts/demo/run_translation_demo.py --embedding_dir "demo/SREBF1/embeddings"
fi
