#! /bin/bash

echo "Running Image2Reg inference pipeline for new imaging data..." | fold -sw 80
echo "---------------------------------------------------" | fold -sw 80
echo ""

echo "Generate metadata for images located in image2reg/test_data/UNKNOWN/images/raw/plate." | fold -sw 80
echo ""
python scripts/demo/run_generate_metadata_for_new_target_demo.py

echo "Preprocess chromatin images..." | fold -sw 80
echo "The output will be saved in the directory test_data/UNKNOWN/images/preprocessed." | fold -sw 80
echo

python run_demo.py --config config/demo/preprocessing/full_image_pipeline_new_target.yml --debug

echo
echo "Image preprocessing complete."
echo "------------------------------"
echo ""

echo "Obtain image embeddings using our convolutional neural network ensemble image encoder model..." | fold -sw 80
echo "The image embeddings for the chromatin images of the nuclei from the new test condition will be saved as test_latents.h5 in the directory image2reg/test_data/embeddings." | fold -sw 80
echo ""
python run_demo.py --config config/demo/image_embeddings/extract_loto_latents_new_target.yml --debug

echo "Inference of the image embeddings complete." | fold -sw 80
echo ""

echo "Translation of the image embeddings to predict the overexpression target out of sample." | fold -sw 80
if [ $1 == "yes" ]
then
	echo ""
	python scripts/demo/run_translation_demo_new_target.py --embedding_dir "test_data/UNKNOWN/embeddings" --random_mode
else
	echo ""
	python scripts/demo/run_translation_demo_new_target.py --embedding_dir "test_data/UNKNOWN/embeddings"
fi
