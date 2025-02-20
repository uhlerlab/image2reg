#! /bin/bash

echo "Running Image2Reg inference pipeline for BRAF..."
echo ""

echo "Preprocess chromatin images..."
echo "The output will be saved in the directory demo/BRAF/images/preprocessed."

python run_demo.py --config config/demo/preprocessing/full_image_pipeline_braf.yml --debug

echo "Image preprocessing complete."
echo ""

echo "Obtain image embeddings using our convolutional neural network ensemble image encoder model..."
echo "The image embeddings for the chromatin images of the nuclei from the BRAF condition will be saved as test_latents.h5 in the directory demo/BRAF/embeddings."
echo ""
python run_demo.py --config config/demo/image_embeddings/extract_loto_latents_braf.yml --debug

echo "Inference of the image embeddings complete."
echo ""

echo "Translation of the image embeddings to predict the overexpression target out of sample."
echo "If you would like to run the translation after randomly permuting the image embeddings which is equivalent to the random baseline, please type in: yes, otherwise simply press enter"
echo ""
read -p "Run pipeline in random mode [no]: " random_mode
random_mode=${random_mode:-no}
echo ""
if [ $random_mode == "yes" ]
then
	echo ""
	python scripts/demo/run_translation_demo.py --embedding_dir "demo/BRAF/embeddings" --random_mode
else
	echo ""
	python scripts/demo/run_translation_demo.py --embedding_dir "demo/BRAF/embeddings"
fi