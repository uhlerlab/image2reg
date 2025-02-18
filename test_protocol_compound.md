# Step-by-step Protocol for our Analyses on Compound Perturbation

---

This step-by-step protocol details how the analyses in Fig. 5 using compound perturbation data can be reproduced after obtaining results in the [previous sections](https://github.com/uhlerlab/image2reg/blob/master/test_protocol.md). To quickly test our Image2Reg pipeline or use it to perform inference on your own data set, please refer to respective documentation described in our [ReadMe file](https://github.com/uhlerlab/image2reg/blob/master/README.md).


---

## Obtain and prepare data

#### Download JUMP-CP data

Start the jupyter server in the conda environment via
```
conda activate image2reg
jupyter notebook
```

Start the jupyter notebook ``notebooks/jump/eda/data_extraction_new_compound.ipynb`` and run all cells to download the image data from the JUMP-CP data set for the selected OE conditions including the illumination corrected images. This script also maps compounds to their known gene targets using DrugBank.
All generated data gets downloaded to ``data/resources/images/jump_compound``.


#### Run nuclear segmentation using the corresponding jupyter notebook

Start the jupyter server in the unet conda environment via
```
conda activate unet
jupyter notebook
```

Open and run the jupyter notebook located in ``unet/notebook/jump_segmentation.ipynb``.
> [!IMPORTANT]
> Please be aware that this is not a path in the image2reg directory but the unet-nuclei directory you have cloned earlier. Please refer to the respective section for the Rohban data set in this protocol for more information.

Run all cells to generate the segmentation masks for all images and stores those in ``image2reg/data_new/resources/images/jump_compound/unet_masks``.


#### Preprocess knockout and compound perturbation imaging data via script

Run the preprocessing script via

```
conda activate image2reg
python run.py --config config/preprocessing/full_image_pipeline_jump_new_compound.yml
python run.py --config config/preprocessing/full_image_pipeline_jump_new_ko.yml
```


This runs all preprocessing steps stores the outputs in a "timestamp" output directory in the directory ``data_new/experiments/jump_compound/images/preprocessing/full_pipeline``.
By default all output directories created as a result of running the run.py will be named after the time point when the script was started.
For the consecutive analyses please copy the content of the timestamp output directory directly to the ``full_pipeline``  directory  and delete the then empty timestamp directory.

## Identify impactful knockout conditions

#### Generate the required data split files

Start the jupyter server in the conda environment via
```
conda activate image2reg
jupyter notebook
```

Start the notebook ``notebooks/rohban/other/cv_screen_data_split_jump_new_impactful.ipynb`` and run all cells in the notebook.
This generates a number of files located in ``data_new/experiments/jump/images/preprocessing/specific_targets_combined_excludeCompound/screen_splits_impactful``.

#### Run the specificity screen

Run the specificity screen to identify impactul overexpression conditions via
```
conda activate image2reg
bash run_screen_ko.sh
```

Finally, rename the output of the screen located in ``data/experiments/jump/images/screen/nuclei_region`` via
```
conda activate image2reg
python scripts/experiments/rename_screen_dirs –root_dir data/experiments/jump/images/screen/nuclei_region
```

#### Analyze the results

Start the jupyter server in the conda environment
```
conda activate image2reg
jupyter notebook
```

Start the notebook ``notebooks/jump/screen/screen_analyses_cv_final.ipynb`` and run all cells.
This creates a summary of the screen results and saves it as ``data_new/experiments/jump/images/screen/nuclei_region/specificity_screen_results_cv.csv``.

# 

## Gene perturbation embeddings

### General setup using all knockout conditions excluding knockouts of potential compound targets

#### Generate data splits

Start the jupyter server in the conda environment via
```
conda activate image2reg
jupyter notebook
```

Start the notebook notebooks/rohban/other/cv_specific_targets_data_split_jump_new_combined_excludeCompound.ipynb and run all cells.
This creates the required metadata csv-files in ``data_new/experiments/jump/images/preprocessing/specific_targets_combined_excludeCompound/``.


#### Run classification using knockout conditions and extract image embeddings of compound perturbations

Run the training of the CNN ensemble on the image data from the JUMP data set in the conda environment via
```
conda activate image2reg
python run.py --config  config/image_embedding/specific_targets/jump_ko_compound/compound_test_excludeCompound.yml
python run.py --config  config/image_embedding/specific_targets/jump_ko_compound/compound_test_others_excludeCompound.yml
```
This step also saves the single-cell image embeddings of compound perturbations.


The results of the analyses are saved in the directory ``data_new/experiments/jump/images/embedding/specificity_target_combined/all_excludeCompound/``.
By default the results are saved in a timestamp subdiretory.
Rename the timestamp directory to ``nuclei_regions``.

#### Analyze the image embeddings

Start the jupyter server in the conda environment via
```
conda activate image2reg
jupyter notebook
```

Start the notebook ``notebooks/jump/ko_embedding/image_embeddings_analysis_combined_excludeCompound.ipynb`` and run all cells
This produces the e.g. the Fig. 5C of the manuscript.

Next, start the notebook ``notebooks/jump/ko_embedding/gene_perturbation_cluster_analysis_excludeCompound.ipynb`` and run all cells to e.g. reproduce the Fig. 5B of the manuscript.


### Performance evaluation - mapping gene perturbation embeddings to regulatory embeddings

Start the jupyter server in the conda environment via
```
conda activate image2reg
jupyter notebook
```

Start the notebook ``notebooks/jump/translation/jump_translation_prediction_new_compound_excludeCompound.ipynb`` and run all cells to perform the complete translation analysis and e.g.generate Fig. 5D-E.

