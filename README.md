# Image2Reg

The structural and regulatory space of cells are directly linked through complex maps that i.a. enable the cell to respond to environmental stimulus in an optimized fashion.
The code can be used to run the experiments exploiting the links to predict the overexpression target from high-throughput fluorescent microscopy images.

---

## Raw data

The raw imaging data is taken from a previous study of [Rohban et al. (2017)](https://doi.org/10.7554/eLife.24060). 
It can be obtained from [IDR0033](https://idr.openmicroscopy.org/webclient/?show=screen-1751).
Please refer to the download instructions of the IDR platform for more information.

The Protein-Protein interaction network data is taken from the [iRefIndex v17](https://irefindex.vib.be/wiki/index.php/iRefIndex).
The single-cell MerFISH gene expression data used to refine the protein-protein-interaction network in addition to the 
[curated cell-specific gene set](https://maayanlab.cloud/Harmonizome/gene_set/U2OS/CCLE+Cell+Line+Gene+Expression+Profiles) is available 
[here](https://www.pnas.org/content/116/39/19490/tab-figures-data).

---

## Preprocessing

### Image data

The raw image data is preprocessed including several filtering steps and nuclear segmentation.
The full preprocessing pipeline can be run via
```
python run.py --config config/preprocessing/full_image_pipeline.yml
```
Please edit the config file to specify the location of the raw imaging data.

### Protein-Protein interaction data

---

## Training of the embedding

### Image data

To train the different models in order to obtain latent, discriminative embeddings of the single-nuclei images run
```
python run.py --config config/image_embedding/<yml_file>
```

### Protein-Protein interaction data

to be added
