# scPER
## Introduction
scPER, which stands for **P**roportions **E**stimated using **s**ingle-**c**ell RNA-seq **R**eference, is a novel computational approach that combines an adversarial autoencoder with extreme gradient boosting. It can robustly estimate the relative compositions of cells and their subtypes within the tumor microenvironment and identify phenotype-associated subclusters. By integrating scRNA-seq datasets from various tumors, scPER constructs comprehensive reference panels encompassing all possible tumor cell types and disentangles confounding factors from true signals within the latent space of the adversarial autoencoder model.

<img src="./overview.png" height=10%>
## Dependencies
Keras 2.10.0
magic-impute 3.0.0
tensorflow 2.10.0
Rmagic 2.0.3
xgboost 1.7.5.1
## Quick Start
The process requires three files, (1) single-cell RNA-seq matrix of the cells you selected and they can come from different sources (2) the labels of the selected cells indicating their cell types, batches (e.g. studies, tissues) (3) bulk RNA-seq samples.
You may also provide clinical phenotypes of the bulk RNA-seq samples if you want to identify the phenotype-associated cell clusters. 

### Preprocess the scRNA-seq data
The first step is to identify the overlap genes between scRNA-seq data and bulk RNA-seq data, impute the scRNA-seq data using MAGIC, and select the top 5k most variable genes in scRNA-seq data. 

```
Rscript preprocess_data.R /example/example_matrix_200_cells_ref.csv /example/Bulk_simulation_100_all_genes.csv
```

### Run adversarial autoencoder for scRNA-seq data and bulk samples

The second step is to train the adversarial autoencoder model to correct the batch effects and generate the latent representations for both scRNA-seq and bulk RNA-seq samples. 

```
python -u main.py /example/reference_top5k_imputation.csv  /example/example_label_200_cells_ref.csv  /example/bulk_5k_genes_matched.csv
```
### Estimate the cell proportions in the bulk samples
The third step is to estimate the cell proportions in bulk samples based on the latent representations generated for scRNA-seq and bulk RNA-seq samples in step 2.

```
Rscript XGBoost.R /results/ADAE_100_latents.tsv /results/bulk_100_latents.tsv /data/example_label_200_cells_ref.csv
```
### (optional) Identify the phenotype-associated cell populations
If you have the clinical phenotypes of the bulk RNA-seq samples, you may generate the plots to identify the phenotype-associated cell clusters. 
