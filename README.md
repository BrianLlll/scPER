# scPER
## Introduction
scPER, which stands for Estimating Cell Proportions using single-cell RNA-seq Reference, is a novel computational approach that combines an adversarial autoencoder with extreme gradient boosting. It can robustly estimate the relative compositions of cells and their subtypes within the tumor microenvironment and identify phenotype-associated subclusters. By integrating scRNA-seq datasets from various tumors, scPER constructs comprehensive reference panels encompassing all possible tumor cell types and disentangles confounding factors from true signals within the latent space of the adversarial autoencoder model.

<img src="./overview.png" height=10%>

## Quick Start

### Preprocess the scRNA-seq data

### Run adversarial autoencoder for the scRNA-seq data

### Embed the bulk sample data

### Estimate the cell proportions in the bulk samples

### Identify the phenotype-associated cell populations
