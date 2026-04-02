# PopGMM: Population Clustering via HDBSCAN and Gaussian Mixture Models

A comprehensive population genetic analysis pipeline for detecting and characterizing population structure in large-scale genomic cohorts using HDBSCAN denoising followed by Gaussian Mixture Model (GMM) clustering.

## Overview

PopGMM is designed for large-scale genomic population studies (e.g., Japanese Biobank, multi-ancestry cohorts). The pipeline identifies population substructure and assigns individuals to ancestral populations using probabilistic models. These ancestry assignments enable:

- **Ancestry inference**: Probabilistically assign individuals to identified population clusters with confidence scores
- **Population stratification**: Control for population structure in downstream association studies
- **Admixture detection**: Identify individuals of mixed ancestry and quantify ancestry proportions
- **Fine-grained ancestry analysis**: Enable ancestry-specific or ancestry-adjusted genome-wide association studies (GWAS)

The pipeline achieves this through:

1. **HDBSCAN Denoising**: Removes sparse outliers/noise from PCA space to retain stable population structure
2. **GMM Clustering**: Uses Bayesian Information Criterion (BIC) to find optimal number of population clusters
3. **Component Merging**: Hierarchically merges nearby GMM components using Mahalanobis distance
4. **High-confidence Assignment**: Assigns individuals to population clusters with posterior probabilities for downstream analysis

## Features

- **Memory-efficient processing**: Chunked data loading for large cohorts (>200k samples)
- **Robust denoising**: HDBSCAN-based noise filtering preserves meaningful population structure
- **Automatic model selection**: BIC-based optimization for K (number of clusters)
- **Confidence scoring**: Posterior probabilities for individual cluster assignments
- **Reproducibility**: Structured configuration objects and detailed audit logs
- **Visualization**: Diagnostic plots for denoising, clustering, and merging steps

## Project Structure

```
PopGMM/
├── workflow.ipynb                       # Main analysis workflow
├── README.md                            # Project documentation
├── .gitignore                           # Git ignore rules
│
├── scripts/                             # Analysis modules
│   ├── data_loading.py                  # Data loading & preprocessing
│   ├── hdbscan_filtering.py             # HDBSCAN denoising
│   ├── gmm_clustering.py                # GMM clustering & BIC search
│   ├── gmm_component_merging.py         # Component merging
│   ├── our_assignment.py                # External cohort assignment
│   ├── high_confidence_visualization.py # Filtering & visualization
│   ├── cluster_all_pcs_kde.py           # KDE validation plots
│   └── gmm_search_audit.py              # Audit logging
│
├── data/                                # Input data (git-ignored)
│   ├── bbj.pca_base.eigenval
│   └── cteph_agp3k_v5_wgs_merged.*.sscore
│
└── results/                             # Outputs by analysis step
    ├── 01_hdbscan_filtering/
    ├── 02_gmm_clustering/
    ├── 03_gmm_component_merging/
    ├── 04_our_assignment/
    ├── 05_high_confidence_visualization/
    └── 06_cluster2_all_pcs_kde_allpcs/
```

## Workflow Pipeline

### Step 0: Data Loading
- Load PCA eigenvalues
- Load BBJ sample scores (no phenotype)
- Load external OUR cohort data with phenotypes
- Separate OUR cases/controls for phenotype analysis

### Step 1: HDBSCAN Denoising
- Apply HDBSCAN clustering on PCA space
- Remove noise points to retain structured population samples
- Output: Non-noise BBJ samples for downstream analysis

### Step 2: GMM Clustering
- Grid search to find optimal K
- Select optimal K by minimum BIC
- Fit final GMM model on selected K
- Output: Cluster labels, posterior probabilities, BIC summary

### Step 3: GMM Component Merging
- Compute pairwise Mahalanobis distances between GMM component means using pooled covariance matrices
- Perform hierarchical clustering on distance matrix
- Cut dendrogram by distance threshold to merge nearby components
- Reassign individuals to merged clusters
- Output: Merged cluster assignments with confidence scores

### Step 4: External Cohort Assignment
- Assign OUR cohort individuals to BBJ-derived population clusters
- Compute assignment probabilities and confidence scores
- Output: OUR samples with BBJ cluster assignments

### Step 5: High-Confidence Filtering
- Filter individuals by assignment confidence threshold
- Subset to high-confidence population assignments
- Output: High-confidence filtered sample list

### Step 6: Validation Visualizations
- KDE plots across all PC dimensions
- Identify sample characteristics per cluster
- Assess cluster stability and separation

## Requirements

- Python 3.10+
- pandas, numpy
- scikit-learn (GaussianMixture, preprocessing)
- hdbscan
- scipy (hierarchical clustering)
- matplotlib, seaborn
