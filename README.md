# PopGMM: Probabilistic Ancestry Inference and Population Stratification Control

**An Unsupervised Learning Approach via PCA + Gaussian Mixture Models (GMM)**

A comprehensive pipeline for large-scale genomic population studies that identifies population substructure, infers ancestral origins, and enables ancestry-based population stratification in association analyses.

## Overview

PopGMM uses probabilistic clustering to:
- **Infer ancestry**: Assign individuals to inferred population clusters with posterior probabilities
- **Control stratification**: Account for population structure in association studies (GWAS, etc.)
- **Detect admixture**: Identify mixed-ancestry individuals and quantify ancestry proportions
- **Enable fine-grained analysis**: Support ancestry-specific or ancestry-adjusted association analyses

The pipeline combines:
1. **HDBSCAN Denoising**: Remove sparse outliers from PCA space to preserve stable population structure
2. **GMM Clustering**: Identify optimal population clusters via Bayesian Information Criterion (BIC)
3. **Component Merging**: Consolidate adjacent clusters using Mahalanobis distance and hierarchical clustering
4. **Probabilistic Assignment**: Assign individuals to merged clusters with confidence scores for downstream use

## Features

- **Scalable**: Chunked data loading for large cohorts (>200k samples)
- **Robust**: HDBSCAN filters noise while preserving true population signal
- **Automated**: BIC-based model selection eliminates manual cluster number specification
- **Interpretable**: Posterior probabilities quantify assignment confidence
- **Reproducible**: Modular configuration objects and detailed audit logs
- **Validated**: Diagnostic visualizations for quality control across all steps

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
