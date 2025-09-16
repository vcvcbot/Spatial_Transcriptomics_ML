# Spatial Transcriptomics ML Project

## Project Overview

This project aims to provide a comprehensive analysis of spatial transcriptomics data using machine learning methods. Spatial transcriptomics technology allows for the measurement of gene expression levels in tissue sections while preserving tissue morphology, offering unprecedented opportunities to understand cellular heterogeneity and tissue microenvironments. The core of this project is a Jupyter Notebook that covers a complete workflow from data loading, preprocessing, and quality control to spatial autocorrelation analysis, dimensionality reduction, and comparative evaluation of multiple clustering methods.

Through this project, we demonstrate how to effectively process and analyze spatial transcriptomics data, identify genes with spatial patterns, and compare the performance of different machine learning clustering algorithms on spatial data. The ultimate goal is to provide researchers with a clear, reproducible analysis framework to extract biological insights from complex spatial transcriptomics data.

## Project Features

*   **Data Loading and Preprocessing**: Uses the `squidpy` library to load Visium H&E stained mouse brain dataset and performs basic data cleaning.
*   **Quality Control (QC)**: Executes filtering of genes and cells, and visualizes key quality metrics through violin plots.
*   **Spatial Autocorrelation Analysis**: Utilizes Moran's I statistic to identify genes with significant spatial expression patterns.
*   **Dimensionality Reduction**: Applies Principal Component Analysis (PCA) and UMAP for data dimensionality reduction, facilitating visualization and clustering.
*   **Comparison of Multiple Clustering Methods**: Implements and compares various machine learning clustering algorithms such as Leiden, K-Means, Louvain, and BIRCH.
*   **Clustering Performance Evaluation**: Quantitatively evaluates the performance of different clustering methods using metrics like Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI).
*   **Rich Visualizations**: Generates various plots, including spatial scatter plots, UMAP dimensionality reduction plots, violin plots, and clustering evaluation bar plots, to intuitively display analysis results.




## Environment Setup and Dependencies

This project requires a Python environment and depends on the following libraries. It is recommended to install them using `pip` or `conda`.

```bash
pip install scanpy squidpy numpy pandas matplotlib seaborn scikit-learn
pip install louvain igraph leidenalg
pip install gseapy
```

## Data Loading, Preprocessing, and Quality Control (Part 1)

### 1. Import Required Libraries

The project begins by importing various Python libraries necessary for spatial transcriptomics data analysis. These include `scanpy` for single-cell data analysis, `squidpy` for spatial omics data analysis, and common data processing and visualization libraries such as `numpy`, `pandas`, `matplotlib`, `seaborn`, and `scikit-learn`. Concurrently, the code checks for and creates a `figures` folder to save all generated images.

### 2. Load Example Dataset

This project uses the Visium H&E stained mouse brain dataset provided by `squidpy` as an example. This is a typical spatial transcriptomics dataset, containing a gene expression matrix and spatial coordinate information for each cell spot. After data loading, the original data dimensions and metadata overview for observations (spots) and variables (genes) are displayed.

### 3. Visualize Ground Truth

The dataset includes a predefined `cluster` column, representing the true regional segmentation of the tissue. We use the `sq.pl.spatial_scatter` function to spatially visualize these true labels, generating an image that shows the actual positions of different cell populations on the tissue section. This plot will serve as a benchmark for subsequent clustering result comparisons.

![Ground Truth Spatial Plot](figures/_ground_truth_spatial.png)

### 4. Data Quality Control (QC) and Gene Expression Statistical Analysis

Quality control is a critical step in spatial transcriptomics data analysis. This project performs the following QC steps:

*   **Calculate QC Metrics**: Uses `sc.pp.calculate_qc_metrics` to compute metrics such as the number of genes and total counts for each cell spot.
*   **Cell Filtering**: Removes cell spots with too few genes (e.g., fewer than 200 genes) to exclude low-quality or empty spots.
*   **Gene Filtering**: Removes genes expressed in too few cell spots (e.g., fewer than 3 cell spots) to exclude low-expression or non-specific genes.

Subsequently, violin plots (`sc.pl.violin`) are used to display the distribution of `n_genes_by_counts` (number of genes per cell spot) and `total_counts` (total reads per cell spot) to assess data quality and filtering effectiveness.

![QC Violin Plots](figures/_qc_violin_plots.png)




## Spatial Autocorrelation Analysis

Spatial autocorrelation analysis aims to identify genes that exhibit non-random distribution patterns in space. This is crucial for understanding tissue structure and the formation of functional regions. This project uses Moran's I statistic to quantify the spatial autocorrelation of gene expression.

### 1. Compute Spatial Neighbors Graph

First, a spatial neighbors graph is constructed using the `sq.gr.spatial_neighbors` function. This graph defines which spatial spots are physically adjacent, typically based on their coordinate distances. Here, `coord_type="grid"` and `n_neighs=6` are used, indicating that each spot is connected to its 6 nearest neighbors, simulating a grid-like spatial relationship.

### 2. Calculate Moran's I Values

Next, Moran's I values are calculated for all highly variable genes. Moran's I is a measure of spatial autocorrelation, ranging from -1 (negative correlation) to 1 (positive correlation). Positive values indicate that similar gene expression values tend to cluster together, while negative values indicate that dissimilar gene expression values tend to be dispersed. `n_perms=100` is used to calculate p-values, assessing the significance of the observed Moran's I values.

### 3. Visualize Highly Spatially Autocorrelated Genes

The analysis results will show the genes with the highest Moran's I values. These genes are considered to have the strongest clustering patterns in space. The project selects several genes with the highest Moran's I values and visualizes their spatial expression patterns using the `sc.pl.spatial` function. This helps to intuitively observe the distribution of these genes on the tissue section, further understanding their potential biological functions.

![Top Spatial Genes](figures/_top_spatial_genes.png)




## Data Normalization, Feature Selection, and Dimensionality Reduction

To prepare the data for clustering analysis, a series of data transformation steps are required, including normalization, feature selection, and dimensionality reduction.

### 1. Data Normalization and Log Transformation

*   **Total Count Normalization**: `sc.pp.normalize_total(adata, target_sum=1e4)` is used to normalize the total counts per cell spot to a constant value (e.g., 10,000), eliminating the influence of sequencing depth differences.
*   **Log Transformation**: Subsequently, `sc.pp.log1p(adata)` applies a log transformation (log(1+x)). This helps stabilize data variance, making gene expression distributions closer to a normal distribution, thereby improving the performance of downstream analyses.

### 2. Feature Selection (Highly Variable Genes)

Not all genes contribute equally to distinguishing cell types or spatial regions. Selecting Highly Variable Genes (HVGs) can reduce data dimensionality while retaining the most important biological information. This project uses the `sc.pp.highly_variable_genes` function to identify HVGs based on their mean expression and variance. After filtering, the dataset will only contain these highly variable genes, further reducing computational complexity.

### 3. Dimensionality Reduction (PCA and UMAP)

High-dimensional data is difficult to visualize and analyze directly, thus requiring dimensionality reduction techniques.

*   **Principal Component Analysis (PCA)**: First, `sc.tl.pca(adata, svd_solver=\'arpack\')` performs PCA. PCA is a linear dimensionality reduction method that projects data onto a set of orthogonal principal components, which capture the maximum variance in the data. This helps remove redundant information.
*   **Neighbor Graph Construction**: In the PCA-reduced space, `sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)` constructs a neighbor graph between cell spots. This graph forms the basis for many graph-based clustering algorithms (e.g., Leiden and Louvain).
*   **UMAP Dimensionality Reduction**: Finally, `sc.tl.umap(adata)` applies Uniform Manifold Approximation and Projection (UMAP). UMAP is a nonlinear dimensionality reduction technique that better preserves both global and local data structures, making it highly suitable for visualizing high-dimensional biological data. The 2D or 3D space projected by UMAP is commonly used to display cell population clustering results.




## Comparison of Multiple Clustering Methods (Part 2)

This project compares the performance of four different machine learning clustering algorithms on spatial transcriptomics data: Leiden, K-Means, Louvain, and BIRCH. Each of these algorithms has unique characteristics and is suitable for different data structures and analytical goals.

### a. Leiden Clustering

The Leiden algorithm is a graph-based clustering method that aims to discover community structures within networks. It typically produces finer and higher-quality clustering results than the Louvain algorithm. This project uses `sc.tl.leiden(adata, resolution=0.5, key_added=\'leiden\')` for clustering, where the `resolution` parameter controls the granularity of the clusters.

### b. K-Means Clustering

K-Means is a classic centroid-based clustering algorithm that partitions data points into K predefined clusters, such that each data point belongs to the cluster with the nearest mean. In this project, the value of K is set to the number of classes in the ground truth labels, and clustering is performed on the PCA-reduced data (`adata.obsm[\'X_pca\']`).

### c. Louvain Clustering

The Louvain algorithm is another popular graph-based community detection algorithm that discovers clusters by optimizing modularity. It is often used for large datasets and can effectively identify hierarchical structures. This project uses `sc.tl.louvain(adata, key_added=\'louvain\')` for clustering.

### d. BIRCH Clustering

BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) is a hierarchical clustering algorithm capable of efficiently handling large datasets by constructing a Clustering Feature (CF) tree to progressively cluster data. In this project, the number of BIRCH clusters is also set to the number of classes in the ground truth labels, and clustering is performed on the PCA-reduced data.




## Data Visualization and Clustering Performance Evaluation (Part 3)

After executing multiple clustering methods, this project proceeds with visual comparison and quantitative evaluation of their results to assess their performance on spatial transcriptomics data.

### 1. Visualization Comparison

Visualization is the most intuitive way to understand clustering results. This project generates two main types of visualization plots:

*   **UMAP Dimensionality Reduction Plot Comparison**: Using the `sc.pl.umap` function, the ground truth labels are compared with the Leiden, K-Means, Louvain, and BIRCH clustering results in the UMAP dimensionality reduction space. This helps observe how well different clustering algorithms preserve and delineate cell population structures in a low-dimensional space.

    ![UMAP Comparison](figures/_umap_comparison.png)

*   **Spatial Scatter Plot Comparison**: Using the `sq.pl.spatial_scatter` function, the ground truth labels and the various clustering results are visualized on their original spatial coordinates. This directly demonstrates the accuracy and rationality of how different clustering algorithms delineate regions on the tissue section, allowing for an intuitive understanding of which algorithms' clustering results better align with biologically true regions.

    ![Spatial Comparison](figures/_spatial_comparison.png)

### 2. Quantitative Evaluation Comparison

In addition to visual comparison, this project employs two commonly used quantitative metrics to evaluate the performance of clustering algorithms:

*   **Adjusted Rand Index (ARI)**: ARI measures the agreement between two clustering results, accounting for chance. Its value ranges from -1 (no agreement) to 1 (perfect agreement). A higher ARI indicates better alignment between the clustering result and the ground truth labels.

*   **Normalized Mutual Information (NMI)**: NMI measures the amount of information shared between two clustering results, also accounting for chance. Its value ranges from 0 (no mutual information) to 1 (perfect agreement). A higher NMI indicates a stronger correlation between the clustering result and the ground truth labels.

The calculated results of these metrics will be organized into a DataFrame and printed, facilitating comparison. Finally, this project generates a bar plot to visually display the scores of different clustering methods on ARI and NMI, thereby providing a clear performance comparison.

![Clustering Method Evaluation Bar Plot](figures/evaluation_barplot.png)



