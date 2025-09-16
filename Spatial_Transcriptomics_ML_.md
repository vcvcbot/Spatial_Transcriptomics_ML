##### =========================================================================
#### PART 0: Environment Setup and Package Installation
##### =========================================================================
#### Install all necessary Python packages
##### =========================================================================


```python
!pip install scanpy squidpy numpy pandas matplotlib seaborn scikit-learn
!pip3 install louvain igraph leidenalg
!pip install gseapy
```

#### =======================================================================
#### PART 1: Data Loading, Preprocessing, and Quality Control
#### 1. Import Required Libraries
#### =======================================================================


```python
import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, Birch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import os

# Create a folder to store the generated images.
if not os.path.exists('figures'):
    os.makedirs('figures')
```

##### 2.Load the example dataset
#####Use the Visium H&E stained mouse brain dataset provided by Squidpy.


```python
adata = sq.datasets.visium_hne_adata()
print(f"Original data dimensions: {adata.shape}")
print("\nObservation (spots) metadata (first 5 rows):")
print(adata.obs.head())
print("\nVariable (genes) metadata (first 5 rows):")
print(adata.var.head())
```

##### 3.Visualize Ground Truth
#####Use the 'cluster' column as the true labels to plot the spatial distribution map.
#####This will generate an image showing the actual positions of different cell clusters on the tissue sample.


```python
print("\nGenerating Ground Truth image...")
sq.pl.spatial_scatter(
    adata,
    color="cluster",      # Specify the 'cluster' column as the color label
    title="Ground Truth", # Set the image title
    figsize=(7, 7),       # Set the image size for clear display
    save="_ground_truth_spatial.png"
)
plt.show() # Display the image in Colab
```

##### 4. Data Quality Control (QC) and Gene Expression Statistical Analysis


```python
sc.pp.calculate_qc_metrics(adata, inplace=True)
print(f"\nData dimensions before filtering: {adata.shape}")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
print(f"Data dimensions after filtering: {adata.shape}")
```

##### Gene Expression Statistical Analysis (Violin Plot)


```python
print("\n=== Gene Expression Statistical Analysis ===")
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts'],
             jitter=0.4, multi_panel=True, save="_qc_violin_plots.png")
plt.show() # Display the image in Colab
```

##### ==============================================================================
##### Spatial Autocorrelation Analysis
##### ==============================================================================


```python
import matplotlib.pyplot as plt

print("\nSpatial Autocorrelation Analysis)")

# First, calculate the spatial neighbors graph
# This defines which spots are adjacent based on their physical distances
sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)

# Calculate Moran's I values for all highly variable genes
# n_perms specifies the number of random permutations used to calculate the p-value; increasing this value improves accuracy but takes longer to compute
sq.gr.spatial_autocorr(
    adata,
    mode="moran",
    genes=adata.var_names,
    n_perms=100,
    n_jobs=1,
)

# Examine the Moran's I results and identify the genes with the highest spatial autocorrelation
moran_i_results = adata.uns["moranI"]
print("\n--- Moran's I Analysis Results (Top 10 Genes with Highest Positive Correlation) ---")
print(moran_i_results.head(10))

# Visualize the spatial expression patterns of the genes with the highest Moran's I values
top_spatial_genes = moran_i_results.index[:4]
print(f"\nVisualizing the spatial expression of genes: {list(top_spatial_genes)}...")
sc.pl.spatial(
    adata,
    color=top_spatial_genes,
    ncols=2,
    cmap="viridis",
    save="_top_spatial_genes.png"
)
plt.show()
```

##### 5. Data Normalization, Feature Selection, and Dimensionality Reduction


```python
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var.highly_variable]
print(f"\nData dimensions after selecting highly variable genes: {adata.shape}")
```

##### PCA


```python
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
```

##### ==============================================================================
##### PART 2: Comparative Analysis of Multiple Clustering Methods
##### ==============================================================================

##### a. Leiden Clustering


```python
print("\n--- Performing Leiden Clustering ---")
sc.tl.leiden(adata, resolution=0.5, key_added="leiden")
```

##### b. K-Means Clustering


```python
kmeans = KMeans(n_clusters=len(adata.obs['cluster'].unique()), random_state=0, n_init=10).fit(adata.obsm['X_pca']) # n_init=10 is the new default
adata.obs['kmeans'] = kmeans.labels_.astype(str)
```

##### c. Louvain Clustering


```python
print("--- Performing Louvain Clustering ---")
sc.tl.louvain(adata, key_added="louvain")
```

##### d. BIRCH Clustering


```python
print("--- Performing BIRCH Clustering ---")
birch = Birch(n_clusters=len(adata.obs['cluster'].unique())).fit(adata.obsm['X_pca'])
adata.obs['birch'] = birch.labels_.astype(str)

print("\n=== Multiple clustering methods completed. ===")
```

##### ==============================================================================
##### PART 3: Data Visualization and Clustering Performance Evaluation
##### ==============================================================================


```python
keys_to_delete = ['leiden_colors', 'kmeans_colors', 'louvain_colors', 'birch_colors']
for key in keys_to_delete:
    if key in adata.uns:
        del adata.uns[key]
```

#### 1. Visualization Comparison


```python
print("\n--- Visualization Comparison ---")
sc.pl.umap(adata, color=['cluster', 'leiden', 'kmeans', 'louvain', 'birch'],
           title=['Ground Truth', 'Leiden', 'K-Means', 'Louvain', 'BIRCH'],
           save="_umap_comparison.png")
plt.show()


```


```python
sq.pl.spatial_scatter(adata, color=['cluster', 'leiden', 'kmeans', 'louvain', 'birch'],
                      title=['Ground Truth', 'Leiden', 'K-Means', 'Louvain', 'BIRCH'],
                      save="_spatial_comparison.png", ncols=3)
plt.show()
```

#### 2. Quantitative Evaluation Comparison


```python
print("\n--- Quantitative Evaluation Comparison ---")
ground_truth_labels = adata.obs['cluster']
leiden_labels = adata.obs['leiden']
kmeans_labels = adata.obs['kmeans']
louvain_labels = adata.obs['louvain']
birch_labels = adata.obs['birch']
```


```python
# Calculate ARI (Adjusted Rand Index)
ari_leiden = adjusted_rand_score(ground_truth_labels, leiden_labels)
ari_kmeans = adjusted_rand_score(ground_truth_labels, kmeans_labels)
ari_louvain = adjusted_rand_score(ground_truth_labels, louvain_labels)
ari_birch = adjusted_rand_score(ground_truth_labels, birch_labels)
```


```python
# Calculate NMI (Normalized Mutual Information)
nmi_leiden = normalized_mutual_info_score(ground_truth_labels, leiden_labels)
nmi_kmeans = normalized_mutual_info_score(ground_truth_labels, kmeans_labels)
nmi_louvain = normalized_mutual_info_score(ground_truth_labels, louvain_labels)
nmi_birch = normalized_mutual_info_score(ground_truth_labels, birch_labels)
```


```python
# Create a DataFrame for the evaluation results
evaluation_df = pd.DataFrame({
    'Method': ['Leiden', 'K-Means', 'Louvain', 'BIRCH'],
    'ARI': [ari_leiden, ari_kmeans, ari_louvain, ari_birch],
    'NMI': [nmi_leiden, nmi_kmeans, nmi_louvain, nmi_birch]
})

print("\nClustering Performance Evaluation Results:")
print(evaluation_df)
```


```python
# Plot a bar chart of the evaluation metrics
evaluation_df.plot(x='Method', y=['ARI', 'NMI'], kind='bar', figsize=(10, 7), grid=True)
plt.title('Clustering Method Evaluation')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.savefig('figures/evaluation_barplot.png')
plt.show()
```
