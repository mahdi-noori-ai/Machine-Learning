---

# Hierarchical Clustering with Agglomerative and Divisive Methods

This notebook demonstrates how to implement and apply **Hierarchical Clustering** using both **Agglomerative** (bottom-up) and **Divisive** (top-down) approaches. Hierarchical clustering is a method of cluster analysis that builds a hierarchy of clusters and is widely used in exploratory data analysis.

## Features

- **Agglomerative Clustering**: A bottom-up approach where each data point starts as its own cluster, and pairs of clusters are merged as one moves up the hierarchy.
- **Divisive Clustering**: A top-down approach where all data points start in one cluster, and clusters are recursively split.
- **Recursive K-Means for Divisive Clustering**: Uses K-Means clustering to perform the recursive divisive clustering.

## Methods Overview

### 1. Agglomerative Clustering (Bottom-up)
Agglomerative clustering is implemented using the `AgglomerativeClustering` class from the `sklearn.cluster` module. This method begins with each data point as its own cluster, and iteratively merges the closest clusters based on a distance metric until all points are part of a single cluster.

### 2. Divisive Clustering (Top-down)
Divisive clustering is implemented using a recursive method where K-Means is applied to each cluster to split it into two smaller clusters. The process continues until the desired number of clusters is reached.

## Dependencies

To run this notebook, ensure the following libraries are installed:

- `numpy`
- `scikit-learn`
- `matplotlib` (for visualizing the clustering)
- `scipy` (for linkage and dendrograms)

You can install these dependencies using pip:

```bash
pip install numpy scikit-learn matplotlib scipy
```

## How It Works

1. **Data Loading**:
   The notebook starts by loading the data for clustering analysis. This could be synthetic data generated using `make_blobs()` or real-world datasets.

2. **Agglomerative Clustering**:
   - Performs bottom-up hierarchical clustering using the `AgglomerativeClustering` class.
   - Computes pairwise distances between points and merges the closest clusters.
   - Uses linkage methods like single, complete, or average linkage to define the distance between clusters.
   
   Example:
   ```python
   from sklearn.cluster import AgglomerativeClustering
   
   agglomerative = AgglomerativeClustering(n_clusters=4, linkage='ward')
   labels = agglomerative.fit_predict(X)
   ```

3. **Divisive Clustering**:
   - A recursive K-Means approach is used to split clusters into two at each step.
   - The splitting continues until the maximum number of clusters is reached.
   
   Example:
   ```python
   from sklearn.cluster import KMeans
   
   def divisive_clustering(X, max_clusters=4):
       # Recursive divisive clustering using KMeans
       # ... (implementation details)
   ```

4. **Visualizing Clusters**:
   - The clusters are visualized using `matplotlib` to plot the clustered data points.
   - The hierarchical structure can be visualized using a dendrogram from the `scipy.cluster.hierarchy` module.

5. **Evaluation**:
   - You can evaluate the clustering results by visual inspection or by using metrics such as the silhouette score.

## Usage

1. **Load Your Data**:
   Replace the synthetic or demo data used in the notebook with your dataset. Ensure your data is numerical and formatted correctly.

2. **Choose a Clustering Method**:
   Select whether you want to use agglomerative clustering or divisive clustering.

3. **Run the Clustering**:
   Execute the clustering function (either agglomerative or divisive) on your data.

4. **Visualize the Results**:
   Use matplotlib to visualize the clusters or a dendrogram to observe the hierarchical structure.

## Example

Here’s an example snippet showing how divisive clustering is applied using recursive K-Means:

```python
from sklearn.cluster import KMeans

def divisive_clustering(X, max_clusters=4):
    clusters = [(X, [i for i in range(len(X))])]
    labels = np.zeros(X.shape[0], dtype=int)

    current_cluster_label = 0
    while len(clusters) < max_clusters:
        new_clusters = []
        for cluster_data, cluster_indices in clusters:
            if len(cluster_data) < 2:
                labels[cluster_indices] = current_cluster_label
                current_cluster_label += 1
                continue
            kmeans = KMeans(n_clusters=2, random_state=42)
            sub_labels = kmeans.fit_predict(cluster_data)
            for sub_cluster_label in range(2):
                indices = [i for i, label in zip(cluster_indices, sub_labels) if label == sub_cluster_label]
                new_clusters.append((X[indices], indices))
                labels[indices] = current_cluster_label
                current_cluster_label += 1
        clusters = new_clusters
    return labels
```

## Conclusion

This notebook provides a clear and concise implementation of both agglomerative and divisive hierarchical clustering techniques. It is a great tool for exploring data and understanding the underlying cluster structure.

---
\
