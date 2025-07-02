## 📊 K-Means Clustering

### 1. K-Means Clustering for 2 Features (k=2)

- **Input**: 2D dataset (12 samples)
- **Clusters**: 2
- **Distance Metric**: Euclidean
- **Output**: Cluster labels, centroids, WCSS, and scatter plot
- **File**: `kmeans_2d_2clusters.py`

### 2. K-Means Clustering for Arbitrary Features (k=3)

- **Input**: Dataset with 4 features (12 samples)
- **Clusters**: 3
- **Distance Metric**: Euclidean
- **Visualization**: PCA for 2D plotting
- **Output**: Cluster labels, centroids, WCSS, and scatter plot
- **File**: `kmeans_nfeatures_kclusters.py`

---

## 📦 Requirements

- Python 3.6+
- NumPy
- Matplotlib
- scikit-learn (for PCA in Task 2 of K-Means)

### Install Requirements:

```bash
pip install numpy matplotlib scikit-learn
python kmeans_2d_2clusters.py
python kmeans_nfeatures_kclusters.py
