import os
import json
import logging
import numpy as np
from sklearn.cluster import DBSCAN
try:
    import hdbscan
    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False

try:
    from .vector_memory_wrapper import _index, EMB_DIM, denoise_embeddings
except ImportError:
    from vector_memory_wrapper import _index, EMB_DIM, denoise_embeddings

logger = logging.getLogger(__name__)


def _load_all_vectors():
    """Load all vectors from the FAISS index."""
    n = _index.ntotal
    if n == 0:
        return np.empty((0, EMB_DIM), dtype='float32')
    vectors = np.vstack([_index.reconstruct(i) for i in range(n)])
    return vectors


def _optimal_eps(vectors, k=5):
    """Compute optimal epsilon for DBSCAN via k-distance graph."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k+1).fit(vectors)
    distances, _ = nn.kneighbors(vectors)
    avg_dist = distances[:, 1:].mean()
    return avg_dist


def merge_tiny_clusters(labels: np.ndarray, vectors: np.ndarray, min_size: int = 6) -> np.ndarray:
    """Merge tiny clusters into nearest larger clusters using cosine similarity.

    Vectors should be L2-normalized. Tiny clusters with size < min_size are
    reassigned to the closest centroid among larger clusters; if none exist,
    they are marked as noise (-1).
    """
    if min_size <= 1:
        return labels

    labels = labels.copy()
    unique_labels = [c for c in sorted(set(labels)) if c != -1]
    if not unique_labels:
        return labels

    # Cluster sizes
    sizes = {c: int((labels == c).sum()) for c in unique_labels}
    big_clusters = [c for c in unique_labels if sizes[c] >= min_size]
    if not big_clusters:
        labels[labels != -1] = -1
        return labels

    # Centroids in normalized space
    centroids = {}
    for c in big_clusters:
        idx = np.where(labels == c)[0]
        centroids[c] = np.mean(vectors[idx], axis=0)
        cn = np.linalg.norm(centroids[c]) + 1e-8
        centroids[c] = centroids[c] / cn

    # Reassign tiny clusters
    for c in unique_labels:
        if sizes[c] >= min_size:
            continue
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        sub_centroid = np.mean(vectors[idx], axis=0)
        sub_centroid /= (np.linalg.norm(sub_centroid) + 1e-8)
        best_c, best_sim = None, -1.0
        for bc, bc_centroid in centroids.items():
            sim = float(np.dot(sub_centroid, bc_centroid))
            if sim > best_sim:
                best_sim = sim
                best_c = bc
        labels[idx] = best_c if best_c is not None else -1

    return labels


def run_dbscan(min_samples: int = None, eps: float = None, use_hdbscan: bool = True):
    """Cluster all memories using denoised embeddings and dimensionality reduction.
    
    Pipeline:
      1. Denoise embeddings (all-but-the-top + mean centering)
      2. PCA pre-reduce to 50-D
      3. HDBSCAN (or DBSCAN fallback)
      4. Merge tiny clusters into nearest large cluster
    """
    vectors = _load_all_vectors()
    if vectors.shape[0] < 5:
        return [-1] * vectors.shape[0]
    
    # 1. Denoise and pre-reduce for better clustering topology
    X = denoise_embeddings(vectors, remove_top_d=3, target_dim=min(50, vectors.shape[0]-1))

    if min_samples is None:
        min_samples = max(2, min(8, len(X) // 20))
    
    if use_hdbscan and _HDBSCAN_AVAILABLE:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_samples,
            min_samples=1,
            metric='euclidean',
            cluster_selection_method='leaf'
        )
    else:
        if eps is None:
            eps = _optimal_eps(X, k=min_samples)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    
    labels = np.array(clusterer.fit_predict(X))
    
    # 2. Merge tiny clusters into nearest large cluster
    labels = merge_tiny_clusters(labels, X, min_size=6)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    logger.info(f"Clustered {len(X)} vectors into {n_clusters} clusters ({n_noise} noise)")
    
    return labels.tolist()


if __name__ == '__main__':
    lbls = run_dbscan()
    print(f"Clustered {len(lbls)} vectors; noise points: {lbls.count(-1)}")
