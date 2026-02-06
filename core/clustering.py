import os
import json
import numpy as np
from sklearn.cluster import DBSCAN
try:
    import hdbscan
    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False

try:
    from .vector_memory_wrapper import _load_index, EMB_DIM
except ImportError:
    from vector_memory_wrapper import _load_index, EMB_DIM


def _load_all_vectors():
    index = _load_index(EMB_DIM)
    n = index.ntotal
    if n == 0:
        return np.empty((0, EMB_DIM))
    vectors = np.vstack([index.reconstruct(i) for i in range(n)])
    return vectors

def _optimal_eps(vectors, k=5):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k+1).fit(vectors)
    distances, _ = nn.kneighbors(vectors)
    avg_dist = distances[:, 1:].mean()
    return avg_dist

def run_dbscan(min_samples: int = None, eps: float = None, use_hdbscan: bool = False):
    """Cluster all memories using DBSCAN (or HDBSCAN)."""
    vectors = _load_all_vectors()
    if vectors.shape[0] == 0:
        return []
    if min_samples is None:
        min_samples = max(2, EMB_DIM // 2)
    if eps is None:
        eps = _optimal_eps(vectors, k=5)
    if use_hdbscan and _HDBSCAN_AVAILABLE:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_samples, metric='euclidean')
    else:
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = clusterer.fit_predict(vectors)
    return labels.tolist()

if __name__ == '__main__':
    lbls = run_dbscan()
    print(f"Clustered {len(lbls)} vectors; noise points: {lbls.count(-1)}")
