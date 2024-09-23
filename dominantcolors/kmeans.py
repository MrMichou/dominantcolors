import numpy as np

MAX_ITERATIONS = 20
THRESHOLD = 3


def kmeans(data: np.ndarray, sample_weights: np.ndarray, n_clusters=2) -> np.ndarray:
    """K-Means Clustering using triangle inequality and weighted averages.
    Returns the centroids sorted by the sum of their weighted assignments"""

    # initialize centroids using k-means++
    rng = np.random.default_rng(seed=42)

    centroids = np.zeros((n_clusters, 3))
    centroids[0] = rng.choice(data, 1, axis=0, replace=False)

    for i in range(1, n_clusters):
        dists = np.min(
            [np.linalg.norm(data - centroid, axis=1) for centroid in centroids[:i]], axis=0
        )
        probs = dists / np.sum(dists)
        centroids[i] = rng.choice(data, 1, p=probs, axis=0, replace=False)

    min_distances = np.full(data.shape[0], fill_value=np.inf, dtype=np.float64)
    assignments = np.zeros(data.shape[0], dtype=np.uint8)

    for _ in range(MAX_ITERATIONS):
        # calculate pairwise distances between centroids
        centroid_distances = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)

        for i, point in enumerate(data):
            oldk = assignments[i]
            for k, centroid in enumerate(centroids):
                # checking triangle inequality
                if centroid_distances[k, oldk] >= 2 * min_distances[i]:
                    continue

                if (dist := np.linalg.norm(point - centroid)) <= min_distances[i]:
                    assignments[i] = k
                    min_distances[i] = dist

        prev_centroids = centroids.copy()
        for i in range(n_clusters):
            assigned_clusters = assignments == i
            if not np.any(assigned_clusters):
                continue
            centroids[i] = np.average(
                data[assigned_clusters], weights=sample_weights[assigned_clusters], axis=0
            )

        if np.linalg.norm(centroids - prev_centroids) < THRESHOLD:
            break

    # sort centroids by sum of their weighted assignments
    weight_per_centroid = np.zeros(n_clusters)
    for i in range(n_clusters):
        weight_per_centroid[i] = np.sum(sample_weights[assignments == i])

    centroids = centroids[np.argsort(-weight_per_centroid)]
    return centroids.astype(np.uint8)
