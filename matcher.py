import faiss
import numpy as np

class FaissMatcher:
    def __init__(self, training_features):
        """Initialize the matcher with training features."""
        self.training_features = training_features
        self.index = faiss.IndexFlatL2(training_features.shape[1])  # L2 Distances
        self.index.add(training_features)  # Add training features to the index

    def search(self, query_features, k, threshold=None):
        """Perform a similarity search for the query features."""
        distances, indices = self.index.search(query_features, k)
        results = []
        for i in range(len(query_features)):
            filtered_indices = []
            filtered_distances = []
            for dist, idx in zip(distances[i], indices[i]):
                if threshold is None or dist <= threshold:
                    filtered_indices.append(idx)
                    filtered_distances.append(dist)
            results.append((filtered_indices, filtered_distances))
        return results

# Example Usage:
# training_data = np.random.rand(100, 128).astype('float32')  # 100 samples, 128-dimensional feature
# query_data = np.random.rand(10, 128).astype('float32')      # 10 query samples
# matcher = FaissMatcher(training_data)
# top_k_results = matcher.search(query_data, k=5, threshold=0.5)  # Search for top-5 matches with a threshold
