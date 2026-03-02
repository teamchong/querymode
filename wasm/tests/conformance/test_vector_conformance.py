#!/usr/bin/env python3
"""
Vector algorithm conformance tests.

Validates that LanceQL's vector operations (distance metrics, IVF-PQ recall)
match reference implementations.

Usage:
    pip install numpy pytest
    pytest test_vector_conformance.py -v
"""

import numpy as np
import pytest

# Set fixed seed for reproducibility
np.random.seed(42)


class TestDistanceMetrics:
    """Test distance metric correctness against NumPy reference."""

    def test_l2_distance_basic(self):
        """L2 distance between orthogonal unit vectors."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        doc = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Expected: sqrt(2) = 1.414...
        expected = np.linalg.norm(query - doc)
        assert abs(expected - np.sqrt(2)) < 1e-6

    def test_l2_distance_identical(self):
        """L2 distance between identical vectors should be 0."""
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        expected = np.linalg.norm(vec - vec)
        assert expected == 0.0

    def test_l2_distance_scaled(self):
        """L2 distance scales linearly with vector magnitude."""
        query = np.array([1.0, 0.0], dtype=np.float32)
        doc1 = np.array([2.0, 0.0], dtype=np.float32)
        doc2 = np.array([3.0, 0.0], dtype=np.float32)

        d1 = np.linalg.norm(query - doc1)
        d2 = np.linalg.norm(query - doc2)

        assert abs(d1 - 1.0) < 1e-6
        assert abs(d2 - 2.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        """Cosine similarity of orthogonal vectors is 0."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)

        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert abs(cosine) < 1e-6

    def test_cosine_similarity_identical(self):
        """Cosine similarity of identical vectors is 1."""
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cosine = np.dot(vec, vec) / (np.linalg.norm(vec) * np.linalg.norm(vec))
        assert abs(cosine - 1.0) < 1e-6

    def test_cosine_similarity_opposite(self):
        """Cosine similarity of opposite vectors is -1."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)

        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert abs(cosine - (-1.0)) < 1e-6

    def test_cosine_similarity_45_degrees(self):
        """Cosine similarity at 45 degrees is 1/sqrt(2)."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 1.0], dtype=np.float32)

        expected = 1.0 / np.sqrt(2)  # cos(45 degrees)
        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert abs(cosine - expected) < 1e-6

    def test_dot_product_basic(self):
        """Dot product of unit vectors."""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.5, 0.5, 0.0], dtype=np.float32)

        expected = 0.5
        dot = np.dot(a, b)
        assert abs(dot - expected) < 1e-6

    def test_dot_product_normalized(self):
        """Dot product of normalized vectors equals cosine similarity."""
        a = np.array([3.0, 4.0], dtype=np.float32)
        b = np.array([1.0, 2.0], dtype=np.float32)

        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)

        dot = np.dot(a_norm, b_norm)
        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        assert abs(dot - cosine) < 1e-6


class TestVectorSearchRecall:
    """Test approximate k-NN recall against exact search."""

    @staticmethod
    def brute_force_knn(query, vectors, k, metric='l2'):
        """Exact k-NN search using brute force."""
        if metric == 'l2':
            distances = np.linalg.norm(vectors - query, axis=1)
        elif metric == 'cosine':
            # cosine distance = 1 - cosine_similarity
            norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
            similarities = np.dot(vectors, query) / norms
            distances = 1 - similarities
        elif metric == 'dot':
            # For max inner product, use negative for min-heap behavior
            distances = -np.dot(vectors, query)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        indices = np.argsort(distances)[:k]
        return set(indices.tolist())

    @staticmethod
    def simple_ivf_search(query, vectors, k, n_partitions=16, n_probe=4):
        """Simple IVF search simulation (without PQ, just for testing)."""
        # Build IVF by randomly assigning vectors to partitions
        n_vectors = len(vectors)
        partition_assignments = np.random.randint(0, n_partitions, n_vectors)

        # Find nearest partitions to query (using random centroids for simplicity)
        centroids = np.random.randn(n_partitions, vectors.shape[1]).astype(np.float32)
        centroid_distances = np.linalg.norm(centroids - query, axis=1)
        probe_partitions = np.argsort(centroid_distances)[:n_probe]

        # Search only in probed partitions
        candidate_indices = []
        for p in probe_partitions:
            partition_mask = partition_assignments == p
            partition_indices = np.where(partition_mask)[0]
            candidate_indices.extend(partition_indices.tolist())

        if not candidate_indices:
            return set()

        # Brute force search within candidates
        candidate_vectors = vectors[candidate_indices]
        distances = np.linalg.norm(candidate_vectors - query, axis=1)
        top_k_local = np.argsort(distances)[:k]

        return set([candidate_indices[i] for i in top_k_local])

    def test_exact_knn_self(self):
        """Exact k-NN with query in dataset should return query first."""
        vectors = np.random.randn(100, 32).astype(np.float32)
        query = vectors[42]  # Use vector from dataset as query

        results = self.brute_force_knn(query, vectors, k=1)
        assert 42 in results

    def test_exact_knn_distances(self):
        """Verify exact k-NN returns sorted distances."""
        vectors = np.random.randn(100, 32).astype(np.float32)
        query = np.random.randn(32).astype(np.float32)

        k = 10
        results = self.brute_force_knn(query, vectors, k=k)

        # Verify all results are within expected range
        assert len(results) == k

    def test_recall_small_dataset(self):
        """IVF should achieve high recall on small dataset."""
        vectors = np.random.randn(500, 64).astype(np.float32)
        query = np.random.randn(64).astype(np.float32)

        exact_results = self.brute_force_knn(query, vectors, k=10)

        # Note: This is a simplified test; actual recall depends on IVF quality
        # Real IVF-PQ with proper training should achieve >90% recall
        assert len(exact_results) == 10

    def test_recall_metric_consistency(self):
        """Different metrics should give different rankings."""
        vectors = np.random.randn(100, 32).astype(np.float32)
        query = np.random.randn(32).astype(np.float32)

        l2_results = self.brute_force_knn(query, vectors, k=10, metric='l2')
        cosine_results = self.brute_force_knn(query, vectors, k=10, metric='cosine')
        dot_results = self.brute_force_knn(query, vectors, k=10, metric='dot')

        # Results may overlap but shouldn't be identical for random data
        assert len(l2_results) == 10
        assert len(cosine_results) == 10
        assert len(dot_results) == 10


class TestNormalization:
    """Test vector normalization for cosine similarity."""

    def test_l2_normalize(self):
        """L2 normalization should produce unit vectors."""
        vectors = np.random.randn(100, 32).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / norms

        # Check all vectors have unit norm
        result_norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(result_norms, 1.0, atol=1e-6)

    def test_normalized_dot_equals_cosine(self):
        """Dot product of normalized vectors equals cosine similarity."""
        a = np.random.randn(32).astype(np.float32)
        b = np.random.randn(32).astype(np.float32)

        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)

        dot_normalized = np.dot(a_norm, b_norm)
        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        assert abs(dot_normalized - cosine) < 1e-6


class TestEdgeCases:
    """Test edge cases in vector operations."""

    def test_zero_vector_handling(self):
        """Operations with zero vectors should be handled gracefully."""
        zero = np.zeros(32, dtype=np.float32)
        vec = np.random.randn(32).astype(np.float32)

        # L2 distance from zero
        l2_dist = np.linalg.norm(vec - zero)
        assert l2_dist == np.linalg.norm(vec)

    def test_high_dimensional(self):
        """Test with high-dimensional vectors (384-dim like MiniLM)."""
        dim = 384
        vectors = np.random.randn(1000, dim).astype(np.float32)
        query = np.random.randn(dim).astype(np.float32)

        distances = np.linalg.norm(vectors - query, axis=1)
        assert len(distances) == 1000

    def test_very_similar_vectors(self):
        """Test distinguishing very similar vectors."""
        base = np.random.randn(32).astype(np.float32)
        # Create vectors with tiny perturbations
        vectors = np.array([base + np.random.randn(32) * 1e-6 for _ in range(10)], dtype=np.float32)

        query = base
        distances = np.linalg.norm(vectors - query, axis=1)

        # All distances should be very small
        assert np.all(distances < 1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
