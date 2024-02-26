import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

def get_euclidean_distance(A_matrix, B_matrix):
    return np.linalg.norm(A_matrix[:, np.newaxis] - B_matrix, axis=2)

class KMeans:
    def __init__(self, num_clusters, distance_measure=get_euclidean_distance, movement_threshold_delta=0):
        self.num_clusters = num_clusters
        self.distance_measure = distance_measure
        self.movement_threshold_delta = movement_threshold_delta
        self.centroids = None
        self.point_cluster = {}
        self.cost = 0

    def get_cost(self):
        return self.cost
    
    def get_pointCluster(self):
        return self.point_cluster

    def fit(self, X):
        self.centroids,clusters = self._perform_k_means_algorithm(X)
        self.cost = self._calculate_cost(self.centroids,clusters)
        print(self.point_cluster)
    
    def _get_initial_centroids(self, X):
        number_of_samples = X.shape[0]
        sample_points_ids = random.sample(range(0, number_of_samples), self.num_clusters)
        print("sample_points_ids")
        print(sample_points_ids)
        # print(X)
        centroids = [tuple(X[id]) for id in sample_points_ids]
        print("the tuples")
        print(centroids[0])
        unique_centroids = list(centroids)

        number_of_unique_centroids = len(unique_centroids)

        while number_of_unique_centroids < self.num_clusters:
            print("Inside While Loop")
            print(number_of_unique_centroids)
            new_sample_points_ids = random.sample(range(0, number_of_samples), self.num_clusters - number_of_unique_centroids)
            new_centroids = [tuple(X[id]) for id in new_sample_points_ids]
            unique_centroids = list(set(unique_centroids + new_centroids))

            number_of_unique_centroids = len(unique_centroids)

        return np.array(unique_centroids)

    def _get_clusters(self, X,centroids):
        clusters = {}
        # print("Centroid")
        # print(centroids)
        distance_matrix = self.distance_measure(X, centroids)
        print("Distance Matrix")
        print(distance_matrix)
        closest_cluster_ids = np.argmin(distance_matrix, axis=1)
        print("closest")
        print(closest_cluster_ids)
        for i in range(self.num_clusters):
            clusters[i] = []

        for i, cluster_id in enumerate(closest_cluster_ids):
            clusters[cluster_id].append(X[i])
            self.point_cluster[i] = cluster_id
        return clusters

    def _has_centroids_covered(self, previous_centroids, new_centroids):
        distances_between_old_and_new_centroids = self.distance_measure(previous_centroids, new_centroids)
        centroids_covered = np.max(distances_between_old_and_new_centroids.diagonal()) <= self.movement_threshold_delta

        return centroids_covered

    def _perform_k_means_algorithm(self, X):
        new_centroids = self._get_initial_centroids(X)
        # print("New Centroid")
        # print(new_centroids)
        centroids_covered = False

        while not centroids_covered:
            previous_centroids = new_centroids
            # print("X")
            # print(X)
            clusters = self._get_clusters(X,previous_centroids)
            # print(clusters)
            new_centroids = np.array([np.mean(clusters[key], axis=0, dtype=X.dtype) for key in sorted(clusters.keys())])

            centroids_covered = self._has_centroids_covered(previous_centroids, new_centroids)

        return new_centroids,clusters

    def visualize_clusters(self, X):
        # Assign each data point to the closest centroid
        clusters = self._get_clusters(X,self.centroids)

        # Plot the data points and centroids
        plt.figure(figsize=(8, 8))
        for i, points in clusters.items():
            points = np.array(points)
            plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {i + 1}')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='x', color='red', label='Centroids')
        plt.title('K-Means Clustering')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()

    def _calculate_cost(self, clusters, centroids):
        total_ssd = 0
        # print("CLuters")
        # print(clusters)
        for i, points in enumerate(clusters):
            centroid = centroids[i]
            total_ssd += np.sum((np.array(points) - centroid) ** 2)
        return total_ssd
    
    def print_cluster_assignments(self, X):
        # Get clusters for each data point
        clusters = self._get_clusters(X,self.centroids)

        # Print cluster assignments for each data point
        for i, points in clusters.items():
            cluster_points = np.array(points)
            print(f"Cluster {i + 1}:")
            for point in cluster_points:
                print(point)
