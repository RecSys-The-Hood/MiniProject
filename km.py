import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

random.seed(7)
np.random.seed(7)

# Load the movie-rating dataset
movie_data = pd.read_csv('MoviesExpanded.csv')

# Preprocess the data to get the feature matrix (X)
X = movie_data.iloc[:, 3:].astype(float).values  # Ensure all elements are of type float

# Modify the get_euclidean_distance function to handle the new feature matrix
def get_euclidean_distance(A_matrix, B_matrix):
    return np.linalg.norm(A_matrix[:, np.newaxis] - B_matrix, axis=2)

def get_initial_centroids(X, k):
    """
    Function picks k random data points from dataset X, recurring points are removed and replaced but new points
    so a result we have array of k unique points. Founded points can be used as initial centroids for k means algorithm
    Args:
        X (numpy.ndarray) : dataset points array, size N:D
        k (int): number of centroids

    Returns:
        (numpy.ndarray): array of k unique initial centroids, size K:D

    """
    number_of_samples = X.shape[0]
    sample_points_ids = random.sample(range(0, number_of_samples), k)

    centroids = [tuple(X[id]) for id in sample_points_ids]
    unique_centroids = list(set(centroids))

    number_of_unique_centroids = len(unique_centroids)

    while number_of_unique_centroids < k:
        new_sample_points_ids = random.sample(range(0, number_of_samples), k - number_of_unique_centroids)
        new_centroids = [tuple(X[id]) for id in new_sample_points_ids]
        unique_centroids = list(set(unique_centroids + new_centroids))

        number_of_unique_centroids = len(unique_centroids)

    return np.array(unique_centroids)


def get_clusters(X, centroids, distance_measuring_method):
    """
    Function finds k centroids and assigns each of the N points of array X to one centroid
    Args:
        X (numpy.ndarray): array of sample points, size N:D
        centroids (numpy.ndarray): array of centroids, size K:D
        distance_measuring_method (function): function taking 2 Matrices A (N1:D) and B (N2:D) and returning distance
        between all points from matrix A and all points from matrix B, size N1:N2

    Returns:
        dict {cluster_number: list_of_points_in_cluster}
    """

    k = centroids.shape[0]

    clusters = {}

    distance_matrix = distance_measuring_method(X, centroids)

    closest_cluster_ids = np.argmin(distance_matrix, axis=1)

    for i in range(k):
        clusters[i] = []

    for i, cluster_id in enumerate(closest_cluster_ids):
        clusters[cluster_id].append(X[i])

    return clusters


def has_centroids_covered(previous_centroids, new_centroids, distance_measuring_method, movement_threshold_delta):
    """
    Function checks if any of centroids moved more than MOVEMENT_THRESHOLD_DELTA if not we assume the centroids were found
    Args:
        previous_centroids (numpy.ndarray): array of k old centroids, size K:D
        new_centroids (numpy.ndarray): array of k new centroids, size K:D
        distance_measuring_method (function): function taking 2 Matrices A (N1:D) and B (N2:D) and returning distance
        movement_threshold_delta (float): threshold value, if centroids move less we assume that algorithm covered


    Returns: boolean True if centroids covered False if not

    """
    distances_between_old_and_new_centroids = distance_measuring_method(previous_centroids, new_centroids)
    centroids_covered = np.max(distances_between_old_and_new_centroids.diagonal()) <= movement_threshold_delta

    return centroids_covered


def perform_k_means_algorithm(X, k, distance_measuring_method, movement_threshold_delta=0):
    """
    Function performs k-means algorithm on a given dataset, finds and returns k centroids
    Args:
        X (numpy.ndarray) : dataset points array, size N:D
        distance_measuring_method (function): function taking 2 Matrices A (N1:D) and B (N2:D) and returning distance
        between all points from matrix A and all points from matrix B, size N1:N2.
        k (int): number of centroids
        movement_threshold_delta (float): threshold value, if centroids move less we assume that algorithm covered

    Returns:
        (numpy.ndarray): array of k centroids, size K:D
    """

    new_centroids = get_initial_centroids(X=X, k=k)

    centroids_covered = False

    while not centroids_covered:
        previous_centroids = new_centroids
        clusters = get_clusters(X, previous_centroids, distance_measuring_method)

        new_centroids = np.array([np.mean(clusters[key], axis=0, dtype=X.dtype) for key in sorted(clusters.keys())])

        centroids_covered = has_centroids_covered(previous_centroids, new_centroids, distance_measuring_method, movement_threshold_delta)

    return new_centroids


def visualize_clusters(data_points, centroids):
    # Assign each data point to the closest centroid
    clusters = get_clusters(data_points, centroids, get_euclidean_distance)

    # Plot the data points and centroids
    plt.figure(figsize=(8, 8))
    for i, points in clusters.items():
        points = np.array(points)
        plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {i + 1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()


def print_cluster_assignments(data_points, centroids):
    """
    Function prints the assignment of clusters for each data point
    Args:
        data_points (numpy.ndarray): array of data points, size N:D
        centroids (numpy.ndarray): array of centroids, size K:D
    """

    # Get clusters for each data point
    clusters = get_clusters(data_points, centroids, get_euclidean_distance)

    # Print cluster assignments for each data point
    for i, points in clusters.items():
        cluster_points = np.array(points)
        print(f"Cluster {i + 1}:")
        for point in cluster_points:
            print(point)


# Main function to test the k-means algorithm on the movie-rating dataset
def test_kmeans_movie_data():
    num_clusters = 5  # Number of clusters
    centroids = perform_k_means_algorithm(X, num_clusters, get_euclidean_distance)

    # Print the centroids
    print("Final centroids:")
    print(centroids)

    # Visualize the data and centroids
    # visualize_clusters(X, centroids)

    # Print cluster assignments for each data point
    print("Cluster Assignments:")
    print_cluster_assignments(X, centroids)

# Test the k-means algorithm on the movie-rating dataset
test_kmeans_movie_data()
