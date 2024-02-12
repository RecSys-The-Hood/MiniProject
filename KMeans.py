import numpy as np
import pandas as pd

class KMeans:
    def __init__(self, k=3, max_iterations=100, distance_metric='euclidean'):
        """
        Initialize KMeans object with specified parameters.
        
        Parameters:
        - k: Number of clusters
        - max_iterations: Maximum number of iterations
        - distance_metric: Distance metric to use ('euclidean' or 'manhattan')
        """
        self.k = k
        self.max_iterations = max_iterations
        self.distance_metric = distance_metric
        self.centroids = None
        self.clusters = None

    def initialize_centroids(self, data):
        """
        Initialize centroids randomly from the dataset.
        
        Parameters:
        - data: DataFrame containing the data points
        
        Sets the centroids attribute to the randomly selected data points.
        """
        self.centroids = data.sample(self.k)

    def assign_clusters(self, data):
        """
        Assign each data point to the nearest centroid.
        
        Parameters:
        - data: DataFrame containing the data points
        
        Computes distances between each data point and centroid,
        then assigns each data point to the cluster with the nearest centroid.
        """
        if self.distance_metric == 'euclidean':
            distances = np.sqrt(((data.values[:, np.newaxis] - self.centroids.values)**2).sum(axis=2))
        elif self.distance_metric == 'manhattan':
            distances = np.abs(data.values[:, np.newaxis] - self.centroids.values).sum(axis=2)
        self.clusters = np.argmin(distances, axis=1)

    def update_centroids(self, data):
        """
        Update centroids based on the mean of data points in each cluster.
        
        Parameters:
        - data: DataFrame containing the data points
        
        Updates centroids to the mean of all data points assigned to each cluster.
        """
        self.centroids = np.array([data[self.clusters == i].mean(axis=0) for i in range(self.k)])

    def compute_error(self, data):
        """
        Compute the error of the clustering.
        
        Parameters:
        - data: DataFrame containing the data points
        
        Returns:
        - Mean error of the clustering
        """
        if self.distance_metric == 'euclidean':
            errors = np.sqrt(((data.values - self.centroids[self.clusters])**2).sum(axis=1))
        elif self.distance_metric == 'manhattan':
            errors = np.abs(data.values - self.centroids[self.clusters]).sum(axis=1)
        return errors.mean()

    def fit(self, data):
        """
        Fit the K-means model to the data.
        
        Parameters:
        - data: DataFrame containing the data points
        
        Initializes centroids, assigns clusters, and updates centroids iteratively
        until convergence or maximum iterations are reached.
        """
        self.initialize_centroids(data)
        for _ in range(self.max_iterations):
            old_centroids = self.centroids.copy()
            self.assign_clusters(data)
            self.update_centroids(data)
            if np.allclose(old_centroids, self.centroids):
                break

# Example usage:
# Assuming 'data' is a pandas DataFrame with numerical columns
# and 'k' is the number of clusters

# Read data from CSV files
users = pd.read_csv("users.csv")
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# Merge dataframes
data = pd.merge(pd.merge(users, ratings), movies)

# Drop non-numeric columns
data = data.drop(columns=["UserID", "MovieID", "Title", "Genres", "Zip-code"])

# Normalize data
data = (data - data.mean()) / data.std()

# Create KMeans instance
kmeans = KMeans(k=5, max_iterations=100, distance_metric='euclidean')

# Fit the model
kmeans.fit(data)

print("Final centroids:\n", kmeans.centroids)
print("Cluster assignments:\n", kmeans.clusters)
print("Final error:", kmeans.compute_error(data))
