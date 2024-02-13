import numpy as np
import pandas as pd

class KMeans:
    def __init__(self, data, k=3, max_iterations=100):
        """
        Initialize KMeans object with the data and parameters.
        
        Parameters:
        - data: DataFrame with numerical columns
        - k: Number of clusters
        - max_iterations: Maximum number of iterations
        """
        self.data = data
        self.train_data = None
        self.test_data = None
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.clusters = None

    def initialize_centroids(self):
        """
        Initialize centroids randomly from the training data.
        """
        self.centroids = self.train_data.sample(self.k)

    def assign_clusters(self):
        """
        Assign data points to the nearest centroid.
        """
        distances = np.sqrt(((self.train_data.values[:, np.newaxis] - self.centroids.values)**2).sum(axis=2))
        self.clusters = np.argmin(distances, axis=1)

    def update_centroids(self):
        """
        Update centroids based on the mean of data points in each cluster.
        """
        self.centroids = np.array([self.train_data[self.clusters == i].mean(axis=0) for i in range(self.k)])

    def train_test_split(self, test_size=0.2):
        """
        Split the data into training and test sets.
        
        Parameters:
        - test_size: Fraction of the data to be used for testing
        """
        np.random.seed(42)  # for reproducibility
        mask = np.random.rand(len(self.data)) < 1 - test_size
        self.train_data = self.data[mask]
        self.test_data = self.data[~mask]

    def evaluate(self):
        """
        Evaluate the K-means model on the test set.
        
        Returns:
        - Accuracy of the predictions
        """
        self.initialize_centroids()
        for _ in range(self.max_iterations):
            old_centroids = self.centroids.copy()
            self.assign_clusters()
            self.update_centroids()
            if np.allclose(old_centroids, self.centroids):
                break
        
        # Calculate accuracy
        correct_predictions = 0
        total_predictions = 0
        for _, row in self.test_data.iterrows():
            # Find the centroid closest to the test point
            test_point = row.dropna().values
            distances = np.sqrt(((test_point - self.centroids)**2).sum(axis=1))
            closest_centroid_idx = np.argmin(distances)
            predicted_cluster = closest_centroid_idx
            true_cluster = row['Cluster']
            total_predictions += 1
            if predicted_cluster == true_cluster:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions
        return accuracy

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

# Create KMeans instance
kmeans = KMeans(data)

# Split data into train and test sets
kmeans.train_test_split(test_size=0.2)

# Evaluate the model
accuracy = kmeans.evaluate()
print("Accuracy:", accuracy)
