import numpy as np
import pandas as pd

class CollaborativeFiltering:
    def __init__(self, data):
        """
        Initialize CollaborativeFiltering object with user-item rating data.
        
        Parameters:
        - data: DataFrame containing user-item ratings
        """
        self.data = data
        self.train_data = None
        self.test_data = None
        self.similarity_matrix = None

    def calculate_similarity_matrix(self):
        """
        Calculate item-item similarity matrix based on user ratings.
        """
        user_item_matrix = self.train_data.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)
        n_movies = user_item_matrix.shape[1]
        similarity_matrix = np.zeros((n_movies, n_movies))
        for i in range(n_movies):
            for j in range(n_movies):
                similarity_matrix[i, j] = np.dot(user_item_matrix.iloc[:, i], user_item_matrix.iloc[:, j]) / (np.linalg.norm(user_item_matrix.iloc[:, i]) * np.linalg.norm(user_item_matrix.iloc[:, j]) + 1e-9)
        self.similarity_matrix = pd.DataFrame(similarity_matrix, index=user_item_matrix.columns, columns=user_item_matrix.columns)

    def predict_ratings(self, user_id):
        """
        Predict ratings for items for a given user.
        
        Parameters:
        - user_id: ID of the user for whom to predict ratings
        
        Returns:
        - DataFrame containing predicted ratings for each item
        """
        user_ratings = self.train_data[self.train_data['UserID'] == user_id]
        predicted_ratings = pd.DataFrame(index=self.similarity_matrix.index, columns=['PredictedRating'])
        for item_id in predicted_ratings.index:
            numerator = 0
            denominator = 0
            for _, rating_row in user_ratings.iterrows():
                similarity = self.similarity_matrix.loc[item_id, rating_row['MovieID']]
                numerator += similarity * rating_row['Rating']
                denominator += similarity
            predicted_ratings.loc[item_id, 'PredictedRating'] = numerator / (denominator + 1e-9)  # Add a small value to avoid division by zero
        return predicted_ratings

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
        Evaluate the Collaborative Filtering model on the test set.
        
        Returns:
        - Mean squared error (MSE) of the predictions
        """
        self.calculate_similarity_matrix()
        mse_sum = 0
        total_predictions = 0
        for user_id in self.test_data['UserID'].unique():
            user_test_ratings = self.test_data[self.test_data['UserID'] == user_id]
            user_predicted_ratings = self.predict_ratings(user_id)
            for _, row in user_test_ratings.iterrows():
                if row['MovieID'] in user_predicted_ratings.index:
                    total_predictions += 1
                    mse_sum += (row['Rating'] - user_predicted_ratings.loc[row['MovieID'], 'PredictedRating']) ** 2
        mse = mse_sum / total_predictions
        return mse

# Example usage:
# Assuming 'data' is a pandas DataFrame with user-item ratings

# Read data from CSV files
users = pd.read_csv("users.csv")
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# Merge dataframes
data = pd.merge(pd.merge(users, ratings), movies)

# Create CollaborativeFiltering instance
cf = CollaborativeFiltering(data)

# Split data into train and test sets
cf.train_test_split(test_size=0.2)

# Evaluate the model
mse = cf.evaluate()
print("Mean Squared Error:", mse)
