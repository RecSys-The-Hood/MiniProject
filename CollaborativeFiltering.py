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
        self.similarity_matrix = None

    def calculate_similarity_matrix(self):
        """
        Calculate item-item similarity matrix based on user ratings.
        """
        # Pivot the data to get a matrix where rows represent users and columns represent items
        user_item_matrix = self.data.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)
        
        # Calculate cosine similarity between items
        item_similarity = np.dot(user_item_matrix.T, user_item_matrix) / (np.linalg.norm(user_item_matrix.T, axis=0) * np.linalg.norm(user_item_matrix, axis=1))
        
        self.similarity_matrix = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

    def predict_ratings(self, user_id):
        """
        Predict ratings for items for a given user.
        
        Parameters:
        - user_id: ID of the user for whom to predict ratings
        
        Returns:
        - DataFrame containing predicted ratings for each item
        """
        # Get ratings of the user for all items
        user_ratings = self.data[self.data['UserID'] == user_id]
        
        # Initialize an empty DataFrame to store predicted ratings
        predicted_ratings = pd.DataFrame(index=self.similarity_matrix.index, columns=['PredictedRating'])
        
        for item_id in predicted_ratings.index:
            # Initialize the numerator and denominator of the prediction formula
            numerator = 0
            denominator = 0
            for _, rating_row in user_ratings.iterrows():
                # Calculate the weighted sum of ratings for similar items
                similarity = self.similarity_matrix.loc[item_id, rating_row['MovieID']]
                numerator += similarity * rating_row['Rating']
                denominator += similarity
            
            # Predict the rating for the item
            predicted_ratings.loc[item_id, 'PredictedRating'] = numerator / (denominator + 1e-9)  # Add a small value to avoid division by zero
        
        return predicted_ratings

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

# Calculate item-item similarity matrix
cf.calculate_similarity_matrix()

# Predict ratings for a specific user (e.g., UserID 1)
user_id = 1
predicted_ratings = cf.predict_ratings(user_id)
print("Predicted ratings for user", user_id, ":\n", predicted_ratings)
