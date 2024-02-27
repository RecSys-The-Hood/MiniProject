import numpy as np
import pandas as pd
import math

class SimilarityMetric:

    def calculateSimilarity(self, u, v, u_mean, v_mean):
        pass

class CosineSimilarity(SimilarityMetric):
    def calculateSimilarity(self, u, v, u_mean, v_mean):
        u = np.nan_to_num(u, nan=0)
        v = np.nan_to_num(v, nan=0)
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)

class PCCSimilarity(SimilarityMetric):
    def calculateSimilarity(self, u, v, u_mean, v_mean):
        return self.__pearson_correlation(u, v, u_mean, v_mean)
    
    def __pearson_correlation(self, x, y, x_mean, y_mean):

        if math.isnan(x_mean) or math.isnan(y_mean):
            return 0
        
        x_df = pd.DataFrame(x, columns = ['a'])
        y_df = pd.DataFrame(y, columns = ['a'])

        x_indices = x_df[x_df['a'].notnull()].index
        y_indices = y_df[y_df['a'].notnull()].index

        indices = x_indices.intersection(y_indices)

        x_reqd = x_df.iloc[indices]
        y_reqd = y_df.iloc[indices]
        
        covariance = sum((x_reqd['a'] - x_mean) * (y_reqd['a'] - y_mean))
        std_x = math.sqrt(sum((x_reqd['a'] - x_mean)**2))
        std_y = math.sqrt(sum((y_reqd['a'] - y_mean)**2))

        if (std_x == 0 or std_y == 0):
            return 0
        
        return covariance / (std_x * std_y)


class CollaborativeFiltering:

    def __init__(self, data, metric:SimilarityMetric):
        """
        Initialize CollaborativeFiltering object with user-item rating data.
        
        Parameters:
        - data: DataFrame containing user-item ratings
        """
        self.data = data
        self.similarity_matrix = None
        self.metric = metric
        self.movies = pd.read_csv("movies.csv")
        self.movies.set_index('MovieID', inplace=True)

    def calculate_similarity_matrix(self):
        pass

    def predict_ratings(self, user_id):
        pass
    
    def getMovies(self, user_id, movie_id_list):

        recommendations = []
    
        for i in movie_id_list:

            user = pd.DataFrame(self.data.loc[user_id], index = self.data.columns, columns = ['a'])
            rating = user.loc[i]
            movie = self.movies.loc[i]
            recommendations.append([i, rating, movie['Title'], movie['Genres']])

        return recommendations
    
    def getSimilarityMatrix(self):
        return self.similarity_matrix

class CollaborativeFilteringItemItem(CollaborativeFiltering):

    def __init__(self, data, metric:SimilarityMetric):
        super().__init__(data, metric)
        self.means = self.data.mean(axis=0)
    
    def calculate_similarity_matrix(self):
        
        # user_item_matrix = self.train_data.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)
        
        n_movies = self.data.shape[1]
        similarity_matrix = np.zeros((n_movies, n_movies))
        for i in range(n_movies):
            for j in range(i+1):
                temp = self.metric.calculateSimilarity(self.data.iloc[:, i], self.data.iloc[:, j], self.means[self.data.columns[i]], self.means[self.data.columns[j]])
                similarity_matrix[i, j] = temp
                similarity_matrix[j, i] = temp

        self.similarity_matrix = pd.DataFrame(similarity_matrix, index=self.data.columns, columns=self.data.columns)

    def predict_ratings(self, user_id):
        """
        Predict ratings for items for a given user.

        Parameters:
        - user_id: ID of the user for whom to predict ratings

        Returns:
        - DataFrame containing predicted ratings for each item
        """

        user_ratings = self.data.loc[user_id]
        predicted_ratings = pd.DataFrame(index=self.data.columns, columns=['PredictedRating'])

        for movie_id in predicted_ratings.index:
            numerator = 0
            denominator = 0

            for other_movie_id in predicted_ratings.index:
                if (other_movie_id != movie_id):
                    similarity = self.similarity_matrix.loc[movie_id, other_movie_id]
                    other_movie_rating = user_ratings[other_movie_id]

                    if not np.isnan(other_movie_rating):
                        numerator += similarity * (other_movie_rating) 
                        denominator += abs(similarity)
            
            predicted_ratings.loc[movie_id, 'PredictedRating'] = numerator / (denominator + 1e-9)
            
        return predicted_ratings
        