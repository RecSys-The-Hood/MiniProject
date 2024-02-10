import csv
import matplotlib.pyplot as plt

ratings_file = "ratings.csv"
users_file = "users.csv"
movies_file = "movies.csv"


movies_dict = {}
ratings_dict = {}
users_dict = {}

header_movies = []
header_ratings = []
header
with open(ratings_file, 'r') as file:
    reader = csv.reader(file)
    
    next(reader)  # Skip header
    for row in reader:
        movies_rated.add(row[1])