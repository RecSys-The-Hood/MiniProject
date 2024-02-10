import pandas as pd

# Given dataset
df = pd.read_csv("movies.csv")

def row_to_dict(row):
    return row.to_dict()

# Convert each row to a dictionary
dataset = df.apply(row_to_dict, axis=1).tolist()

# Function to split genres
def split_genres(movie):
    genres = movie["Genres"].split('|')
    return [{"MovieID": movie["MovieID"], "Title": movie["Title"], "Genre": genre} for genre in genres]

# Splitting the dataset
split_dataset = [split_genres(movie) for movie in dataset]

# Flattening the list of lists
flattened_dataset = [item for sublist in split_dataset for item in sublist]

# Writing to CSV files for each genre
for genre in set(movie["Genre"] for movie in flattened_dataset):
    genre_data = [movie for movie in flattened_dataset if movie["Genre"] == genre]
    df = pd.DataFrame(genre_data)
    filename = f"src/{genre}.csv"
    df.to_csv(filename, index=False)

print("CSV files written successfully.")
