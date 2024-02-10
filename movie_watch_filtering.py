import csv

# Read ratings CSV file
ratings_file = 'ratings.csv'
movies_rated = set()
with open(ratings_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        movies_rated.add(row[1])

# Read movies CSV file
movies_file = 'movies.csv'
movies_data = []
with open(movies_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        movie_id = row[0]
        if movie_id in movies_rated:
            movies_data.append(row)

# Write filtered movies to another text file
filtered_movies_file = 'filtered_movies.csv'
with open(filtered_movies_file, 'w') as file:
    for movie in movies_data:
        file.write(','.join(movie) + '\n')

print("Filtered movies have been written to", filtered_movies_file)
