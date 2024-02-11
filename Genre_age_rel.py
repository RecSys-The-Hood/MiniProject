import csv
import matplotlib.pyplot as plt

ratings_file = "ratings.csv"
users_file = "users.csv"
movies_file = "movies.csv"

movies_dict = {}
ratings_dict = {}
users_dict = {}
genre_count_user={}
def find_genres_for_movie(genre_movie_dict, movie_id):
    genres_for_movie = []
    for genre, movie_ids in genre_movie_dict.items():
        if movie_id in movie_ids:
            genres_for_movie.append(genre)
    return genres_for_movie
# Read movies CSV file
with open(movies_file, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        movie_id = row['MovieID']
        genres = row['Genres'].split('|')
        for genre in genres:
            if genre not in movies_dict:
                movies_dict[genre] = []
            movies_dict[genre].append(movie_id)

# print(movies_dict)
# Read users CSV file to get user age
with open(users_file, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        users_dict[row['UserID']] = row['Age']
# print(users_dict)
# Read ratings CSV file
with open(ratings_file, 'r') as file:
    reader = csv.DictReader(file)
    reader1=csv.DictReader(file)
    for row in reader:
        user_id = row['UserID']
        movie_id = row['MovieID']
        rating = int(row['Rating'])
        
        # Get user age
        age_group = users_dict.get(user_id)
        if age_group is None:
            continue
        
        # Get genres for the movie
        genres = find_genres_for_movie(movies_dict,movie_id)
        
        # Update genre_ratings_by_age dictionary
        for genre in genres:
            # print(genre)
            if genre not in ratings_dict:
                ratings_dict[genre] = {age_group: 0}
                genre_count_user[genre]={age_group : 0}
                
            if age_group not in ratings_dict[genre]:
                ratings_dict[genre][age_group] = 0
                genre_count_user[genre][age_group]=0

            ratings_dict[genre][age_group] += rating
            genre_count_user[genre][age_group]+=1
    
    age_groups = sorted(set(users_dict.values()))
    genres = sorted(movies_dict.keys())
    for genre in genres:
        for age_group in age_groups:
            ratings_dict[genre][age_group]=ratings_dict[genre][age_group]/genre_count_user[genre][age_group]

# Plot stacked bar chart
age_groups = sorted(set(users_dict.values()))
genres = sorted(movies_dict.keys())
print(ratings_dict)
import matplotlib.pyplot as plt

# Extracting age groups and genres from the dictionary
age_groups = sorted(set(ratings_dict['Drama'].keys()))
genres = sorted(ratings_dict.keys())

# Initialize a dictionary to store ratings for each genre and age group
genre_ratings = {genre: [ratings_dict[genre][age_group] for age_group in age_groups] for genre in genres}

# Plotting the stacked bar chart
plt.figure(figsize=(10, 6))
bottom = [0] * len(age_groups)  # Bottom position for the bars

for genre in genres:
    plt.bar(age_groups, genre_ratings[genre], label=genre, bottom=bottom)
    bottom = [bottom[i] + genre_ratings[genre][i] for i in range(len(age_groups))]

plt.xlabel('Age Group')
plt.ylabel('Average Rating')
plt.title('Genre Ratings by Age Group')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

