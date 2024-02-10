import csv

# Read ratings CSV file
ratings_file = 'ratings.csv'
movies_rated = set()
with open(ratings_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        movies_rated.add(row[0])

# Read movies CSV file
users_file = 'users.csv'
users_data = []
with open(users_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        user_id = row[0]
        if user_id in movies_rated:
            users_data.append(row)

# Write filtered movies to another text file
filtered_users_file = 'filtered_users.csv'
with open(filtered_users_file, 'w') as file:
    for user in users_data:
        file.write(','.join(user) + '\n')

print("Filtered users have been written to", filtered_users_file)
