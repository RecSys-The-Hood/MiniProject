import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os

def read_csv_files(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Filter only CSV files
    csv_files = [file for file in files if file.endswith('.csv')]
    
    # Read each CSV file into a DataFrame
    dataframes = []
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        dataframes.append(df)
    
    return dataframes

def MergeCSV(movieDf):

    df1=pd.read_csv("ratings.csv")
    df2=movieDf
    dfusers=pd.read_csv("convertedusers.csv")
    merged_df = pd.merge(df1, df2, on='MovieID')
    df=merged_df.drop(columns=["Timestamp"])
    merge2df=pd.merge(df,dfusers, on="UserID")
    # merge2df.to_csv("Task2.csv")
    return merge2df


def row_to_dict(row):
    return row.to_dict()

# Function to split genres
def split_genres(movie):
    genres = movie["Genres"].split('|')
    return [{"MovieID": movie["MovieID"], "Title": movie["Title"], "Genre": genre} for genre in genres]

def generateSplitCSV(): 
# Convert each row to a dictionary
    df = pd.read_csv("movies.csv")
    dataset = df.apply(row_to_dict, axis=1).tolist()

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


def GenresExpandedCSV():
    df2=pd.read_csv("movies.csv")
    genres = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western"
    ]
    for genre in genres:
        df2[genre]=0

    dataset = df2.apply(row_to_dict, axis=1).tolist()
    for movie in dataset:
        genres = movie["Genres"].split('|')
        for genre in genres:
            movie[genre]=1

    newMoviedf=pd.DataFrame(dataset)
    newMoviedf=newMoviedf.drop(columns=['Genres'],axis=1)
    newMoviedf.to_csv("MoviesExpanded.csv")
    return newMoviedf

def MergedCSV2(dfMovies):
    df1=pd.read_csv("ratings.csv")
    dfusers=pd.read_csv("convertedusers.csv")
    merged_df = pd.merge(df1, dfMovies, on='MovieID')
    df=merged_df.drop(columns=["Timestamp"])
    merge2df=pd.merge(df,dfusers, on="UserID")

    return merge2df

# Method 1

# generateSplitCSV()
# dfs=read_csv_files("src/")
# mergedDfs=[]

# for df in dfs:
#     mergedDfs.append(MergeCSV(df))

# combined_df = pd.concat(mergedDfs, axis=0, ignore_index=True)
# combined_df.to_csv("Task2.csv")

#Method 2

df=GenresExpandedCSV()
newDf=MergedCSV2(df)
newDf.to_csv("Combined.csv")