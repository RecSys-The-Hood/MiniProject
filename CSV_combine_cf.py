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
    dfusers=pd.read_csv("users.csv")
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
    df2.drop(df2.tail(1).index,inplace=True) # drop last row
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
    # print(dataset)
    for movie in dataset:
        genres = movie['Genres'].split('|')
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
    genderMapping={
        'M': 0,
        'F' : 1
    }
    
    occupationMapping = {
        "other or not specified":int(0),
        "academic/educator": 1,
        "artist": 2,
        "clerical/admin": 3,
        "college/grad student": 4,
        "customer service": 5,
        "doctor/health care": 6,
        "executive/managerial": 7,
        "farmer": 8,
        "homemaker": 9,
        "K-12 student": 10,
        "lawyer": 11,
        "programmer": 12,
        "retired": 13,
        "sales/marketing": 14,
        "scientist": 15,
        "self-employed": 16,
        "technician/engineer": 17,
        "tradesman/craftsman": 18,
        "unemployed": 19,
        "writer": 20
    }

    merge2df['Occupation']=merge2df['Occupation'].map(occupationMapping)
    merge2df['Gender']=merge2df['Gender'].map(genderMapping)

    # merge2df=merge2df.dropna(subset=['Occupation'])
    merge2df['UserID']=merge2df['UserID'].astype(np.int64)
    merge2df['MovieID']=merge2df['MovieID'].astype(np.int64)
    # data compression
    intCompress=["Action","Adventure","Animation","Children's","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western","Gender","Age","Rating"]

    merge2df["UserID"]=merge2df["UserID"].astype(np.int16)
    merge2df["MovieID"]=merge2df["MovieID"].astype(np.int16)
   
    for col in intCompress:
        merge2df[col]=merge2df[col].astype(np.int8)


    merge2df=merge2df.drop(columns=["Zip-code","Title"],axis=1)
    print("New")
    print(merge2df.dtypes)
    return merge2df

def convert(filepath):
    df1=pd.read_csv(filepath)
    mp={
        0:  "other",
        1:  "academic/educator",
        2:  "artist",
        3:  "clerical/admin",
        4:  "college/grad student",
        5:  "customer service",
        6:  "doctor/health care",
        7:  "executive/managerial",
        8:  "farmer",
        9:  "homemaker",
        10:  "K-12 student",
        11:  "lawyer",
        12:  "programmer",
        13:  "retired",
        14:  "sales/marketing",
        15:  "scientist",
        16:  "self-employed",
        17:  "technician/engineer",
        18:  "tradesman/craftsman",
        19:  "unemployed",
        20:  "writer"
    }

    df1['Occupation']=df1['Occupation'].replace(mp)
    df1.to_csv("convertedusers.csv",index=False)

# convert("users.csv")
# df=pd.read_csv("convertedusers.csv")
# grouped = df.groupby(['Zip-code', 'Occupation']).size().unstack(fill_value=0)
# Method 1

# generateSplitCSV()
# dfs=read_csv_files("src/")
# mergedDfs=[]

# for df in dfs:
#     mergedDfs.append(MergeCSV(df))

# combined_df = pd.concat(mergedDfs, axis=0, ignore_index=True)
# combined_df.to_csv("Task2.csv")

#Method 2

convert("users.csv")
df=GenresExpandedCSV()
newDf=MergedCSV2(df)
newDf.to_csv("EncodedCombined1.csv",index=False)
# newDf.to_parquet('file.parquet', compression='gzip')  # For Parquet
