import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

df1=pd.read_csv("ratings.csv")
df2=pd.read_csv("src/Action.csv")
# do for each genre
dfusers=pd.read_csv("convertedusers.csv")

merged_df = pd.merge(df1, df2, on='MovieID')
df=merged_df.drop(columns=["Timestamp"])
merge2df=pd.merge(df,dfusers[['UserID','Occupation']], on="UserID")


print(merge2df)
merge2df.to_csv("Task2.csv")