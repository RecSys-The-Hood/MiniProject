import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
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

convert("users.csv")
df=pd.read_csv("convertedusers.csv")
grouped = df.groupby(['Zip-code', 'Occupation']).size().unstack(fill_value=0)

# Plot stacked bar graph
# grouped.plot(kind='bar', stacked=True)
# plt.xlabel('Zip Code')
# plt.ylabel('Count')
# plt.title('Occupation distribution by Zip Code')
# plt.legend(title='Occupation', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()