import numpy as np 
import pandas as pd 

def convertDatToCSV(inputFile,fields,outputFileName):
    with open(inputFile, 'r',errors='ignore') as file:
        data = file.read()
        data=data.split("\n")
        data=[x.split("::") for x in data]
        df = pd.DataFrame(data, columns=fields)
        csv_file_path = outputFileName
        df.to_csv(csv_file_path,index=False)


convertDatToCSV('ml-1m/users.dat',['UserID',"Gender","Age","Occupation","Zip-code"],'users.csv')
convertDatToCSV('ml-1m/movies.dat',['MovieID',"Title","Genres"],'movies.csv')
convertDatToCSV('ml-1m/ratings.dat',['UserID',"MovieID","Rating","Timestamp"],'ratings.csv')

