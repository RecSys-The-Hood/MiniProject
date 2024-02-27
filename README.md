# Recommendation Systems Mini Project

This project has been submitted by 

1. Siddharth Kothari - IMT2021019
2. Munagala Kalyan Ram - IMT2021023
3. Sankalp Kothari - IMT2021028
4. M Srinivasan - IMT2021058

## Instructions to run the Code

You should first install numpy and pandas in your system. You can refer to the following websites for instructions to install the code.

1. https://numpy.org/install/
2. https://pandas.pydata.org/docs/getting_started/install.html


Following this, you should first run some files to get the data required. 

1. First run the Generate_csv.py file. This will convert the dat files to generate 3 csv files - movies.csv, users.csv, ratings.csv.
2. Then run the CSV_combine.py and the CSV_combine_cf.py file. This will give us the EncodedCombined2.csv and the EncodedCombined1.csv files, along with other files which will be required for EDA.

## File Structure

1. CombinedEDA.twb is the tableau workbook containing the plots we made for EDA.
2. The EDA folder contains these images.
3. ml_1m contains the .dat files. 
4. results contains some results that we have stored from previous runs of the code.
5. CollaborativeFiltering.ipynb contains the codes for Collaborative Filtering. You can comment in the required configuration that you wish to run. The possible configurations are - 
    1. Cosine Similarity + Item-Item
    2. PCC Similarity + Item-Item
    3. Cosine Similarity + User-User
    4. PCC Similarity + User-User
    5. IDF Weighted Similarity + User-User
    6. Variance Weighted Similarity + User-User

6. km_plusplus.py contains the code for the KMeans++ algorithm, km.py contains the code for the normal KMeans algorithm, kmode.py contains the code for the KModes algorithm, while SVD.py contains the code for the svd implementation. We have implemented normal and reduced svd.

7. These codes are imported by the following files, and hence have been separated into separate modules to allow for reuse.
    1. KModes_SVD.ipynb - contains the code for KModes Algorithm followed by SVD. KModes is run on the entire ratings matrix to get clusters. We then select the cluster which the target user belongs, and run SVD on it. We then pick out the 5 movies with the highest rating for the given user (this rating is obtained from svd), and display them.
    2. SVD_Kmeans_updated.ipynb - contains the code for SVD applied on the user-genre average rating matrix, to get the user representations, and then apply KMeans to cluster users.
    3. SVD_KMeans++Entire.ipynb - contains the same approach, but this time using the KMeans++ algorithm.

8. Each of these 3 files can be run to see tge recommendations provided.

## Report 

You can refer to the ppt (attached as pdf) for further details on implementations and inferences.