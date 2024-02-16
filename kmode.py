# importing necessary libraries
# from scipy import stats # for calculating mode
import random # for random number generation
import warnings
import numpy as np
warnings.filterwarnings("ignore") # ignoring warnings


class KModes:

    def __init__(self,k):
        self.k=k
        self.cluster_centroids = [] # list of cluster centroids
        self.ind_list = [] # list of data-point indices, selected as cluster centroids
        self.y = [] # target label after clustering
        self.flag = 1 # flag variable for checking whether there is any change in the clusters, hence terminating the training process
        self.iterations = 0 # counting the number of training iterations
    # delta function or delta metric
    def delta(self,a, b):
        return len([1 for i in range(len(a)) if a[i] != b[i]])

# implementation of k-Mode Clustering
    def fit(self,data):  
        # selection of k random cluster centroids among the given data-points

        ## assigning clusters
        for i in range(self.k):
            ind = random.randint(0, len(data) - 1) # selection of a data-point index, for the data-point to be a cluster centroid
            while(ind in self.ind_list): # if the data-point chosen randomly, is already a selected cluster centroid 
                ind = random.randint(0, len(data) - 1)
            self.cluster_centroids.append(data[ind]) # updating the list of cluster centroids
            self.ind_list = [data.index(x) for x in self.cluster_centroids] # updating the list of data-point indices, selected as cluster centroids
            
        # k-Mode Clustering Algorithm ...
        # The loop will assign the cluster or will reassign and store in list "y"
        while(1):
            cost = 0 # variable for calculating total cost for each iteration
            self.iterations += 1
            for i in range(len(data)):
                dis = [self.delta(centroid, data[i]) for centroid in self.cluster_centroids] # delta metric values for k clusters of a data-point
                cost += min(dis) # adding the cost
                if len(self.y) < len(data): # in case of 1st iteration or 1st pass
                    self.y.append(dis.index(min(dis))) # cluster assignment step
                else: # in case of iterations after the 1st pass
                    if self.y[i] == dis.index(min(dis)): # no change in cluster assignment
                        self.flag = 0
                    else:
                        self.y[i] = dis.index(min(dis)) # cluster re-assignment
                        self.flag = 1
                        
            # Displaying the Cost for each iteration
            print('Cost i.e., sum of delta metrics for all data-points for Iteration ' + str(self.iterations) + ': ', cost)
            if self.flag == 0: # for all the data-points, there is no change in the clusters, hence, training is terminated
                break
 
            # Cluster Centroid Updation
            for label in range(self.k):
                data_filter = [data[i] for i in range(len(data)) if self.y[i] == label] # filtering data-points that are assigned a particular cluster label
                # data filter has the list of rows which belong to the given label/node
                modes = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=data_filter)
                self.cluster_centroids[label]=modes.tolist()
                # cluster_centroids[label] = list(stats.mode(data_filter, axis = 0)[0][0]) # cluster updation by taking mode
                # above line will update the label based on mode
           
        return self.y, self.cluster_centroids # returning cluster labels and cluster centroids

    def predict(self,data):
        predicted_clusters = []
        for new_customer in data:
            # Calculate distances between new data point and cluster centroids
            distances = []
            for centroid in self.cluster_centroids:
                distance = 0
                for i in range(len(centroid)):
                    if centroid[i] != new_customer[i]:
                        distance += 1
                distances.append(distance)

            # Find the nearest cluster
            nearest_cluster = distances.index(min(distances))

            # Predict the cluster
            predicted_clusters.append(nearest_cluster)

        return predicted_clusters