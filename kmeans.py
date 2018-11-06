## the following code implements k-means algorithm using python. it then runs it for a given dataset cluster_data. 
## the number of clusters required is taken as user input limited to 11 for proof of concept purposes but easily extendable 
## it then plots each cluster and centroid with different colors and markers and labels them as required   


# importing required packages and libraries
import pandas as pd
from pandas import *
import random as rand
from random import *
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
import math
from math import *

warnings.simplefilter("ignore")

# defining a class for kmeans
class Kmeans:

	# constructor with parameters for number of clusters, number of iterations and tolerance value (default value = 0.000000001)
	def __init__(self, k, number_of_iterations,tolerance = 0.000000001):
		self.k = k
		self.tolerance = tolerance
		self.number_of_iterations = number_of_iterations

	# the following function reads the input data file and returns the dataset as a dataframe 
	def getFeatures (self):
		
		df = pd.read_table('cluster_data.txt', delim_whitespace=True, names = ['index','x','y'])
		
		# drops the index column 
		df.drop(columns = ['index'], inplace = True)
		
		X = df.values		
		return X

	# the following function is the implementation of the kmeans algorithm. it creates a dictionary to store the centroids
	# and a dictionary to store the clusters. It then randomly initializes the cluster centroids and assigns each feature in 
	# the dataset to one of the clusters based on the closest cluster as per Euclidean distance 
	# once all features are assigned a cluster, the cluster centroid is updated according to the mean of all datapoints in the 
	# corresponding cluster. This cluster assignment and centroid update is repeated iteratively for the required number of 
	# iterations or till the update to the centroids is below the tolerance value
	def formClusters(self, data):
		
		# creating a dictionary of centroids 
		self.centroids = {}
		
		# initializing random cluster centroids 
		for i in range(self.k):
			self.centroids[i] = data[i]

		# repeating cluster assignment and centroid update for the required number of iterations
		for i in range(self.number_of_iterations):
			
			# creating empty clusters
			self.clusters = {}
			for j in range (self.k):
				self.clusters[j] = []

			# assigning features to clusters
			for feature in data:
				# creating array to store Euclidean distances 
				distances = []
				
				# calculating euclidean distances between features and centroids
				for j in self.centroids:
					distances.append(np.linalg.norm(feature - self.centroids[j]))
				
				# finding the index of the nearest and assigning it to cluster index
				clusterIndex = (distances.index(min(distances)))
		
				# appending the feature to the nearest cluster. so each key of clusters will be the cluster label and will contains 
				# values corresponding to the features in that cluster  
				self.clusters[clusterIndex].append(feature)
			
			# storing the centroid in previousCentroid since current centroid will be updated 			
			previousCentroid = dict(self.centroids)

			
			# updating all centroids based on mean values of all data points within cluster
			for clusterIndex in self.clusters:
				# considering only non empty clusters, since centroids of emptty clusters would remain unchanged
				if  self.clusters[clusterIndex]:
					self.centroids[clusterIndex] = np.nanmean(self.clusters[clusterIndex], axis = 0)
				
			# the following code snippet checks for the amount of total change between 2 successive values of centroids
			# if the total change is below tolerance then the kmeans algorithm has converged. For this all centroids need
			# to be updated by a value less than tolerance and only then will the algorithm stop
			belowTolerance = 0 
			for clusterIndex in self.centroids:
				
				# getting current and preceding values of each centroid
				curr = self.centroids[clusterIndex]
				prev = previousCentroid[clusterIndex]

				# calculating difference for each coordinate of centroid. here calculating difference in x and y coords
				difference = 0
				for m in range(2):
					difference += curr[m] - prev[m]  

				# checking if the difference is below tolerance for each centroid
				if abs(difference) < self.tolerance:
					belowTolerance += 1

			# checking if all centroids have undergone change lesser than tolerance. If yes: break
			if belowTolerance == self.k:
				break

# the following function is the function to run the kmeans for the cluster_data dataset		
def main():
	
	# taking user input for required parameters
	k = int(input("Enter number of clusters"))
	number_of_iterations = int(input("Enter number of iterations"))
	
	# creating a kmeans object with required initializations
	km = Kmeans(k, number_of_iterations)
	
	# getting the input dataset
	data = km.getFeatures()
	
	# forming clusters using kmeans algorithm
	km.formClusters(data)

	# defining colors and markers for the various clusters
	colors = ["r", "g", "c", "b", "k", "m", "y", "darkblue", "crimson", "darkred", "olive"]
	markers = [".","^","s","*", "D","P","p","1","2","3","4","8"]

	# defining cluster labels as 'Cluster1', 'Cluster2', etc.
	text = [0]*k
	for i in range(k):
		text[i] = "Cluster"+str(i+1)

	# the following snippet plots the final centroids and annotates it with the appropriate cluster labels 
	counter = 0
	for centroid in km.centroids:
		color = colors[centroid]
		plt.scatter(km.centroids[centroid][0], km.centroids[centroid][1], color = color, s = 200, marker = "x")
		plt.annotate(text[counter], km.centroids[centroid])
		counter += 1

	# printing required plot labels 
	plt.xlabel('Length')
	plt.ylabel('Width')
	
	# the following snippet plots the various clusters as a scatter plot. each cluster has a different color and marker 
	for clusterIndex in km.clusters:
		color = colors[clusterIndex]
		marker = markers[clusterIndex]
		for features in km.clusters[clusterIndex]:
			plt.scatter(features[0], features[1], color = color,s = 5, marker=marker)

	plt.show()


if __name__ == "__main__":
	main()
