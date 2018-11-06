## the following code applies the 3 clustering techniques to the handwritten digits dataset at 
## http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html.
## Here for kmeans, I have used both scikit learn as well as the code implemented from scratch. the code, in addition to clustering, 
## relabels the cluster labels as per the most occuring digit in the dataset. there are 2 confusion matrices generated for each 
## clustering algortihm: 1. maps actual digit values against default cluster labels from the algorithm 2. maps actual digit values 
## against predicted digit values 
## number of iterations for kmeans has been taken as 200
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from pandas import *
import random as rand
from random import *
import random
import numpy as np
from matplotlib import style
import warnings
import math
from math import *
from sklearn import metrics
from statistics import mode
np.set_printoptions(threshold=np.inf)


# load the dataset of handwritten digits and split it into features and targets (actual values) 
digits = load_digits() 
features = digits.data
targets = digits.target

corrected_labels = {}

# the following section implements kmeans using scikit learn
################# kmeans #####################
print ("#######################kmeans implementation from scikit learn begins###################### ")
# create kmeans object with 10 clusters
km = KMeans(n_clusters = 10)

# predict the cluster labels for each of the image data point in features input 
predictions = km.fit_predict(features)

# create confusion matrix using predicted cluster labels and actual targert values 
cm = confusion_matrix(targets, predictions, labels = [0,1,2,3,4,5,6,7,8,9])

# we now print the confusion matrix using the cluster labels as obtained by default from the algorithm
print ("Confusion matrix using default cluster labels as given by kmeans algorithm : ")
print (cm)

# the following snippet goes through the each column of confusion matrix(row of transpose of cm) and finds the 
# largest number. this corresponds to the digit which occurs for the most number of times in that particular cluster. 
# a mapping dictionary is then created which maps each cluster label to the digit which dominates that cluster    

for i in range(10):
	# find largest number in each cluster (column)
	max_occured = (cm.transpose()[i].max())
	
	# find out the index of the largest number which is the number which dominates that cluster
	index_max = cm.transpose()[i].tolist().index(max_occured)

	# enter the mapping in the dictionary
	corrected_labels[i] = index_max

# print the mapping 
print ("corrected label mapping for kmeans using scikit learn: ")
print (corrected_labels)

# printing the mapping in verbose
for i in corrected_labels:
	print ("Cluster", i," will be labeled ",corrected_labels[i])

# the kmeans algorithm gives as output the cluster labels
# however the following snippet actually creates the array value_predictions which predicts the image as predicted 
# for each of the input handwritten digit 
value_predictions = [0]*len(predictions)

# looping through all predicted cluster labels
for i in range(len(predictions)):

	# we take in each value of predicted cluster label, then from the mapping created, we obtain the digit that it 
	# represents and add it to the value_predictions array
	value_predictions[i] = corrected_labels[predictions[i]]

# uncomment to see the actual predicted digit values
# print (value_predictions)

# create the modified confusion matrix. this matrix maps the actual digit values to the predicted digit 'values' instead of labels
# this matrix is diagonally dominant indicating good accuracy of the algorithm
mod_cm = confusion_matrix(targets, value_predictions, labels = [0,1,2,3,4,5,6,7,8,9])
print ("modified confusion matrix mapping actual digit values against predicted digit values")
print (mod_cm)

# uncomment to see score. doesnt run on linux but tested on a windows machine 
print ("Accuracy of kmeans skleanrn: ",metrics.fowlkes_mallows_score(targets, value_predictions))

# ############### agglomerative clustering begins ############################

# run Agglomerative Clustering using scikit learn for 10 clusters
print (" ")
print ("######## AgglomerativeClustering starts #################")
agglo = AgglomerativeClustering(n_clusters = 10)

# predict cluster labels for input digits
predictions = agglo.fit_predict(features)

# create confusion matrix using predicted cluster labels and actual targert values 
cm = confusion_matrix(targets, predictions, labels = [0,1,2,3,4,5,6,7,8,9])

print ("Confusion matrix for default cluster labels as given by agglomerative clustering algorithm: ")
print (cm)

# the following snippet goes through the each column of confusion matrix(row of transpose of cm) and finds the 
# largest number. this corresponds to the digit which occurs for the most number of times in that particular cluster. 
# a mapping dictionary is then created which maps each cluster label to the digit which dominates that cluster    
corrected_labels = {}
for i in range(10):
	max_occured = (cm.transpose()[i].max())
	index_max = cm.transpose()[i].tolist().index(max_occured)
	corrected_labels[i] = index_max

# print the mapping 
print ("corrected label mapping:")
print (corrected_labels)

# print the mapping in verbose
for i in corrected_labels:
	print ("Cluster", i," will be labeled ",corrected_labels[i])

# the agglomerative clustering algorithm gives as output the cluster labels
# however the following snippet actually creates the array value_predictions which predicts the image as predicted 
# for each of the input handwritten digit 
value_predictions = [0]*len(predictions)

# looping through all predicted cluster labels
for i in range(len(predictions)):
	
	# we take in each value of predicted cluster label, then from the mapping created, we obtain the digit that it 
	# represents and add it to the value_predictions array
	value_predictions[i] = corrected_labels[predictions[i]]

# uncomment to see the actual predicted digit values
#print (value_predictions)

# create the modified confusion matrix. this matrix maps the actual digit values to the predicted digit 'values' instead of labels
# this matrix is diagonally dominant indicating good accuracy of the algorithm
mod_cm = confusion_matrix(targets, value_predictions, labels = [0,1,2,3,4,5,6,7,8,9])
print ("modified confusion matrix mapping actual digit values against predicted digit values ")
print (mod_cm)

# uncomment to see score. doesnt run on linux but tested on a windows machine 
print ("Accuracy of Agglomerative Clustering: ",metrics.fowlkes_mallows_score(targets, value_predictions))

################ affinity propagation starts ################
print(" ")
print ("######## affinity propagation starts #################")

# predict cluster labels for input digits using affinity propagation
affinity = AffinityPropagation()
predictions = affinity.fit_predict(features)

# printing confusion matrix for all clusters generated by affinity propagation
print ("confusion matrix for affinity propagation based on default cluster labels:")
cm = confusion_matrix(targets, predictions)
print (cm)

# affinity propagation creates clusters whose number is not pre determined. so the number of clusters can become huge
# however multiple clusters contain the same digit and hence can be mapped by the dominant number in each cluster

# creating an empty dictionary containing as many keys as there are clusters
affinity_dict = {}
for j in range (predictions.max()+1):
	affinity_dict[j] = []

# in the created dictionary, we map all numbers contained in a cluster to the corresponding cluster number
counter  = 0
# loop through all cluster labels
for i in predictions:
	# for the key corresponding to the cluster label in prediction, append the digit which is corresponds to this particular label
	affinity_dict[i].append(targets[counter])
	counter += 1

# the following snippet replaces each cluster label by the digit which is present the most number of times in that cluster
for i in affinity_dict:
	affinity_dict[i] = mode(affinity_dict[i])

# the following code snippet actually creates the array value_predictions which predicts the image as predicted 
# for each of the input handwritten digit 
value_predictions = [0]*len(predictions)

# looping through all predicted cluster labels
for i in range(len(predictions)):
	
	# we take in each value of predicted cluster label, then from the mapping created, we obtain the digit that it 
	# represents and add it to the value_predictions array
	value_predictions[i] = affinity_dict[predictions[i]]

# uncomment to see the actual predicted digit values
#print (value_predictions)

# create the modified confusion matrix. this matrix maps the actual digit values to the predicted digit 'values' instead of labels
# this matrix is diagonally dominant indicating good accuracy of the algorithm
mod_cm = confusion_matrix(targets, value_predictions, labels = [0,1,2,3,4,5,6,7,8,9])
print ("modified confusion matrix mapping actual digit values against predicted digit values: ")
print (mod_cm)

print ("Accuracy of Affinity Propagation: ",metrics.fowlkes_mallows_score(targets, value_predictions))

################ kmeans implemented in question 1 #############33
print (" ")
print ("###### kmeans question1 implementation starts #############")
warnings.simplefilter("ignore")

# the following is the kmeans code implemented in question 1 and hence kindly refer to the comments therein for any clarification
def formClusters(k, data, labels, number_of_iterations, tolerance):
	centroids = {}
	predictions = {}

	for j in range (k):
			predictions[j] = []

	# random initial clusters 
	j = random.choice(range(10))
	for i in range(k):
		centroids[i] = data[j]
		j+=1


	for i in range(number_of_iterations):
		clusters = {}
		predictions = {}
		for j in range (k):
			clusters[j] = []
			# here we additionally create a predictions dictionary which contains as keys the various cluster labels 
			# and contains as values: the actual digit values stored in the cluster
			predictions[j] = []

		# cluster assignment
		counter = 0
		for feature in data:
			distances = []
			for j in centroids:
				distances.append(np.linalg.norm(feature - centroids[j]))
			
			clusterIndex = (distances.index(min(distances)))
			
			clusters[clusterIndex].append(feature)
			# append the corresponding digit value to the cluster label in the predictions dictionary
			predictions[clusterIndex].append(labels[counter])
			counter+=1

		
		previousCentroid = dict(centroids)

		
		# centroid update
		for clusterIndex in clusters:
			
			if  clusters[clusterIndex]:
				centroids[clusterIndex] = np.nanmean(clusters[clusterIndex], axis = 0)
			
		belowTolerance = 0 
		for clusterIndex in centroids:
			curr = centroids[clusterIndex]
			prev = previousCentroid[clusterIndex]

			difference = 0
			for m in range(k):
				difference += curr[m] - prev[m]  

			if abs(difference) < tolerance:
				belowTolerance += 1

		if belowTolerance == k:
			print ("iterations = ", i)
			break

	return predictions

k = 10
number_of_iterations = 200

# run kmeans to form clusters
predictions = formClusters(k, features, targets, number_of_iterations, 0.00001)

# creating confusion matrix:

# here we create an empty confusion matrix of size 10*10 since we have 10 clusters
cm = np.zeros([10,10])

# for all cluster labels in predictions
for i in predictions:

	# for each digit value in predictions[i] 
	for j in predictions[i]:
		# update the confusion matrix with the count of the digit in each cluster
		cm[i][j]+=1

# the matrix contains rows as cluster labels and actual digit values as columns. thus printing transpose to get 
# confusion matrix in standard form  
print ("confusion_matrix using default cluster labels from kmeans implemented in question1")
print (cm.transpose())


# the following snippet goes through the output dictionary from kmeans. since each entry in the dictionary
# corresponds to the actual digit value present in the cluster, finding the mode of the row will give us the 
# number which dominates that cluster. 
# the mapping dictionary thus maps each cluster label to the digit which dominates that cluster    
corrected_labels = {}
for i in predictions:
	corrected_labels[i] = mode(predictions[i])

# print the mapping 
print ("corrected label mapping:")
print (corrected_labels)

# print the mapping in verbose
for i in corrected_labels:
	print ("Cluster", i," will be labeled ",corrected_labels[i])

# create the modified confusion matrix. this matrix maps the actual digit values to the predicted digit 'values' instead of labels
# this matrix is diagonally dominant indicating good accuracy of the algorithm
mod_cm = np.zeros([10,10])
minus_one = -1*np.ones(10)

# rearranging the confusion matrix in the order of the digit values
for i in range(len(cm)):
	# viewing a dictionary as a pandas series to obtain the key from a given value 
	ser = pd.Series(corrected_labels)
	# a is the key for value = i
	a = ser[ser.values == i].index
	# handling the edge case when more than one cluster is dominated by a single digit
	# in this case: to represent the modified confusion matrix, the dominated digit would be represented by that cluster 
	# where it appears more often. also in this case few of the digit would not dominate any of the cluster, so there wouldn't
	# be any cluster represented by that label and hence that particular column would be represented by -1
	
	# checking if one digit dominates multiple clusters
	if len(a)>1:
		# assigning the column as the cluster with more occurence of that digit
		if max(cm[a[0]]) > max(cm[a[1]]):
			a = a[0]
		else:
			a = a[1]	
		mod_cm[i] = cm[a]
	
	# if a particular digit is missing, assigning that column as -1
	elif len(a) == 0:
		mod_cm[i] = minus_one
	
	# our standard case 
	else:
		mod_cm[i] = cm[a]
print ("modified confusion matrix mapping actual digit values against predicted digit values: ")
print (mod_cm.transpose())