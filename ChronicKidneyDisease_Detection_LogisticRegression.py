## the following program implements logistic regression. it runs for the given dataset of chronic kidney disease
## it runs for different values of the regularization parameter and plots the f-measure against the regularization 
## parameter for training and test data. It repeats the same after standardization if the dataset. The code can be 
## run for different values of learning rate and number of iterations  

from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing.imputation import Imputer
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter("ignore")
np.set_printoptions(threshold=np.inf)

# the following function reads data from the given dataset and converts the categorical values to numeric values
# presence of disease is 1 and absence is marked as 0. also all factors which indicate presence of the disease like
# poor appetite, presence of coronary artery disease, etc. have been tagged 1 and normal factors have been tagged 0
# factors contributing/not contributing to the disease was found by internet research 
def cleanse_data():
	ckd = pd.read_csv('chronic_kidney_disease_full.csv')
	
	ckd.replace("ckd", 1, inplace = True)
	ckd.replace("notckd", 0, inplace = True)
	ckd.replace("normal", 0, inplace = True)
	ckd.replace("abnormal", 1, inplace = True)
	ckd.replace("good", 0, inplace = True)
	ckd.replace("poor", 1, inplace = True)
	ckd.replace("present", 1, inplace = True)
	ckd.replace("notpresent", 0, inplace = True)
	ckd.replace("yes", 1, inplace = True)
	ckd.replace("no", 0, inplace = True)

	# replacing all NA values with median of the column
	ckd.fillna(ckd.median(), inplace = True)
	
	# return processed dataset 
	return ckd

# the following function implements logistic regression for the passed parameters in the argument list
# the following algorithm takes into account regularization parameter and returns the finalized weights after running
# for the required number of iterations. kindly refer to the report for calculation of the gradient 
def logistic_regression(features, labels, learning_rate, reg_para, number_of_iterations):
	m = features.shape[0]
	n = features.shape[1]
	
	# initialise weights as array of 0s
	weights = [0]*n
	
	gradient = []
	
	# running for the required number of times
	for i in range(number_of_iterations):
		
		# calculating dot product of features and weights
		values  = features.dot(weights)
		
		# converting values to probabilities using the sigmoid activation function	
		probabilities = 1/(1 + np.exp(-values))
		
		# calculating error for each of the predicted probabilities
		error  = probabilities - labels
		
		# calculating gradient values using the derivative of the cost function
		gradient = features.T.dot(error) 	
		gradient += np.dot(reg_para , weights)
		gradient /= m
		
		# udpate weights using gradient descent
		weights -= learning_rate*gradient
	
	# return updated weights after the required number of iterations
	return weights

# the following function calculates the f-measure for the given instance of logistic regression. 
# it takes in the features and the labels and the finalized weights. it calculates the probabilities and 
# predictions using the given features, weights and labels 
def calc_fmeasure(features, weights, labels):
	
	# calculates values uing features and weights obtained from logistic regression
	values  = features.dot(weights)

	# calculates probabilities using sigmoid activation function
	probabilities = 1/(1 + np.exp(-values))
	
	# calculates output labels as: if probability < 0.5, label is 0 else label is 1
	predictions =[]
	for i in range(probabilities.shape[0]):
		if probabilities.iloc[i] < 0.5:
			predictions.append(0)
		else:
			predictions.append(1)
	
	# calculates tp,tn,fp,fn as per their definitions
	tn = fn = fp = tp = f_measure = pre = rec = 0
	
	for i in range(len(predictions)):
		if predictions[i] == 0 and labels.iloc[i] == 0:
			tn+=1
		elif predictions[i] == 0 and labels.iloc[i] == 1:
			fn+=1
		elif predictions[i] == 1 and labels.iloc[i] == 0:
			fp+=1
		elif predictions[i] == 1 and labels.iloc[i] == 1:
			tp+=1
	
	# calculates pre and rec using the tp,fp,tn,fn values. in case tp and fp equal 0 then pre is set as 0.
	# similarly if tp and fn equal 0 then rec is set as 0. f-measure is calulated as per its definition and
	# is set to 0 if pre and rec equal 0
	if tp ==0 and fp == 0:
		pre =0
	else:
		pre = tp/(tp+fp)
	if tp == 0 and fn == 0:
		rec = 0
	else:
		rec = tp/(tp+fn)
	if pre == 0 and rec == 0:
		f_measure=0
	else:
		f_measure = 2*pre*rec/(pre+rec)
	return f_measure

# the following function calculates the f-measure for the given instance of logistic regression after standardization 
# it takes in the features and the labels and the finalized weights. it calculates the probabilities and 
# predictions using the given features, weights and labels. separate function is used since there are different data
# structures are used with and without standardization
def calc_fmeasure_std(features, weights, labels):
	# calculations similar as above function
	values  = features.dot(weights)
	probabilities = 1/(1 + np.exp(-values))
	
	predictions =[]
	for i in range(probabilities.shape[0]):
		if probabilities[i] < 0.5:
			predictions.append(0)
		else:
			predictions.append(1)
	
	tn = fn = fp = tp = f_measure = pre = rec = 0
	
	for i in range(len(predictions)):
		if predictions[i] == 0 and labels.iloc[i] == 0:
			tn+=1
		elif predictions[i] == 0 and labels.iloc[i] == 1:
			fn+=1
		elif predictions[i] == 1 and labels.iloc[i] == 0:
			fp+=1
		elif predictions[i] == 1 and labels.iloc[i] == 1:
			tp+=1

	# calculates pre and rec using the tp,fp,tn,fn values. in case tp and fp equal 0 then pre is set as 0.
	# similarly if tp and fn equal 0 then rec is set as 0. f-measure is calulated as per its definition and
	# is set to 0 if pre and rec equal 0
	if tp ==0 and fp == 0:
		pre =0
	else:
		pre = tp/(tp+fp)
	if tp == 0 and fn == 0:
		rec = 0
	else:
		rec = tp/(tp+fn)
	if pre == 0 and rec == 0:
		f_measure=0
	else:
		f_measure = 2*pre*rec/(pre+rec)
	return f_measure


# the following function yields the counter variable of a for loop in float increments
def float_range(start, stop, step):
    i = 0
    while start + i * step < stop:
        yield start + i * step
        i += 1

# the following function reads the dataset, splits it in the desired ratio. it then runs logistic regression for 
# the various values of the regularization parameter and calculates the f-measure and plots them. it repeats the same
# after standardization of the dataset  

def main():
	
	# cleanses and processes the input data set  
	ckd = cleanse_data()
	
	# split the dataset into train and test data 
	train, test = train_test_split(ckd, test_size=0.2)
	n = train.shape[1] 
	
	# obtain the target labels for the training data 
	Y_train = train.iloc[:,n-1]
	# obtain the feature data points for the training data  
	X_train = train.iloc[:,0:n-1] 
	
	# append the intercept column to the training data. inserted after standardized to ensure intecept column 
	#doesn't become all 0s
	X_train.insert(0, 'x0',1)
	
	# obtain the target labels for the test data 
	Y_test =  test.iloc[:,n-1]
	
	# obtain the feature data points for the test data  
	X_test =  test.iloc[:,0:n-1]
	
	# append the intercept column to the test data 
	X_test.insert(0, 'x0',1) 
	
	# taking user input for learning rate and number of iterations
	learning_rate = float(input('Enter learning rate '))
	number_of_iterations = int(input('Enter no. of number of iterations '))

	# creating empty arrays for f-measure and regularization parameter
	f_measure_trg = []
	f_measure_test = []

	reg_para_array = []

	# running logistic regression for the required values of the regularization parameter on training data
	for reg_para in float_range(-2,4,0.2):
		reg_para_array.append(reg_para)

		# calculating final weights for the given value of regularization parameter on the training data 
		weights = logistic_regression(X_train, Y_train, learning_rate, reg_para, number_of_iterations)

		# calculate f-measure for the training data and given value of regularization parameter
		f_measure_trg.append(calc_fmeasure(X_train, weights, Y_train))
	
	# running logistic regression for the required values of the regularization parameter on test data
	for reg_para in float_range(-2,4,0.2):

		# calculating final weights for the given value of regularization parameter using training data 
		weights = logistic_regression(X_train, Y_train, learning_rate, reg_para, number_of_iterations)
		
		# calculate f-measure for the test data and given value of regularization parameter
		f_measure_test.append(calc_fmeasure(X_test, weights, Y_test))
	

	print ("f-measure training: ",f_measure_trg)
	print ("f-measure testing : ",f_measure_test)

	# plotting f-measure vs regularization parameter for training data
	plt.subplot(221)
	plt.plot(reg_para_array, f_measure_trg)
	plt.xlabel('regularization parameter')
	plt.ylabel('f-measure - training')

	# plotting f-measure vs regularization parameter for test data
	plt.subplot(222)
	plt.plot(reg_para_array, f_measure_test)
	plt.xlabel('regularization parameter')
	plt.ylabel('f-measure - testing')
	

	# creating a StandardScaler object to standardize using the mean and standard deviation of each data column
	scaler = StandardScaler(copy = False)
	
	# obtaining the training features
	X_train = train.iloc[:,0:n-1] 
	
	# standardizing training features
	X_train= scaler.fit_transform(X_train)
	
	# appending intercept column of 1s
	m = X_train.shape[0]	
	a = np.ones(m)
	X_train = np.insert(X_train, 0, a, axis = 1)

	# obtaining the test features
	X_test =  test.iloc[:,0:n-1]
	
	# standardizing test features
	X_test= scaler.fit_transform(X_test)
	
	# appending intercept column of 1s
	m = X_test.shape[0]
	a = np.ones(m)
	X_test = np.insert(X_test, 0, a, axis = 1)

	f_measure_trg = []
	f_measure_test = []

	# running logistic regression for the required values of the regularization parameter on standardized training data
	for reg_para in float_range(-2,4,0.2):

		# calculating final weights for the given value of regularization parameter on the standardized training data 
		weights = logistic_regression(X_train, Y_train, learning_rate, reg_para, number_of_iterations)

		# calculate f-measure for the standardized training data and given value of regularization parameter
		f_measure_trg.append(calc_fmeasure_std(X_train, weights, Y_train))
	
	# running logistic regression for the required values of the regularization parameter on standardized test data
	for reg_para in float_range(-2,4,0.2):

		# calculating final weights for the given value of regularization parameter using the standardized training data 
		weights = logistic_regression(X_train, Y_train, learning_rate, reg_para, number_of_iterations)
		
		# calculate f-measure for the standardized test data and given value of regularization parameter
		f_measure_test.append(calc_fmeasure_std(X_test, weights, Y_test))
	

	print ("f-measure of standardized training data: ",f_measure_trg)
	print ("f-measure of standardized test data",f_measure_test)

	reg_para_array = np.asarray(reg_para_array)

	f_measure_trg = np.asarray(f_measure_trg)
	f_measure_test = np.asarray(f_measure_test)
	
	# plotting f-measure vs regularization parameter for standardized training data
	plt.subplot(223)
	plt.plot(reg_para_array, f_measure_trg)
	plt.xlabel('regularization parameter')
	plt.ylabel('f-measure - trg with std')

	# plotting f-measure vs regularization parameter for standardized test data
	plt.subplot(224)
	plt.plot(reg_para_array, f_measure_test)
	plt.xlabel('regularization parameter')
	plt.ylabel('f-measure - testing with std')

	plt.show()


if __name__ == "__main__":
	main()



