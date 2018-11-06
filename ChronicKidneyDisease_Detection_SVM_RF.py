## the following code runs the linear SVM, RBF SVM and Random Forest Classifier on the ckd training data after preprocessing
## it also calculates f-measures of all 3 algorithms on training as well as test data 

import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.simplefilter("ignore")

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

# the following function calculates the f-measure for the given instance of logistic regression. 
# it takes in the features and the labels and the finalized weights. it calculates the probabilities and 
# predictions using the given features, weights and labels 
def calc_fmeasure(length, predictions, labels):
	
	# calculates tp,tn,fp,fn as per their definitions
	tn = fn = fp = tp = f_measure = pre = rec = 0
	for i in range(length):
		if predictions[i] == 0 and labels[i] == 0:
			tn+=1
		elif predictions[i] == 0 and labels[i] == 1:
			fn+=1	
		elif predictions[i] == 1 and labels[i] == 0:
			fp+=1
		elif predictions[i] == 1 and labels[i] == 1:
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
	f_measure = 2*pre*rec/(pre+rec)
	
	return f_measure

def main():
	
	# cleanses and processes the input data set  
	ckd = cleanse_data()
	
	# create a MinMax scaler object woth default range of [0,1]
	scaler = MinMaxScaler(copy = False)
	
	# scale the data input data 
	ckd = scaler.fit_transform(ckd)

	# split the input data into training and test data 
	trg_data, testing_data = train_test_split(ckd, test_size=0.2)
	
	# obtain training labels and features 
	n = trg_data.shape[1] 
	Y_train = trg_data[:,n-1]
	X_train = trg_data[:,0:n-1] 
	
	# obtain test labels and features 
	Y_test =  testing_data[:,n-1]
	X_test =  testing_data[:,0:n-1]
	

	# train the linear SVM classifier on the training data
	svm  = SVC(kernel = 'linear')
	svm.fit (X_train, Y_train)
	
	# predict the SVM classifier output on test data
	test_predictions = svm.predict(X_test)
	
	# predict the SVM classifier output on training data
	trg_predictions = svm.predict(X_train)


	trg_length = len(trg_predictions)
	test_length = len(test_predictions)

	# calculate f-measure for training data
	f_measure_trg = calc_fmeasure(trg_length, trg_predictions, Y_train)
	
	# calculate f-measure for test data
	f_measure_test = calc_fmeasure(test_length, test_predictions, Y_test)

	print ("f-measure of Linear SVM on training data : ", f_measure_trg)
	print ("f-measure of Linear SVM on test data: ", f_measure_test)

	# train the rbf SVM classifier on the training data
	svm  = SVC()
	svm.fit (X_train, Y_train)
	
	# predict the rbf SVM classifier output on test data
	test_predictions = svm.predict(X_test)
	
	# predict the rbf SVM classifier output on training data
	trg_predictions = svm.predict(X_train)

	trg_length = len(trg_predictions)
	test_length = len(test_predictions)

	# calculate f-measure for training data
	f_measure_trg = calc_fmeasure(trg_length, trg_predictions, Y_train)
	
	# calculate f-measure for test data	
	f_measure_test = calc_fmeasure(test_length, test_predictions, Y_test)

	print ("f-measure of RBF SVM on training data : ", f_measure_trg)
	print ("f-measure of RBF SVM on test data: ", f_measure_test)

	# train the Random forest classifier on the training data
	rand_forest = RandomForestClassifier()
	rand_forest.fit(X_train, Y_train)
	
	# predict the Random forest classifier output on test data
	test_predictions = rand_forest.predict(X_test)
	
	# predict the Random forest classifier output on training data
	trg_predictions = rand_forest.predict(X_train)

	trg_length = len(trg_predictions)
	test_length = len(test_predictions)

	# calculate f-measure for training data
	f_measure_trg = calc_fmeasure(trg_length, trg_predictions, Y_train)
	
	# calculate f-measure for training data
	f_measure_test = calc_fmeasure(test_length, test_predictions, Y_test)

	print ("f-measure of RandomForestClassifier on training data : ", f_measure_trg)
	print ("f-measure of RandomForestClassifier on test data: ", f_measure_test)

if __name__ == "__main__":
	main()

