# Author: Haonan Tian 
# Date: 07/26/2018
# All Rights Reserved

################################# Description ########################################
# This is a demo of applying logistic regression to sample data set. The description 
# the data set applied in this program can be found at the following link:
# http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
######################################################################################

################################# Initialization ########################################
import time
import math
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

workclass = {"Private": 8, 
			 "Self-emp-not-inc": 7, 
			 'Self-emp-inc': 6, 
			 'Federal-gov': 5, 
			 "State-gov": 4,
			 "Local-gov": 3, 
			 "Without-pay": 2, 
			 "Never-worked": 1, 
			 "?": 0}
marital_status = {"Married-civ-spouse": 1, 
				  "Divorced": 2, 
				  "Never-married": 3,
				  "Separated": 4, 
				  "Widowed": 5, 
				  "Married-spouse-absent": 6, 
				  "Married-AF-spouse": 7, 
				  "?": 0}
occupation = {"Tech-support": 1, 
              "Craft-repair": 2, 
              "Other-service": 3, 
              "Sales": 4, 
              "Exec-managerial": 5, 
              "Prof-specialty": 6, 
              "Handlers-cleaners": 7, 
              "Machine-op-inspct": 8, 
              "Adm-clerical": 9, 
              "Farming-fishing": 10, 
              "Transport-moving": 11,  
              "Priv-house-serv": 12, 
              "Protective-serv": 13, 
              "Armed-Forces": 14, 
              "?": 0}
relationship = {"Wife": 1, 
				"Own-child": 2, 
				"Husband": 3, 
				"Not-in-family": 4, 
				"Other-relative": 5, 
				"Unmarried": 6, 
				"?": 0}
race = {"White": 1, 
		"Asian-Pac-Islander": 2, 
		"Amer-Indian-Eskimo": 3, 
		"Other": 4, 
		"Black": 5, 
		"?": 0}
sex = {"Female": 1, 
       "Male": 2, 
       "?": 0}
native_country = {"United-States": 1, 
				  "Cambodia": 2, 
				  "England": 3, 
				  "Puerto-Rico": 4, 
				  "Canada": 5,
				  "Germany": 6,
				  "Outlying-US(Guam-USVI-etc)": 7,
				  "India": 8,
				  "Japan": 9,
				  "Greece": 10,
				  "South": 11,
				  "China": 12,
				  "Cuba": 13,
				  "Iran": 14,
				  "Honduras": 15,
				  "Philippines": 16,
				  "Italy": 17,
				  "Poland": 18,
				  "Jamaica": 19,
				  "Vietnam": 20,
				  "Mexico": 21,
				  "Portugal": 22,
				  "Ireland": 23,
				  "France": 24,
				  "Dominican-Republic": 25,
				  "Laos": 26,
				  "Ecuador": 27,
				  "Taiwan": 28,
				  "Haiti": 29,
				  "Columbia": 30,
				  "Hungary": 31,
				  "Guatemala": 32,
				  "Nicaragua": 33,
				  "Scotland": 34,
				  "Thailand": 35,
				  "Yugoslavia": 36,
				  "El-Salvador": 37,
				  "Trinadad&Tobago": 38,
				  "Peru": 39,
				  "Hong": 40,
				  "Holand-Netherlands": 41,
				  "?": 0}
income = {">50K": 1, 
		  "<=50K": 0}
######################################################################################

################################# Load Data Set ########################################

def detectWidth(inputFile):
	fin = open(inputFile, 'r')
	line = fin.readline()
	line = line.strip().split(',')
	num_x = len(line) - 1 # last column is the column of output y
	fin.close()
	return num_x

def detectNum(inputFile):
	fin = open(inputFile, 'r')
	counter = 0
	for line in fin:
		if line is not "\n":
			counter = counter + 1
	fin.close()
	return counter

def loadData(inputFile):
	fin = open(inputFile, 'r')
	num_x = detectWidth(inputFile) - 1 # The education can be represented by edu-num
	m = detectNum(inputFile)
	x = np.zeros((m, num_x))
	y = np.zeros((m, 1))
	for i in range(m):
		line = fin.readline()
		tempLine = line.strip().split(",")
		for k in range(len(tempLine)):
			tempLine[k] = tempLine[k].strip()
			tempLine[k] = tempLine[k].strip(".")
		for j in range(len(tempLine)):
			if j == 0:
				x[i, j] = tempLine[j]
			elif j == 2:
				x[i, j] = tempLine[j]
			elif j == 1:
				x[i, j] = workclass[tempLine[j]]
			elif j == 4 or j ==10 or j == 11 or j ==12:
				x[i, j-1] = tempLine[j]
			elif j == 5:
				x[i, j-1] = marital_status[tempLine[j]]
			elif j == 6:
				x[i, j-1] = occupation[tempLine[j]]
			elif j == 7:
				x[i, j-1] = relationship[tempLine[j]]
			elif j == 8:
				x[i, j-1] = race[tempLine[j]]
			elif j == 9:
				x[i, j-1] = sex[tempLine[j]]
			elif j == 13:
				x[i, j-1] = native_country[tempLine[j]]
			if j == 14:
				y[i, 0] = income[tempLine[j]]
		'''if i % 5000 == 0:
			print("Finished line " + str(i))'''
	fin.close()
	print("Shape of x = " + str(x.shape) + " Shape of y = " + str(y.shape) + "\n")
	return x, y, num_x + 1, m
######################################################################################

################################# Logistic Module ########################################
def normalization(X):
	m = X.shape[0]
	num_x = X.shape[1]
	mean = np.sum(X, axis = 0) / m
	var = np.sum(np.square(X - mean), axis = 0) / m
	output = (X - mean) / var
	return output

def initialize_parameters(num_x): # Return grad and initial weights
	return np.zeros((num_x, 1))


def sigmoid(Z): # input may be a scaler or an arrary
	g = 1 / (1 + np.exp(-Z))
	return g

def compute_cost(X, W, y):
	m = X.shape[0]
	A = sigmoid(np.dot(X, W))
	J = -(1 / m) * (np.dot(y.T, np.log(A) + np.dot((1 - y).T, np.log(1 - A))))
	return J

def gradient_update(X, W, y, learning_rate):
	m = X.shape[0]
	result = W - learning_rate/ m  * np.dot(X.T, sigmoid(np.dot(X, W)) - y) 
	return result 

def predict(X, W, y):
	A = sigmoid(np.dot(X, W))
	for i in range(A.shape[0]):
		if A[i,0] <= 0.5:
			A[i,0] = 0
		else:
			A[i,0] = 1
	A = A - y
	A = A == 0
	unique, counts = np.unique(A, return_counts = True)
	diction = dict(zip(unique, counts))
	return diction[True] / X.shape[0]
######################################################################################


def main():
	print("Start running trainning set\n")
	time_start = time.time()
	X_train, y_train, num_x_train, m_train = loadData("adult.data.txt")
	X_train = normalization(X_train)
	X_train = np.append(X_train, np.ones((m_train,1)), 1)
	W_train = initialize_parameters(num_x_train)
	for i in range(num_iterations):
		J = compute_cost(X_train, W_train, y_train)
		if i % 40 == 0:
			print("Cost for iteration " + str(i) + " is " + str(J) + "\n")
		W_train = gradient_update(X_train, W_train, y_train, learning_rate)
	print("Accuracy for trainning set = " + str(predict(X_train, W_train, y_train)) + "\n\n")
	print("--------------------------------------------------------------------------\n")
	print("Start running test set\n")
	X_test, y_test, num_x_test, m_test = loadData("adult.test.txt")
	X_test = normalization(X_test)
	X_test = np.append(X_test, np.ones((m_test,1)), 1)
	W_test = initialize_parameters(num_x_test)
	for i in range(num_iterations):
		J = compute_cost(X_test, W_test, y_test)
		if i % 40 == 0:
			print("Cost for iteration " + str(i) + " is " + str(J) + "\n")
		W_test = gradient_update(X_test, W_test, y_test, learning_rate)
	print("Accuracy for test set = " + str(predict(X_test, W_test, y_test)) + "\n\n")
	print("--------------------------------------------------------------------------\n")
	print("Start trainning by API\n")  # running the same data set by API 
	classifier = LogisticRegression(random_state=0)
	classifier.fit(X_train, y_train.ravel()) # ravel() convert the array to shape [n,]
	y_pred = classifier.predict(X_test)
	confusion_matrix_1 = confusion_matrix(y_test, y_pred.ravel())
	print("Start printing cofusion matrix:\n")
	print(confusion_matrix_1)
	print('\nAccuracy of logistic regression classifier on test set: {:.2f}'.
		format(classifier.score(X_test, y_test)) + "\n")
	time_end = time.time()
	total_time = time_end - time_start
	print("Total time used = " + str(total_time) + "\n")
	print("Done!\n")
	return 0

if __name__ == "__main__":
	learning_rate = 1
	num_iterations = 400
	main()


