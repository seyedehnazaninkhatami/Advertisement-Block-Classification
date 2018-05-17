"""

@author: nazkh
"""
import pandas as pd
import numpy as np
import sklearn.datasets
import time

StartTime = time.time()

# Load datasets in the svmlight format into sparse CSR matrix. Text-based with one sample per line
from sklearn.datasets import load_svmlight_file

# Loading Train Data
TrainData = sklearn.datasets.load_svmlight_file('Desktop\MachineLearning\COMPSCI-589-HW2\Data\HW2.train.txt',multilabel=True)

# Separating the class label and the features into arrays

# Array of Train Features
XTrain = TrainData[0]
# To handle Sparse Matrix
XTrain = XTrain.todense()

# List of Class Label for Train Data
YTrain = TrainData[1]
# List to Array
YTrain= np.asarray(YTrain)


# Loading Test Data
TestData = sklearn.datasets.load_svmlight_file('Desktop\MachineLearning\COMPSCI-589-HW2\Data\HW2.test.txt',multilabel=True)

# Separating the class label and the features into arrays

# Array of Test Features 
XTest = TestData[0]
# To handle Sparse Matrix
XTest = XTest.todense()

# List of Class Label for Test Data
YTest = TestData[1]
#List to Array
YTest=np.asarray(YTest)

# Loading Kaggle Data
KaggleData = sklearn.datasets.load_svmlight_file('Desktop\MachineLearning\COMPSCI-589-HW2\Data\HW2.kaggle.txt',multilabel=True)

# Separating the class label and the features into arrays
# Array of Test Features 
XKaggle = KaggleData[0]
# To handle Sparse Matrix
XKaggle = XKaggle.todense()

# Array of Class Label for Test Data
#YKaggle = KaggleData[1]
LoadDataTime= time.time()-StartTime 

###Question 3
StartTime = time.time()
####
# Classifier Number 1: SVM
from sklearn import svm
# Scaling the Data result in lower run-time
from sklearn import preprocessing
XTrain = preprocessing.scale(XTrain)
XTest = preprocessing.scale(XTest)

# SupportVectorMachine = svm.SVC(kernel="linear") accuracy:0.87147826086956526
# SupportVectorMachine = svm.SVC(kernel="poly", degree = 2) accuracy 0.84
#SupportVectorMachine = svm.SVC(kernel="poly", degree = 3) #accuracy 0.87304
SupportVectorMachine = svm.SVC()
SupportVectorMachine.fit(XTrain,YTrain.ravel())

SVMTrainingTime = time.time() - StartTime
print(f'{SVMTrainingTime} is the SVM Training Time')
StartTime = time.time()

predicted = SupportVectorMachine.predict(XTest)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(YTest.ravel(), predicted.ravel())
print(f'{accuracy:.3} is the Accuracy for SVM Classifier')

SVMRunningTime = time.time() - StartTime
print(f' {SVMRunningTime} is the SVM Running Time')

StartTime = time.time()
####
# Classifier Number 2: KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
KNN = KNeighborsClassifier()
KNN.fit(XTrain, YTrain.ravel())

KNNTrainingTime = time.time() - StartTime 
print(f'{KNNTrainingTime} is the KNN training Time')

StartTime = time.time() 

KNNPrediction = KNN.predict(XTest)
accuracy = accuracy_score(YTest, KNNPrediction)
print(f'{accuracy:.3} is the Accuracy for KNN Classifier')


KNNRunningTime = time.time() - StartTime
print(f' {KNNRunningTime} is the KNN Running Time')

StartTime = time.time()

#### 
# Classifier Number 3: Random Forest
from sklearn.ensemble import RandomForestClassifier
randf = RandomForestClassifier()
randf.fit(XTrain, YTrain.ravel())

RFTrainingTime = time.time() - StartTime
print(f'{RFTrainingTime} is the RF Training Time')

StartTime = time.time()

from sklearn.metrics import accuracy_score
predicted = randf.predict(XTest)
accuracy = accuracy_score(YTest, predicted)
print(f'{accuracy:.3}: is the Accuracy for RF Classifier')

RFRunningTime = time.time()- StartTime
print(f' {RFRunningTime} is the RF Running Time')

### Random Forest for Kaggle Competition
#from sklearn.ensemble import RandomForestClassifier
#randf = RandomForestClassifier()
#randf.fit(XTrain, YTrain.ravel())
#
#from sklearn.metrics import accuracy_score
#predicted = randf.predict(XKaggle)
#accuracy = accuracy_score(YTest, predicted)
#print(f'Mean accuracy score: {accuracy:.3}')
#predictions = predicted

# kaggle.py module

#import csv
#import numpy as np

#Save prediction vector in Kaggle CSV format
#Input must be a Nx1, 1XN, or N long numpy vector
#def kaggleize(predictions,file):
#
#	if(len(predictions.shape)==1):
#		predictions.shape = [predictions.shape[0],1]
#
#	ids = 1 + np.arange(predictions.shape[0])[None].T
#	kaggle_predictions = np.hstack((ids,predictions)).astype(int)
#	writer = csv.writer(open(file, 'w'))
#	writer.writerow(['# id','Prediction'])
#	writer.writerows(kaggle_predictions)

# kaggleize (predicted,"Nazanin.csv")

###Question 4
### Grid Search Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# Number of trees in random forest
n_estimators = [1000]
#[int(x) for x in np.linspace(start = 200, stop = 3000, num = 1)]
# Number of features to consider at every split
max_features = ['auto']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 300, num = 10)]

min_samples_split=[2, 5, 10]
# Create the parameter grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'min_samples_split': min_samples_split,
               'max_depth': max_depth}
               

# Creating the classifier
randf = RandomForestClassifier()
# Grid sweep of parameters, using 3 fold cross validation
# We use all the available cores by setting n_jobs=-1
randf_random = GridSearchCV(estimator = randf, param_grid  = param_grid, cv = 3, verbose=2, n_jobs = -1)
# Fit the Grid search model
randf_random.fit(XTrain, YTrain.ravel())


print(f'{randf_random.best_params_} are the Bes Parameters')

### Plotting

X=randf_random.grid_scores_
A = np.zeros(len(X))
B = np.zeros(len(X))
C = np.zeros(len(X))
for i in range(len(X)):
  A[i]=1-X[i].mean_validation_score
  B[i]=X[i].parameters.get('min_samples_split')
  C[i]=X[i].parameters.get('max_depth')
  
import matplotlib.pyplot as plt
plt.plot(B,A)
plt.xlabel('Min Sample Split')
plt.ylabel('Training CrossValidation Error')
plt.title('Optimizing Hyperparameter')
plt.savefig('Desktop\MachineLearning\COMPSCI-589-HW2\Submission\Figures\MinSampleSplit.png')
plt.clf()

plt.plot(C,A)
plt.xlabel('Maximim Depth of the Trees')
plt.ylabel('Training CrossValidation Error')
plt.title('Optimizing Hyperparameter')
plt.savefig('Desktop\MachineLearning\COMPSCI-589-HW2\Submission\Figures\TreesDepthRF.png')

print(f'It took {time.time()-start} seconds.')
