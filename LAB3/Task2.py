# -*- coding: utf-8 -*-


import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
#Load dataset
cancer = datasets.load_breast_cancer()

# print the names of the features
print("Features: ", cancer.feature_names)

# print the label type of cancer
print("Labels: ", cancer.target_names)

# print data(feature)shape
cancer.data.shape

print("\nData: ",cancer.data)
print("\nTarget: ",cancer.target)

from sklearn.model_selection import train_test_split

#split data set into train and test sets
data_train, data_test, target_train, target_test = train_test_split(cancer.data,
                        cancer.target, test_size = 0.4, random_state = 26)

gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(data_train, target_train)

#Predict the response for test dataset
target_pred = gnb.predict(data_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(target_test, target_pred))

#Import confusion_matrix from scikit-learn metrics module for confusion_matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(target_test, target_pred)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

precision = precision_score(target_test, target_pred, average=None)
recall = recall_score(target_test, target_pred, average=None)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))

