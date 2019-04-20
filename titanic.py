# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:04:47 2019

@author: Rakshita
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import csv
import json

# Importing the datasets
train_set = pd.read_csv('train.csv')
test_set= pd.read_csv('test.csv')

X = train_set.iloc[:, [2, 4,5,9]].values ##independent variables of Training set
y = train_set.iloc[:, 1].values #dependent variable of training set
test_data= test_set.iloc[:, [1,3,4,8]].values #independent variable of Test set
pID_test= test_set.iloc[:, 0].values
## Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

from sklearn.preprocessing import Imputer
imputer_train = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_train = imputer_train.fit(X[:, [2]])
X[:, [2]] = imputer_train.transform(X[:, [2]])
imputer_test = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_test = imputer_test.fit(test_data[:, [2]])
test_data[:, [2]] = imputer_test.transform(test_data[:, [2]])
imputer_test1 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_test1 = imputer_test1.fit(test_data[:, [3]])
test_data[:, [3]] = imputer_test.transform(test_data[:, [3]])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_train = LabelEncoder()
X[:, 1] = labelencoder_train.fit_transform(X[:, 1])
onehotencoder_train = OneHotEncoder(categorical_features = [1])
X = onehotencoder_train.fit_transform(X).toarray()

labelencoder_test = LabelEncoder()
test_data[:, 1] = labelencoder_test.fit_transform(test_data[:, 1])
onehotencoder_test = OneHotEncoder(categorical_features = [1])
test_data = onehotencoder_test.fit_transform(test_data).toarray()

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X, y)

# Predicting the Test set results
pred = classifier.predict(test_data)

#file_writer= csv.writer(open('output4.csv','w'))
#file_writer.writerow(["PassengerID", "Survived"])
#for i in range(0,len(test_data)):
#    file_writer.writerow([pID_test[i], pred[i]])
row = ['PassengerID', ' Survived']
with open('output.csv', 'w', newline='') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(row)
    for i in range(0,len(test_data)):
        writer.writerow([pID_test[i], pred[i]])
        
        
        
writeFile.close()

