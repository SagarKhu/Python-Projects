#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:03:40 2019
@author: Khurana, Sagar
        M.B.A. 2nd yr
"""

# importing necessary packages: 
import sys
import os
targetDirectory = os.path.abspath("C:/PythonCode/introML")
sys.path.append(targetDirectory)

# Load helpers
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt


print("Question 1")
print("LOGISTIC REGRESSION")

# function to make random classes
def make_Class(n): # making n size random smaple of X and y 
    X = np.random.uniform(low=0.,high=0.5,size=(n,10))
    p = np.sum(X[:,0:2],axis=1)
    y = (np.random.uniform(low=0.,high=1.,size=n)<p)
    return X,y

nmc = 1000 # monte-carlo loop
for C, marker in zip([10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001], ['o', '^', 'v', 'o', '^', 'v', 'o', '^', 'v', 'o']):
    for i in range(nmc): 
        X, y = make_Class(40) 
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, test_size=0.5)
    lr_l1 = LogisticRegression(C=C, penalty="l1").fit(X_train, y_train)
    print("ACCURACY:")
    print("Training with C={:.4f}: {:.2f}".format(C, lr_l1.score(X_train, y_train)))
    print("Test with     C={:.4f}: {:.2f}".format(C, lr_l1.score(X_test, y_test)))
    plt.semilogx(lr_l1.coef_.T, marker, label="C={:.4f}".format(C))
    xlims = plt.xlim()
    plt.hlines(0, xlims[0], xlims[1])
    plt.xlim(-1,10)
    plt.legend(loc=1)


print("Question 2")
print("LinearSVC")

for C, marker in zip([10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001], ['o', '^', 'v', 'o', '^', 'v', 'o', '^', 'v', 'o']):
    for i in range(mnc): 
        X, y = make_Class(40) 
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, test_size=0.5)
    lr_l1 = LinearSVC(C=C).fit(X_train, y_train)
    print("ACCURACY:")
    print("Training with C={:.4f}: {:.2f}".format(C, lr_l1.score(X_train, y_train)))
    print("Test with     C={:.4f}: {:.2f}".format(C, lr_l1.score(X_test, y_test)))
    plt.semilogx(lr_l1.coef_.T, marker, label="C={:.4f}".format(C))
    xlims = plt.xlim()
    plt.hlines(0, xlims[0], xlims[1])
    plt.xlim(-1,10)
    plt.legend(loc=1)



print("Question 3")

# load credit card data set
credit_card = pd.read_csv("C:/Users/Nannu/OneDrive/Spring 2019/Subjects/Machine Learning/Problem Sets/Problem Set 3/defaultBal1.csv")

# Display the top 5 records of the dataframe
credit_card.head()

# Set-up data for Scikit-learn
y = credit_card.default
Xall= credit_card.values[:,1:24]
X = Xall.copy()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
yGuess = np.mean(y)


# Let's try Monte-Carlo for GaussianNB()
nmc = 50
trainScore = np.zeros(nmc)
testScore  = np.zeros(nmc)
gnb = GaussianNB()
for i in range(nmc):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
    trainFit = gnb.fit(X_train,y_train)
    trainScore[i] = trainFit.score(X_train,y_train)
    testScore[i] =  trainFit.score(X_test,y_test)
print(np.mean(trainScore))
print(np.std(trainScore))
print(np.mean(testScore))
print(np.std(trainScore))
print(np.mean(testScore>yGuess))


# Q 3.1 Linear DIscriminant Classifier
print("3.1: Monte-Carlo with Linear Discriminant Classifier")

nmc = 50
trainScore = np.zeros(nmc)
testScore  = np.zeros(nmc)
lda = LinearDiscriminantAnalysis()
for i in range(nmc):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    trainFit = lda.fit(X_train,y_train)
    trainScore[i] = trainFit.score(X_train,y_train)
    testScore[i] =  trainFit.score(X_test,y_test)
print(np.mean(trainScore))
print(np.std(trainScore))
print(np.mean(testScore))
print(np.std(trainScore))
print(np.mean(testScore>yGuess))


# Q 3.2 Logistic Regression
print("3.2: Monte-Carlo with Logistic Regression")

nmc = 50
trainScore = np.zeros(nmc)
testScore  = np.zeros(nmc)
for Cval in [100., 1., 0.01]:
    lgr = LogisticRegression(C=Cval)
    for k in range(nmc):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        model = LogisticRegression(solver='liblinear')
        trainFit = lgr.fit(X_train,y_train)
        trainScore[k] = trainFit.score(X_train,y_train)
        testScore[k] =  trainFit.score(X_test,y_test)
    print('C=',Cval)
    print(np.mean(trainScore))
    print(np.std(trainScore))
    print(np.mean(testScore))
    print(np.std(trainScore))
    print(np.mean(testScore>yGuess))


# Q 3.3 Linear SVC
print("3.3: Monte-Carlo with Linear SVC")

nmc = 50
trainScore = np.zeros(nmc)
testScore  = np.zeros(nmc)
for Cval in [100., 1., 0.01]:
    lsvc = LinearSVC(C=Cval)
    for i in range(nmc):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
        trainFit = lsvc.fit(X_train,y_train)
        trainScore[i] = trainFit.score(X_train,y_train)
        testScore[i] =  trainFit.score(X_test,y_test)
    print('C=',Cval)
    print(np.mean(trainScore))
    print(np.std(trainScore))
    print(np.mean(testScore))
    print(np.std(trainScore))
    print(np.mean(testScore>yGuess))
    

# Q 3.4 KNN
print("3.4: Monte-Carlo with K Nearest Neighbor")

nmc = 50
trainScore = np.zeros(nmc)
testScore = np.zeros(nmc)
for Nval in [3, 11, 25]:
    knc = KNeighborsClassifier(n_neighbors = Nval)
    for k in range(nmc):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
        trainFit = knc.fit(X_train, y_train)
        trainScore[k] = trainFit.score(X_train, y_train)
        testScore[k] = trainFit.score(X_test, y_test)
    print("K Neighbors: ", Nval)
    print(np.mean(trainScore))
    print(np.std(trainScore))
    print(np.mean(testScore))
    print(np.std(testScore))
    print(np.mean(testScore>yGuess))
    

# Q 3.5 Decision Tree
print("3.5 Monte-Carlo with Decision Tree")

nmc = 50
trainScore = np.zeros(nmc)
testScore  = np.zeros(nmc)
for maxdepth in [5,10,25]:
    dtc = DecisionTreeClassifier(max_depth=maxdepth)
    for i in range(nmc):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
        trainFit = model.fit(X_train,y_train)
        trainScore[i] = trainFit.score(X_train,y_train)
        testScore[i] =  trainFit.score(X_test,y_test)
    print('depth: ',maxdepth)
    print(np.mean(trainScore))
    print(np.std(trainScore))
    print(np.mean(testScore))
    print(np.std(trainScore))
    print(np.mean(testScore>yGuess))


# Q 3.6 Linear Discriminant with Real Valued Data
print("3.6 Monte-Carlo on Linear Discriminant with Real Valued Data")

Xreal = Xall[:,12:24].copy()
nmc = 50
trainScore = np.zeros(nmc)
testScore = np.zeros(nmc)
lda2 = LinearDiscriminantAnalysis()
for i in range(nmc):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    trainFit = lda2.fit(X_train, y_train)
    trainScore[i] = trainFit.score(X_train, y_train)
    testScore[i] = trainFit.score(X_test, y_test)
print(np.mean(trainScore))
print(np.std(trainScore))
print(np.mean(testScore))
print(np.std(testScore))
print(np.mean(testScore>yGuess))


# Q 3.7 Best method
print("On comparing the mean test scores of the above 6 methods, I feel that Linear Discriminant Analysis is the best method so far as the mean test score for this analysis is higher than the others. As Linear Discriminant has the best mean score, the Linear Discriminant nalysis with real valued data falls 2nd in line followed by the Decision Tree method. Decision Tree may have better results if we maintain the depth accordingly. Logistic Regression and Linear SVC shows almost similar scores with Logistic having an edge over SVC. KNN and Gaussian have the worst mean scores.")
print("The best method based on the mean test score is Linear Discriminant Analysis." )