#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from sklearn import metrics

train = pd.read_csv('main.csv')

feature_cols = ['timestamp','axis1','axis2','axis3']
X = train[feature_cols]
Y = train['Activity']

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.7,random_state=0)

bounds = {
    'C':(1,50)
}


def lr(C):
    params = {
        'C': int(C)
    }

    # instantiate the model (using the default parameters)
    logreg = LogisticRegression(C=int(C))

    # fit the model with data
    logreg.fit(X_train,y_train)

    y_pred=logreg.predict(X_test)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    acc = metrics.accuracy_score(y_test, y_pred)
    return acc

optimizer = BayesianOptimization(
    f=lr,
    pbounds=bounds,
    random_state=1,
    )

optimizer.maximize(n_iter=50)

accuracy = []
Cpara = []
for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))
    accuracy.append(optimizer.res[i]['target'])
    Cpara.append(optimizer.res[i]['params']['C'])

plt.plot(Cpara,accuracy, label = 'Accuracy')
plt.xlabel("C parameter")
plt.ylabel("Performance")
plt.title("Bayesian Optimization Performance for optimal C parameter")




