# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:21:52 2020

@author: msi
"""

# A person has affair or not

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

affair = pd.read_csv("F:\\Data Science\\Assignemnts\\Brindan\\logistic regression\\affairs.csv")
affair
affair.shape
affair.columns
affair.isnull().sum() # there are no missing values

np.mean(affair)

import seaborn as sb

plt.boxplot(affair.affairs)
plt.hist(affair.affairs)
pd.crosstab(affair.affairs, affair.rating).plot(kind='bar')

from sklearn.linear_model import LogisticRegression

import seaborn as sb
sb.countplot(affair.gender)
sb.countplot(affair.children)
sb.countplot(affair.affairs)
sb.countplot(affair.rating)

help(sb.boxplot)
sb.boxplot(data = affair,orient = "v")


pd.crosstab(affair.gender,affair.children).plot(kind="bar")
pd.crosstab(affair.affairs, affair.children).plot(kind='bar')
pd.crosstab(affair.affairs, affair.rating).plot(kind='bar')

affair.shape
x= affair.iloc[:,[1,2,3,4,5,6,7,8]]
y = affair.iloc[:,0]

model=LogisticRegression()
model.fit(x,y)

affair_dummies = pd.get_dummies(affair[["gender","children"]])
affair.drop(['gender','children'], inplace=True,axis = 1)

affair = pd.concat([affair,affair_dummies],axis=1)

affair.shape
x= affair.iloc[:,[1,2,3,4,5,6,7,8,9,10]]
y = affair.iloc[:,0]
model.fit(x,y)
model.coef_
y_prob=model.predict_proba(x)

y_pred = model.predict(x)
affair["y_pred"] = y_pred

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y,y_pred)
print (confusion_matrix)


pd.crosstab(affair.affairs, affair.y_pred).plot(kind='bar')

from sklearn.metrics import accuracy_score
accuracy_score(y,y_pred) #0.75