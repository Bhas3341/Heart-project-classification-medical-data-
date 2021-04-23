#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 17:01:03 2021

@author: bhaskaryuvaraj
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
header_row = ['age','sex','pain','BP','chol','fbs','ecg','maxhr','eiang',
              'eist','slope','vessels','thal','diagnosis']
heart = (pd.read_table('/Users/bhaskaryuvaraj/Library/Mobile Documents/com~apple~CloudDocs/Documents/data_science work_files/Heart Disease data.txt',
                       sep=',', names=header_row))

heart.dtypes
heart.columns
heart['vessels']=pd.to_numeric(heart['vessels'], errors='coerce')
heart['thal']=pd.to_numeric(heart['thal'], errors='coerce')

heart.isnull().sum()
heart.dropna(inplace=True)

heart['diagnosis'].unique()
heart['diagnosis'].replace([1,2,3,4],[1,1,1,1],inplace=True)

#------------------------------------------------------EDA-----------------------------------------------

heart.groupby(['age','diagnosis'])['diagnosis'].count().plot.hist(bins=20,stacked=True)
plt.figure()
heart[['age','diagnosis']].plot.hist(bins=12)
pyplot.hist('age',bins=6,alpha=0.5,lable='age')
pyplot.hist('diagnosis',bins=2,alpha=0.5,lable='diagnosis')
bins = numpy.linspace(1, 100)
pyplot.legend(loc='upper right')
pyplot.show()

heart['age'].hist(by=heart['diagnosis'],bins=5)
#if you notice from the graph it is clear that people between age 55 to 60 are most affected from heart
#problems

heart.groupby(['sex','diagnosis'])['diagnosis'].count().plot(kind='bar')
#mostly sex with number 1 is affected the most and sex with number 0 are affected the least

heart.plot(kind='scatter',x='chol',y='BP')
heart.groupby('BP')['chol'].median().plot(kind='line')
#from the graph it is not clear about the relationship btw BP and cholestrol
heart.groupby([pd.cut(x=heart['BP'],bins=(90,119,139,159,189,209,229)),'diagnosis'])['diagnosis'].count().plot(kind='bar')
#people with BP btw 119-139 are mostly affected 
heart['chol'].unique()
heart.groupby([pd.cut(x=heart['chol'],bins=(100,149,199,249,299,349,399,449,499,549,599)),'diagnosis'])['diagnosis'].count().plot(kind='bar')
#people with chol between 200-300 are mostly affected

heart.groupby(['pain','diagnosis'])['diagnosis'].count().plot(kind='bar')
#it is obvious that people with more pain are mostly affected

heart.groupby(['fbs','diagnosis'])['diagnosis'].count().plot(kind='bar')
#most people are affected if fbs=0

heart.groupby(['ecg','diagnosis'])['diagnosis'].count().plot(kind='bar')
#most people are affected if ecg is 0 and 2 and very few people are affected with ecg=1

heart.groupby(['eiang','diagnosis'])['diagnosis'].count().plot(kind='bar')
#most people are affected with eiang=0

heart.groupby(['eist','diagnosis'])['diagnosis'].count().plot(kind='line')
#most people are affected with eist as 0

heart.groupby(['slope','diagnosis'])['diagnosis'].count().plot(kind='bar')
#most people are affected with slope of 2

heart.groupby(['vessels','diagnosis'])['diagnosis'].count().plot(kind='bar')
#most people are affected with vessels 0 and 1

heart.groupby(['thal','diagnosis'])['diagnosis'].count().plot(kind='bar')
#most people are affected with thal of 7
heart['maxhr'].unique()
heart.groupby([pd.cut(x=heart['maxhr'],bins=(70,99,119,139,159,179,199,219)),'diagnosis'])['diagnosis'].count().plot(kind='bar')
#most people are affected with maxhr btw 139-159

#-----------------------------------------------------EDA end---------------------------------------------------

#now to check for outliers, outliers are checked only for age,BP, chol, maxhr,eist,
plt.boxplot(heart['age'])
plt.boxplot(heart['BP'])     #has outlier
plt.boxplot(heart['chol'])  #has outlier
plt.boxplot(heart['maxhr'])  #has outlier
plt.boxplot(heart['eist'])   #has outlier

#to remove outlier
def remove_outlier(d,c):
    q1=d[c].quantile(0.25)
    q3=d[c].quantile(0.75)
    iqr=q3-q1
    ub=q3+1.53*iqr
    lb=q1-1.53*iqr
    result=d[(d[c]>lb) & (d[c]<ub)]
    return result

heart=remove_outlier(heart,'BP')  
plt.boxplot(heart['BP'])  
heart=remove_outlier(heart,'chol')
plt.boxplot(heart['chol'])  
heart=remove_outlier(heart,'maxhr')
plt.boxplot(heart['maxhr']) 
heart=remove_outlier(heart,'eist')
plt.boxplot(heart['eist'])

#---------------------Feature Selection-------------------------------------------------
#they provide the same information. Hence we will remove them
correlated_features = set()
correlation_matrix = heart.drop('diagnosis', axis=1).corr()

for i in range(len(correlation_matrix.columns)):

    for j in range(i):

        if abs(correlation_matrix.iloc[i, j]) > 0.8:

            colname = correlation_matrix.columns[i]

            correlated_features.add(colname)
#Check correlated features            
print(correlated_features)
#none of the columns are highly coreleated
#seperating the dependent and independent variables
y=heart['diagnosis'].copy()
x=heart.drop('diagnosis', axis=1)

#create training and test data by splitting x and y into 70:30 ratio
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=6)
x_train.columns
#---------------------------------Decision tree starts---------------------------------------------------
#classification with scikit-learn decision tree
from sklearn import tree
d_tree = tree.DecisionTreeClassifier()
d_tree.fit(x_train, y_train)
         
dtree_pred=d_tree.predict(x_test)
d_tree.score(x_test,y_test)
#accuracy=0.6904761904761905
#---------------------------------Decision tree ends---------------------------------------------------