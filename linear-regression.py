#!/usr/bin/env python
# coding: utf-8

import pandas as pd

df = pd.read_csv('datasheet.csv')
df.info()
df.shape
df.isnull().sum()
print(df)

#Separate the null values from dataframe (df) and create a variable “test data”
test_data = df[df['C'].isnull()]

#Drop the null values from the dataframe (df) and represent as ‘train data”
df.dropna(inplace=True)


#Create “x_train” & “y_train” from train data.
x_train = df.drop('C',axis=1)
y_train = df['C']


#Build the linear regression model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)


x_test = test_data[['A','B']]


#Apply the model on x_test of test data to make predictions.
y_pred = lr.predict(x_test)

#Replace the missing values with predicted values.
test_data['y_pred'] = y_pred
print(test_data)
