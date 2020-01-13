# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 22:54:08 2019

@author: sharo
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error 
from sklearn.linear_model import Ridge

diabetes = datasets.load_diabetes()

#dict_keys(['data', 'target', 'DESCR', 'feature_names',
#'data_filename', 'target_filename'])
"""Ten baseline variables, age, sex, body mass index, average blood
pressure, and six blood serum measurements were obtained for each of n =
442 diabetes patients"""


#diabetes_X = diabetes.data[:,np.newaxis,2] # for selecting one feature
diabetes_X = diabetes.data # for all features of data
#print(diabetes_X)

diabetes_X_train = diabetes_X[:-40]
diabetes_X_test = diabetes_X[-40:]

diabetes_y_train = diabetes.target[:-40]
diabetes_y_test = diabetes.target[-40:]

ridge_reg = Ridge(alpha=10, solver = "auto")
ridge_reg.fit(diabetes_X_train, diabetes_y_train)

# 2nd Model Trained with ridge_reg2
ridge_reg2 = Ridge(alpha=5, solver = "auto")
ridge_reg2.fit(diabetes_X_train, diabetes_y_train)

predict = ridge_reg.predict(diabetes_X_test)

print("Mean Squared error is: ",mean_squared_error(predict, diabetes_y_test))

print("weight10: ", ridge_reg.coef_) #model 1 weights
print("weight5: ", ridge_reg2.coef_) # model 2 weights
print("Intercept: ", ridge_reg.intercept_)

#plt.figure(figsize=(20, 6))
#plt.scatter(diabetes_X_test, diabetes_y_test)s
plt.bar([1,2,3,4,5,6,7,8,9,10],ridge_reg2.coef_, label= "alpha = 5")
plt.bar([1,2,3,4,5,6,7,8,9,10],ridge_reg.coef_, label= "alpha = 10")
plt.axis([0,11,-100,150])
plt.legend()
plt.show()

