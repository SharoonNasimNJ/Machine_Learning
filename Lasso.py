# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 23:27:45 2019

@author: sharo
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 22:54:08 2019

@author: sharo
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error 
from sklearn.linear_model import Lasso

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

Lasso_reg = Lasso(alpha=1)
Lasso_reg.fit(diabetes_X_train, diabetes_y_train)

Lasso_reg2 = Lasso(alpha=0.01)
Lasso_reg2.fit(diabetes_X_train, diabetes_y_train)

predict = Lasso_reg.predict(diabetes_X_test)
predict2 = Lasso_reg2.predict(diabetes_X_test)

print("Mean Squared error is(1): ",mean_squared_error(predict, diabetes_y_test))
print("Mean Squared error(0.01) is: ",mean_squared_error(predict2, diabetes_y_test))

print("weight1: ", Lasso_reg.coef_)
print("weight0.01: ", Lasso_reg2.coef_)

#print("Intercept: ", Lasso_reg.intercept_)

#plt.figure(figsize=(20, 6))
#plt.scatter(diabetes_X_test, diabetes_y_test)s
plt.bar([1,2,3,4,5,6,7,8,9,10],Lasso_reg2.coef_, label="1")
plt.bar([1,2,3,4,5,6,7,8,9,10],Lasso_reg.coef_ , label = "2")
plt.axis([0,11,-500,500])
plt.legend()
#ax = plt.gca() # gets the active axis
#ax.set_aspect()
plt.show()

