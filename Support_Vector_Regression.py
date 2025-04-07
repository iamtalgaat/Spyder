# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 15:33:25 2025

@author: iamtalgaat
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
y = y.reshape(len(y), 1)


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1, 1))

#Vusualizing the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(sc_X.inverse_transform(X), 
         sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)),
         color='blue')
plt.title("SVR results")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()