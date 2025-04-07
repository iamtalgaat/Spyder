# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 11:27:19 2025

@author: iamtalgaat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitting Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=6)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizing the Linear Regression
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title("Linear Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Visualizing the Polynormial Regression results
plt.scatter(X, y, color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title("Polynormial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

lin_reg.predict([[6.5]])

lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
