# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 22:52:43 2025

@author: iamtalgaat
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

corrMatrix = dataset.drop('State', axis=1).corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)

# Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(drop='first'), [3])
], remainder='passthrough')

ct.fit(X_train)

X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1),
                      y_test.reshape(len(y_test), 1)), 1))

sns.set_style('whitegrid')
plt.plot(y_test, color='green', marker='o')
plt.plot(y_pred, color='red', marker='x')

plt.title("Output")
plt.show()

regressor.predict([[0, 0, 2500000, 100000, 200000]])


