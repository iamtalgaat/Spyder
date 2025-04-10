# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 19:58:54 2025

@author: iamtalgaat
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25, 
    random_state=0
)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier,
                             X = X_train,
                             y = y_train,
                             cv = 10)
print("Accuracy: {:.2f}%".format(accuracies.mean()*100))
print("Accuracy Deviation: {:.2f}%".format(accuracies.std()*100))


from sklearn.model_selection import GridSearchCV
parameters = [{"C": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], # здесь он перепробует только 4 варианта
               "kernel": ['Linear']},
              {"C": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], # здесь он перепробует все варианты C и Гаммы
               "kernel": ['rbf'],
               "gamma": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}] #Гамма только с 'rbf'

grid_search = GridSearchCV(
    estimator=classifier, # Модель
    param_grid=parameters, # Параметры которые мы указали наверху
    scoring='accuracy', # Оценка будет давать точность
    cv = 10, # Трен будет проходит используя Cross Validation на 10 фолдов
    n_jobs=-1) # -1 - значит что мы будем использовать все свободные ядер процесс

grid_search.fit(X_train, y_train) # Подгоняем
best_accuracy = grid_search.best_score_ # Лучшая точность
best_parameters = grid_search.best_params_ # Лучшие параметры
print("Best Accuracy: {:.2f}%".format(best_accuracy*100))
print("Best Parameters:", best_parameters)


from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start= X_set[:, 0].min() - 1,
                               stop = X_set[:, 0].max() + 1,
                               step = 1.5),
                     np.arange(start= X_set[:, 1].min() - 1,
                               stop = X_set[:, 1].max() + 1,
                               step = 1.5))

plt.contourf(X1,
             X2,
             classifier.predict(
                 sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.65,
             cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X2.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0],
                X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i),
                label = j)
    
plt.title("Kernel SVM (Training Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()















