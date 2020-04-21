# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:52:30 2020

@author: Flo
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('data\\Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values


#Variables catégorique qui nécessite un réencodage - celle qui ont plus de deux catégories diff de 0 / 1 (Pays  et genre par ex)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1=LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:, 1])


labelencoder_X_2=LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:, 2])

#catégorisation de pays : Trois variable oui ou non (Espagne France Allemagne)
onehotencoder = OneHotEncoder(categorical_features = [2])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


#Split de données entre partie entrainement et test

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state= 0)

# changement d'echelle des variables

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Partie 2 Construire le Réseau de neurones


from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()


# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, Y_train, batch_size = 10, epochs = 2)
Y_pred = classifier.predict(X_test)

"""Predire si le client va quitter la banque:
    
    
    

    
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000

Création du vecteur"""

new_prediction = classifier.predict(sc.transform(np.array([[1, 600, 0, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

from sklearn.metrics import confusion_matrix






