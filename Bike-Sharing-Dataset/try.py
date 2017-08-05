#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 12:17:08 2017

@author: susovan
"""

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


X = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(123)
y = np.sin(X) + np.random.normal(0, .15, len(X))
data = pd.DataFrame(np.column_stack([X,y]), columns=['X', 'y'])
plt.scatter(X, y, color = 'red')


for i in range(2,16):
    colname = 'X_%d'%i
    data[colname] = data['X']**i
    
print(data.head())

from sklearn.linear_model import LinearRegression
def linear_regression(data, power, models_to_plot):
    predictors = ['X']
    if power>=2:
        predictors.extend(['X_%d'%i for i in range(2, power+1)])
        
    regressor = LinearRegression(normalize=True)
    regressor.fit(data[predictors], data['y'])
    y_pred = regressor.predict(data[predictors])
    
    if power in models_to_plot:
        plt.subplot(models_to_plot[power])
        plt.tight_layout()
        plt.plot(data['X'], y_pred)
        plt.plot(data['X'], data['y'], '.')
        plt.title('Plot for power: %d'%power)
        
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([regressor.intercept_])
    ret.extend(regressor.coef_)
    return ret


#Initialize a dataframe to store the results:
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['model_pow_%d'%i for i in range(1,16)]
coef_matrix_simple = pd.DataFrame(index=ind, columns=col)

#Define the powers for which a plot is required:
models_to_plot = {1:231,3:232,6:233,9:234,12:235,15:236}

#Iterate through all powers and assimilate results
for i in range(1,16):
    coef_matrix_simple.iloc[i-1,0:i+2] = linear_regression(data, power=i, models_to_plot=models_to_plot)














# Import the dataset
dataset = pd.read_csv('hour.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Encode categorical data
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
# Avoid Dummy Variable Trap
X = X[:, 1:]"""


# Split dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


# Fit Regression Model (Create regressor)

# Sample multiple linear regression
"""from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)"""

# Sample ploynomial regression
"""from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_fetr = PolynomialFeatures(degree=3)
X_poly = poly_fetr.fit_transform(X)
regressor = LinearRegression()
regressor.fit(X_poly, y)"""


# Predict result
y_pred = regressor.predict(420)

# Visualize result
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('ML Study (Regression Model)')
plt.xlabel('X Lable')
plt.ylabel('Y Lable')
plt.show()

# Visualize result in higher resolution with smoother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('ML Study (Regression Model)')
plt.xlabel('X Lable')
plt.ylabel('Y Lable')
plt.show()
