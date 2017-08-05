#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import modified dataset
train = pd.read_csv("train_modified.csv")
test = pd.read_csv("test_modified.csv")



# =============================================================================
# baseline model
# =============================================================================
mean_sales = train['Item_Outlet_Sales'].mean()

#Define a dataframe with IDs for submission:
base1 = test[['Item_Identifier','Outlet_Identifier']]
base1['Item_Outlet_Sales'] = mean_sales

#Export submission file
base1.to_csv("alg__baseline-model.csv",index=False)



# =============================================================================
# generic function to fit algorithm and data
# =============================================================================
#Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
from sklearn import cross_validation, metrics
def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Perform cross-validation:
    cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)
    

    
# =============================================================================
# Linear Regression Model
# =============================================================================
from sklearn.linear_model import LinearRegression
predictors = [x for x in train.columns if x not in [target]+IDcol]
# print predictors
alg1 = LinearRegression(normalize=True)
modelfit(alg1, train, test, predictors, target, IDcol, 'alg1.csv')
coef1 = pd.Series(alg1.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')



# =============================================================================
# Ridge Regression Model
# =============================================================================
from sklearn.linear_model import Ridge
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg2 = Ridge(alpha=0.05,normalize=True)
modelfit(alg2, train, test, predictors, target, IDcol, 'alg2.csv')
coef2 = pd.Series(alg2.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')



# =============================================================================
# Decision Tree Model
# =============================================================================
from sklearn.tree import DecisionTreeRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg3 = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(alg3, train, test, predictors, target, IDcol, 'alg3.csv')
coef3 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances')

# after parameter tuning
predictors = ['Item_MRP','Outlet_Type_0','Outlet_5','Outlet_Years']
alg4 = DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)
modelfit(alg4, train, test, predictors, target, IDcol, 'alg4.csv')
coef4 = pd.Series(alg4.feature_importances_, predictors).sort_values(ascending=False)
coef4.plot(kind='bar', title='Feature Importances')



# =============================================================================
# Random Forest Model
# =============================================================================
from sklearn.ensemble import RandomForestRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg5 = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=-1)
modelfit(alg5, train, test, predictors, target, IDcol, 'alg5.csv')
coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending=False)
coef5.plot(kind='bar', title='Feature Importances')

# after parameter tuning
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg6 = RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=100,n_jobs=-1)
modelfit(alg6, train, test, predictors, target, IDcol, 'alg6.csv')
coef6 = pd.Series(alg6.feature_importances_, predictors).sort_values(ascending=False)
coef6.plot(kind='bar', title='Feature Importances')

# after parameter tuning #2
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg7 = RandomForestRegressor(n_estimators=500,max_depth=6,min_samples_leaf=100,n_jobs=-1)
modelfit(alg7, train, test, predictors, target, IDcol, 'alg7.csv')
coef7 = pd.Series(alg7.feature_importances_, predictors).sort_values(ascending=False)
coef7.plot(kind='bar', title='Feature Importances')


# =============================================================================
# Grid Search
# =============================================================================
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


alg8 = RandomForestRegressor(n_estimators=500,n_jobs=-1)
# use a full grid over all parameters
param_grid = {"max_depth": [7,6,5],
              "min_samples_leaf": [40, 50, 60, 80]}

grid_search = GridSearchCV(estimator=alg8, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=-1)

predictors = [x for x in train.columns if x not in [target]+IDcol]
grid_search.fit(train[predictors], train[target])
grid_search.best_params_
grid_search.best_estimator_


# Random forest from findings
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg8 = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=6,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_split=1e-07, min_samples_leaf=10,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=500, n_jobs=-1, oob_score=True, random_state=None,
           verbose=0, warm_start=False)
modelfit(alg8, train, test, predictors, target, IDcol, 'alg8.csv')
coef8 = pd.Series(alg8.feature_importances_, predictors).sort_values(ascending=False)
coef8.plot(kind='bar', title='Feature Importances')
