#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:56:28 2019

@author: xinning.w
"""
import pickle
import json
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 

import train_utils

addresses = train_utils.getDirs()

def findBestParam(regressor, param_grid, df_splitted):  
    X_train, X_test, y_train, y_test = [x for x in list(df_splitted.values())]
    
    grid_obj = GridSearchCV(regressor, param_grid, cv=5)
    grid_obj = grid_obj.fit(X_train, y_train)
    
    regressor = grid_obj.best_estimator_
    
    return regressor

def getModelPerformances(y_train, y_train_pred, y_test, y_test_pred):
    metric_names = ['MSE', 'MAE', 'R-Square']
    metric_values_train = [mean_squared_error(y_train, y_train_pred),
                           mean_absolute_error(y_train, y_train_pred),
                           r2_score(y_train, y_train_pred)]
    metric_values_test = [mean_squared_error(y_test, y_test_pred),
                          mean_absolute_error(y_test, y_test_pred),
                          r2_score(y_test, y_test_pred)]
    all_metrics = {name: {'train': train, 'test': test} for name, train, test in zip(metric_names, metric_values_train, metric_values_test)}
    
    return all_metrics

def trainModel(regressor, df_splitted, param_grid):
    algorithm_name = str(regressor).split('(')[0]
    regressor = findBestParam(regressor, param_grid, df_splitted)
    X_train, X_test, y_train, y_test = [x for x in list(df_splitted.values())]
    regressor.fit(X_train, y_train)
    
    y_train_pred = regressor.predict(X_train)
    y_test_pred = regressor.predict(X_test)
    
    all_metrics = getModelPerformances(y_train, y_train_pred, y_test, y_test_pred)
    
    with open(addresses['model'] + algorithm_name + '.pkl', 'wb') as f:
        pickle.dump(regressor, f)
    with open(addresses['performance'] + algorithm_name + '.json', 'w') as f:
        json.dump(all_metrics, f)

def trainLinearRegression(df_splitted):
    regressor = LinearRegression()
    param_grid = {'fit_intercept': [True, False],
                  'normalize': [True, False],
                  'n_jobs': [-1]}
    
    trainModel(regressor, df_splitted, param_grid)
    
def trainRidge(df_splitted):
    regressor = Ridge()
    param_grid = {'fit_intercept': [True, False],
                  'normalize': [True, False],
                  'alpha': [0.01, 0.1, 1.0],
                  'random_state': [42]}
    
    trainModel(regressor, df_splitted, param_grid)
    
def trainLasso(df_splitted):
    regressor = Lasso()
    param_grid = {'fit_intercept': [True, False],
                  'normalize': [True, False],
                  'alpha': [0.01, 0.1, 1.0],
                  'random_state': [42]}
    
    trainModel(regressor, df_splitted, param_grid)
    
def trainDecisionTree(df_splitted):
    regressor = DecisionTreeRegressor()
    param_grid = {'criterion': ['mse', 'mae'],
                  'min_samples_leaf': [3, 5, 10, 20, 50],
                  'random_state': [42]}
    
    trainModel(regressor, df_splitted, param_grid)
    
def trainRandomForest(df_splitted):
    regressor = RandomForestRegressor()
    param_grid = {'n_estimators': [50, 100, 200],
                  'min_samples_leaf': [3, 5, 10, 20, 50],
                  'random_state': [42],
                  'n_jobs': [-1]}
    
    trainModel(regressor, df_splitted, param_grid)
    
def trainGBDT(df_splitted):
    regressor = GradientBoostingRegressor()
    param_grid = {'n_estimators': [50, 100, 200],
                  'max_depth': [3, 5, 10],
                  'random_state': [42]}
    
    trainModel(regressor, df_splitted, param_grid)
    
    