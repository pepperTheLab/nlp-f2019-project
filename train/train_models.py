#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:56:28 2019

@author: xinning.w
"""
import pickle
import json
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 

import train_utils

addresses = train_utils.getDirs()

def findBestParam(clf, param_grid, df_splitted):  
    X_train, X_test, y_train, y_test = [x for x in list(df_splitted.values())]
    
    grid_obj = GridSearchCV(clf, param_grid, cv=5)
    grid_obj = grid_obj.fit(X_train, y_train)
    
    clf = grid_obj.best_estimator_
    
    return clf

def getModelPerformances(y_train, y_train_pred, y_test, y_test_pred):
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    metric_values_train = [accuracy_score(y_train, y_train_pred),
                           precision_score(y_train, y_train_pred),
                           recall_score(y_train, y_train_pred),
                           f1_score(y_train, y_train_pred)]
    metric_values_test = [accuracy_score(y_test, y_test_pred),
                          precision_score(y_test, y_test_pred),
                          recall_score(y_test, y_test_pred),
                          f1_score(y_train, y_train_pred)]
    all_metrics = {name: {'train': train, 'test': test} for name, train, test in zip(metric_names, metric_values_train, metric_values_test)}
    
    return all_metrics

def trainModel(clf, df_splitted, param_grid):
    algorithm_name = str(clf).split('(')[0]
    print(f'Start Training {algorithm_name}...')
    clf = findBestParam(clf, param_grid, df_splitted)
    print('Best Parameters Found!')
    X_train, X_test, y_train, y_test = [x for x in list(df_splitted.values())]
    clf.fit(X_train, y_train)
    
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    all_metrics = getModelPerformances(y_train, y_train_pred, y_test, y_test_pred)
    print(f'{algorithm_name} Completed!')
    with open(addresses['model'] + algorithm_name + '.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open(addresses['performance'] + algorithm_name + '.json', 'w') as f:
        json.dump(all_metrics, f)

def trainLogisticRegression(df_splitted):
    clf = LogisticRegression()
    param_grid = {'penalty': ['l2'],
                  'C': [0.01, 0.1, 1.0],
                  'solver': ['newton-cg', 'lbfgs'],
                  'multi_class': ['multinomial'],
                  'n_jobs': [-1]}
    
    trainModel(clf, df_splitted, param_grid)
    
def trainDecisionTree(df_splitted):
    clf = DecisionTreeClassifier()
    param_grid = {'min_samples_leaf': [3, 5, 10, 20, 30]}
    
    trainModel(clf, df_splitted, param_grid)
    
#def trainSVM(df_splitted):
#    regressor = SVC()
#    param_grid = {'fit_intercept': [True, False],
#                  'normalize': [True, False],
#                  'alpha': [0.01, 0.1, 1.0],
#                  'random_state': [42]}
#    
#    trainModel(regressor, df_splitted, param_grid)
    
def trainRandomForrest(df_splitted):
    clf = RandomForestClassifier()
    param_grid = {'n_estimators': [50, 100, 150, 200], 
                  'min_samples_leaf': [3, 5, 10, 20, 30],
                  'oob_score': [True, False], 
                  'n_jobs': [-1], 
                  'random_state': [42]}
    
    trainModel(clf, df_splitted, param_grid)
    
def trainKNN(df_splitted):
    clf = KNeighborsClassifier()
    param_grid = {'n_neighbors': [3, 5, 10, 20], 
                  'weights': ['uniform', 'distance']}
    
    trainModel(clf, df_splitted, param_grid)
    
def trainGBDT(df_splitted):
    clf = GradientBoostingClassifier()
    param_grid = {'loss': ['deviance'],
                  'learning_rate': [0.01, 0.1, 1.0],
                  'n_estimators': [50, 100, 150, 200], 
                  'max_depth': [3, 5, 10],
                  'random_state': [42]}
    
    trainModel(clf, df_splitted, param_grid)
    
    