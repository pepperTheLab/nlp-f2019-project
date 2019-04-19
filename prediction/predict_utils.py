# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd

def getDirs():
    current = os.getcwd()
    addresses = {'raw': current + '/../data/raw/',
                 'processed': current + '/../data/preprocessed/',
                 'untagged': current + '/../data/untagged/',
                 'meta': current + '/../data/meta/',
                 'model': current + '/../data/models/',
                 'performance': current + '/../data/performances/'}
    
    for folder in addresses.values():
        if not os.path.exists(folder):
            os.mkdir(folder)
    
    return addresses

addresses = getDirs()

def loadModels():
    files = [file for file in os.listdir(addresses['model']) if file.split('.')[-1]=='pkl']
    models = {}
    for file in files:
        name = file.split('.')[0]
        if name not in ['medoids', 'vocab']:
            with open(addresses['model'] + file, 'rb') as f:
                clf = pickle.load(f)
            models[name] = clf
        
    return models

def loadMedoids():
    with open(addresses['model'] + 'medoids.pkl', 'rb') as f:
        medoids = pickle.load(f)
    
    return medoids

def loadVocab():
    with open(addresses['model'] + 'vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    return vocab

def loadTextData():
    csvs = [file for file in os.listdir(addresses['processed']) if file.split('.')[-1]=='pkl']
    text_files = [file for file in csvs if file.split('.')[0].split('_')[-1]=='text']
    df = None
    for file in text_files:
        with open(addresses['processed']+file, 'rb') as f:
            df_new = pickle.load(f)
        if df is None:
            df = df_new
        else:
            df = pd.concat([df, df_new])
            df.reset_index(inplace=True, drop=True)
            
    return df

def loadNumData():
    csvs = [file for file in os.listdir(addresses['processed']) if file.split('.')[-1]=='pkl']
    text_files = [file for file in csvs if file.split('.')[0].split('_')[-1]=='num']
    df = None
    for file in text_files:
        with open(addresses['processed']+file, 'rb') as f:
            df_new = pickle.load(f)
        if df is None:
            df = df_new
        else:
            df = pd.concat([df, df_new])
            df.reset_index(inplace=True, drop=True)
            
    return df

def voteCalc(row):
    sum_vote = row['GradientBoostingClassifier_prediction'] + row['KNeighborsClassifier_prediction'] + row['LogisticRegression_prediction'] + row['RandomForestClassifier_prediction'] + row['DecisionTreeClassifier_prediction']
    if sum_vote > 2.5:
        return 1
    else:
        return 0





















    
