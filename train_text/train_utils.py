#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:55:45 2019

@author: xinning.w
"""
import os
import pickle
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def getDirs():
    current = os.getcwd()
    addresses = {'raw': current + '/../data/raw/',
                 'processed': current + '/../data/preprocessed/',
                 'meta': current + '/../data/meta/',
                 'model': current + '/../data/models/',
                 'performance': current + '/../data/performances/'}
    
    for folder in addresses.values():
        if not os.path.exists(folder):
            os.mkdir(folder)
    
    return addresses

addresses = getDirs()

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

def extractVocab(df):
    vocab = [word for t in df['Tokens'] for word in t]
    vocab = [word for word in vocab if len(word)>1]
    counter = Counter(vocab)
    vocab = [word for word, count in counter.most_common(3000)]
    
    return vocab

def extractDocuments(df):
    documents = [' '.join(tokens) for tokens in df['Tokens']]
    
    return documents

def extractVectorMatrix(df):
    vocab = extractVocab(df)
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    matrix = vectorizer.fit_transform(extractDocuments(df)).todense()
    vectors = pd.DataFrame(matrix)
    vectors.columns = vocab
    df.reset_index(inplace=True, drop=True)
    vectors.reset_index(inplace=True, drop=True)
    df = pd.concat([df, vectors], axis=1)
    df.drop('Tokens', axis=1, inplace=True)
    
    return df

def trainTestSplit(df):
    X = df.drop('AwardedAmountToDate', axis=1).values
    y = df['AwardedAmountToDate'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42)
    df_splitted = {'X_train':X_train,
                   'X_test':X_test,
                   'y_train':y_train,
                   'y_test':y_test}
    
    return df_splitted
    
    
