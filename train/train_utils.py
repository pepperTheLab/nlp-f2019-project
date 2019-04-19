#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:55:45 2019

@author: xinning.w
"""
import os
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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

def downSampleMajor(df):
    minor = df[df['AwardedAmountToDate']==1]
    n_sample = minor.shape[0]
    downsampled = df[df['AwardedAmountToDate']==0].sample(n=n_sample, random_state=1)
    df = pd.concat([minor, downsampled])
    
    return shuffle(df)

def upSampleMinor(df):
    major = df[df['AwardedAmountToDate']==0]
    n_sample = major.shape[0]
    upsampled = df[df['AwardedAmountToDate']==1].sample(n=n_sample, random_state=1)
    df = pd.concat([major, upsampled])
    
    return shuffle(df)

def extractVocab(df):
    vocab = [word for t in df['Tokens'] for word in t]
    vocab = [word for word in vocab if len(word)>1]
    counter = Counter(vocab)
    vocab = [word for word, count in counter.most_common(3000)]
    with open(addresses['model'] + 'vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    
    return vocab

def extractDocuments(df):
    documents = [' '.join(tokens) for tokens in df['Tokens']]
    
    return documents

def extractVectorMatrix(df):
    print('Extracting Vector Matrix...')
    vocab = extractVocab(df)
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    matrix = vectorizer.fit_transform(extractDocuments(df)).todense()
    vectors = pd.DataFrame(matrix)
    vectors.columns = vocab
    df.reset_index(inplace=True, drop=True)
    vectors.reset_index(inplace=True, drop=True)
    df = pd.concat([df, vectors], axis=1)
    df.drop('Tokens', axis=1, inplace=True)
    print('Vector Matrix Extraction Complete!')
    
    return df

def generateMedoids(df):
    print('Generating Medoids...')
    df1 = df[df['AwardedAmountToDate']==1].drop('AwardedAmountToDate', axis=1)
    df0 = df[df['AwardedAmountToDate']==0].drop('AwardedAmountToDate', axis=1)
    medoid1 = df1.median().values
    medoid0 = df0.median().values
    medoids = {1: medoid1,
               0: medoid0}
    with open(addresses['model'] + 'medoids.pkl', 'wb') as f:
        pickle.dump(medoids, f)
    print('Medoids Generation Completed!')
    
    return medoids
        
def cosineSimilarity(medoid, a):
    try:
        compare = np.array([a.tolist(), medoid.tolist()])
        compare_sparse = sparse.csr_matrix(compare)
        similarity = cosine_similarity(compare_sparse)[0][1]
    except:
        return 0.0
    
    return similarity

def computeSimilarities(df, medoids):
    print('Calculating Similarity Scores...')
    df_sim = pd.DataFrame(columns = ['similarity_0', 'similarity_1'])
    for index, row in  enumerate(df.drop('AwardedAmountToDate', axis=1).iterrows()):
        df_sim.loc[index, 'similarity_0'] = cosineSimilarity(medoids[0], row[1].values)
        df_sim.loc[index, 'similarity_1'] = cosineSimilarity(medoids[1], row[1].values)
    print('Similarity Calculation Completed!')
    
    return df_sim
    
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
    
    
