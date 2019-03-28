#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 10:57:50 2019

@author: xinning.w
"""
import os
import pandas as pd
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


def loadData(path):
    df = None
    for data in [file for file in os.listdir(path) if file[-4:]=='.csv']:
        df_batch = pd.read_csv(path + data, encoding='ISO-8859-1')
        if df == None:
            df = df_batch
        else:
            df = pd.concat([df, df_batch], axis=1)
    
    return df

def loadStem():
    stem = SnowballStemmer('english')
    
    return stem

def loadLemma():
    lemma = WordNetLemmatizer()
    
    return lemma

def loadStopwords():
    stop_words = set(stopwords.words('english'))
    
    return stop_words

def loadVectorizer(vocab):
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    
    return vectorizer

