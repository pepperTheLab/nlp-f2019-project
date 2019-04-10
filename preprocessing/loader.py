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

def getDirs():
    current = os.getcwd()
    addresses = {'raw': current + '/../data/raw/',
                 'processed': current + '/../data/preprocessed/'}
    
    return addresses

def loadData(file):
    df = pd.read_csv(file, encoding='ISO-8859-1')
    
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

