#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 10:53:40 2019

@author: xinning.w
"""
import re
from nltk.tokenize import word_tokenize

import loader

stem = loader.loadStem()
lemma = loader.loadLemma()
stop_words = loader.loadStopwords()

def textWrapper(orig_func):
    def wrapper(*args, **kwargs):
        try:
            return orig_func(*args, **kwargs)
        except TypeError:
            return empty()
    
    return wrapper
            
def empty():
    return ''

@textWrapper
def htmlTagRemover(text):
    return re.sub(r'<.+>', '', text)

@textWrapper
def characterRemover(text):
    return re.sub(r'\W', ' ', text)

@textWrapper
def tokenizer(text):
    return word_tokenize(text)

def unTokenize(tokens):
    return ' '.join(tokens)

def tokenStemAndLemma(token):
    return stem.stem(lemma.lemmatize(token))

def tokensStemAndLemma(tokens):
    return [tokenStemAndLemma(token) for token in tokens]

def stopwordsRemover(tokens):
    return [token for token in tokens if token not in stop_words]

