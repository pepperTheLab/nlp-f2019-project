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

def htmlTagRemover(text):
    return re.sub(r'<.+>', '', text)

def characterRemover(text):
    return re.sub(r'\W', ' ', text)

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

