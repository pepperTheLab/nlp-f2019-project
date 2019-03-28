#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 10:59:50 2019

@author: xinning.w
"""
import loader
import preProcessing

PATH_TO_DATA = 'Awards_data/'
df = loader.loadData(PATH_TO_DATA)

df = preProcessing.htmlTagRemover(df)
df = preProcessing.characterRemover(df)
df = preProcessing.tokenizer(df)
df = preProcessing.stemAndLemma(df)
df = preProcessing.stopwordsRemover(df)
df = preProcessing.extractVectorMatrix(df)
df = preProcessing.nonPredictiveFeatureRemover(df)
df = preProcessing.processDateFeatures(df)
df = preProcessing.processCategoricalFeatures(df)

df.to_csv('processed.csv', index=False)

