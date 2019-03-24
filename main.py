#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 10:59:50 2019

@author: xinning.w
"""
import loader
import preProcessing


df = loader.loadData()
df = preProcessing.htmlTagRemover(df)
df = preProcessing.characterRemover(df)
df = preProcessing.tokenizer(df)
df = preProcessing.stemAndLemma(df)
df = preProcessing.stopwordsRemover(df)
# df = preProcessing.extractVectorMatrix(df)

df.to_csv('processed.csv', index=False)

