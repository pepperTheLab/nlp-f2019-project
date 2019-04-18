#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 10:59:50 2019

@author: xinning.w
"""
import os
import pickle

import loader
import preProcessing

addresses = loader.getDirs()

def processFile(file):
    file_name = file.split('/')[-1].split('.')[0]
    
    df = loader.loadData(file)
    
    df = preProcessing.targetToNum(df)
    df = preProcessing.createLabel(df)
    df_text = df[['AwardedAmountToDate', 'Abstract']]
    df_num = df.drop('Abstract', axis=1)
    
    df_text = preProcessing.htmlTagRemover(df_text)
    df_text = preProcessing.characterRemover(df_text)
    df_text = preProcessing.tokenizer(df_text)
    df_text = preProcessing.stemAndLemma(df_text)
    df_text = preProcessing.stopwordsRemover(df_text)
    
    df_num = preProcessing.nonPredictiveFeatureRemover(df_num)
    df_num = preProcessing.processDateFeatures(df_num)
    df_num = preProcessing.processCategoricalFeatures(df_num)
    
    text_file_name = file_name + '_text.pkl'
    num_file_name = file_name + '_num.pkl'
    with open(addresses['processed'] + text_file_name, 'wb') as f:
        pickle.dump(df_text, f)
    with open(addresses['processed'] + num_file_name, 'wb') as f:
        pickle.dump(df_num, f)

def runPreprocessing():
    for file in os.listdir(addresses['raw']):
        if file.split('.')[-1]=='csv':
            processFile(addresses['raw'] + file)
            
if __name__ == '__main__':
    runPreprocessing()
            
