# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd


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
            df.append(df_new).fillna(None)
            df.reset_index(inplace=True, drop=True)
            
    return df
