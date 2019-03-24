#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 11:22:40 2019

@author: xinning.w
"""
import loader
import base


def htmlTagRemover(df):
    df['Abstract'] = df['Abstract'].apply(lambda x: base.htmlTagRemover(x))
    
    return df

def characterRemover(df):
    df['Abstract'] = df['Abstract'].apply(lambda x: base.characterRemover(x))
    
    return df

def tokenizer(df):
    df['Tokens'] = df['Abstract'].apply(lambda x: base.tokenizer(x))
    
    return df

def stemAndLemma(df):
    df['Tokens'] = df['Tokens'].apply(lambda x: base.tokensStemAndLemma(x))
    
    return df

def stopwordsRemover(df):
    df['Tokens'] = df['Tokens'].apply(lambda x: base.stopwordsRemover(x))
    
    return df

def extractVocab(df):
    vocab = [word for ab in df.Tokens for word in ab]
    vocab = list(set(vocab))
    
    return vocab

def extractDosuments(df):
    documents = [base.unTokenize(tokens) for tokens in df['Tokens']]
    
    return documents

def extractVectorMatrix(df):
    vocab = extractVocab(df)
    vectorizer = loader.loadVectorizer(vocab)
    matrix = vectorizer.fit_transform(extractDosuments(df)).todense()
    
    return matrix
    

