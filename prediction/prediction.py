# -*- coding: utf-8 -*-
import json
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 

import predict_utils

medoids = predict_utils.loadMedoids()
vocab = predict_utils.loadVocab()
addresses = predict_utils.getDirs()

def extractDocuments(df):
    documents = [' '.join(tokens) for tokens in df['Tokens']]
    
    return documents

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
    for index, row in  enumerate(df.iterrows()):
        df_sim.loc[index, 'similarity_0'] = cosineSimilarity(medoids[0], row[1].values)
        df_sim.loc[index, 'similarity_1'] = cosineSimilarity(medoids[1], row[1].values)
    print('Similarity Calculation Completed!')
    
    return df_sim

def prepareData():
    df_text = predict_utils.loadTextData()
    df_num = predict_utils.loadNumData()
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    matrix = vectorizer.fit_transform(extractDocuments(df_text)).todense()
    matrix = pd.DataFrame(matrix, columns=vocab)
    df_sim = computeSimilarities(df_text, medoids)
    df = pd.concat([df_num, df_sim], axis=1, sort=False)
    
    return df

def voteForFinalPrediction(df):
    df['Final'] = df.apply(lambda row: predict_utils.voteCalc(row), axis=1)
    
    return df
    
def eveluatePrediction(df):
    y_real = df['AwardedAmountToDate'].values
    y_pred = df['Final'].values
    metrics = {'Accuracy_Score': accuracy_score(y_real, y_pred),
               'Precision_Score': precision_score(y_real, y_pred),
               'Recall_Score': recall_score(y_real, y_pred),
               'F1_Score': f1_score(y_real, y_pred)}
    with open(addresses['performance'] + 'Voting.json', 'w') as f:
        json.dump(metrics, f)





