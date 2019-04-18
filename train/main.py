# -*- coding: utf-8 -*-
import train_utils
import train_models

import pandas as pd

def runPreparation():
    df_text = train_utils.loadTextData()
    df_num = train_utils.loadNumData()
    df_text = train_utils.extractVectorMatrix(df_text)
    medoids = train_utils.generateMedoids(df_text)
    df_sim = train_utils.computeSimilarities(df_text, medoids)
    df = pd.concat([df_num, df_sim], axis=1, sort=False)
    df = train_utils.downSampleMajor(df)
#    df = train_utils.upSampleMinor(df)
#   df = train_utils.upSampleMinor(df)
    df = train_utils.extractVectorMatrix(df)
    df_splitted = train_utils.trainTestSplit(df)
    
    return df_splitted

def runTraining():
    df_splitted = runPreparation()
    train_models.trainLogisticRegression(df_splitted)
    train_models.trainDecisionTree(df_splitted)
    train_models.trainRandomForrest(df_splitted)
    train_models.trainKNN(df_splitted)
    train_models.trainGBDT(df_splitted)
    
if __name__ == '__main__':
    runTraining()



