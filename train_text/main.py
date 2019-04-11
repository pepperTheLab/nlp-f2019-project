# -*- coding: utf-8 -*-
import train_utils
import train_models

def runPreparation():
    df = train_utils.loadTextData()
    df = train_utils.extractVectorMatrix(df)
    df_splitted = train_utils.trainTestSplit(df)
    
    return df_splitted

def runTraining():
    df_splitted = runPreparation()
    train_models.trainLinearRegression(df_splitted)
    train_models.trainRidge(df_splitted)
    train_models.trainLasso(df_splitted)
    train_models.trainDecisionTree(df_splitted)
    train_models.trainRandomForest(df_splitted)
    train_models.trainGBDT(df_splitted)
    
if __name__ == '__main__':
    runTraining()



