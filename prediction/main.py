# -*- coding: utf-8 -*-
import prediction
import predict_utils

def runPrediction():
    df = prediction.prepareData()
    X = df.drop('AwardedAmountToDate', axis=1).values
    models = predict_utils.loadModels()
    print('Making Predictions...')
    for name, clf in models.items():
        df[f'{name}_prediction'] = clf.predict(X)
    print('Prediction Completed!')
    df = prediction.voteForFinalPrediction(df)
    prediction.eveluatePrediction(df)

if __name__ == '__main__':
    runPrediction()

