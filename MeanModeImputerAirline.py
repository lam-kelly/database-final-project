import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

def mean_imputation(actualCSV, testCSV):
    print()
    print("Mean Imputation on", testCSV, "with", actualCSV)
    df = pd.read_csv(testCSV)
    df = df.drop(['flight', 'airline', 'airportfrom', 'airportto'], axis=1)

    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(df)
    pred = imp_mean.transform(df)

    actual = pd.read_csv(actualCSV)
    actual = actual.drop(['flight', 'airline', 'airportfrom', 'airportto'], axis=1)

    # root mean squared error
    mrse = np.sqrt(mean_squared_error(actual, pred))
    print("MRSE", mrse)

    mae = mean_absolute_error(actual, pred)
    print("MAE", mae)
    return mrse, mae

def mode_imputation(actualCSV, testCSV):
    print()
    print("Mode Imputation on", testCSV, "with", actualCSV)
    df = pd.read_csv(testCSV)
    df = df.drop(['flight'], axis=1)
    nanInd = df.loc[pd.isna(df['airportfrom']), :].index

    imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imp_mean.fit(df)
    pred = imp_mean.transform(df)
    imputed = pred.loc[nanInd]['airportfrom']

    actual = pd.read_csv(actualCSV)
    actual = actual.loc[nanInd]['airportfrom']

    # Accuracy = (true positive + true negative) / (true positive + false positive + true negative + false negative)
    print("Accuracy: ", accuracy_score(actual, imputed))

    # Precision = true positives / (true positives + false positive)
    labels = actual.unique()
    print("Precision: ", precision_score(actual, imputed, labels=labels, average='micro'))

    # Recall = true positive / (true positive + false negatives)
    print("Recall: ", recall_score(actual, imputed, labels=labels, average='micro'))

    # F1 score = harmonic means of precision and recall = 2 * (precision * recall) / (precision + recall)
    print("F1 score: ", f1_score(actual, imputed, labels=labels, average='micro'))

