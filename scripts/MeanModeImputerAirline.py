import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

def mean_imputation(actualCSV, testCSV):
    print()
    print("Mean Imputation on", testCSV, "with", actualCSV)
    df = pd.read_csv(testCSV)
    df = df.drop(['flight', 'airline', 'airportfrom', 'airportto'], axis=1)
    nanInd = df.loc[pd.isna(df['time']), :].index

    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_mean.fit(df)
    pred = imp_mean.transform(df)
    pred = pd.DataFrame(pred, columns=['time','length', 'dayofweek','delay'])
    pred = pred.loc[nanInd]['time']

    actual = pd.read_csv(actualCSV)
    actual = actual.drop(['flight', 'airline', 'airportfrom', 'airportto'], axis=1)
    actual = actual.loc[nanInd]['time']

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

    imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imp_mode.fit(df)
    pred = imp_mode.transform(df)
    pred = pd.DataFrame(pred, columns=['time','length','airline','airportfrom','airportto','dayofweek','delay'])
    imputed = pred.loc[nanInd]['airportfrom']

    actual = pd.read_csv(actualCSV)
    actual = actual.loc[nanInd]['airportfrom']

    # Accuracy = (true positive + true negative) / (true positive + false positive + true negative + false negative)
    accuracy = accuracy_score(actual, imputed)
    print("Accuracy: ", accuracy)

    # Precision = true positives / (true positives + false positive)
    labels = actual.unique()
    precision = precision_score(actual, imputed, labels=labels, average='micro')
    print("Precision: ", precision)

    # Recall = true positive / (true positive + false negatives)
    recall = recall_score(actual, imputed, labels=labels, average='micro')
    print("Recall: ", recall)

    # F1 score = harmonic means of precision and recall = 2 * (precision * recall) / (precision + recall)
    f1score = f1_score(actual, imputed, labels=labels, average='micro')
    print("F1 score: ", f1score)
    return accuracy, precision, recall, f1score

