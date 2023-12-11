import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score

def knn_imputation_time(actualCSV, testCSV):
    print()
    print("KNN Imputation on", testCSV, "with", actualCSV)
    df = pd.read_csv(testCSV)
    df = df.drop(['flight'], axis=1)
    nanInd = df.loc[pd.isna(df['time']), :].index

    # one-hot encode categorical columns
    cat_variables = df[['airline', 'airportfrom', 'airportto']]
    cat_dummies = pd.get_dummies(cat_variables, drop_first=True)

    df = df.drop(['airline', 'airportfrom', 'airportto'], axis=1)
    df = pd.concat([df, cat_dummies], axis=1)

    # scale data values
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # impute
    imputer = KNNImputer(n_neighbors=5)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    imputed = df.loc[nanInd]['time']

    # compare with original
    actual = pd.read_csv(actualCSV)
    actual = actual.loc[nanInd]['time']

    mrse = np.sqrt(mean_squared_error(actual, imputed))
    print("MRSE", mrse)

    mae = mean_absolute_error(actual, imputed)
    print("MAE", mae)
    return mrse, mae

def knn_imputation_airportfrom(actualCSV, testCSV):
    print("KNN Imputation on ", testCSV, " with ", actualCSV)
    df = pd.read_csv(testCSV)
    df = df.drop(['flight'], axis=1)
    nanInd = df.loc[pd.isna(df['airportfrom']), :].index

    # one-hot encode categorical columns
    cat_variables = df[['airline', 'airportfrom', 'airportto']]
    cat_dummies = pd.get_dummies(cat_variables, drop_first=True)
    cat_dummies.loc[nanInd, cat_dummies.columns.str.startswith('airportfrom_')]= np.nan

    df = df.drop(['airline', 'airportfrom', 'airportto'], axis=1)
    df = pd.concat([df, cat_dummies], axis=1)

    # scale data values
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # impute
    imputer = KNNImputer(n_neighbors=5)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # reverse one-hot encode by picking column with the largest imputed value
    imputed = df.drop(['time', 'length', 'dayofweek', 'delay'], axis=1)
    imputed = imputed[imputed.columns[imputed.columns.str.startswith('airportfrom_')]]
    imputed = imputed.loc[nanInd]
    imputed['airport_from'] = imputed.idxmax(axis=1)
    imputed = imputed['airport_from'].str.slice(start=12)

    # compare with original
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

    # print(classification_report(original, imputed))
