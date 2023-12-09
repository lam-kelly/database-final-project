import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def knn_imputation(nrows):
    df = pd.read_csv("airlines.csv", nrows=nrows)
    # randomly delete 10% of airportfrom
    df.loc[df.sample(frac=0.1).index, "airportfrom"] = np.nan
    df = df.drop(['flight'], axis=1)
    df.isnull().sum()
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
    original = pd.read_csv("airlines.csv", nrows=nrows)
    original = original.loc[nanInd]['airportfrom']

    # Accuracy = (true positive + true negative) / (true positive + false positive + true negative + false negative)
    print("Accuracy: ", accuracy_score(original, imputed))

    # Precision = true positives / (true positives + false positive)
    labels = original.unique()
    print("Precision: ", precision_score(original, imputed, labels=labels, average='micro'))

    # Recall = true positive / (true positive + false negatives)
    print("Recall: ", recall_score(original, imputed, labels=labels, average='micro'))

    # F1 score = harmonic means of precision and recall = 2 * (precision * recall) / (precision + recall)
    print("F1 score: ", f1_score(original, imputed, labels=labels, average='micro'))

    # print(classification_report(original, imputed))

# up to 5000 runs in reasonable time
knn_imputation(1000)