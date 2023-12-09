import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

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
    original = pd.read_csv("airlines.csv", nrows=5000)
    original = original.loc[nanInd]['airportfrom']

    # Accuracy
    accuracy = 1-original.compare(imputed).shape[0]/original.shape[0]
    print("Accuracy: ", accuracy)

    # Precision = true positives / (true positives + false positive)

    # Recall = true positive / (true positive + false negatives)

    # Accuracy = (true positive + true negative) / (true positive + false positive + true negative + false negative)

    # F1 score = harmonic means of precision and recall = 2 * (precision * recall) / (precision + recall)

# up to 5000 runs in reasonable time
knn_imputation(1000)