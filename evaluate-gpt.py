import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score

def evaluate():
    actual = pd.read_csv("airportfrom_MCAR/actual/airportfrom_MCAR_1_actual.csv")
    pred = pd.read_csv("GPT/airportfrom_MCAR_1.0_1_imputed.csv")
    df_test = pd.read_csv("airportfrom_MCAR/test/airportfrom_MCAR_1.0_1.csv")

    # test_data = df_test[df_test['time'].isnull()]
    # y_test = test_data['time']
    # print(y_test)

    # # Get the correct answers from original dataset
    # actual = actual.loc[y_test.index, "time"]
    # y_pred = pred.loc[y_test.index, "time"]

    # # root mean squared error
    # mrse = sqrt(mean_squared_error(actual, y_pred))
    # print("MRSE", mrse)

    # mae = mean_absolute_error(actual, y_pred)
    # print("MAE", mae)
    # return mrse, mae

    test_data = df_test[df_test['airportfrom'].isnull()]
    nanInd = test_data['airportfrom'].index

    # compare with original
    actual = actual.loc[nanInd]['airportfrom']
    imputed = pred.loc[nanInd]['airportfrom']

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

evaluate()