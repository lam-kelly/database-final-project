import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from math import sqrt

def evaluate():
    actual = pd.read_csv("time_MAR/actual/time_MAR_1_actual.csv")
    pred = pd.read_csv("GPT/time_MAR_1.0_1_imputed.csv")
    df_test = pd.read_csv("time_MAR/test/time_MAR_1.0_1.csv")

    test_data = df_test[df_test['time'].isnull()]
    y_test = test_data['time']
    print(y_test)

    # Get the correct answers from original dataset
    actual = actual.loc[y_test.index, "time"]
    y_pred = pred.loc[y_test.index, "time"]

    # root mean squared error
    mrse = sqrt(mean_squared_error(actual, y_pred))
    print("MRSE", mrse)

    mae = mean_absolute_error(actual, y_pred)
    print("MAE", mae)
    return mrse, mae

evaluate()