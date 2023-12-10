import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer

def mean_imputation(nrows):
    df = pd.read_csv("airlines.csv", nrows=nrows)
    df.loc[df.sample(frac=0.1).index, "time"] = np.nan

    df = df.drop(['flight', 'airline', 'airportfrom', 'airportto'], axis=1)

    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(df)
    pred = imp_mean.transform(df)

    actual = pd.read_csv("airlines.csv", nrows=nrows)
    actual = actual.drop(['flight', 'airline', 'airportfrom', 'airportto'], axis=1)

    # root mean squared error
    mse = np.sqrt(mean_squared_error(actual, pred))
    print("MSE", mse)

    mae = np.sqrt(mean_absolute_error(actual, pred))
    print("MAE", mae)

def mode_imputation(nrows):
    df = pd.read_csv("airlines.csv", nrows=nrows)
    df.loc[df.sample(frac=0.1).index, "time"] = np.nan

    df = df.drop(['flight', 'airline', 'airportfrom', 'airportto'], axis=1)

    imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imp_mean.fit(df)
    pred = imp_mean.transform(df)

    actual = pd.read_csv("airlines.csv", nrows=nrows)
    actual = actual.drop(['flight', 'airline', 'airportfrom', 'airportto'], axis=1)

    # root mean squared error
    mse = np.sqrt(mean_squared_error(actual, pred))
    print("MSE", mse)

    mae = np.sqrt(mean_absolute_error(actual, pred))
    print("MAE", mae)

# up to 5000 runs in reasonable time
mean_imputation(1000)
mode_imputation(1000)