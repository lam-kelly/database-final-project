import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def knn_imputation(nrows):
    df = pd.read_csv("airlines.csv", nrows=nrows)
    df.loc[df.sample(frac=0.1).index, "time"] = np.nan

    df = df.drop(['flight', 'airline', 'airportfrom', 'airportto'], axis=1)

    mean_value = df.mean()
    mean_imputation = df.fillna(mean_value)

    actual = pd.read_csv("airlines.csv", nrows=nrows)
    actual = actual.drop(['flight', 'airline', 'airportfrom', 'airportto'], axis=1)

    # root mean squared error
    rms = np.sqrt(mean_squared_error(actual, mean_imputation))
    print(rms)

# up to 5000 runs in reasonable time
knn_imputation(1000)