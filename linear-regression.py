import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

def linear_regression_imputation(missing_percent, length=None, time=None):
    df = pd.read_csv("airlines.csv")

    if length:
        # delete 10% of a specified column based on another column
        n = int(missing_percent*df.shape[0])
        df.loc[df[df["length"].gt(length)].sample(n).index, "time"] = np.nan
    elif time:
        # delete 10% of a specified column based on own column
        n = int(missing_percent*df.shape[0])
        df.loc[df[df["time"].gt(time)].sample(n).index, "time"] = np.nan
    else:
        # delete 10% of specified column
        df.loc[df.sample(frac=missing_percent).index, "time"] = np.nan

    print(df.isnull().sum())

    # Drop flight column since it has identifier that shouldn't affect other columns
    df = df.drop(['flight'], axis=1)

    # One hot encode categorical values
    cat_variables = df[['airline', 'airportfrom', 'airportto']]
    cat_dummies = pd.get_dummies(cat_variables, drop_first=True)
    # Replace categorical columns with one-hot encoded columns
    df = df.drop(['airline', 'airportfrom', 'airportto'], axis=1)
    df = pd.concat([df, cat_dummies], axis=1)

    # Separate the null value rows from dataframe (df) and create a variable “test data”
    test_data = df[df['time'].isnull()]

    # Drop the null values from the dataframe (df) and represent as ‘train data”
    df.dropna(inplace=True)

    # Use all other columns to predict time
    x_train = df.drop(['time'], axis=1)
    x_test = test_data.drop(['time'], axis=1)
    # target variable is time
    y_train = df['time']
    y_test = test_data['time']

    # Build the linear regression model
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    # Apply the model on x_test of test data to make predictions.
    y_pred = lr.predict(x_test)

    # Get the correct answers from original dataset
    original = pd.read_csv("airlines.csv")
    original = original.loc[y_test.index, "time"]

    # root mean squared error
    mse = np.sqrt(mean_squared_error(original, y_pred))
    print(mse)

    mae = np.sqrt(mean_absolute_error(original, y_pred))
    print("MAE", mae)

for p in [0.3, 0.5]:
    # eventually, put this in for-loop for various missing percentages
    print(p)
    linear_regression_imputation(p)
    # if the airline = ..., then delete time
    linear_regression_imputation(p, length=0)
    linear_regression_imputation(p, time=0)