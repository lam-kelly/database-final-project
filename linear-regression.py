import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def linear_regression_imputation(missing_percent):
    df = pd.read_csv("airlines.csv")

    # delete 10% of specified column
    df.loc[df.sample(frac=missing_percent).index, "time"] = np.nan

    # Drop flight column since it has identifier that shouldn't affect other columns
    df = df.drop(['flight'], axis=1)

    # Nne hot encode categorical values
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
    rms = np.sqrt(mean_squared_error(original, y_pred))
    print(rms)

# eventually, put this in for-loop for various missing percentages
linear_regression_imputation(.01)