from MeanModeImputerAirline import mean_imputation, mode_imputation
from LinearRegressionAirline import linear_regression_imputation
from KnnImputerAirline import knn_imputation_time, knn_imputation_airportfrom
import os
import json

actual_files_time = ["/time_MCAR/time_MCAR_0.1_1_answer.csv", "time_MCAR_0.1_1_actual.csv"]
test_files_time = ["time_MCAR_0.1_1_answer.csv", "time_MCAR_0.1_1_actual.csv"]
result = {'experiments': []}

for test_file in os.listdir("time_MCAR/test"):
    test_file_name = "time_MCAR/test/" + test_file
    actual_file = "time_MCAR/actual/time_MCAR_" + str(test_file[-5]) + "_actual.csv"
    imputations = {"testfile": test_file, 'actual_file': actual_file, 'imputations': []}
    mrse, mae = mean_imputation(actual_file, test_file_name)
    imputations["imputations"].append({"imputation": 'mean', 'MRSE': mrse, 'MAE': mae})
    mrse, mae = linear_regression_imputation(actual_file, test_file_name)
    imputations["imputations"].append({"imputation": 'linear regression', 'MRSE': mrse, 'MAE': mae})
    mrse, mae = knn_imputation_time(actual_file, test_file_name)
    imputations["imputations"].append({"imputation": 'KNN', 'MRSE': mrse, 'MAE': mae})
    result['experiments'].append(imputations)

with open('results.json', 'w') as fp:
    json.dump(result, fp)

# actual_files_airportfrom = ["time_MCAR_0.1_1_answer.csv", "time_MCAR_0.1_1_actual.csv"]
# test_files_airportfrom = ["time_MCAR_0.1_1_answer.csv", "time_MCAR_0.1_1_actual.csv"]

# for actual_file in actual_files_airportfrom:
#     for test_file in test_files_airportfrom:
#         mode_imputation(actual_file, test_files_time)
#         knn_imputation_airportfrom(actual_file, test_files_time)
