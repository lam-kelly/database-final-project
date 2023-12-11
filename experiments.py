from MeanModeImputerAirline import mean_imputation, mode_imputation
from LinearRegressionAirline import linear_regression_imputation
from KnnImputerAirline import knn_imputation_time, knn_imputation_airportfrom

actual_files_time = ["/time_MCAR/time_MCAR_0.1_1_answer.csv", "time_MCAR_0.1_1_actual.csv"]
test_files_time = ["time_MCAR_0.1_1_answer.csv", "time_MCAR_0.1_1_actual.csv"]

for actual_file in actual_files_time:
    for test_file in test_files_time:
        mean_imputation(actual_file, test_files_time)
        linear_regression_imputation(actual_file, test_files_time)
        knn_imputation_time(actual_file, test_files_time)

actual_files_airportfrom = ["time_MCAR_0.1_1_answer.csv", "time_MCAR_0.1_1_actual.csv"]
test_files_airportfrom = ["time_MCAR_0.1_1_answer.csv", "time_MCAR_0.1_1_actual.csv"]

for actual_file in actual_files_airportfrom:
    for test_file in test_files_airportfrom:
        mean_imputation(actual_file, test_files_time)
        linear_regression_imputation(actual_file, test_files_time)
        knn_imputation_time(actual_file, test_files_time)
