Data Imputation using ML and GPT-4
==================
## Contents

- [scripts](#scripts)
- [experimental_datasets](#experimental_datasets)
- [ML_results](#ML_results)
- [GPT_imputations](#GPT_imputations)

## <a name="scripts">scripts</a>
All of the scripts used in this project are contained in this folder.

- `create-datasets.ipynb`: notebook that generates datasets for each combination of missing ratio and missingness pattern
- `experiments.py`: script to run all of the traditional machine learning methods on the datasets and save results in a json, contained in `ML_results`
- `evaluate-gpt.py`: script to evaluate GPT-4 results from `GPT_imputations`
- `KnnImputerAirline.py`: implements KNN model for numerical and categorical imputations
- `LinearRegressionAirline.py`: implements linear regression model
- `MeanModeImputerAirline.py`: implements mean and mode simple imputers

## <a name="experimental_datasets">experimental_datasets</a>
This folder contains all of the generated datasets from `create-datasets.ipynb`. An interpretation of the following structure in this folder is as follows:
```
├── time_MAR
│   ├── actual
│   │   ├── time_MAR_1_actual.csv
|   |   ├── ...
│   ├── test
│   |   ├── time_MAR_1.0_1.csv
|   |   ├── ...
``````
The top level folder identifies the column that has missing values and the missingness pattern. The datasets are divided into the complete datasets in `actual` and the ones with deleted values `test`. For each complete dataset, each of the missing percentage is deleted. For example, `time_MAR_1.0_1.csv` means this dataset has 1% of the time column missing at random. The other files follow the same convention.

## <a name="ML_results">ML_results</a>
This folder contains all of the evaluation results for traditional ML models. There is one file per column/missingness pattern.

## <a name="GPT_imputations">GPT_imputations</a>
This folder contains the imputed files generated by ChatGPT-4. Each file corresponds with a dataset from `experimental_datasets`. They can be identified by the same naming convention, with an `imputed` at the end.