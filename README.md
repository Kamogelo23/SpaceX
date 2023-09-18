# Space X Falcon 9 First Stage Landing Prediction
Overview
This project aims to predict the landing outcome of Space X Falcon 9's first stage using machine learning models. The prediction is based on various features and data collected from previous launches.

## Objectives
Exploratory Data Analysis (EDA): Understand the underlying data and its structure.
Data Preprocessing:
Create a column for the class.
Standardize the data.
Split the data into training and test sets.
Hyperparameter Tuning: Determine the best hyperparameters for SVM, Classification Trees, and Logistic Regression.
Model Evaluation: Find out which method performs best using test data.
Libraries and Dependencies
pip install numpy pandas seaborn requests
### Libraries Used
pandas: For data manipulation and analysis.
numpy: Supports large, multi-dimensional arrays and matrices.
matplotlib: Provides a MATLAB-like plotting framework.
seaborn: For drawing attractive and informative statistical graphics.
sklearn: For data preprocessing, model training, and evaluation.
Data Loading
Data is loaded from provided URLs using the requests library and then read into pandas DataFrames.

import requests
import io
import pandas as pd
### Load the data frame
URL1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
resp1 = requests.get(URL1)
df1 = pd.read_csv(io.StringIO(resp1.text))
Model Training and Evaluation
Multiple models including Logistic Regression, SVM, Decision Trees, and KNN are trained using GridSearchCV for hyperparameter tuning. The models are then evaluated on test data to determine their accuracy and other metrics.

#### Confusion Matrix
A custom function plot_confusion_matrix is used to visualize the confusion matrix for each model, helping in understanding the model's performance in terms of false positives, false negatives, etc.

### Results
After training and evaluating all models, the K-Nearest Neighbors (KNN) model was found to be the optimal model for this dataset.
