![image](https://github.com/user-attachments/assets/374de57f-dee9-4c91-9811-d092e185ad16)

# Supreme Ventures - Pick 2 [Predicition Model]
![ChatGPT](https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Microsoft Excel](https://img.shields.io/badge/Microsoft_Excel-217346?style=for-the-badge&logo=microsoft-excel&logoColor=white) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

This model utilizies several statistical libraries such as Pandas and Scikit. The data provided was collected from the Supreme Ventures draw results located on their website



## Program Breakdown

### Importing Libraries

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import time
```

- **pandas**: For data manipulation and analysis.
- **sklearn.model_selection**: For splitting data into training and testing sets, and for hyperparameter tuning.
- **sklearn.ensemble**: For using the RandomForestClassifier (although not used in the final implementation).
- **sklearn.metrics**: For evaluating the performance of the models.
- **sklearn.preprocessing**: For normalizing features.
- **xgboost**: For using the XGBClassifier, a gradient boosting framework.
- **time**: For measuring the time taken for each step.

### Loading Data

```python
# Load data
file_path = 'p2_draws.csv'
data = pd.read_csv(file_path)
```

- **file_path**: Specifies the path to the CSV file containing the data.
- **data**: Loads the data from the CSV file into a pandas DataFrame.

### Preprocessing Data

```python
start_time = time.time()
data['Date'] = pd.to_datetime(data['Date'])
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month
```

- **data['Date']**: Converts the 'Date' column to datetime format.
- **data['DayOfWeek']**: Extracts the day of the week from the 'Date' column.
- **data['Month']**: Extracts the month from the 'Date' column.

### Cleaning Data

```python
# Clean the 'Winning Numbers' column
data['Winning Numbers'] = data['Winning Numbers'].str.strip()
data = data.dropna(subset=['Winning Numbers'])
```

- **data['Winning Numbers']**: Strips any leading or trailing whitespace from the 'Winning Numbers' column.
- **data.dropna**: Drops rows where 'Winning Numbers' is NaN (missing).

### Splitting and Cleaning Winning Numbers

```python
# Ensure the winning numbers are split correctly and remove rows with issues
split_numbers = data['Winning Numbers'].str.split(expand=True)
split_numbers = split_numbers.dropna()
split_numbers = split_numbers[split_numbers.apply(lambda row: row.map(str.isdigit).all(), axis=1)]
```

- **split_numbers**: Splits the 'Winning Numbers' column into two separate columns.
- **split_numbers.dropna**: Drops rows with NaN values in the split columns.
- **split_numbers.apply**: Keeps only rows where all split values are digits.

### Updating Data with Cleaned Split Numbers

```python
# Update the original data with cleaned split numbers
data = data.loc[split_numbers.index]
data[['WinningNumber1', 'WinningNumber2']] = split_numbers.astype(int)
```

- **data.loc**: Keeps only the rows in the original DataFrame that have valid split numbers.
- **data[['WinningNumber1', 'WinningNumber2']]**: Assigns the cleaned split numbers to new columns and converts them to integers.

### Adjusting Labels

```python
# Adjust the labels to fit the expected range
data['WinningNumber1'] -= 1
data['WinningNumber2'] -= 1
```

- **data['WinningNumber1'] -= 1**: Adjusts the first winning number to be zero-based.
- **data['WinningNumber2'] -= 1**: Adjusts the second winning number to be zero-based.

### Feature Engineering

```python
# Feature engineering
data['WinningNumber1_Freq'] = data['WinningNumber1'].map(data['WinningNumber1'].value_counts())
data['WinningNumber2_Freq'] = data['WinningNumber2'].map(data['WinningNumber2'].value_counts())
```

- **data['WinningNumber1_Freq']**: Creates a feature for the frequency of the first winning number.
- **data['WinningNumber2_Freq']**: Creates a feature for the frequency of the second winning number.

### Preparing Features and Labels

```python
X = data[['DayOfWeek', 'Month', 'WinningNumber1_Freq', 'WinningNumber2_Freq']]
y1 = data['WinningNumber1']
y2 = data['WinningNumber2']
```

- **X**: Features matrix containing 'DayOfWeek', 'Month', 'WinningNumber1_Freq', and 'WinningNumber2_Freq'.
- **y1**: Labels for the first winning number.
- **y2**: Labels for the second winning number.

### Normalizing Features

```python
# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

- **scaler**: Initializes a StandardScaler to normalize features.
- **X_scaled**: Normalizes the feature matrix.

### Splitting Data into Training and Testing Sets

```python
# Split data into training and testing sets
X_train_1, X_test_1, y1_train, y1_test = train_test_split(X_scaled, y1, test_size=0.2, random_state=42)
X_train_2, X_test_2, y2_train, y2_test = train_test_split(X_scaled, y2, test_size=0.2, random_state=42)
preprocessing_time = time.time() - start_time
print(f'Preprocessing Time: {preprocessing_time:.2f} seconds')
```

- **train_test_split**: Splits the data into training and testing sets for both winning numbers.
- **preprocessing_time**: Measures and prints the time taken for preprocessing.

### Hyperparameter Tuning and Model Training

```python
# Initialize and train models with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'learning_rate': [0.01, 0.1]
}

model1 = XGBClassifier(random_state=42)
grid_search_1 = GridSearchCV(model1, param_grid, cv=3, scoring='accuracy')
start_time = time.time()
grid_search_1.fit(X_train_1, y1_train)
training_time_1 = time.time() - start_time
```

- **param_grid**: Defines the hyperparameters to be tuned.
- **model1**: Initializes the XGBClassifier for the first winning number.
- **grid_search_1**: Initializes GridSearchCV to perform hyperparameter tuning for the first model.
- **grid_search_1.fit**: Fits the model to the training data and performs hyperparameter tuning.
- **training_time_1**: Measures and prints the time taken for training the first model.

```python
model2 = XGBClassifier(random_state=42)
grid_search_2 = GridSearchCV(model2, param_grid, cv=3, scoring='accuracy')
start_time = time.time()
grid_search_2.fit(X_train_2, y2_train)
training_time_2 = time.time() - start_time

print(f'Model 1 Training Time: {training_time_1:.2f} seconds')
print(f'Model 2 Training Time: {training_time_2:.2f} seconds')
```

- **model2**: Initializes the XGBClassifier for the second winning number.
- **grid_search_2**: Initializes GridSearchCV to perform hyperparameter tuning for the second model.
- **grid_search_2.fit**: Fits the model to the training data and performs hyperparameter tuning.
- **training_time_2**: Measures and prints the time taken for training the second model.

### Evaluating Models

```python
# Best models from grid search
best_model1 = grid_search_1.best_estimator_
best_model2 = grid_search_2.best_estimator_

# Evaluate the models
y1_pred = best_model1.predict(X_test_1)
y2_pred = best_model2.predict(X_test_2)

# Output model performance
accuracy_1 = accuracy_score(y1_test, y1_pred)
report_1 = classification_report(y1_test, y1_pred)

accuracy_2 = accuracy_score(y2_test, y2_pred)
report_2 = classification_report(y2_test, y2_pred)

print(f'Model 1 Accuracy: {accuracy_1}')
print('Model 1 Classification Report:')
print(report_1)

print(f'Model 2 Accuracy: {accuracy_2}')
print('Model 2 Classification Report:')
print(report_2)
```

- **best_model1**: Retrieves the best estimator from the grid search for the first model.
- **best_model2**: Retrieves the best estimator from the grid search for the second model.
- **y1_pred**: Makes predictions on the test set for the first winning number.
- **y2_pred**: Makes predictions on the test set for the second winning number.
- **accuracy_score**: Computes the accuracy of the predictions.
- **classification_report**: Generates a detailed classification report.

### Making Predictions

```python
# Making predictions
# Example of predicting a specific date
new_data = pd.DataFrame({'DayOfWeek': [2], 'Month': [8], 'WinningNumber1_Freq': [0], 'WinningNumber2_Freq': [0]})  # Example date feature
new_data_scaled = scaler.transform(new_data)
start_time = time.time()
predicted_number1 = best_model1.predict(new_data_scaled)
predicted_number2 = best_model2.predict(new_data_scaled)
prediction_time = time.time() - start_time
```

# Running SP-ALGO
This model can be ran Google Colab, which is a free cloud based Jupyter Notebook. Datasets can be uploaded via google drive and be referenced in code. 
![image](https://github.com/user-attachments/assets/38727107-7a26-4e1c-9e6b-f61705dc3516)


# Information 
Created by Damoy Skeene, utilizing self-taught principles and 3rd party AI tools such as ChatGpt for refining model behavior and incorporating advance principles which are outside of my expertise. This is a experimental project. Please use this program at your own risk.


