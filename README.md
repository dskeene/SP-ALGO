![image](https://github.com/user-attachments/assets/374de57f-dee9-4c91-9811-d092e185ad16)

# Supreme Ventures - Pick 2 [Predicition Model]
![ChatGPT](https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Microsoft Excel](https://img.shields.io/badge/Microsoft_Excel-217346?style=for-the-badge&logo=microsoft-excel&logoColor=white) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

SP-ALGO is an experimental predicition model, which intends to predict the Supreme Ventures Pick 2 draw results, using frequency analysis and probability based on historical frequency, as well as other advance statistical concepts implemented using Artificial Intelligence. This model utilizies several statistical libraries such as Pandas and Scikit. The data provided was collected from the Supreme Ventures draw results located on their website


## DISCLAIMER 
This project was created by Damoy Skeene, as a experiemental past time project. It is in no shape or form, stable enough to encourage any bets based on numbers predicted. Please use at your own risk. Thank you.

## Program Breakdown

Sure, I'll break down each significant line of your script and explain its purpose:

```python
import pandas as pd
```
- **Imports the pandas library**, useful for data manipulation and analysis.

```python
from sklearn.model_selection import train_test_split, GridSearchCV
```
- **Imports tools from scikit-learn**: `train_test_split` for splitting data into training and testing sets, `GridSearchCV` for optimizing model parameters.

```python
from sklearn.metrics import accuracy_score, classification_report
```
- **Imports performance metrics**: `accuracy_score` to measure the accuracy of predictions, `classification_report` to show a detailed classification report including precision, recall, f1-score, etc.

```python
from sklearn.preprocessing import StandardScaler
```
- **Imports `StandardScaler`** for normalizing/standardizing features.

```python
from xgboost import XGBClassifier
```
- **Imports `XGBClassifier` from XGBoost**, a powerful machine learning algorithm that implements gradient boosting framework.

```python
import time
```
- **Imports the time module** to measure execution time.

```python
import psutil
```
- **Imports psutil**, a cross-platform library for accessing system details and process utilities.

```python
file_path = 'csvfilehere'
data = pd.read_csv(file_path)
```
- **Loads data from a CSV file** into a pandas DataFrame.

```python
start_time = time.time()
```
- **Records the start time** for measuring the duration of the preprocessing step.

```python
data['Date'] = pd.to_datetime(data['Date'])
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month
```
- **Converts the 'Date' column to datetime format** and extracts useful features like `DayOfWeek` and `Month`.

```python
data['Winning Numbers'] = data['Winning Numbers'].str.strip()
data = data.dropna(subset=['Winning Numbers'])
```
- **Cleans the 'Winning Numbers' column** by stripping leading/trailing spaces and dropping any rows where 'Winning Numbers' are NaN.

```python
split_numbers = data['Winning Numbers'].str.split(expand=True)
split_numbers = split_numbers.dropna()
split_numbers = split_numbers[split_numbers.apply(lambda row: row.map(str.isdigit).all(), axis=1)]
```
- **Splits the 'Winning Numbers' into separate columns** and ensures that each entry is a digit, removing any rows with non-digit characters.

```python
data = data.loc[split_numbers.index]
data[['WinningNumber1', 'WinningNumber2']] = split_numbers.astype(int)
```
- **Updates the original data** with cleaned and split numbers, converting them to integers.

```python
data['WinningNumber1'] -= 1
data['WinningNumber2'] -= 1
```
- **Adjusts the winning numbers** by subtracting 1 (shift range for zero-indexing purposes).

```python
data['WinningNumber1_Freq'] = data['WinningNumber1'].map(data['WinningNumber1'].value_counts())
data['WinningNumber2_Freq'] = data['WinningNumber2'].map(data['WinningNumber2'].value_counts())
```
- **Creates frequency features** for how often each number has won.

```python
X = data[['DayOfWeek', 'Month', 'WinningNumber1_Freq', 'WinningNumber2_Freq']]
y1 = data['WinningNumber1']
y2 = data['WinningNumber2']
```
- **Defines features (X) and labels (y)** for the model.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
- **Normalizes the features** to have zero mean and unit variance.

```python
X_train_1, X_test_1, y1_train, y1_test = train_test_split(X_scaled, y1, test_size=0.2, random_state=42)
X_train_2, X_test_2, y2_train, y2_test = train_test_split(X_scaled, y2, test_size=0.2, random_state=42)
```
- **Splits the data into training and testing sets** for two models.

```python
preprocessing_time = time.time() - start_time
print(f'Preprocessing Time: {preprocessing_time:.2f} seconds')
```
- **Calculates and prints the preprocessing time**.

```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'learning_rate': [0.01, 0.1]
}
```
- **Defines a grid of hyperparameters** for tuning the models.

```python
memory_before, cpu_before = track_resource_usage()
```
- **Tracks memory and CPU usage** before model training.

```python
grid_search_1 = GridSearchCV(model1, param_grid, cv=3
```
```python
grid_search_1 = GridSearchCV(model1, param_grid, cv=3, scoring='accuracy')
start_time = time.time()
grid_search_1.fit(X_train_1, y1_train)
training_time_1 = time.time() - start_time
```
- **Initializes `GridSearchCV`** for `model1` with the parameter grid, using 3-fold cross-validation and accuracy as the scoring metric.
- **Records the start time** for model training.
- **Fits `grid_search_1`** to the training data (`X_train_1`, `y1_train`).
- **Calculates the training time** for the first model.

```python
memory_after_1, cpu_after_1 = track_resource_usage()
```
- **Tracks memory and CPU usage** after training the first model.

```python
model2 = XGBClassifier(random_state=42)
grid_search_2 = GridSearchCV(model2, param_grid, cv=3, scoring='accuracy')
start_time = time.time()
grid_search_2.fit(X_train_2, y2_train)
training_time_2 = time.time() - start_time
```
- **Initializes `GridSearchCV`** for `model2` with the parameter grid, using 3-fold cross-validation and accuracy as the scoring metric.
- **Records the start time** for model training.
- **Fits `grid_search_2`** to the training data (`X_train_2`, `y2_train`).
- **Calculates the training time** for the second model.

```python
memory_after_2, cpu_after_2 = track_resource_usage()
```
- **Tracks memory and CPU usage** after training the second model.

```python
print(f'Model 1 Training Time: {training_time_1:.2f} seconds')
print(f'Model 1 Memory Usage: {memory_after_1 - memory_before:.2f} MB')
print(f'Model 1 CPU Usage: {cpu_after_1 - cpu_before:.2f}%')
```
- **Prints the training time, memory usage, and CPU usage** for the first model.

```python
print(f'Model 2 Training Time: {training_time_2:.2f} seconds')
print(f'Model 2 Memory Usage: {memory_after_2 - memory_before:.2f} MB')
print(f'Model 2 CPU Usage: {cpu_after_2 - cpu_before:.2f}%')
```
- **Prints the training time, memory usage, and CPU usage** for the second model.

```python
best_model1 = grid_search_1.best_estimator_
best_model2 = grid_search_2.best_estimator_
```
- **Retrieves the best models** found by `GridSearchCV`.

```python
y1_pred = best_model1.predict(X_test_1)
y2_pred = best_model2.predict(X_test_2)
```
- **Makes predictions** on the test data using the best models.

```python
accuracy_1 = accuracy_score(y1_test, y1_pred)
report_1 = classification_report(y1_test, y1_pred)
```
- **Calculates the accuracy** and **generates a classification report** for the first model.

```python
accuracy_2 = accuracy_score(y2_test, y2_pred)
report_2 = classification_report(y2_test, y2_pred)
```
- **Calculates the accuracy** and **generates a classification report** for the second model.

```python
print(f'Model 1 Accuracy: {accuracy_1}')
print('Model 1 Classification Report:')
print(report_1)
```
- **Prints the accuracy** and **classification report** for the first model.

```python
print(f'Model 2 Accuracy: {accuracy_2}')
print('Model 2 Classification Report:')
print(report_2)
```
- **Prints the accuracy** and **classification report** for the second model.

```python
unique_dates = data['Date'].unique()
predictions = []
```
- **Retrieves unique dates** from the data.
- **Initializes a list to store predictions**.

```python
for date in unique_dates:
    day_of_week = date.dayofweek
    month = date.month
    winning_number1_freq = data['WinningNumber1_Freq'].mean()
    winning_number2_freq = data['WinningNumber2_Freq'].mean()
    
    new_data = pd.DataFrame({
        'DayOfWeek': [day_of_week], 
        'Month': [month], 
        'WinningNumber1_Freq': [winning_number1_freq], 
        'WinningNumber2_Freq': [winning_number2_freq]
    })
    
    new_data_scaled = scaler.transform(new_data)
    predicted_number1 = best_model1.predict(new_data_scaled)[0] + 1
    predicted_number2 = best_model2.predict(new_data_scaled)[0] + 1
    predictions.append((date, predicted_number1, predicted_number2))
```
- **Loops through each unique date** to generate predictions.
- **Extracts features** (day of the week, month) for the date.
- **Uses average frequencies** of winning numbers as placeholders.
- **Creates a DataFrame** with the new data for prediction.
- **Scales the new data** using the previously fitted scaler.
- **Makes predictions** using the best models and adjusts the predictions back to the original range by adding 1.
- **Appends the predictions** to the list.

```python
for date, num1, num2 in predictions:
    print(f'Date: {date}, Predicted Winning Numbers: {num1}, {num2}')
```
- **Prints the predictions** for each date.

```python
new_data = pd.DataFrame({'DayOfWeek': [2], 'Month': [8], 'WinningNumber1_Freq': [0], 'WinningNumber2_Freq': [0]})
new_data_scaled = scaler.transform(new_data)
start_time = time.time()
predicted_number1 = best_model1.predict(new_data_scaled)
predicted_number2 = best_model2.predict(new_data_scaled)
prediction_time = time.time() - start_time
predicted_number1 += 1
predicted_number2 += 1
print(f'Predicted Winning Numbers: {predicted_number1[0]}, {predicted_number2[0]}')
print(f'Prediction Time: {prediction_time:.2f} seconds')
```
- **Creates a DataFrame with example features** to make a specific prediction.
- **Scales the new data**.
- **Records the start time** for making predictions.
- **Makes predictions** using the best models.
- **Calculates the prediction time**.
- **Adjusts predictions** back to the original range.
- **Prints the predicted winning numbers and the time taken for the prediction**.

# Running SP-ALGO
This model can be ran Google Colab, which is a free cloud based Jupyter Notebook. Datasets can be uploaded via google drive and be referenced in code. 
![image](https://github.com/user-attachments/assets/38727107-7a26-4e1c-9e6b-f61705dc3516)


# Information 
Created by Damoy Skeene, utilizing self-taught principles and 3rd party AI tools such as ChatGpt for refining model behavior and incorporating advance principles which are outside of my expertise. This is a experimental project. Please use this program at your own risk.


