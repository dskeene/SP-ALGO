#Program was created using experiemental Artificial Intelligence systems, as a well as entry-mid level expertise by Damoy Skeene. Programs will consist of bugs and is currently in it's semi-developed form.
# Pick 2 Prediction, based on historical data collected from vendor website.

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import time
import psutil

# Load data
file_path = 'csvfilehere'
data = pd.read_csv(file_path)

# Preprocess data
start_time = time.time()
data['Date'] = pd.to_datetime(data['Date'])
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month

# Clean the 'Winning Numbers' column
data['Winning Numbers'] = data['Winning Numbers'].str.strip()
data = data.dropna(subset=['Winning Numbers'])

# Ensure the winning numbers are split correctly and remove rows with issues
split_numbers = data['Winning Numbers'].str.split(expand=True)
split_numbers = split_numbers.dropna()
split_numbers = split_numbers[split_numbers.apply(lambda row: row.map(str.isdigit).all(), axis=1)]

# Update the original data with cleaned split numbers
data = data.loc[split_numbers.index]
data[['WinningNumber1', 'WinningNumber2']] = split_numbers.astype(int)

# Adjust the labels to fit the expected range
data['WinningNumber1'] -= 1
data['WinningNumber2'] -= 1

# Feature engineering
data['WinningNumber1_Freq'] = data['WinningNumber1'].map(data['WinningNumber1'].value_counts())
data['WinningNumber2_Freq'] = data['WinningNumber2'].map(data['WinningNumber2'].value_counts())

X = data[['DayOfWeek', 'Month', 'WinningNumber1_Freq', 'WinningNumber2_Freq']]
y1 = data['WinningNumber1']
y2 = data['WinningNumber2']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train_1, X_test_1, y1_train, y1_test = train_test_split(X_scaled, y1, test_size=0.2, random_state=42)
X_train_2, X_test_2, y2_train, y2_test = train_test_split(X_scaled, y2, test_size=0.2, random_state=42)
preprocessing_time = time.time() - start_time
print(f'Preprocessing Time: {preprocessing_time:.2f} seconds')

# Initialize and train models with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'learning_rate': [0.01, 0.1]
}

def track_resource_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024), process.cpu_percent(interval=1)

memory_before, cpu_before = track_resource_usage()

model1 = XGBClassifier(random_state=42)
grid_search_1 = GridSearchCV(model1, param_grid, cv=3, scoring='accuracy')
start_time = time.time()
grid_search_1.fit(X_train_1, y1_train)
training_time_1 = time.time() - start_time

memory_after_1, cpu_after_1 = track_resource_usage()

model2 = XGBClassifier(random_state=42)
grid_search_2 = GridSearchCV(model2, param_grid, cv=3, scoring='accuracy')
start_time = time.time()
grid_search_2.fit(X_train_2, y2_train)
training_time_2 = time.time() - start_time

memory_after_2, cpu_after_2 = track_resource_usage()

print(f'Model 1 Training Time: {training_time_1:.2f} seconds')
print(f'Model 1 Memory Usage: {memory_after_1 - memory_before:.2f} MB')
print(f'Model 1 CPU Usage: {cpu_after_1 - cpu_before:.2f}%')

print(f'Model 2 Training Time: {training_time_2:.2f} seconds')
print(f'Model 2 Memory Usage: {memory_after_2 - memory_before:.2f} MB')
print(f'Model 2 CPU Usage: {cpu_after_2 - cpu_before:.2f}%')

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

# Making predictions for each day
unique_dates = data['Date'].unique()
predictions = []

for date in unique_dates:
    day_of_week = date.dayofweek
    month = date.month
    # Use average frequencies as placeholders
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

# Print predictions for each day
for date, num1, num2 in predictions:
    print(f'Date: {date}, Predicted Winning Numbers: {num1}, {num2}')

# Making a prediction for a specific date example
new_data = pd.DataFrame({'DayOfWeek': [2], 'Month': [8], 'WinningNumber1_Freq': [0], 'WinningNumber2_Freq': [0]})  # Example date feature
new_data_scaled = scaler.transform(new_data)
start_time = time.time()
predicted_number1 = best_model1.predict(new_data_scaled)
predicted_number2 = best_model2.predict(new_data_scaled)
prediction_time = time.time() - start_time
# Adjust predictions back to the original range
predicted_number1 += 1
predicted_number2 += 1
print(f'Predicted Winning Numbers: {predicted_number1[0]}, {predicted_number2[0]}')
print(f'Prediction Time: {prediction_time:.2f} seconds')
