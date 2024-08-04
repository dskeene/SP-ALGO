![image](https://github.com/user-attachments/assets/374de57f-dee9-4c91-9811-d092e185ad16)

# Supreme Ventures - Pick 2 [Prediction Model]
![ChatGPT](https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Microsoft Excel](https://img.shields.io/badge/Microsoft_Excel-217346?style=for-the-badge&logo=microsoft-excel&logoColor=white) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

SP-ALGO is an experimental prediction model, which intends to predict the Supreme Ventures Pick 2 draw results, using frequency analysis and probability based on historical frequency, as well as other advance statistical concepts implemented using Artificial Intelligence. This model utilizies several statistical libraries such as Pandas and Scikit. The data provided was collected from the Supreme Ventures draw results located on their website.


## Current Model Accuracy 
``` Current: V.1 : 25%```
``` Upcoming: V.2: Untested```


## DISCLAIMER 
This project was created by Damoy Skeene, as a experiemental past time project. It is in no shape or form, stable enough to encourage any bets based on numbers predicted. Please use at your own risk. Thank you.

## Program Breakdown (V.2 - LTSM Model)

```python
import pandas as pd
```
- **Imports pandas**: A library for data manipulation and analysis.

```python
import numpy as np
```
- **Imports NumPy**: A library for numerical operations.

```python
from sklearn.preprocessing import MinMaxScaler
```
- **Imports MinMaxScaler**: A tool to normalize data within a given range.

```python
from sklearn.model_selection import train_test_split
```
- **Imports train_test_split**: A function to split data into training and testing sets.

```python
from tensorflow.keras.models import Sequential
```
- **Imports Sequential**: A Keras model type for linear stacking of layers.

```python
from tensorflow.keras.layers import LSTM, Dense, Dropout
```
- **Imports LSTM, Dense, Dropout**: Layers to build the LSTM neural network. LSTM is for the recurrent layer, Dense is for the fully connected layer, and Dropout is for regularization.

```python
from tensorflow.keras.callbacks import EarlyStopping
```
- **Imports EarlyStopping**: A callback to stop training when a monitored metric has stopped improving.

```python
from kerastuner.tuners import RandomSearch
```
- **Imports RandomSearch**: A tuner for hyperparameter optimization using random search.

```python
from datetime import datetime
```
- **Imports datetime**: A module for manipulating dates and times.

### Load and Preprocess Data
```python
file_path = '/content/p2_draws.csv'
data = pd.read_csv(file_path)
```
- **Loads the data**: Reads the CSV file into a pandas DataFrame.

```python
data['Date'] = pd.to_datetime(data['Date'])
data['Winning Numbers'] = data['Winning Numbers'].str.strip()
data = data.dropna(subset=['Winning Numbers'])
```
- **Processes the data**: Converts 'Date' to datetime, strips whitespace from 'Winning Numbers', and drops rows where 'Winning Numbers' is NaN.

### Ensure Exactly Two Numbers Per Entry
```python
data = data[data['Winning Numbers'].str.count(' ') == 1]
```
- **Filters data**: Keeps only rows where 'Winning Numbers' contains exactly two numbers separated by a space.

### Split Winning Numbers into Separate Columns
```python
data[['WinningNumber1', 'WinningNumber2']] = data['Winning Numbers'].str.split(expand=True).astype(int)
```
- **Splits 'Winning Numbers'** into two columns, 'WinningNumber1' and 'WinningNumber2', and converts them to integers.

### Feature Engineering
```python
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month
```
- **Extracts features**: Adds 'DayOfWeek' and 'Month' columns based on 'Date'.

```python
data = data.loc[:, ['DayOfWeek', 'Month', 'WinningNumber1', 'WinningNumber2']]
```
- **Selects relevant columns**: Keeps only the columns needed for the model.

### Prepare Features and Labels
```python
X = data[['DayOfWeek', 'Month', 'WinningNumber1', 'WinningNumber2']].values
y1 = data['WinningNumber1'].values
y2 = data['WinningNumber2'].values
```
- **Prepares the feature matrix (X)** and the labels (y1 for 'WinningNumber1' and y2 for 'WinningNumber2').

### Normalize the Data
```python
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
```
- **Scales the features**: Normalizes X to a range of 0 to 1.

### Create Sequences
```python
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)
```
- **Defines a function to create sequences**: Converts the feature matrix and labels into sequences of specified length (`time_steps`).

```python
time_steps = 10
X1_seq, y1_seq = create_sequences(X_scaled, y1, time_steps)
X2_seq, y2_seq = create_sequences(X_scaled, y2, time_steps)
```
- **Creates sequences** for both 'WinningNumber1' and 'WinningNumber2'.

### Split Data into Training and Testing Sets
```python
split = int(0.8 * len(X1_seq))
X1_train, X1_test = X1_seq[:split], X1_seq[split:]
y1_train, y1_test = y1_seq[:split], y1_seq[split:]

X2_train, X2_test = X2_seq[:split], X2_seq[split:]
y2_train, y2_test = y2_seq[:split], y2_seq[split:]
```
- **Splits the sequences** into training and testing sets (80% training, 20% testing).

### Hyperparameter Tuning Function
```python
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32),
                   return_sequences=True, input_shape=(time_steps, X_scaled.shape[1])))
    model.add(Dropout(hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32)))
    model.add(Dropout(hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
```
- **Defines the model-building function** for hyperparameter tuning. This function builds a Sequential model with:
  - An LSTM layer with tunable units and dropout.
  - A second LSTM layer with tunable units and dropout.
  - A Dense output layer.
  - Uses 'adam' optimizer and 'mean_squared_error' loss.

### Hyperparameter Tuning with Keras Tuner
```python
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=2,
    directory='tuner',
    project_name='lstm_tuning'
)
```
- **Sets up RandomSearch** for hyperparameter tuning:
  - `objective`: Minimize validation loss.
  - `max_trials`: Try up to 10 different hyperparameter combinations.
  - `executions_per_trial`: Train each combination 2 times.
  - `directory`: Save results to the 'tuner' directory.
  - `project_name`: Name the project 'lstm_tuning'.

```python
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
```
- **Sets up early stopping** to stop training when validation loss stops improving, with patience of 10 epochs, and restore the best weights.

### Train the Models
```python
tuner.search(X1_train, y1_train, epochs=50, batch_size=16, validation_split=0.2, callbacks=[early_stopping])
best_model1 = tuner.get_best_models(num_models=1)[0]

tuner.search(X2_train, y2_train, epochs=50, batch_size=16, validation_split=0.2, callbacks=[early_stopping])
best_model2 = tuner.get_best_models(num_models=1)[0]
```
- **Performs hyperparameter search and training**:
  - Searches for the best model for `y1`.
  - Searches for the best model for `y2`.
  - Trains each model for up to 50 epochs with a batch size of 16, using 20% of the training data for validation.
  - Uses early stopping to prevent overfitting.
  - Retrieves the best model from the search.

### Evaluate the Models
```python
loss1 = best_model1.evaluate(X1_test, y1_test)
loss2 = best_model2.evaluate(X2_test, y2_test)
print(f'Model 1 Test Loss: {loss1}')
print(f'Model 2 Test Loss: {loss2}')
```
- **Evaluates the models** on the test sets and prints the test loss for both models.

### Make Predictions
```python
def predict_next(model, data, time_steps):
    prediction = model.predict(data[-time_steps:].reshape(1, time_steps, data.shape[1]))
    return prediction[0][0]
```
- **Defines a function to make predictions**: Uses the trained model to predict the next number based on the last `time_steps` of data.

```python
next_number1 = predict_next(best_model1, X_scaled, time_steps)
next_number2 = predict_next(best_model2, X_scaled, time_steps)

print(f"Predicted Winning Numbers: {round(next_number1)}, {round(next_number2)}")
```
- **Uses the best models to predict the next winning numbers** and prints them.

# Running SP-ALGO
This model can be ran Google Colab, which is a free cloud based Jupyter Notebook. Datasets can be uploaded via google drive and be referenced in code. 
![image](https://github.com/user-attachments/assets/38727107-7a26-4e1c-9e6b-f61705dc3516)


# Information 
Created by Damoy Skeene, utilizing self-taught principles and 3rd party AI tools such as ChatGpt for refining model behavior and incorporating advance principles which are outside of my expertise. This is a experimental project. Please use this program at your own risk.


