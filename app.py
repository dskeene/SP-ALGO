import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch
from datetime import datetime

# Load and preprocess data
file_path = '/content/p2_draws.csv'
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])
data['Winning Numbers'] = data['Winning Numbers'].str.strip()
data = data.dropna(subset=['Winning Numbers'])

# Ensure exactly two numbers per entry
data = data[data['Winning Numbers'].str.count(' ') == 1]

# Split winning numbers into two separate columns
data[['WinningNumber1', 'WinningNumber2']] = data['Winning Numbers'].str.split(expand=True).astype(int)

# Feature engineering
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month

# Combine features
data = data.loc[:, ['DayOfWeek', 'Month', 'WinningNumber1', 'WinningNumber2']]

# Prepare features and labels
X = data[['DayOfWeek', 'Month', 'WinningNumber1', 'WinningNumber2']].values
y1 = data['WinningNumber1'].values
y2 = data['WinningNumber2'].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# Create sequences
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10
X1_seq, y1_seq = create_sequences(X_scaled, y1, time_steps)
X2_seq, y2_seq = create_sequences(X_scaled, y2, time_steps)

# Split data into training and testing sets
split = int(0.8 * len(X1_seq))
X1_train, X1_test = X1_seq[:split], X1_seq[split:]
y1_train, y1_test = y1_seq[:split], y1_seq[split:]

X2_train, X2_test = X2_seq[:split], X2_seq[split:]
y2_train, y2_test = y2_seq[:split], y2_seq[split:]

# Hyperparameter tuning function
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

# Hyperparameter tuning with Keras Tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=2,
    directory='tuner',
    project_name='lstm_tuning'
)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the models
tuner.search(X1_train, y1_train, epochs=50, batch_size=16, validation_split=0.2, callbacks=[early_stopping])
best_model1 = tuner.get_best_models(num_models=1)[0]

tuner.search(X2_train, y2_train, epochs=50, batch_size=16, validation_split=0.2, callbacks=[early_stopping])
best_model2 = tuner.get_best_models(num_models=1)[0]

# Evaluate the models
loss1 = best_model1.evaluate(X1_test, y1_test)
loss2 = best_model2.evaluate(X2_test, y2_test)
print(f'Model 1 Test Loss: {loss1}')
print(f'Model 2 Test Loss: {loss2}')

# Make predictions
def predict_next(model, data, time_steps):
    prediction = model.predict(data[-time_steps:].reshape(1, time_steps, data.shape[1]))
    return prediction[0][0]

next_number1 = predict_next(best_model1, X_scaled, time_steps)
next_number2 = predict_next(best_model2, X_scaled, time_steps)

print(f"Predicted Winning Numbers: {round(next_number1)}, {round(next_number2)}")
