import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Load Dataset
df = pd.read_csv('weather.csv', parse_dates=['Date'])
df = df.sort_values('Date')

# 2. Basic Preprocessing
df = df.dropna()
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# 3. Feature Selection
features = ['Humidity', 'Rainfall', 'WindSpeed', 'Year', 'Month', 'Day']
target = 'Temperature'

X = df[features]
y = df[target]

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Models

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# LSTM requires time series formatting
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[features + [target]])

X_lstm = []
y_lstm = []
n_steps = 10

for i in range(n_steps, len(scaled_features)):
    X_lstm.append(scaled_features[i-n_steps:i, :-1])
    y_lstm.append(scaled_features[i, -1])

X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
split_index = int(0.8 * len(X_lstm))

X_train_lstm, X_test_lstm = X_lstm[:split_index], X_lstm[split_index:]
y_train_lstm, y_test_lstm = y_lstm[:split_index], y_lstm[split_index:]

# LSTM Model
lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, X_lstm.shape[2])),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=16, verbose=0)
lstm_pred_scaled = lstm_model.predict(X_test_lstm)

# Inverse scale LSTM predictions
inv_scale = scaler.inverse_transform(
    np.hstack([X_test_lstm[:, -1, :-1], lstm_pred_scaled])
)
lstm_pred = inv_scale[:, -1]

# 6. Evaluation
def evaluate_model(y_true, y_pred, name):
    print(f"\n{name} Evaluation:")
    print(f"MAE:  {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print(f"R²:   {r2_score(y_true, y_pred):.2f}")

evaluate_model(y_test, lr_pred, "Linear Regression")
evaluate_model(y_test, rf_pred, "Random Forest")
evaluate_model(df[target].values[split_index + n_steps:], lstm_pred, "LSTM")

# 7. Plotting
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:100], label='Actual', linestyle='--')
plt.plot(lr_pred[:100], label='Linear Regression')
plt.plot(rf_pred[:100], label='Random Forest')
plt.plot(lstm_pred[:100], label='LSTM')
plt.title('Temperature Prediction (Sample)')
plt.xlabel('Sample Index')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
