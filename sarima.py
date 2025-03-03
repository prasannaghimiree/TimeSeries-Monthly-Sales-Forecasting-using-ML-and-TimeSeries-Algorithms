import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
data = pd.read_csv('csv_data/merge.csv', parse_dates=True, index_col='date')
data = data.sort_index()

plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(data['sales'], lags=24, ax=plt.gca())
plt.title("ACF of Sales")
plt.subplot(122)
plot_pacf(data['sales'], lags=24, ax=plt.gca())
plt.title("PACF of Sales")
plt.show()

seasonality_period = 12  

sarima_model = auto_arima(data['sales'], exogenous=data[['qty']], seasonal=True, m=seasonality_period,
                          stepwise=True, trace=True, error_action='ignore', suppress_warnings=True,
                          max_p=5, max_q=5, max_order=10)

print(sarima_model.summary())

forecast_sarima = sarima_model.predict(n_periods=12, exogenous=data[['qty']][-12:])

mae_sarima = mean_absolute_error(data['sales'][-12:], forecast_sarima)
r2_sarima = r2_score(data['sales'][-12:], forecast_sarima)
print(f'SARIMA -> MAE: {mae_sarima}, R^2: {r2_sarima}')

def create_lagged_features(data_sales, data_qty, lags):
    X, y = [], []
    for i in range(max(lags), len(data_sales)):
        X.append([data_sales[i-lag] for lag in lags] + [data_qty[i-lag] for lag in lags])
        y.append(data_sales[i])
    return np.array(X), np.array(y)

scaler_sales = MinMaxScaler(feature_range=(0, 1))
scaler_qty = MinMaxScaler(feature_range=(0, 1))

scaled_sales = scaler_sales.fit_transform(data[['sales']])
scaled_qty = scaler_qty.fit_transform(data[['qty']])

selected_lags = [1, 12, 24, 6]
X, y = create_lagged_features(scaled_sales[:, 0], scaled_qty[:, 0], selected_lags)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define LSTM model 
model_lstm = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    BatchNormalization(),
    LSTM(256, return_sequences=True),
    Dropout(0.3),
    BatchNormalization(),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(64),
    Dense(1)
])

model_lstm.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train the LSTM model
model_lstm.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

# Make predictions with LSTM
predictions_lstm = model_lstm.predict(X_test)


# Inverse scale the 'sales' predictions and actual 'sales' values
predictions_lstm_exp = scaler_sales.inverse_transform(predictions_lstm.reshape(-1, 1)) 
y_test_exp = scaler_sales.inverse_transform(y_test.reshape(-1, 1)) 

# Evaluate LSTM performance
mae_lstm = mean_absolute_error(y_test_exp, predictions_lstm_exp)
r2_lstm = r2_score(y_test_exp, predictions_lstm_exp)
print(f'LSTM -> MAE: {mae_lstm}, R^2: {r2_lstm}')

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['sales'], label='Actual Sales', color='blue')
plt.plot(data.index[-12:], forecast_sarima, label='SARIMA Predictions', linestyle='dashed', color='red')
plt.plot(data.index[-len(y_test_exp):], predictions_lstm_exp, label='LSTM Predictions', linestyle='dashed', color='green')
plt.legend()
plt.show()
