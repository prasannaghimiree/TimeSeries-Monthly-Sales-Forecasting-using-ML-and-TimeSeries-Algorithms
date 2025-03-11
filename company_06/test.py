import joblib
#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r"merge_06.csv", parse_dates=["date"], index_col="date")
df = df.sort_index()
df.dropna(inplace=True)

#loading saved model and standard scaler saved as scaler_X.pkl and scaler_y.pkl
rf_model = joblib.load("new_model.pkl")
scaler_X=joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# Generate future dates for next 6 months
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='MS')

# Create a temporary DataFrame for future forecasts
future_df = pd.DataFrame(index=future_dates)
future_df['year'] = future_df.index.year
future_df['month'] = future_df.index.month
future_df['quarter'] = future_df.index.quarter

# Combine historical data with future dates
full_df = pd.concat([df, future_df], axis=0)

# Features needed for forecasting
required_features = ['year', 'month', 'quarter', 'lag_1', 'lag_11', 'lag_12','lag_13',
                    'rolling_mean_3', 'rolling_mean_6']

# Copy to store predictions
forecast_df = full_df.copy()

# Initialize lag_1 for the first future date using the last historical sales value
forecast_df.loc[future_dates[0], 'lag_1'] = df['sales'].iloc[-1]  

for i, date in enumerate(future_dates):
    # Update lag_1 for subsequent dates using prior forecasts
    if i > 0:
        prev_date = future_dates[i-1]
        forecast_df.loc[date, 'lag_1'] = forecast_df.loc[prev_date, 'sales']
    
    # Handle 11/12-month lags (use historical data if available)
    try:
        forecast_df.loc[date, 'lag_11'] = forecast_df.loc[date - pd.DateOffset(months=11), 'sales']
        forecast_df.loc[date, 'lag_12'] = forecast_df.loc[date - pd.DateOffset(months=12), 'sales']
        forecast_df.loc[date, 'lag_13'] = forecast_df.loc[date - pd.DateOffset(months=13), 'sales']
    except KeyError:
        
        # Fallback to mean if lag exceeds historical data
        forecast_df.loc[date, 'lag_11'] = forecast_df['lag_11'].mean()
        forecast_df.loc[date, 'lag_12'] = forecast_df['lag_12'].mean()
        forecast_df.loc[date, 'lag_13'] = forecast_df['lag_13'].mean()

    
    # Calculate rolling means using all available data (historical + forecasts)
    available_data = forecast_df['sales'].dropna()
    if len(available_data) >= 3:
        forecast_df.loc[date, 'rolling_mean_3'] = available_data[-3:].mean()
    if len(available_data) >= 6:
        forecast_df.loc[date, 'rolling_mean_6'] = available_data[-6:].mean()
    
    # Prepare features and predict
    X_future = forecast_df.loc[date, required_features].values.reshape(1, -1)
    X_future_scaled = scaler_X.transform(X_future)
    y_future = scaler_y.inverse_transform(rf_model.predict(X_future_scaled).reshape(-1, 1))
    print("Future data are : \n", X_future)
    print('sales are : \n',y_future)
    
    # Store prediction
    forecast_df.loc[date, 'sales'] = y_future[0][0]
    print(f"Forecast for {date.strftime('%Y-%m')}: {y_future[0][0]:.2f}")