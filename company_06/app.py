from flask import Flask, jsonify, Response
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import plotly.express as px
import plotly.io as pio
from datetime import datetime

app = Flask(__name__)

# Load data and models
df = pd.read_csv("merge_06.csv", parse_dates=["date"], index_col="date")
df = df.sort_index().dropna()

# loading saved model
model = joblib.load("new_model.pkl")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

def generate_forecast():
    # Forecasting logic
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='MS')
    
    future_df = pd.DataFrame(index=future_dates)
    future_df['year'] = future_df.index.year
    future_df['month'] = future_df.index.month
    future_df['quarter'] = future_df.index.quarter
    
    full_df = pd.concat([df, future_df])
    required_features = ['year', 'month', 'quarter', 'lag_1', 'lag_11', 
                        'lag_12', 'lag_13', 'rolling_mean_3', 'rolling_mean_6']
    
    forecast_df = full_df.copy()
    forecast_df.loc[future_dates[0], 'lag_1'] = df['sales'].iloc[-1]
    
    for i, date in enumerate(future_dates):
        if i > 0:
            prev_date = future_dates[i-1]
            forecast_df.loc[date, 'lag_1'] = forecast_df.loc[prev_date, 'sales']
        
        for offset in [11, 12, 13]:
            try:
                forecast_df.loc[date, f'lag_{offset}'] = forecast_df.loc[date - pd.DateOffset(months=offset), 'sales']
            except KeyError:
                forecast_df.loc[date, f'lag_{offset}'] = forecast_df[f'lag_{offset}'].mean()
        
        available_data = forecast_df['sales'].dropna()
        for window in [3, 6]:
            if len(available_data) >= window:
                forecast_df.loc[date, f'rolling_mean_{window}'] = available_data[-window:].mean()
        
        X = forecast_df.loc[date, required_features].values.reshape(1, -1)
        X_scaled = scaler_X.transform(X)
        prediction = scaler_y.inverse_transform(model.predict(X_scaled).reshape(-1, 1))
        forecast_df.loc[date, 'sales'] = prediction[0][0]
    
    return forecast_df.loc[future_dates, 'sales']

@app.route('/forecast', methods=['GET'])
def get_forecast():
    forecast = generate_forecast()
    result = {date.strftime('%Y-%m-%d'): round(value, 2) 
             for date, value in forecast.items()}
    return jsonify(result)

@app.route('/plot/matplotlib', methods=['GET'])
def matplotlib_plot():
    forecast = generate_forecast()
    
    plt.figure(figsize=(12, 6))
    forecast.plot(kind='line', marker='o', title='6-Month Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.grid(True)
    plt.tight_layout()
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()
    
    return Response(img_buffer.getvalue(), mimetype='image/png')

@app.route('/plot/plotly', methods=['GET'])
def plotly_visualization():
    forecast = generate_forecast().reset_index()
    forecast.columns = ['Date', 'Sales']
    
    fig = px.line(forecast, x='Date', y='Sales', 
                 title='6-Month Sales Forecast',
                 markers=True, template='plotly_dark')
    fig.update_layout(hovermode='x unified')
    
    return pio.to_html(fig, full_html=False)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)