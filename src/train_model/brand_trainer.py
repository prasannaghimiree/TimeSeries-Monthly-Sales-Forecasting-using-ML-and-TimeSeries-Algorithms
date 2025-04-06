# src/train_model/brand_trainer.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.stattools import acf, pacf
import joblib

class BrandTrainer:
    def __init__(self, historical_data_path, model_path_prefix, forecast_path_prefix):
        self.historical_data_path = historical_data_path
        self.model_path_prefix = model_path_prefix
        self.forecast_path_prefix = forecast_path_prefix
        os.makedirs(os.path.dirname(model_path_prefix), exist_ok=True)
        os.makedirs(os.path.dirname(forecast_path_prefix), exist_ok=True)

    def train_brand_model(self, brand):
        """Train a Random Forest model for a specific brand."""
        # Read the CSV with BS_YEAR_MONTH as a column
        df = pd.read_csv(self.historical_data_path)
        df["BS_YEAR_MONTH"] = pd.to_datetime(df["BS_YEAR_MONTH"])
        df.set_index("BS_YEAR_MONTH", inplace=True)

        if brand not in df.columns:
            print(f"Brand {brand} not found in historical data.")
            return None, None, None, None

        # Initialize brand_data with the brand's sales column
        brand_data = df[[brand]].copy()  # Use copy to avoid modifying original df

        # Drop NaN values
        brand_data.dropna(inplace=True)

        if len(brand_data) < 6:
            print(f"Skipping {brand}: insufficient data ({len(brand_data)} rows)")
            return None, None, None, None

        # Feature engineering
        brand_data["year"] = brand_data.index.year
        brand_data["month"] = brand_data.index.month
        brand_data["quarter"] = brand_data.index.quarter

        acf_values = acf(brand_data[brand], nlags=12, fft=False)
        pacf_values = pacf(brand_data[brand], nlags=12)
        lags = [f"lag_{i}" for i in range(1, 13) if abs(acf_values[i]) > 0.2 or abs(pacf_values[i]) > 0.2]
        if not lags:
            lags = ["lag_1"]

        for lag in lags:
            lag_num = int(lag.split("_")[1])
            brand_data[lag] = brand_data[brand].shift(lag_num)

        brand_data["rolling_mean_3"] = brand_data[brand].rolling(window=3).mean()
        brand_data["rolling_mean_6"] = brand_data[brand].rolling(window=6).mean()
        brand_data.dropna(inplace=True)

        if len(brand_data) < 6:
            print(f"Skipping {brand}: insufficient data after feature engineering ({len(brand_data)} rows)")
            return None, None, None, None
        
        features = ["year", "month", "quarter"] + lags + ["rolling_mean_3", "rolling_mean_6"]

        X = brand_data[features]
        y = brand_data[brand]

        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

        param_grid = {
            "n_estimators": [100, 250, 500],
            "max_depth": [3, 7, 10],
            "min_samples_split": [5, 10, 15],
            "min_samples_leaf": [3, 5, 10],
        }
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
        grid_search.fit(X_scaled, y_scaled)

        model_path = f"{self.model_path_prefix}{brand}.pkl"
        scaler_X_path = f"{self.model_path_prefix}{brand}_scaler_X.pkl"
        scaler_y_path = f"{self.model_path_prefix}{brand}_scaler_y.pkl"
        joblib.dump(grid_search.best_estimator_, model_path)
        joblib.dump(scaler_X, scaler_X_path)
        joblib.dump(scaler_y, scaler_y_path)

        print(f"Trained model for {brand} saved at {model_path}")
        return grid_search.best_estimator_, scaler_X, scaler_y, lags

    def generate_forecast(self, brand, model, scaler_X, scaler_y, lags):
        """Generate forecast for a brand and save it."""
        df = pd.read_csv(self.historical_data_path)
        df["BS_YEAR_MONTH"] = pd.to_datetime(df["BS_YEAR_MONTH"])
        df.set_index("BS_YEAR_MONTH", inplace=True)

        if brand not in df.columns:
            print(f"Brand {brand} not found in historical data.")
            return None

        brand_data = df[[brand]].rename(columns={brand: "sales"})
        if brand_data.empty:
            print(f"No data available for {brand} to generate forecast.")
            return None

        brand_data = brand_data.sort_index()
        brand_data.dropna(inplace=True)

        if len(brand_data) < 1:
            print(f"Insufficient data for {brand} after cleaning to generate forecast.")
            return None

        last_date = brand_data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq="MS")
        future_df = pd.DataFrame(index=future_dates)
        future_df["year"] = future_df.index.year
        future_df["month"] = future_df.index.month
        future_df["quarter"] = future_df.index.quarter

        full_df = pd.concat([brand_data, future_df])
        forecast_df = full_df.copy()

        # Ensure all lags exist in forecast_df
        for lag in lags:
            lag_num = int(lag.split("_")[1])
            forecast_df[lag] = forecast_df["sales"].shift(lag_num)

        forecast_df["rolling_mean_3"] = forecast_df["sales"].rolling(window=3).mean()
        forecast_df["rolling_mean_6"] = forecast_df["sales"].rolling(window=6).mean()

        # Features should match those used in training
        features = ["year", "month", "quarter"] + lags + ["rolling_mean_3", "rolling_mean_6"]
        for i, date in enumerate(future_dates):
            if i == 0:
                forecast_df.loc[date, "lag_1"] = brand_data["sales"].iloc[-1]
            else:
                forecast_df.loc[date, "lag_1"] = forecast_df.loc[future_dates[i - 1], "sales"]

            for lag in lags[1:]:  # Skip lag_1 since it's already set
                lag_num = int(lag.split("_")[1])
                forecast_df.loc[date, lag] = forecast_df["sales"].shift(lag_num).iloc[-1] if pd.notna(forecast_df["sales"].shift(lag_num).iloc[-1]) else forecast_df[lag].mean()

            available_data = forecast_df["sales"].dropna()
            for window in [3, 6]:
                if len(available_data) >= window:
                    forecast_df.loc[date, f"rolling_mean_{window}"] = available_data[-window:].mean()
                else:
                    forecast_df.loc[date, f"rolling_mean_{window}"] = available_data.mean() if not available_data.empty else 0

            # Keep X as a DataFrame with column names
            X = pd.DataFrame([forecast_df.loc[date, features]], columns=features)
            X_scaled = scaler_X.transform(X)
            prediction = scaler_y.inverse_transform(model.predict(X_scaled).reshape(-1, 1))[0][0]
            forecast_df.loc[date, "sales"] = prediction

        forecast_result = forecast_df.loc[future_dates, ["sales"]].reset_index()
        forecast_result.columns = ["forecast_date", "forecast_sales"]
        forecast_result.to_csv(f"{self.forecast_path_prefix}{brand}.csv", index=False)
        print(f"Forecast for {brand} saved at {self.forecast_path_prefix}{brand}.csv")
        return forecast_result

    def train_and_forecast(self, brand):
        """Train and generate forecast if model doesn't exist."""
        model_path = f"{self.model_path_prefix}{brand}.pkl"
        forecast_path = f"{self.forecast_path_prefix}{brand}.csv"
        
        if not os.path.exists(model_path) or not os.path.exists(forecast_path):
            model, scaler_X, scaler_y, lags = self.train_brand_model(brand)
            if model:
                return self.generate_forecast(brand, model, scaler_X, scaler_y, lags)
        return None