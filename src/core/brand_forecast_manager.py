import os
import pandas as pd
import numpy as np
import nepali_datetime
from statsmodels.tsa.stattools import acf, pacf
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import linregress
import joblib
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from src.utils.utils import BaseForecastManager, fetch_db_data
from src.utils.db_queries import BRAND_SALES_QUERY
import logging

logger = logging.getLogger(__name__)

class BrandSalesForecastManager(BaseForecastManager):
    def __init__(self):
        super().__init__()
        self.historical_data_path = r"data\dataset\historical_sales_latest_12_months_brands.csv"
        self.model_path_prefix = r"models\brand_model_"
        self.forecast_path_prefix = r"brand_dataset\forecast-"
        os.makedirs(os.path.dirname(self.historical_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.forecast_path_prefix), exist_ok=True)
        os.makedirs(os.path.dirname(self.model_path_prefix), exist_ok=True)
        self.forecast_cache = {}
        self.brands = self._fetch_and_prepare_data()

    def _fetch_and_prepare_data(self):
        combined_df = fetch_db_data(self.db_config, BRAND_SALES_QUERY, is_brand=True)
        combined_df["BS_YEAR_MONTH"] = pd.to_datetime(combined_df["BS_YEAR_MONTH"], format="%Y-%m")
        current_bs_date = nepali_datetime.date.today()
        current_bs_year_month = f"{current_bs_date.year:04d}-{current_bs_date.month:02d}"
        combined_df = combined_df[combined_df["BS_YEAR_MONTH"] < current_bs_year_month]

        latest_date = combined_df["BS_YEAR_MONTH"].max()
        one_year_ago = latest_date - pd.DateOffset(months=12)
        latest_12_months_data = combined_df[combined_df["BS_YEAR_MONTH"] >= one_year_ago]
        unique_brands = latest_12_months_data["BRAND_NAME"].unique()
        historical_data = combined_df[combined_df["BRAND_NAME"].isin(unique_brands)]
        all_months = pd.date_range(start=combined_df["BS_YEAR_MONTH"].min(), end=combined_df["BS_YEAR_MONTH"].max(), freq="MS")
        pivoted_data = historical_data.pivot_table(index="BS_YEAR_MONTH", columns="BRAND_NAME", values="SALES_VALUE", aggfunc="sum", fill_value=0)
        pivoted_data = pivoted_data.reindex(all_months, fill_value=0).reset_index().rename(columns={"index": "BS_YEAR_MONTH"})
        pivoted_data.to_csv(self.historical_data_path, index=False)
        print(f"New dataset created and saved as '{self.historical_data_path}'")
        return unique_brands.tolist()

    def preprocess_data(self, brand_data, brand):
        """Handle outliers using IQR method."""
        Q1, Q3 = brand_data[brand].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        brand_data[brand] = brand_data[brand].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
        return brand_data

    def train_brand_model(self, brand, n_lags=12, threshold=0.2):
        """Train an XGBoost model for a specific brand."""
        df = pd.read_csv(self.historical_data_path)
        df["BS_YEAR_MONTH"] = pd.to_datetime(df["BS_YEAR_MONTH"])
        df.set_index("BS_YEAR_MONTH", inplace=True)
        if brand not in df.columns:
            print(f"Brand {brand} not found in historical data.")
            return None, None, None, None

        brand_data = df[[brand]].copy()
        brand_data = self.preprocess_data(brand_data, brand)
        target = brand
        brand_data = brand_data.dropna(subset=[target])
        if len(brand_data) < n_lags:
            print(f"Skipping {brand}: insufficient data ({len(brand_data)} rows)")
            return None, None, None, None

        # Feature engineering
        brand_data["time_index"] = range(len(brand_data))
        brand_data["year"] = brand_data.index.year
        brand_data["month"] = brand_data.index.month
        brand_data["month_sin_12"] = np.sin(2 * np.pi * brand_data["month"] / 12)
        brand_data["month_cos_12"] = np.cos(2 * np.pi * brand_data["month"] / 12)
        brand_data["month_sin_6"] = np.sin(2 * np.pi * brand_data["month"] / 6)
        brand_data["month_cos_6"] = np.cos(2 * np.pi * brand_data["month"] / 6)
        brand_data["month_sin_3"] = np.sin(2 * np.pi * brand_data["month"] / 3)
        brand_data["month_cos_3"] = np.cos(2 * np.pi * brand_data["month"] / 3)
        brand_data["quarter"] = brand_data.index.quarter
        brand_data["time_index_sq"] = brand_data["time_index"] ** 2
        brand_data["time_index_cube"] = brand_data["time_index"] ** 3
        slope, intercept, _, _, _ = linregress(brand_data["time_index"], brand_data[target])
        brand_data["trend"] = brand_data["time_index"] * slope + intercept
        brand_data["trend_month_sin_12"] = brand_data["trend"] * brand_data["month_sin_12"]
        brand_data["trend_month_cos_12"] = brand_data["trend"] * brand_data["month_cos_12"]

        acf_values = acf(brand_data[target], nlags=n_lags, fft=False)
        pacf_values = pacf(brand_data[target], nlags=n_lags)
        combined_lags = list(set(
            [f"lag_{i}" for i in range(1, len(acf_values)) if abs(acf_values[i]) > threshold] +
            [f"lag_{i}" for i in range(1, len(pacf_values)) if abs(pacf_values[i]) > threshold]
        ))
        if not combined_lags:
            combined_lags = [f"lag_{i}" for i in range(1, n_lags + 1)]
            print(f"No significant lags for {brand}, using default lags: {combined_lags}")
        lags = sorted(combined_lags, key=lambda x: int(x.split("_")[1]))
        for lag in lags:
            lag_num = int(lag.split("_")[1])
            brand_data[lag] = brand_data[target].shift(lag_num)

        brand_data["rolling_mean_3"] = brand_data[target].rolling(window=3).mean()
        brand_data["rolling_mean_6"] = brand_data[target].rolling(window=6).mean()
        brand_data["rolling_mean_12"] = brand_data[target].rolling(window=12).mean()
        brand_data["rolling_std_3"] = brand_data[target].rolling(window=3).std()
        brand_data = brand_data.dropna()
        if len(brand_data) < 6:
            print(f"Skipping {brand}: insufficient data after feature engineering ({len(brand_data)} rows)")
            return None, None, None, None

        features = ["time_index", "time_index_sq", "time_index_cube", "trend", "year",
                    "month_sin_12", "month_cos_12", "month_sin_6", "month_cos_6",
                    "month_sin_3", "month_cos_3", "quarter", "trend_month_sin_12",
                    "trend_month_cos_12"] + lags + \
                   ["rolling_mean_3", "rolling_mean_6", "rolling_mean_12", "rolling_std_3"]
        X = brand_data[features]
        y = brand_data[target]

        scaler_X = RobustScaler()
        X_scaled = scaler_X.fit_transform(X)
        param_grid = {
            "n_estimators": [500, 1000, 1500],
            "max_depth": [10, 15, 20],
            "learning_rate": [0.001, 0.005, 0.01],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_lambda": [0, 0.01],
            "reg_alpha": [0, 0.01]
        }
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(xgb_model, param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1)
        grid_search.fit(X_scaled, y)

        model_path = f"{self.model_path_prefix}{brand}.pkl"
        scaler_X_path = f"{self.model_path_prefix}{brand}_scaler_X.pkl"
        joblib.dump(grid_search.best_estimator_, model_path)
        joblib.dump(scaler_X, scaler_X_path)
        print(f"Trained and saved model for {brand} at {model_path}")
        print(f"Train MAE for {brand}: {mean_absolute_error(y, grid_search.best_estimator_.predict(X_scaled)):.2f}")
        print(f"Train MSE for {brand}: {mean_squared_error(y, grid_search.best_estimator_.predict(X_scaled)):.2f}")
        print(f"Train R² for {brand}: {r2_score(y, grid_search.best_estimator_.predict(X_scaled)):.2f}")

        return grid_search.best_estimator_, scaler_X, lags, features

    def generate_forecast(self, brand, model, scaler_X, lags, features, forecast_periods=12):
        """Generate recursive forecast for a brand."""
        df = pd.read_csv(self.historical_data_path)
        df["BS_YEAR_MONTH"] = pd.to_datetime(df["BS_YEAR_MONTH"])
        df.set_index("BS_YEAR_MONTH", inplace=True)
        if brand not in df.columns:
            print(f"Brand {brand} not found in historical data.")
            return None

        brand_data = df[[brand]].rename(columns={brand: "sales"}).sort_index().dropna()
        if len(brand_data) < 1:
            print(f"Insufficient data for {brand} to generate forecast.")
            return None

        brand_data["time_index"] = range(len(brand_data))
        brand_data["year"] = brand_data.index.year
        brand_data["month"] = brand_data.index.month
        brand_data["month_sin_12"] = np.sin(2 * np.pi * brand_data["month"] / 12)
        brand_data["month_cos_12"] = np.cos(2 * np.pi * brand_data["month"] / 12)
        brand_data["month_sin_6"] = np.sin(2 * np.pi * brand_data["month"] / 6)
        brand_data["month_cos_6"] = np.cos(2 * np.pi * brand_data["month"] / 6)
        brand_data["month_sin_3"] = np.sin(2 * np.pi * brand_data["month"] / 3)
        brand_data["month_cos_3"] = np.cos(2 * np.pi * brand_data["month"] / 3)
        brand_data["quarter"] = brand_data.index.quarter
        slope, intercept, _, _, _ = linregress(brand_data["time_index"], brand_data["sales"])
        brand_data["trend"] = brand_data["time_index"] * slope + intercept
        brand_data["time_index_sq"] = brand_data["time_index"] ** 2
        brand_data["time_index_cube"] = brand_data["time_index"] ** 3
        brand_data["trend_month_sin_12"] = brand_data["trend"] * brand_data["month_sin_12"]
        brand_data["trend_month_cos_12"] = brand_data["trend"] * brand_data["month_cos_12"]

        for lag in lags:
            lag_num = int(lag.split("_")[1])
            brand_data[lag] = brand_data["sales"].shift(lag_num)

        brand_data["rolling_mean_3"] = brand_data["sales"].rolling(window=3).mean()
        brand_data["rolling_mean_6"] = brand_data["sales"].rolling(window=6).mean()
        brand_data["rolling_mean_12"] = brand_data["sales"].rolling(window=12).mean()
        brand_data["rolling_std_3"] = brand_data["sales"].rolling(window=3).std()

        forecast_df = brand_data.copy()
        last_date = brand_data.index[-1]
        last_time_index = brand_data["time_index"].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_periods, freq="MS")
        print(f"\nForecasting for {brand}:")
        forecast_values = []
        for i, date in enumerate(future_dates):
            new_row = pd.Series(index=[date], dtype=float)
            new_row["time_index"] = last_time_index + 1 + i
            new_row["time_index_sq"] = (last_time_index + 1 + i) ** 2
            new_row["time_index_cube"] = (last_time_index + 1 + i) ** 3
            new_row["trend"] = (last_time_index + 1 + i) * slope + intercept
            new_row["year"] = date.year
            new_row["month"] = date.month
            new_row["month_sin_12"] = np.sin(2 * np.pi * date.month / 12)
            new_row["month_cos_12"] = np.cos(2 * np.pi * date.month / 12)
            new_row["month_sin_6"] = np.sin(2 * np.pi * date.month / 6)
            new_row["month_cos_6"] = np.cos(2 * np.pi * date.month / 6)
            new_row["month_sin_3"] = np.sin(2 * np.pi * date.month / 3)
            new_row["month_cos_3"] = np.cos(2 * np.pi * date.month / 3)
            new_row["quarter"] = date.quarter
            new_row["trend_month_sin_12"] = new_row["trend"] * new_row["month_sin_12"]
            new_row["trend_month_cos_12"] = new_row["trend"] * new_row["month_cos_12"]
            for lag in lags:
                lag_num = int(lag.split("_")[1])
                new_row[lag] = pd.concat([forecast_df["sales"], pd.Series(forecast_values)]).iloc[-lag_num]
            available_data = pd.concat([forecast_df["sales"], pd.Series(forecast_values)])
            for window in [3, 6, 12]:
                new_row[f"rolling_mean_{window}"] = available_data[-window:].mean() if len(available_data) >= window else available_data.mean()
                if window == 3:
                    new_row["rolling_std_3"] = available_data[-window:].std() if len(available_data) >= window else (available_data.std() if len(available_data) > 1 else 0)
            X = pd.DataFrame([new_row[features]], columns=features)
            X_scaled = scaler_X.transform(X)
            prediction = model.predict(X_scaled)[0]
            forecast_values.append(prediction)
            forecast_df.loc[date, "sales"] = prediction
            print(f"Date: {date}, Forecast: {prediction:.2f}")

        forecast_result = forecast_df.loc[future_dates, ["sales"]].reset_index()
        forecast_result.columns = ["forecast_date", "forecast_sales"]
        forecast_result.to_csv(f"{self.forecast_path_prefix}{brand}.csv", index=False)
        print(f"Forecast for {brand} saved at {self.forecast_path_prefix}{brand}.csv")
        return forecast_result

    def train_and_forecast(self, brand):
        """Train, forecast, and visualize."""
        model_path = f"{self.model_path_prefix}{brand}.pkl"
        forecast_path = f"{self.forecast_path_prefix}{brand}.csv"
        if not os.path.exists(model_path) or not os.path.exists(forecast_path):
            model, scaler_X, lags, features = self.train_brand_model(brand)
            if model:
                return self.generate_forecast(brand, model, scaler_X, lags, features)
        forecast_result = pd.read_csv(forecast_path, parse_dates=["forecast_date"])
        return forecast_result

    def train_all_brands(self):
        for brand in self.brands:
            self.train_and_forecast(brand)

    def _load_forecast_data(self, brand):
        forecast_path = f"{self.forecast_path_prefix}{brand}.csv"
        if not os.path.exists(forecast_path):
            self.train_and_forecast(brand)
        if os.path.exists(forecast_path) and os.path.getsize(forecast_path) > 0:
            return pd.read_csv(forecast_path, parse_dates=["forecast_date"])
        return pd.DataFrame(columns=["forecast_date", "forecast_sales"])

    def query_csv(self, structured_query):
        try:
            brand = structured_query.get("brand")
            if not brand or brand not in self.brands:
                return self._build_response(structured_query, {}, f"Invalid or unknown brand: {brand}", structured_query.get("type", "invalid"))

            if brand not in self.forecast_cache:
                self.forecast_cache[brand] = self._load_forecast_data(brand)
            forecast_df = self.forecast_cache[brand]

            if forecast_df.empty:
                return self._build_response(structured_query, {}, f"No forecast data found for brand {brand}.", structured_query.get("type", "invalid"))

            query_type = structured_query.get("type")
            operation = structured_query.get("operation", "none").lower()
            year = str(structured_query.get("year") or forecast_df["forecast_date"].dt.year.max())

            if query_type == "single":
                month = structured_query.get("month", "").lower()
                if month not in self.month_name_to_number:
                    return self._build_response(structured_query, {}, "Invalid month name.", query_type)
                month_num = self.month_name_to_number[month]
                formatted_date = f"{year}-{month_num}-01"
                result = forecast_df[forecast_df["forecast_date"] == formatted_date]
                if not result.empty:
                    forecast_value = round(float(result["forecast_sales"].iloc[0]), 2)
                    data = {"Date": [f"{year}-{self.month_number_to_name[month_num]}"], "Sales": [str(forecast_value)]}
                    return self._build_response(structured_query, data, self._get_random_response(), query_type)
                return self._build_response(structured_query, {}, "No forecast data found for this date.", query_type)

            elif query_type == "range":
                start_month = structured_query.get("start_month", "").lower()
                end_month = structured_query.get("end_month", "").lower()
                if start_month not in self.month_name_to_number or end_month not in self.month_name_to_number:
                    return self._build_response(structured_query, {}, "Invalid month name in range.", query_type)
                start_num = int(self.month_name_to_number[start_month])
                end_num = int(self.month_name_to_number[end_month])
                month_nums = ([f"{i:02d}" for i in range(start_num, 13)] + [f"{i:02d}" for i in range(1, end_num + 1)]) if start_num > end_num else [f"{i:02d}" for i in range(start_num, end_num + 1)]
                data = {"Date": [], "Sales": []}
                for month_num in month_nums:
                    formatted_date = f"{year}-{month_num}-01"
                    result = forecast_df[forecast_df["forecast_date"] == formatted_date]
                    if not result.empty:
                        data["Date"].append(f"{year}-{self.month_number_to_name[month_num]}")
                        data["Sales"].append(str(float(result["forecast_sales"].iloc[0])))
                desc = self._get_random_response() if data["Date"] else "No forecast data found for this range."
                if operation in ["total", "average"] and data["Sales"]:
                    data, desc = self.calculate(data, operation, query_type, brand, start_month, end_month)
                return self._build_response(structured_query, data, desc, query_type)

            elif query_type == "quarter":
                quarter = str(structured_query.get("quarter", ""))
                if quarter not in self.quarter_to_months:
                    return self._build_response(structured_query, {}, "Invalid quarter name.", query_type)
                month_nums = self.quarter_to_months[quarter]
                data = {"Date": [], "Sales": []}
                for month_num in month_nums:
                    formatted_date = f"{year}-{month_num}-01"
                    result = forecast_df[forecast_df["forecast_date"] == formatted_date]
                    if not result.empty:
                        data["Date"].append(f"{year}-{self.month_number_to_name[month_num]}")
                        data["Sales"].append(str(float(result["forecast_sales"].iloc[0])))
                desc = self._get_random_response() if data["Date"] else "No forecast data found for this quarter."
                if operation in ["total", "average"] and data["Sales"]:
                    data, desc = self.calculate(data, operation, query_type, brand, quarter_num=quarter)
                return self._build_response(structured_query, data, desc, query_type)

            return self._build_response(structured_query, {}, "Query not related to brand sales.", query_type)

        except Exception as e:
            logger.error(f"Error processing brand query: {str(e)}")
            return self._build_response(structured_query, {}, f"Error processing query: {str(e)}", structured_query.get("type", "invalid"))