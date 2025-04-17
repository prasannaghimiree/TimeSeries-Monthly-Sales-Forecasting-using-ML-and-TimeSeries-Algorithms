import os
import pandas as pd
import numpy as np
import nepali_datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.stattools import acf, pacf
import joblib
from src.utils.utils import BaseForecastManager, fetch_db_data
from src.utils.db_queries import SALES_QUERY

class SalesForecastManager(BaseForecastManager):
    def __init__(self):
        super().__init__()
        self.historical_data_path = r"data\dataset\total_data.csv"
        self.forecast_output_path = r"data\forecast_output\forecast_data.csv"
        self.model_path = r"models\random_forest_model.pkl"
        self.scaler_X_path = r"models\scaler_X.pkl"
        self.scaler_y_path = r"models\scaler_y.pkl"
        os.makedirs(os.path.dirname(self.historical_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.forecast_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.forecast_df = self._load_forecast_data()

    def _load_forecast_data(self):
        if os.path.exists(self.forecast_output_path) and os.path.getsize(self.forecast_output_path) > 0:
            return pd.read_csv(self.forecast_output_path, parse_dates=["Date"])
        return pd.DataFrame(columns=["Date", "Sales"])

    def fetch_latest_data(self):
        combined_df = fetch_db_data(self.db_config, SALES_QUERY)
        current_bs_date = nepali_datetime.date.today()
        current_bs_year_month = f"{current_bs_date.year:04d}-{current_bs_date.month:02d}"
        combined_df = combined_df[combined_df["bs_year_month"] < current_bs_year_month]
        combined_df.to_csv(self.historical_data_path, index=False)
        return combined_df

    def train_model(self):
        df = self.fetch_latest_data()
        df["bs_year_month"] = pd.to_datetime(df["bs_year_month"])
        df.set_index("bs_year_month", inplace=True)
        df.dropna(inplace=True)

        # Feature engineering
        df["year"] = df.index.year
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        acf_values = acf(df['sales'], nlags=13, fft=False)
        pacf_values = pacf(df["sales"], nlags=13)
        acf_features = [f"lag_{i}" for i in range(1, len(acf_values)) if abs(acf_values[i]) > 0.2]
        pacf_features = [f"lag_{i}" for i in range(1, len(pacf_values)) if abs(pacf_values[i]) > 0.2]
        combined_acf_pacf = list(set(acf_features + pacf_features))
        lags = sorted(combined_acf_pacf, key=lambda x: int(x.split("_")[1]))
        for lag in lags:
            lag_num = int(lag.split("_")[1])
            df[lag] = df['sales'].shift(lag_num)
        df["rolling_mean_3"] = df["sales"].rolling(window=3).mean()
        df["rolling_mean_6"] = df["sales"].rolling(window=6).mean()
        df.dropna(inplace=True)

        X = df.drop(columns=["sales"])
        y = df["sales"]
        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

        param_grid = {
            "n_estimators": [100, 250, 500], "max_depth": [3, 7, 10],
            "min_samples_split": [5, 10, 15], "min_samples_leaf": [3, 5, 10]
        }
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
        grid_search.fit(X_scaled, y_scaled)

        joblib.dump(grid_search.best_estimator_, self.model_path)
        joblib.dump(scaler_X, self.scaler_X_path)
        joblib.dump(scaler_y, self.scaler_y_path)
        self.generate_forecast(lags)

    def generate_forecast(self, lags):
        df = pd.read_csv(self.historical_data_path, parse_dates=["bs_year_month"], index_col="bs_year_month").sort_index().dropna()
        model = joblib.load(self.model_path)
        scaler_X, scaler_y = joblib.load(self.scaler_X_path), joblib.load(self.scaler_y_path)

        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq="MS")
        future_df = pd.DataFrame(index=future_dates)
        future_df["year"], future_df["month"], future_df["quarter"] = future_df.index.year, future_df.index.month, future_df.index.quarter
        full_df = pd.concat([df, future_df])
        features = ["year", "month", "quarter"] + lags + ["rolling_mean_3", "rolling_mean_6"]
        forecast_df = full_df.copy()

        for i, date in enumerate(future_dates):
            if i == 0:
                forecast_df.loc[date, "lag_1"] = df["sales"].iloc[-1]
            else:
                forecast_df.loc[date, "lag_1"] = forecast_df.loc[future_dates[i - 1], "sales"]
            for lag in lags:
                lag_num = int(lag.split("_")[1])
                if lag_num != 1:
                    forecast_df.loc[date, lag] = forecast_df.get("sales").shift(lag_num).iloc[-1] or forecast_df[lag].mean()
            available_data = forecast_df["sales"].dropna()
            for window in [3, 6]:
                if len(available_data) >= window:
                    forecast_df.loc[date, f"rolling_mean_{window}"] = available_data[-window:].mean()
            X = forecast_df.loc[date, features].values.reshape(1, -1)
            X_scaled = scaler_X.transform(X)
            prediction = scaler_y.inverse_transform(model.predict(X_scaled).reshape(-1, 1))[0][0]
            forecast_df.loc[date, "sales"] = prediction

        forecast_result = forecast_df.loc[future_dates, ["sales"]].reset_index()
        forecast_result.columns = ["Date", "Sales"]
        forecast_result.to_csv(self.forecast_output_path, index=False)
        self.forecast_df = forecast_result

    def query_csv(self, structured_query):
        if self.forecast_df.empty:
            self.train_model()
        
        query_type = structured_query.get("type")
        operation = structured_query.get("operation", "none").lower()
        year = structured_query.get("year") or str(self.forecast_df["Date"].dt.year.max())

        if query_type == "single":
            month = structured_query["month"].lower()
            if month not in self.month_name_to_number:
                return self._build_response(structured_query, {}, "Invalid month name.", query_type)
            month_num = self.month_name_to_number[month]
            formatted_date = f"{year}-{month_num}-01"
            result = self.forecast_df[self.forecast_df["Date"] == formatted_date]
            if not result.empty:
                forecast_value = round(float(result["Sales"].iloc[0]), 2)
                data = {"Date": [f"{year}-{self.month_number_to_name[month_num]}"], "Sales": [str(forecast_value)]}
                return self._build_response(structured_query, data, self._get_random_response(), query_type)
            return self._build_response(structured_query, {}, "No forecast data found for this date.", query_type)

        elif query_type == "range":
            start_month, end_month = structured_query["start_month"].lower(), structured_query["end_month"].lower()
            if start_month not in self.month_name_to_number or end_month not in self.month_name_to_number:
                return self._build_response(structured_query, {}, "Invalid month name in range.", query_type)
            start_num, end_num = self.month_name_to_number[start_month], self.month_name_to_number[end_month]
            start_idx, end_idx = int(start_num), int(end_num)
            month_nums = ([f"{i:02d}" for i in range(start_idx, 13)] + [f"{i:02d}" for i in range(1, end_idx + 1)]) if start_idx > end_idx else [f"{i:02d}" for i in range(start_idx, end_idx + 1)]
            data = {"Date": [], "Sales": []}
            for month_num in month_nums:
                formatted_date = f"{year}-{month_num}-01"
                result = self.forecast_df[self.forecast_df["Date"] == formatted_date]
                if not result.empty:
                    data["Date"].append(f"{year}-{self.month_number_to_name[month_num]}")
                    data["Sales"].append(str(float(result["Sales"].iloc[0])))
            desc = self._get_random_response() if data["Date"] else "No forecast data found for this range."
            if operation in ["total", "average"] and data["Sales"]:
                data, desc = self.calculate(data, operation, query_type, start_month=start_month, end_month=end_month)
            return self._build_response(structured_query, data, desc, query_type)

        elif query_type == "quarter":
            quarter = str(structured_query["quarter"])
            if quarter not in self.quarter_to_months:
                return self._build_response(structured_query, {}, "Invalid quarter name.", query_type)
            month_nums = self.quarter_to_months[quarter]
            data = {"Date": [], "Sales": []}
            for month_num in month_nums:
                formatted_date = f"{year}-{month_num}-01"
                result = self.forecast_df[self.forecast_df["Date"] == formatted_date]
                if not result.empty:
                    data["Date"].append(f"{year}-{self.month_number_to_name[month_num]}")
                    data["Sales"].append(str(float(result["Sales"].iloc[0])))
            desc = self._get_random_response() if data["Date"] else "No forecast data found for this quarter."
            if operation in ["total", "average"] and data["Sales"]:
                data, desc = self.calculate(data, operation, query_type, quarter_num=quarter)
            return self._build_response(structured_query, data, desc, query_type)
        
        return self._build_response(structured_query, {}, "Query not related to sales.", query_type)