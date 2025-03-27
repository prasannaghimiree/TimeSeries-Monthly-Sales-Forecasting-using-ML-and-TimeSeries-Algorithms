import os
import cx_Oracle
import pandas as pd
from dotenv import load_dotenv
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.stattools import acf, pacf
import re
import random
import nepali_datetime
import json


class SalesForecastManager:
    def __init__(
        self,
        config_file="config.json",
        random_response="random_response.json",
        months_mapping="months_mapping.json",
    ):
        # loading database credentials from config.json
        self.load_config = config_file
        with open(config_file, "r") as file:
            config = json.load(file)

        self.DB_USERS = [db["user"] for db in config["databases"]]
        self.DB_PASSES = [db["password"] for db in config["databases"]]
        self.HOSTS = [db["host"] for db in config["databases"]]
        self.PORTS = [db["port"] for db in config["databases"]]
        self.SERVICES = [db["service"] for db in config["databases"]]

        # loading random responses from random_response.json
        self.load_random = random_response
        with open(random_response, "r") as file:
            random = json.load(file)
        self.random_response = random["responses"]

        # loading month details from months_mapping.json
        self.load_months = months_mapping
        with open(months_mapping, "r") as file:
            month_details = json.load(file)

        self.month_name_to_number = month_details["month_name_to_number"]
        self.month_number_to_name = month_details["month_number_to_name"]

        self.historical_data_path = r"Dataset\total_data.csv"
        self.forecast_output_path = r"forecast_output\forecast_data.csv"
        self.model_path = r"models\random_forest_model.pkl"
        self.scaler_X_path = r"models\scaler_X.pkl"
        self.scaler_y_path = r"models\scaler_y.pkl"

        if (
            os.path.exists(self.forecast_output_path)
            and os.path.getsize(self.forecast_output_path) > 0
        ):
            try:
                self.forecast_df = pd.read_csv(
                    self.forecast_output_path, parse_dates=["forecast_date"]
                )
            except pd.errors.EmptyDataError:
                self.forecast_df = pd.DataFrame(
                    columns=["forecast_date", "forecast_sales"]
                )
        else:
            self.forecast_df = pd.DataFrame(columns=["forecast_date", "forecast_sales"])

    def fetch_latest_data(self):
        # funtion to fetch database through select query
        all_data = []
        for user, password, host, port, service in zip(
            self.DB_USERS, self.DB_PASSES, self.HOSTS, self.PORTS, self.SERVICES
        ):
            dsn_tns = cx_Oracle.makedsn(host, port, service_name=service)
            connection = cx_Oracle.connect(user=user, password=password, dsn=dsn_tns)
            query = """
            SELECT 
                SUBSTR(BS_DATE(SALES_DATE), 1, 7) AS BS_YEAR_MONTH, 
                SUM(NVL(QUANTITY * NET_GROSS_RATE, 0)) AS SALES_VALUE
            FROM SA_SALES_INVOICE
            WHERE DELETED_FLAG = 'N' 
            AND COMPANY_CODE IN ('06', '0')   
            GROUP BY SUBSTR(BS_DATE(SALES_DATE), 1, 7)
            ORDER BY 1
            """
            df = pd.read_sql(query, con=connection)
            connection.close()
            all_data.append(df)

        combined_df = pd.concat(all_data)
        combined_df = combined_df.groupby("BS_YEAR_MONTH").sum().reset_index()
        combined_df.columns = ["bs_year_month", "sales"]

        current_nepali_date = nepali_datetime.date.today()
        current_bs_year_month = (
            f"{current_nepali_date.year:04d}-{current_nepali_date.month:02d}"
        )
        # donot take data from current month because it is still running and is not completed
        combined_df = combined_df[combined_df["bs_year_month"] < current_bs_year_month]
        # combined_df is the dataframe that that contains date until previous month but not present month
        combined_df.to_csv(self.historical_data_path, index=False)
        return combined_df

    def train_model(self):
        # this function is called either if there is no any model or at starting ofevery month. It always fetch the latest data, directly from the database
        df = self.fetch_latest_data()
        df["bs_year_month"] = pd.to_datetime(df["bs_year_month"])
        df.set_index("bs_year_month", inplace=True)
        df.dropna(inplace=True)

        df["year"] = df.index.year
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter

        acf_values = acf(df["sales"], nlags=13, fft=False)
        pacf_values = pacf(df["sales"], nlags=13)
        lags = [1, 11, 12, 13]

        for lag in lags:
            df[f"lag_{lag}"] = df["sales"].shift(lag)
        df["rolling_mean_3"] = df["sales"].rolling(window=3).mean()
        df["rolling_mean_6"] = df["sales"].rolling(window=6).mean()

        df.dropna(inplace=True)

        X = df.drop(columns=["sales"])
        y = df["sales"]

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

        param_grid = {
            "n_estimators": [100, 250, 500],
            "max_depth": [3, 7, 10],
            "min_samples_split": [5, 10, 15],
            "min_samples_leaf": [3, 5, 10],
        }
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1
        )
        grid_search.fit(X_scaled, y_scaled)

        joblib.dump(grid_search.best_estimator_, self.model_path)
        joblib.dump(scaler_X, self.scaler_X_path)
        joblib.dump(scaler_y, self.scaler_y_path)

        self.generate_forecast()

    def generate_forecast(self):
        # function to generate forecast for 6 months.
        df = pd.read_csv(
            self.historical_data_path,
            parse_dates=["bs_year_month"],
            index_col="bs_year_month",
        )
        df = df.sort_index().dropna()

        model = joblib.load(self.model_path)
        scaler_X = joblib.load(self.scaler_X_path)
        scaler_y = joblib.load(self.scaler_y_path)

        last_date = df.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1), periods=6, freq="MS"
        )

        future_df = pd.DataFrame(index=future_dates)
        future_df["year"] = future_df.index.year
        future_df["month"] = future_df.index.month
        future_df["quarter"] = future_df.index.quarter

        full_df = pd.concat([df, future_df])
        required_features = [
            "year",
            "month",
            "quarter",
            "lag_1",
            "lag_11",
            "lag_12",
            "lag_13",
            "rolling_mean_3",
            "rolling_mean_6",
        ]

        forecast_df = full_df.copy()
        forecast_df.loc[future_dates[0], "lag_1"] = df["sales"].iloc[-1]

        for i, date in enumerate(future_dates):
            if i > 0:
                forecast_df.loc[date, "lag_1"] = forecast_df.loc[
                    future_dates[i - 1], "sales"
                ]

            for offset in [11, 12, 13]:
                try:
                    forecast_df.loc[date, f"lag_{offset}"] = forecast_df.loc[
                        date - pd.DateOffset(months=offset), "sales"
                    ]
                except KeyError:
                    forecast_df.loc[date, f"lag_{offset}"] = forecast_df[
                        f"lag_{offset}"
                    ].mean()

            available_data = forecast_df["sales"].dropna()
            for window in [3, 6]:
                if len(available_data) >= window:
                    forecast_df.loc[date, f"rolling_mean_{window}"] = available_data[
                        -window:
                    ].mean()

            X = forecast_df.loc[date, required_features].values.reshape(1, -1)
            X_scaled = scaler_X.transform(X)
            prediction = scaler_y.inverse_transform(
                model.predict(X_scaled).reshape(-1, 1)
            )
            forecast_df.loc[date, "sales"] = prediction[0][0]

        forecast_result = forecast_df.loc[future_dates, ["sales"]].reset_index()
        forecast_result.columns = ["forecast_date", "forecast_sales"]
        forecast_result.to_csv(self.forecast_output_path, index=False)
        self.forecast_df = forecast_result

    def query_csv(self, structured_query):
        """Fetch data based on structured query from LLM."""
        # if forecasted values are not present then forecast the values at first
        if self.forecast_df.empty:
            self.generate_forecast()

        # loading month conversion dictionary from months_mapping.py
        month_name_to_number = self.month_name_to_number
        month_number_to_name = self.month_number_to_name

        # for single query type. i.e. if forecasted value is asked only for a paticular month
        query_type = structured_query.get("type")
        if query_type == "single":
            month = structured_query["month"].lower()
            year = structured_query["year"] or str(
                self.forecast_df["forecast_date"].dt.year.max()
            )
            # check if month present in nepalu_month dictionary
            if month not in month_name_to_number:
                return {
                    "query": structured_query["original"],
                    "question": f"Forecasted sales for {structured_query['original']}?",
                    "graph_keys": [],
                    "desc": "Invalid month name.",
                    "for_data": [],
                }
            month_num = month_name_to_number[month]
            formatted_date = f"{year}-{month_num}-01"
            result = self.forecast_df[
                self.forecast_df["forecast_date"] == formatted_date
            ]

            if not result.empty:
                forecast_value = float(result["forecast_sales"].iloc[0])
                desc = self.get_random_response()
                return {
                    "query": structured_query["original"],
                    "question": f"Forecasted sales for {year}-{month_number_to_name[month_num]}?",
                    "graph_keys": [["forecast_date", "forecast_sales"]],
                    "desc": desc,
                    "for_data": [
                        {
                            "forecast_date": f"{year}-{month_number_to_name[month_num]}",
                            "forecast_sales": forecast_value,
                        }
                    ],
                }
            else:
                return {
                    "query": structured_query["original"],
                    "question": f"Forecasted sales for {year}-{month_number_to_name[month_num]}?",
                    "graph_keys": [],
                    "desc": "No forecast data found for this date.",
                    "for_data": [],
                }

        elif query_type == "range":
            # for range type query like : give me the forecast from baisakh to shrawan
            start_month = structured_query["start_month"].lower()
            end_month = structured_query["end_month"].lower()
            year = structured_query["year"] or str(
                self.forecast_df["forecast_date"].dt.year.max()
            )

            if (
                start_month not in month_name_to_number
                or end_month not in month_name_to_number
            ):
                return {
                    "query": structured_query["original"],
                    "question": f"Forecasted sales for {structured_query['original']}?",
                    "graph_keys": [],
                    "desc": "Invalid month name in range.",
                    "for_data": [],
                }

            start_num = month_name_to_number[start_month]
            end_num = month_name_to_number[end_month]
            start_idx = int(start_num)
            end_idx = int(end_num)

            # if ask range from same year or start range is less then end
            if start_idx <= end_idx:
                month_nums = [f"{i:02d}" for i in range(start_idx, end_idx + 1)]
            else:
                # for eg: start month greater then end month like start month = falgun(11) and end is shrawan(04), then months be (11,12)+ (1,2,3,4) = (11,12,1,2,3,4)
                month_nums = [f"{i:02d}" for i in range(start_idx, 13)] + [
                    f"{i:02d}" for i in range(1, end_idx + 1)
                ]

            for_data = []
            for month_num in month_nums:
                formatted_date = f"{year}-{month_num}-01"
                result = self.forecast_df[
                    self.forecast_df["forecast_date"] == formatted_date
                ]
                if not result.empty:
                    forecast_value = float(result["forecast_sales"].iloc[0])
                    for_data.append(
                        {
                            "forecast_date": f"{year}-{month_number_to_name[month_num]}",
                            "forecast_sales": forecast_value,
                        }
                    )

            desc = (
                self.get_random_response()
                if for_data
                else "No forecast data found for this range."
            )
            desc = self.get_random_response() if for_data else "Forecast not found"
            return {
                "query": structured_query["original"],
                "question": f"Forecasted sales from {year}-{month_number_to_name[start_num]} to {year}-{month_number_to_name[end_num]}?",
                "graph_keys": [["forecast_date", "forecast_sales"]],
                "desc": desc,
                "for_data": for_data,
            }

        return {
            "query": structured_query["original"],
            "question": f"Forecasted sales for {structured_query['original']}?",
            "graph_keys": [],
            "desc": "Query not related to sales or upcoming sales.",
            "for_data": [],
        }

    def get_random_response(self):

        return random.choice(self.random_response)
