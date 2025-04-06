import os
import json
import numpy as np
import random
import pandas as pd
import cx_Oracle
import nepali_datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.stattools import acf, pacf
import joblib
from src.utils.db_queries import SALES_QUERY
from src.utils.constants import MONTH_NUMBER_TO_NAME, DEFAULT_RESPONSE


class SalesForecastManager:
    def __init__(self, config_file="src/config/config.json", random_response="src/config/random_response.json",
                 months_mapping="src/config/months_mapping.json"):
        # Load configurations
        with open(config_file, "r") as f:
            config = json.load(f)
        self.db_config = config["databases"]
        
        # for random response as a description
        with open(random_response, "r") as f:
            self.random_responses = json.load(f)["responses"]
        
        # for month name to number and number to month
        with open(months_mapping, "r") as f:
            months = json.load(f)

        self.month_name_to_number = months["month_name_to_number"]
        self.month_number_to_name = months["month_number_to_name"]
        self.quarter_to_months = months["quarter_to_months"]

        self.month_name = months["month_name_to_number"]
        self.month_number = months["month_number_to_name"]
        self.quarter_month = months["quarter_to_months"]
 
        # define paths for datasets and models
        self.historical_data_path = r"data\dataset\total_data.csv"
        self.forecast_output_path = r"data\forecast_output\forecast_data.csv"
        self.model_path = r"models\random_forest_model.pkl"
        self.scaler_X_path = r"models\scaler_X.pkl"
        self.scaler_y_path = r"models\scaler_y.pkl"

        # Load forecast data if exists
        self.forecast_df = self._load_forecast_data()

    def _load_forecast_data(self):
        # It loads forecast data only if data exists
        if os.path.exists(self.forecast_output_path) and os.path.getsize(self.forecast_output_path) > 0:
            return pd.read_csv(self.forecast_output_path, parse_dates=["Date"])
        return pd.DataFrame(columns=["Date", "Sales"])

    def fetch_latest_data(self):
        #database connection
        all_data = []
        for db in self.db_config:
            dsn_tns = cx_Oracle.makedsn(db["host"], db["port"], service_name=db["service"])
            conn = cx_Oracle.connect(user=db["user"], password=db["password"], dsn=dsn_tns)
            df = pd.read_sql(SALES_QUERY, con=conn)
            conn.close()
            all_data.append(df)

        combined_df = pd.concat(all_data).groupby("BS_YEAR_MONTH").sum().reset_index()
        combined_df.columns = ["bs_year_month", "sales"]
        
        current_bs_date = nepali_datetime.date.today()
        current_bs_year_month = f"{current_bs_date.year:04d}-{current_bs_date.month:02d}"
        #dont take current month data as a input
        combined_df = combined_df[combined_df["bs_year_month"] < current_bs_year_month]
        combined_df.to_csv(self.historical_data_path, index=False)
        return combined_df

    def train_model(self):
        # It is a training pipeline
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

        acf_features=[]
        pacf_features=[]

        for i in range(1,len(acf_values)):
            if abs(acf_values[i])>0.2:
                index=i
                lag=f"lag_{index}"
                acf_features.append(lag)

        for i in range(1, len(pacf_values)):
            if abs(pacf_values[i])>0.2:
                index =i
                lag=f"lag_index"
                acf_features.append(lag)


        combined_acf_Pacf= (list(set(acf_features+pacf_features)))
  
        combined_acf_Pacf.remove("lag_index")

        lag_numbers= []

        for lag in combined_acf_Pacf:
            if lag.startswith('lag_') and lag != 'lag_index':  
                lag_num = int(lag.split('_')[1])  
                lag_numbers.append(lag_num)

                df[lag] = df['sales'].shift(lag_num)
        
        # sorting and calculatinglag index as 'lag_1','lag_2' ..... etc
        lag_numbers.sort()
        lags = []
        for lag in lag_numbers:
            format=f"lag_{lag}"
            lags.append(format)
        
        print("##############################################################################")
        print(lags)        
        print("##############################################################################")

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
        self.generate_forecast(lags, lag_numbers)

    def generate_forecast(self, lags, lag_numbers):
        # forecast data for 12 months
        df = pd.read_csv(self.historical_data_path, parse_dates=["bs_year_month"], index_col="bs_year_month").sort_index().dropna()
        model = joblib.load(self.model_path)
        scaler_X, scaler_y = joblib.load(self.scaler_X_path), joblib.load(self.scaler_y_path)

        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq="MS")
        future_df = pd.DataFrame(index=future_dates)
        future_df["year"], future_df["month"], future_df["quarter"] = future_df.index.year, future_df.index.month, future_df.index.quarter

        full_df = pd.concat([df, future_df])
        lags = lags
        lag_numbers = list(lag_numbers)
        
        lag_numbers.remove(1)
        print("**********************************************************")
        print(lag_numbers)
        print("***********************************************************")
        
        # required_features = ["year", "month", "quarter", "lag_1", "lag_11", "lag_12", "lag_13", "rolling_mean_3", "rolling_mean_6"]
        data_features = ['year', 'month', 'quarter']
        features= data_features + lags
        mean_features = ["rolling_mean_3", "rolling_mean_6"]
        required_features = features + mean_features

        forecast_df = full_df.copy()

        for i, date in enumerate(future_dates):
            if i == 0:
                forecast_df.loc[date, "lag_1"] = df["sales"].iloc[-1]
            else:
                forecast_df.loc[date, "lag_1"] = forecast_df.loc[future_dates[i - 1], "sales"]

            for offset in lag_numbers:
                forecast_df.loc[date, f"lag_{offset}"] = forecast_df.get("sales").shift(offset).iloc[-1] or forecast_df[f"lag_{offset}"].mean()
            
            available_data = forecast_df["sales"].dropna()
            for window in [3, 6]:
                if len(available_data) >= window:
                    forecast_df.loc[date, f"rolling_mean_{window}"] = available_data[-window:].mean()

            X = forecast_df.loc[date, required_features].values.reshape(1, -1)
            X_scaled = scaler_X.transform(X)
            prediction = scaler_y.inverse_transform(model.predict(X_scaled).reshape(-1, 1))[0][0]
            forecast_df.loc[date, "sales"] = prediction

        forecast_result = forecast_df.loc[future_dates, ["sales"]].reset_index()
        forecast_result.columns = ["Date", "Sales"]
        forecast_result.to_csv(self.forecast_output_path, index=False)
        self.forecast_df = forecast_result

    def calculate(self, data, operation, query_type, start_month=None, end_month=None, quarter_num=None):
        # function to calculate total, average for month range as well as quarter
      
        if operation in ["total","sum", "overall"]:
            total = round(sum(float(s) for s in data["Sales"]), 2)
            if query_type == "range":
                desc = f"Total sales form {start_month} to {end_month} is : {total}"
                data["Date"] = [f"{start_month}-{end_month} Total"]
            elif query_type=="quarter":
                desc = f"Total sales for Quarter {quarter_num} is : {total}"
                data["Date"] = [f"Quarter {quarter_num} Total"]

            data["Sales"] = [str(total)]
            
                
        elif operation in ["average", "mean", "avg"]:
            avg = round(np.mean([float(s) for s in data["Sales"]]), 2)
            desc = f"Average sales for Quarter {quarter_num}: {avg}"
            data["Sales"] = [str(avg)]
            data["Date"] = [f"Quarter {quarter_num} Average"]
        return data, desc


    def _build_response(self, structured_query, result_data, desc):
        response = DEFAULT_RESPONSE.copy()
        response["query"] = structured_query["original"]
        response["question"] = f"Forecasted sales for {structured_query['original']}?"
        response["graph_keys"] = [["Date", "Sales"]] if result_data else []
        response["desc"] = desc
        response["data"] = result_data

        return response

    def query_csv(self, structured_query):
        if self.forecast_df.empty:
            self.generate_forecast()
        
        month_name_to_number = self.month_name_to_number
        month_number_to_name = self.month_number_to_name
        quarter_to_months = self.quarter_to_months

        query_type = structured_query.get("type")
        operation = structured_query.get("operation", "none").lower()
        year = structured_query.get("year") or str(self.forecast_df["Date"].dt.year.max())

        if query_type == "single":
            month = structured_query["month"].lower()
            if month not in self.month_name_to_number:
                return self._build_response(structured_query, {}, "Invalid month name.")
            
            month_num = self.month_name_to_number[month]    
            formatted_date = f"{year}-{month_num}-01"
            result = self.forecast_df[self.forecast_df["Date"] == formatted_date]
            
            if not result.empty:
                forecast_value = round(float(result["Sales"].iloc[0]), 2)
                data = {"Date": [f"{year}-{self.month_number_to_name[month_num]}"], "Sales": [str(forecast_value)]}
                return self._build_response(structured_query, data, self._get_random_response())
            return self._build_response(structured_query, {}, "No forecast data found for this date.")

        elif query_type == "range":
            start_month, end_month = structured_query["start_month"].lower(), structured_query["end_month"].lower()
            if start_month not in self.month_name_to_number or end_month not in self.month_name_to_number:
                return self._build_response(structured_query, {}, "Invalid month name in range.")

            start_num, end_num = self.month_name_to_number[start_month], self.month_name_to_number[end_month]
            start_idx, end_idx = int(start_num), int(end_num)
            # month_nums = ([f"{i:02d}" for i in range(start_idx, 13)] + [f"{i:02d}" for i in range(1, end_idx + 1)]) if start_idx > end_idx else [f"{i:02d}" for i in range(start_idx, end_idx + 1)]
            if start_idx <= end_idx:
                month_nums = [f"{i:02d}" for i in range(start_idx, end_idx + 1)]
            else:
                month_nums = [f"{i:02d}" for i in range(start_idx, 13)] + [
                    f"{i:02d}" for i in range(1, end_idx + 1)
                ]

            data = {"Date": [], "Sales": []}
            for month_num in month_nums:
                formatted_date = f"{year}-{month_num}-01"
                result = self.forecast_df[self.forecast_df["Date"] == formatted_date]
                if not result.empty:
                    data["Date"].append(f"{year}-{self.month_number_to_name[month_num]}")
                    data["Sales"].append(str(float(result["Sales"].iloc[0])))

            desc = self._get_random_response() if data["Date"] else "No forecast data found for this range."
            if (operation == 'total' or operation == 'average') and data["Sales"]:
                
                data, desc = self.calculate(data, operation, query_type=query_type, start_month=start_month, end_month=end_month)
            return self._build_response(structured_query, data, desc)
        
        elif query_type == "quarter":
            quarter = str(structured_query["quarter"])
            year = structured_query["year"] or str(self.forecast_df["Date"].dt.year.max())
            if quarter not in quarter_to_months:
                return self._build_response(structured_query, {}, "Invalid quarter name.")
            
            # map months number from quarter number
            month_nums = quarter_to_months[quarter]
            data = {"Date":[], "Sales":[]}
            for month_num in month_nums:
                formatted_date = f"{year}-{month_num}-01"
                result = self.forecast_df[self.forecast_df["Date"]==formatted_date]
                if not result.empty:
                    forecast_value = float(result["Sales"].iloc[0])
                    data["Date"].append(f"{year}-{month_number_to_name[month_num]}")
                    data["Sales"].append(str(forecast_value))

                else:
                    desc = "No Forecast data found for this quarter" 
   

                desc = self._get_random_response()
                
                if (operation == 'total' or operation == 'average') and data["Sales"]:
                
                    data, desc = self.calculate(data, operation,  query_type=query_type, quarter_num= str(structured_query["quarter"]))        


            
       
            return self._build_response(structured_query, data, desc)
        return self._build_response(structured_query, {}, "Query not related to sales or upcoming sales.") 
               
        
    def _get_random_response(self):
        return random.choice(self.random_responses)
    
    

