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


load_dotenv()

USER7778 = os.getenv("USER7778")
USER7879 = os.getenv("USER7879")
USER7980 = os.getenv("USER7980")
USER8081 = os.getenv("USER8081")
USER8182 = os.getenv("USER8182")

PASS7778 = os.getenv("PASS7778")
PASS7879 = os.getenv("PASS7879")
PASS7980 = os.getenv("PASS7980")
PASS8081 = os.getenv("PASS8081")
PASS8182 = os.getenv("PASS8182")

SER_HOST = os.getenv('SER_HOST')
LOCAL_HOST = os.getenv('LOCAL_HOST')

PORT = os.getenv("PORT")

SER_SERVICE = os.getenv("SER_SERVICE")
LOCAL_SERVICE = os.getenv("LOCAL_SERVICE")

class SalesForecastManager:
    def __init__(self):
        self.DB_USERS = [USER7778, USER7879, USER7980, USER8081, USER8182]
        self.DB_PASSES = [PASS7778, PASS7879, PASS7980, PASS8081, PASS8182]
        self.HOSTS = [SER_HOST, LOCAL_HOST, LOCAL_HOST, LOCAL_HOST, LOCAL_HOST]
        self.PORTS = [PORT, PORT, PORT, PORT, PORT]
        self.SERVICES = [SER_SERVICE, LOCAL_SERVICE, LOCAL_SERVICE, LOCAL_SERVICE, LOCAL_SERVICE]
        
        self.historical_data_path = r"Dataset\total_data.csv"
        self.forecast_output_path = r"forecast_output\forecast_data.csv"
        self.model_path = r"models\random_forest_model.pkl"
        self.scaler_X_path = r"models\scaler_X.pkl"
        self.scaler_y_path = r"models\scaler_y.pkl"

        if os.path.exists(self.forecast_output_path) and os.path.getsize(self.forecast_output_path) > 0:
            try:
                self.forecast_df = pd.read_csv(self.forecast_output_path, parse_dates=["forecast_date"])
            except pd.errors.EmptyDataError:
                self.forecast_df = pd.DataFrame(columns=["forecast_date", "forecast_sales"])
        else:
            self.forecast_df = pd.DataFrame(columns=["forecast_date", "forecast_sales"])

    def fetch_latest_data(self):
        # This function Fetch data every single time model is trained. This function is called by train_model function.
        """Fetch data from all databases and combine into a single as a historical dataset from FY2077 - Present."""
        all_data = []
        for user, password, host, port, service in zip(self.DB_USERS, self.DB_PASSES, self.HOSTS, self.PORTS, self.SERVICES):
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
        
        combined_df = combined_df.groupby("bs_year_month").sum().reset_index()
        
        current_nepali_date = nepali_datetime.date.today()
        current_bs_year_month = f"{current_nepali_date.year:04d}-{current_nepali_date.month:02d}"
        combined_df = combined_df[combined_df["bs_year_month"] < current_bs_year_month] 
        combined_df.to_csv(self.historical_data_path, index=False)
        return combined_df

    def train_model(self):
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
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
        grid_search.fit(X_scaled, y_scaled)

        joblib.dump(grid_search.best_estimator_, self.model_path)
        joblib.dump(scaler_X, self.scaler_X_path)
        joblib.dump(scaler_y, self.scaler_y_path)

        self.generate_forecast()

    def generate_forecast(self):
        df = pd.read_csv(self.historical_data_path, parse_dates=["bs_year_month"], index_col="bs_year_month")
        df = df.sort_index().dropna()

        model = joblib.load(self.model_path)
        scaler_X = joblib.load(self.scaler_X_path)
        scaler_y = joblib.load(self.scaler_y_path)

        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='MS')

        future_df = pd.DataFrame(index=future_dates)
        future_df['year'] = future_df.index.year
        future_df['month'] = future_df.index.month
        future_df['quarter'] = future_df.index.quarter

        full_df = pd.concat([df, future_df])
        required_features = ['year', 'month', 'quarter', 'lag_1', 'lag_11', 'lag_12', 'lag_13', 
                            'rolling_mean_3', 'rolling_mean_6']

        forecast_df = full_df.copy()
        forecast_df.loc[future_dates[0], 'lag_1'] = df['sales'].iloc[-1]

        for i, date in enumerate(future_dates):
            if i > 0:
                forecast_df.loc[date, 'lag_1'] = forecast_df.loc[future_dates[i - 1], 'sales']

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

        forecast_result = forecast_df.loc[future_dates, ['sales']].reset_index()
        forecast_result.columns = ['forecast_date', 'forecast_sales']
        forecast_result.to_csv(self.forecast_output_path, index=False)
        self.forecast_df = forecast_result

    def query_csv(self, query):
        if self.forecast_df.empty:
            self.generate_forecast()

        query = query.lower().strip()
        nepali_months = {
            "baisakh": "01", "baisak": "01", "baishakh": "01", "baishak": "01","baaisakh":"01",
            "jestha": "02", "jeshtha": "02", "jesth": "02", "jeth": "02",
            "asadh": "03", "asad": "03", "ashad": "03", "ashadh": "03", "asaadh": "03", "asar": "03",
            "shrawan": "04", "saun": "04", "shawan": "04", "sawan": "04", "shraban": "04",
            "bhadra": "05", "bhadau": "05", "bhadaw": "05", "bhad": "05",
            "asoj": "06", "ashoj": "06", "ashwin": "06","ashoz":"06",
            "kartik": "07", "kattik": "07","katik":"07","kaatik":"07",
            "mangsir": "08", "mangshir": "08", "mansir": "08","mangser":"08", "mangseer":"08",
            "poush": "09", "push": "09", "pous": "09", "posh": "09",
            "magh": "10", "mag": "10","maag":"10","marga":"10",
            "falgun": "11", "phalgun": "11", "falgoon": "11", "fagun": "11","flagun":"11",
            "chaitra": "12", "chait": "12"
            }
        month_number_to_name = {
            "01": "Baisakh", "02": "Jestha", "03": "Ashad", "04": "Shrawan", "05": "Bhadra",
            "06": "Ashoj", "07": "Kartik", "08": "Mangsir", "09": "Poush", "10": "Magh",
            "11": "Falgun", "12": "Chaitra"
        }

        # Check if it's a range query (e.g., "baisakh to shrawan" or "2082 baisakh to shrawan")
        range_match = re.search(r"(\d{4}\s+)?(\w+)\s+to\s+(\w+)", query)
        single_match = re.search(r"(\d{4})[-\s]?(\d{2}|\w+)", query)

        if range_match:
            year, start_month, end_month = range_match.groups()
            year = year.strip() if year else self.forecast_df["forecast_date"].max().year  # Default to latest year if not specified
            start_idx = list(nepali_months.keys()).index(start_month)
            end_idx = list(nepali_months.keys()).index(end_month)
            
            if start_idx > end_idx:  # Handle wrap-around (e.g., Poush to Jestha)
                months = list(nepali_months.keys())[start_idx:] + list(nepali_months.keys())[:end_idx + 1]
            else:
                months = list(nepali_months.keys())[start_idx:end_idx + 1]

            for_data = []
            for month in months:
                month_num = nepali_months[month]
                formatted_date = f"{year}-{month_number_to_name[month_num]}"
                result = self.forecast_df[self.forecast_df["forecast_date"] == f"{year}-{month_num}-01"]
                if not result.empty:
                    forecast_value = float(result['forecast_sales'].values[0])
                    for_data.append({"forecast_date": formatted_date, "forecast_sales": forecast_value})

            desc = self.get_random_response() if for_data else "No forecast data found for this range."
            return {
                "query": query,
                "question": f"Forecasted sales from {year}-{month_number_to_name[nepali_months[start_month]]} to {year}-{month_number_to_name[nepali_months[end_month]]}?",
                "graph_keys": [["forecast_date", "forecast_sales"]],
                "desc": desc,
                "for_data": for_data
            }
        elif single_match:
            year, month = single_match.groups()
            if month in nepali_months:
                month = nepali_months[month]
            month = month.zfill(2)
            result = self.forecast_df[self.forecast_df["forecast_date"] == f"{year}-{month}-01"]

            if not result.empty:
                forecast_value = float(result['forecast_sales'].values[0])
                formatted_date = f"{year}-{month_number_to_name[month]}"
                desc = self.get_random_response()
                return {
                    "query": query,
                    "question": f"Forecasted sales for {formatted_date}?",
                    "graph_keys": [["forecast_date", "forecast_sales"]],
                    "desc": desc,
                    "for_data": [{"forecast_date": formatted_date, "forecast_sales": forecast_value}]
                }
            else:
                formatted_date = f"{year}-{month_number_to_name[month]}"
                return {
                    "query": query,
                    "question": f"Forecasted sales for {formatted_date}?",
                    "graph_keys": [],
                    "desc": "No forecast data found for this date.",
                    "for_data": []
                }
        return {
            "query": query,
            "question": "Sorry, I couldn't understand the query.",
            "graph_keys": [],
            "desc": "Could not process the request.",
            "for_data": []
        }

    def get_random_response(self):
        responses = [
        "Here's your freshly compiled list.",
        "Your requested summary is ready below.",
        "The latest data you asked for is now available.",
        "Here's the detailed response you requested.",
        "Your list has been generated and is displayed below.",
        "We've put together your summary; check it out below.",
        "Below is the response we've prepared for you.",
        "Your results are in! See your data below.",
        "The analyzed data is ready for you; find it below.",
        "Here's the summary you asked for, ready and waiting below.",
        "Your list is ready! Check out the details below.",
        "We've gathered the information you needed; view your summary below.",
        "Here's your customized list, freshly prepared.",
        "The response you requested is now available below.",
        "Your latest data insights are ready; see them below.",
        "Below is the detailed summary you've been waiting for.",
        "We've generated the data you needed; review it below.",
        "Your detailed list is prepared and ready for you.",
        "Find your compiled response below, ready for review.",
        "The results you requested are now available below."
        ]
        return random.choice(responses)