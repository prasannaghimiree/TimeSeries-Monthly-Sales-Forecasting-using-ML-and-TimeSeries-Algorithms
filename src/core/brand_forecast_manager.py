# src/core/brand_forecast_manager.py
import os
import json
import numpy as np
import random
import pandas as pd
import cx_Oracle
import nepali_datetime
from src.utils.db_queries import BRAND_SALES_QUERY
from src.utils.constants import DEFAULT_RESPONSE
from src.train_model.brand_trainer import BrandTrainer

class BrandSalesForecastManager:
    def __init__(self, config_file="src/config/config.json", months_mapping="src/config/months_mapping.json"):
        # Load configurations
        with open(config_file, "r") as f:
            config = json.load(f)
        self.db_config = config["databases"]

        # Load month mappings
        with open(months_mapping, "r") as f:
            months = json.load(f)
        self.month_name_to_number = months["month_name_to_number"]
        self.month_number_to_name = months["month_number_to_name"]
        self.quarter_to_months = months["quarter_to_months"]

        # Define paths
        self.historical_data_path = r"data\dataset\historical_sales_latest_12_months_brands.csv"
        self.model_path_prefix = r"models\brand_model_"
        self.forecast_path_prefix = r"brand_dataset\forecast-"
        os.makedirs("data/dataset", exist_ok=True)
        os.makedirs("brand_dataset", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        # Load random responses
        with open("src/config/random_response.json", "r") as f:
            self.random_responses = json.load(f)["responses"]

        # Initialize forecast cache and trainer
        self.forecast_cache = {}
        self.brands = self._fetch_and_prepare_data()
        self.trainer = BrandTrainer(self.historical_data_path, self.model_path_prefix, self.forecast_path_prefix)

    def _fetch_and_prepare_data(self):
        """Fetch brand-wise sales data, filter for latest 12 months brands, and prepare pivoted historical data."""
        all_data = []
        for db in self.db_config:
            dsn_tns = cx_Oracle.makedsn(db["host"], db["port"], service_name=db["service"])
            conn = cx_Oracle.connect(user=db["user"], password=db["password"], dsn=dsn_tns)
            df = pd.read_sql(BRAND_SALES_QUERY, con=conn)
            conn.close()
            all_data.append(df)

        # Combine data from all databases
        combined_df = pd.concat(all_data).groupby(["BS_YEAR_MONTH", "BRAND_NAME"]).sum().reset_index()
        combined_df.columns = ["BS_YEAR_MONTH", "BRAND_NAME", "SALES_VALUE"]

        # Convert BS_YEAR_MONTH to datetime
        combined_df["BS_YEAR_MONTH"] = pd.to_datetime(combined_df["BS_YEAR_MONTH"], format="%Y-%m")

        # Exclude current month
        current_bs_date = nepali_datetime.date.today()
        current_bs_year_month = f"{current_bs_date.year:04d}-{current_bs_date.month:02d}"
        combined_df = combined_df[combined_df["BS_YEAR_MONTH"] < current_bs_year_month]

        # Get the latest 12 months of data
        latest_date = combined_df["BS_YEAR_MONTH"].max()
        one_year_ago = latest_date - pd.DateOffset(months=12)
        latest_12_months_data = combined_df[combined_df["BS_YEAR_MONTH"] >= one_year_ago]

        # Identify unique brands active in the last 12 months
        unique_brands = latest_12_months_data["BRAND_NAME"].unique()

        # Filter historical data to include only these brands
        historical_data = combined_df[combined_df["BRAND_NAME"].isin(unique_brands)]

        # Create a complete date range from min to max BS_YEAR_MONTH
        all_months = pd.date_range(start=combined_df["BS_YEAR_MONTH"].min(),
                                   end=combined_df["BS_YEAR_MONTH"].max(),
                                   freq="MS")

        # Pivot the data with brands as columns
        pivoted_data = historical_data.pivot_table(index="BS_YEAR_MONTH",
                                                   columns="BRAND_NAME",
                                                   values="SALES_VALUE",
                                                   aggfunc="sum",
                                                   fill_value=0)

        # Reindex to include all months, filling missing values with 0
        pivoted_data = pivoted_data.reindex(all_months, fill_value=0)

        # Reset index and explicitly name the column as BS_YEAR_MONTH
        pivoted_data = pivoted_data.reset_index()
        pivoted_data = pivoted_data.rename(columns={"index": "BS_YEAR_MONTH"})  # Ensure column name is set correctly

        # Save the pivoted data with BS_YEAR_MONTH as a column
        pivoted_data.to_csv(self.historical_data_path, index=False)
        print(f"New dataset created and saved as '{self.historical_data_path}'")

        return unique_brands.tolist()

    def train_all_brands(self):
        """Train models and generate forecasts for all brands."""
        for brand in self.brands:
            self.trainer.train_and_forecast(brand)

    def _load_forecast_data(self, brand):
        """Load forecast data for a specific brand, training if necessary."""
        forecast_path = f"{self.forecast_path_prefix}{brand}.csv"
        if not os.path.exists(forecast_path):
            self.trainer.train_and_forecast(brand)
        if os.path.exists(forecast_path) and os.path.getsize(forecast_path) > 0:
            return pd.read_csv(forecast_path, parse_dates=["forecast_date"])
        return pd.DataFrame(columns=["forecast_date", "forecast_sales"])

    def calculate(self, data, operation, query_type, brand, start_month=None, end_month=None, quarter_num=None):
        """Calculate total or average for a brand's sales data."""
        if operation in ["total", "sum", "overall"]:
            total = round(sum(float(s) for s in data["Sales"]), 2)
            if query_type == "range":
                desc = f"Total sales for {brand} from {start_month} to {end_month} is: {total}"
                data["Date"] = [f"{start_month}-{end_month} Total"]
            elif query_type == "quarter":
                desc = f"Total sales for {brand} in Quarter {quarter_num} is: {total}"
                data["Date"] = [f"Quarter {quarter_num} Total"]
            data["Sales"] = [str(total)]
        elif operation in ["average", "mean", "avg"]:
            avg = round(np.mean([float(s) for s in data["Sales"]]), 2)
            if query_type == "range":
                desc = f"Average sales for {brand} from {start_month} to {end_month} is: {avg}"
                data["Date"] = [f"{start_month}-{end_month} Average"]
            elif query_type == "quarter":
                desc = f"Average sales for {brand} in Quarter {quarter_num} is: {avg}"
                data["Date"] = [f"Quarter {quarter_num} Average"]
            data["Sales"] = [str(avg)]
        return data, desc

    def _build_response(self, structured_query, result_data, desc):
        """Build the response structure."""
        response = DEFAULT_RESPONSE.copy()
        response["query"] = structured_query["original"]
        response["question"] = f"Forecasted sales for {structured_query['original']}?"
        response["graph_keys"] = [["Date", "Sales"]] if result_data else []
        response["desc"] = desc
        response["data"] = result_data
        return response

    def query_csv(self, structured_query):
        """Process the structured query for brand-specific forecasts."""
        brand = structured_query.get("brand")
        if not brand or brand not in self.brands:
            return self._build_response(structured_query, {}, f"Invalid or unknown brand: {brand}")

        # Load forecast data for the brand (trains if not present)
        if brand not in self.forecast_cache:
            self.forecast_cache[brand] = self._load_forecast_data(brand)
        forecast_df = self.forecast_cache[brand]

        if forecast_df.empty:
            return self._build_response(structured_query, {}, f"No forecast data found for brand {brand}.")

        query_type = structured_query.get("type")
        operation = structured_query.get("operation", "none").lower()
        year = structured_query.get("year") or str(forecast_df["forecast_date"].dt.year.max())

        if query_type == "single":
            month = structured_query["month"].lower()
            if month not in self.month_name_to_number:
                return self._build_response(structured_query, {}, "Invalid month name.")
            
            month_num = self.month_name_to_number[month]
            formatted_date = f"{year}-{month_num}-01"
            result = forecast_df[forecast_df["forecast_date"] == formatted_date]
            
            if not result.empty:
                forecast_value = round(float(result["forecast_sales"].iloc[0]), 2)
                data = {"Date": [f"{year}-{self.month_number_to_name[month_num]}"], "Sales": [str(forecast_value)]}
                return self._build_response(structured_query, data, self._get_random_response())
            return self._build_response(structured_query, {}, "No forecast data found for this date.")

        elif query_type == "range":
            start_month = structured_query["start_month"].lower()
            end_month = structured_query["end_month"].lower()
            if start_month not in self.month_name_to_number or end_month not in self.month_name_to_number:
                return self._build_response(structured_query, {}, "Invalid month name in range.")

            start_num = self.month_name_to_number[start_month]
            end_num = self.month_name_to_number[end_month]
            start_idx, end_idx = int(start_num), int(end_num)
            if start_idx <= end_idx:
                month_nums = [f"{i:02d}" for i in range(start_idx, end_idx + 1)]
            else:
                month_nums = [f"{i:02d}" for i in range(start_idx, 13)] + [f"{i:02d}" for i in range(1, end_idx + 1)]

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
            return self._build_response(structured_query, data, desc)

        elif query_type == "quarter":
            quarter = str(structured_query["quarter"])
            if quarter not in self.quarter_to_months:
                return self._build_response(structured_query, {}, "Invalid quarter name.")
            
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
            return self._build_response(structured_query, data, desc)

        return self._build_response(structured_query, {}, "Query not related to brand sales.")

    def _get_random_response(self):
        """Return a random response description."""
        return random.choice(self.random_responses)