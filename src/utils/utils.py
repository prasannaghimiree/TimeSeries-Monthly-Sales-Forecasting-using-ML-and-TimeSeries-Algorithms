import os
import json
import random
import pandas as pd
import numpy as np
import cx_Oracle
from src.utils.constants import DEFAULT_RESPONSE
from src.utils.db_queries import SALES_QUERY, BRAND_SALES_QUERY

class BaseForecastManager:
    def __init__(self, config_file="src/config/config.json", months_mapping="src/config/months_mapping.json",
                 random_response="src/config/random_response.json"):
        # Loading configurations
        with open(config_file, "r") as f:
            config = json.load(f)
        self.db_config = config["databases"]

        with open(months_mapping, "r") as f:
            months = json.load(f)
        self.month_name_to_number = months["month_name_to_number"]
        self.month_number_to_name = months["month_number_to_name"]
        self.quarter_to_months = months["quarter_to_months"]

        with open(random_response, "r") as f:
            self.random_responses = json.load(f)["responses"]

    def calculate(self, data, operation, query_type, brand=None, start_month=None, end_month=None, quarter_num=None):
        """Calculate total or average for sales data for rangeand quarter type"""
        if operation in ["total", "sum", "overall"]:
            total = round(sum(float(s) for s in data["Sales"]), 2)

            if query_type == "range":
                desc = f"Total sales{' for ' + brand if brand else ''} from {start_month} to {end_month} is: {total}"
                data["Date"] = [f"{start_month}-{end_month} Total"]
            elif query_type == "quarter":
                desc = f"Total sales{' for ' + brand if brand else ''} in Quarter {quarter_num} is: {total}"
                data["Date"] = [f"Quarter {quarter_num} Total"]
            data["Sales"] = [str(total)]

        elif operation in ["average", "mean", "avg"]:
            avg = round(np.mean([float(s) for s in data["Sales"]]), 2)

            if query_type == "range":
                desc = f"Average sales{' for ' + brand if brand else ''} from {start_month} to {end_month} is: {avg}"
                data["Date"] = [f"{start_month}-{end_month} Average"]

            elif query_type == "quarter":
                desc = f"Average sales{' for ' + brand if brand else ''} in Quarter {quarter_num} is: {avg}"
                data["Date"] = [f"Quarter {quarter_num} Average"]
            data["Sales"] = [str(avg)]
        return data, desc

    def _build_response(self, structured_query, result_data, desc, query_type):
        """Build the response structure."""
        response = DEFAULT_RESPONSE.copy()
        response["query"] = structured_query["original"]
        brand = structured_query.get("brand", None)
        
        if query_type == "single":
            month = structured_query.get("month", "unknown month")
            response["question"] = f"Forecasted sales{' for ' + brand if brand else ''} in {month}?"

        elif query_type == "range":
            start_month = structured_query.get("start_month", "unknown start")
            end_month = structured_query.get("end_month", "unknown end")
            response["question"] = f"Forecasted sales{' for ' + brand if brand else ''} from {start_month} to {end_month}?"

        elif query_type == "quarter":
            quarter = structured_query.get("quarter", "unknown quarter")
            response["question"] = f"Forecasted sales{' for ' + brand if brand else ''} in Quarter {quarter}?"
            
        else:
            response["question"] = f"Forecasted sales for {structured_query['original']}?"

        response["graph_keys"] = [["Date", "Sales"]] if result_data else []
        response["desc"] = desc
        response["data"] = result_data
        return response

    def _get_random_response(self):
        """Return a random response description."""
        return random.choice(self.random_responses)

def fetch_db_data(db_config, query, is_brand=False):
    """Fetch data from databases."""
    all_data = []
    for db in db_config:
        dsn_tns = cx_Oracle.makedsn(db["host"], db["port"], service_name=db["service"])
        conn = cx_Oracle.connect(user=db["user"], password=db["password"], dsn=dsn_tns)
        df = pd.read_sql(query, con=conn)
        conn.close()
        all_data.append(df)
    
    if is_brand:
        combined_df = pd.concat(all_data).groupby(["BS_YEAR_MONTH", "BRAND_NAME"]).sum().reset_index()
        combined_df.columns = ["BS_YEAR_MONTH", "BRAND_NAME", "SALES_VALUE"]
    else:
        combined_df = pd.concat(all_data).groupby("BS_YEAR_MONTH").sum().reset_index()
        combined_df.columns = ["bs_year_month", "sales"]
    
    return combined_df