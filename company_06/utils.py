import os
import re
import joblib
import pandas as pd
import pandas as pd
from dotenv import load_dotenv
import random
import json

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Load data and models
data= pd.read_csv(r"Dataset\merge_06.csv", parse_dates=["date"], index_col="date")
data = data.sort_index().dropna()

# loading saved model
model = joblib.load(r"models\new_model.pkl")
scaler_X = joblib.load(r"models\scaler_X.pkl")
scaler_y = joblib.load(r"models\scaler_y.pkl")

def generate_forecast():
    # Forecasting logic
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='MS')

    future_df = pd.DataFrame(index=future_dates)
    future_df['year'] = future_df.index.year
    future_df['month'] = future_df.index.month
    future_df['quarter'] = future_df.index.quarter

    full_df = pd.concat([data, future_df])
    required_features = ['year', 'month', 'quarter', 'lag_1', 'lag_11', 
                         'lag_12', 'lag_13', 'rolling_mean_3', 'rolling_mean_6']

    forecast_df = full_df.copy()
    forecast_df.loc[future_dates[0], 'lag_1'] = data['sales'].iloc[-1]

    for i, date in enumerate(future_dates):
        if i > 0:
            prev_date = future_dates[i - 1]
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

    # Create a DataFrame with renamed columns
    forecast_result = forecast_df.loc[future_dates, ['sales']].reset_index()
    forecast_result.columns = ['forecast_date', 'forecast_sales']

    return forecast_result

df = generate_forecast()
print("********************************************************************")
print(type(df))
print(df.columns)
df.to_csv(r'forecast_output\forecast_final.csv')
# df.to_excel('forecast.xlsx')
print("**********************************************************************")

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
    "falgun": "11", "phalgun": "11", "falgoon": "11", "fagun": "11",
    "chaitra": "12", "chait": "12"
}

month_number_to_name={
    "01":"Baisakh",
    "02":"Jestha",
    "03": "Ashad",
    "04":"Shrawan",
    "05":"Bhadra",
    "06":"Ashoj",
    "07":"Kartik",
    "08":"Mangsir",
    "09":"Poush",
    "10":"Magh",
    "11":"Falgun",
    "12":"chaitra"
    }

def query_csv(query):
    query = query.lower()
    match = re.search(r"(\d{4})[-\s]?(\d{2}|\w+)", query)

    if match:
        year, month = match.groups()
        
        if month in nepali_months:
            month = nepali_months[month]

        

        month = month.zfill(2)
        result = df[df["forecast_date"] == f"{year}-{month}"]

        #######################################################################################



        # Convert to JSON
        his_data = data.to_json(orient="records", indent=4)


        #######################################################################
        

        if not result.empty:
            forecast_value = result['forecast_sales'].values[0]
            desc = get_random_response()
            return {
                # "data": his_data,
                "query": query,
                "question": f"Forecasted sales for {year}-{month_number_to_name[month]}?",
                "graph_keys": [["forecast_date", "forecast_sales"]],
                "desc": desc,
                "for_data" : [{"forecast_date": f"{year}-{month_number_to_name[month]}", "forecast_sales": forecast_value}]
            }
        else:
            return {
                # "data": his_data,
                "query": query,
                "question": f"Forecasted sales for {year}-{month_number_to_name[month]}?",
                "graph_keys": [],
                "desc": "No forecast data found for this date.",
                "for_data":[]
            }

    elif "average" in query or "mean" in query:
        months = re.findall(r"(\d{2}|\w+)", query)
        numeric_months = []

        for month in months:
            if month in nepali_months:
                numeric_months.append(nepali_months[month])
            elif month.isdigit():
                numeric_months.append(month.zfill(2))
        
        filtered_df = df[df["forecast_date"].str[-2:].isin(numeric_months)]

        if not filtered_df.empty:
            avg_sales = filtered_df["forecast_sales"].mean()
            desc = get_random_response()
            return {
                # "data":his_data,
                "query": query,
                "question": f"Average forecasted sales from {months[0]} to {months[-1]}?",
                "graph_keys": [["forecast_date", "forecast_sales"]],
                "desc": desc,
                "for_data": [{"forecast_date": "Average", "forecast_sales": avg_sales}]
            }
        else:
            return {
                # "data": his_data,
                "query": query,
                "question": f"Average forecasted sales from {months[0]} to {months[-1]}?",
                "graph_keys": [],
                "desc": "No data available for the given months.",
                "for_data": [],
            }

    return {
        # "data": his_data,
        "query": query,
        "question": "Sorry, I couldn't understand the query.",
        "graph_keys": [],
        "desc": "Could not process the request.",
        "for_data": [],
    }


def get_random_response():
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

