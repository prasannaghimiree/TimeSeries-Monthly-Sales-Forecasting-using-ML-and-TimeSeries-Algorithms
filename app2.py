import os
import json
import logging
import pandas as pd
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
import google.generativeai as genai
from utils2 import SalesForecastManager
import nepali_datetime
from pytz import timezone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = Flask(__name__)

# Initialize Gemini model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

forecast_manager = SalesForecastManager()

def schedule_train():
    """Check if today is the first day of the Nepali month and train the model if it is."""
    current_nepali_date = nepali_datetime.date.today()
    if current_nepali_date.day == 1:
        logger.info("First day of Nepali month. Training model...")
        forecast_manager.train_model()
    else:
        logger.info("Not the first day of Nepali month. Skipping training.")

# Initialize scheduler to run schedule_train daily at night at 12:00 AM
scheduler = BackgroundScheduler(timezone=timezone("Asia/Kathmandu"))
scheduler.add_job(schedule_train, "cron", hour=0, minute=0)
scheduler.start()

# Month number to name mapping
month_number_to_name = {
    "01": "Baisakh", "02": "Jestha", "03": "Ashad", "04": "Shrawan", "05": "Bhadra",
    "06": "Ashoj", "07": "Kartik", "08": "Mangsir", "09": "Poush", "10": "Magh",
    "11": "Falgun", "12": "Chaitra"
}

def parse_query_with_llm(query):
    """Use Gemini LLM to parse user query into a structured format."""
    # calculation of current and next month date
    current_date = nepali_datetime.date.today()

    cur_year = current_date.year
    cur_month_number = current_date.month
    current_month = month_number_to_name[f"{cur_month_number:02d}"]

    if cur_month_number == 12:  
        next_month_number = 1
        next_year = cur_year + 1
    else:
        next_month_number = cur_month_number + 1
        next_year = cur_year

    # next_date = f"{next_year}-{next_month}"
    
    next_month = month_number_to_name[f"{next_month_number:02d}"]


    prompt = f"""
    Analyze the following user query and determine if it is asking about sales or upcoming sales. 
    If it is, convert it into a structured JSON format that includes:
    - "type": "single" for a single month or "range" for a range of months
    - "month": the month name (e.g., "Baisakh") for single queries
    - "start_month" and "end_month": for range queries
    - "year": the year if specified, otherwise null
    - "original": the original query string
    If the query is not about sales or upcoming sales, return:
    - "type": "invalid"
    - "original": the original query string

    Current Nepali month: {current_month} ({current_date.year})
    Next Nepali month: {next_month} ({next_year})
    Valid months: Baisakh, Jestha, Ashad, Shrawan, Bhadra, Ashoj, Kartik, Mangsir, Poush, Magh, Falgun, Chaitra

    Query: "{query}"

    Examples:
    - "forecast me the sales of baisakh" → {{"type": "single", "month": "Baisakh", "year": null, "original": "forecast me the sales of baisakh"}}
    - "sales from baisakh to shrawan" → {{"type": "range", "start_month": "Baisakh", "end_month": "Shrawan", "year": null, "original": "sales from baisakh to shrawan"}}
    - "2082 baisakh sales" → {{"type": "single", "month": "Baisakh", "year": "2082", "original": "2082 baisakh sales"}}
    - "what's the weather like" → {{"type": "invalid", "original": "what's the weather like"}}
    - "upcoming sales for jestha" → {{"type": "single", "month": "Jestha", "year": null, "original": "upcoming sales for jestha"}}
    - "this/current/this-month/ sales" → {{"type": "single", "month": "{current_month}", "year": "{current_date.year}", "original": "this month sales"}}
    - "next/upcomming/comming/after-this month sales" → {{"type": "single", "month": "{next_month}", "year": "{next_year}", "original": "next month sales"}}
    - "If forecast me the sales of shrawan is asked then give result of upcomming shrawan"
    - "Understand the user intent, if user is asking about forecast, prediction or estimated value of particular month then only give answer, If user is not asking sales value of that month then donot give answer"
    - "If the user is not asking about sales or sales related keywords then donot give the answer, query might contain: sales in shrawan, forecasted valus in shrawan, what might be the sales in shrawan, shrawan sales values like such keywords"
    """
    
    response = model.generate_content(prompt)
    try:
        structured_query = json.loads(response.text.strip("```json\n").strip("```"))
        return structured_query
    except Exception as e:
        logger.error(f"Error parsing LLM response: {str(e)}")
        return {"type": "invalid", "original": query}

@app.route("/assistant", methods=["POST"])
def assistant():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Query parameter is required"}), 400

    query = data["query"]
    logger.info(f"Received query: {query}")

    # Parse query using LLM
    structured_query = parse_query_with_llm(query)
    logger.info(f"Structured query: {structured_query}")

    try:
        # Fetch data using query_csv with structured query
        tool_output = forecast_manager.query_csv(structured_query)
        logger.info(f"Tool output: {tool_output}")

        load_response = {
            "query": tool_output["query"],
            "question": tool_output["question"],
            "graph_keys": tool_output["graph_keys"],
            "desc": tool_output["desc"],
            "for_data": tool_output["for_data"]
        }

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        load_response = {
            "query": query,
            "question": f"Forecasted sales for {query}?",
            "graph_keys": [],
            "desc": "Opps! Something is wrong. I couldn't fetch the data.",
            "for_data": []
        }
    
    # append histrical data to the response
    try:
        historical_data = pd.read_csv(forecast_manager.historical_data_path)
        historical_data["year"] = historical_data["bs_year_month"].str[:4]
        historical_data["month_num"] = historical_data["bs_year_month"].str[5:7]
        historical_data["month"] = historical_data["month_num"].map(month_number_to_name)
        historical_data["formatted_date"] = historical_data["year"] + "-" + historical_data["month"]
        load_response["data"] = {
            "date": historical_data["formatted_date"].tolist(),
            "sales": historical_data["sales"].astype(float).tolist()
        }
    except Exception as e:
        logger.error(f"Error loading historical data: {str(e)}")
        load_response["data"] = {"date": [], "sales": []}

    final_response = json.dumps(load_response)
    logger.info(f"Final response: {final_response}")
    return final_response

if __name__ == "__main__":
    if not os.path.exists(forecast_manager.model_path) or forecast_manager.forecast_df.empty:
        forecast_manager.train_model()
    app.run(debug=True, host="0.0.0.0", port=5000)