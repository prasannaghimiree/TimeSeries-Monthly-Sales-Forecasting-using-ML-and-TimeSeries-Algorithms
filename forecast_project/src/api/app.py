import os
import json
import logging
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
import google.generativeai as genai
from pytz import timezone
import pandas as pd
import nepali_datetime
from src.core.forecast_manager import SalesForecastManager
from src.core.query_parser import QueryParser
from src.utils.constants import MONTH_NUMBER_TO_NAME

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = Flask(__name__)

# Initialize dependencies
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")
forecast_manager = SalesForecastManager()
query_parser = QueryParser(model)

def schedule_train():
    if nepali_datetime.date.today().day == 1:
        logger.info("First day of Nepali month. Training model...")
        forecast_manager.train_model()
    else:
        logger.info("Not the first day of Nepali month. Skipping training.")

scheduler = BackgroundScheduler(timezone=timezone("Asia/Kathmandu"))
scheduler.add_job(schedule_train, "cron", hour=0, minute=0)
scheduler.start()

@app.route("/assistant", methods=["POST"])
def assistant():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Query parameter is required"}), 400

    query = data["query"]
    logger.info(f"Received query: {query}")

    structured_query = query_parser.parse(query)
    logger.info(f"Structured query: {structured_query}")

    try:
        tool_output = forecast_manager.query_csv(structured_query)
        response = tool_output
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        response = {
            "query": query, "question": f"Forecasted sales for {query}?",
            "graph_keys": [], "desc": "Oops! Something went wrong.", "data": {}
        }

    # Append historical data
    try:
        historical_data = pd.read_csv(forecast_manager.historical_data_path)
        historical_data["year"] = historical_data["bs_year_month"].str[:4]
        historical_data["month_num"] = historical_data["bs_year_month"].str[5:7]
        historical_data["month"] = historical_data["month_num"].map(MONTH_NUMBER_TO_NAME)
        response["history"] = {
            "Date": (historical_data["year"] + "-" + historical_data["month"]).tolist(),
            "Sales": historical_data["sales"].astype(str).tolist()
        }
    except Exception as e:
        logger.error(f"Error loading historical data: {str(e)}")
        response["history"] = {"Date": [], "Sales": []}

    final_response = json.dumps(response)
    return f"'''{final_response}'''"

if __name__ == "__main__":
    if not os.path.exists(forecast_manager.model_path) or forecast_manager.forecast_df.empty:
        forecast_manager.train_model()
    app.run(debug=True, host="0.0.0.0", port=5000)