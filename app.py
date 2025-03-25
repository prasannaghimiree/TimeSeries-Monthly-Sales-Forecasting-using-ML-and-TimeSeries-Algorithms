import os
import json
import logging
import pandas as pd
import re
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from langchain_together import ChatTogether
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from utils import SalesForecastManager
import nepali_datetime
from pytz import timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = Flask(__name__)

forecast_manager = SalesForecastManager()


def schedule_train():
    # It is checked by scheduler at mid night 12:00 AM
    """Check if today is the first day of the Nepali month and train the model if iot is the first day"""
    current_nepali_date = nepali_datetime.date.today()
    if current_nepali_date.day == 1:
        logger.info("First day of Nepali. So, now Training model...")
        forecast_manager.train_model()
    else:
        print("************************************************")
        print("Its not day 1 of nepali date so, model should not get trained")
        print("************************************************")


# initializing scheduler to run the schedule_train function daily
scheduler = BackgroundScheduler(timezone=timezone("Asia/Kathmandu"))
# call schedule_train function everyday at 12:00 AM
scheduler.add_job(schedule_train, "cron", hour=0, minute=0)
scheduler.start()

# Initialization of LLM
llm = ChatTogether(
    model="meta-llama/Llama-3-70b-chat-hf",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("TOGETHER_API_KEY"),
)

# month number into month name mapping for the output
month_number_to_name = {
    "01": "Baisakh",
    "02": "Jestha",
    "03": "Ashad",
    "04": "Shrawan",
    "05": "Bhadra",
    "06": "Ashoj",
    "07": "Kartik",
    "08": "Mangsir",
    "09": "Poush",
    "10": "Magh",
    "11": "Falgun",
    "12": "Chaitra",
}


# calculating current nepali month for calulating relative keywords like "this month", "upcomming month", "currenct month", "two months after"
def calculate_nepali_month():
    current_nepali_date = nepali_datetime.date.today()
    current_bs_year_month = (
        f"{current_nepali_date.year:04d}-{current_nepali_date.month:02d}"
    )

    return current_bs_year_month


latest_nepali_month = calculate_nepali_month()


# initializing langchain tool which reads CSV file and extract data based on description
tools = [
    Tool(
        name="CSV Reader",
        func=forecast_manager.query_csv,
        description=(
            f"""Extracts forecasted sales data from the latest forecast CSV based on user queries. 
            Understands single months (e.g., '2082 Baisakh') or ranges (e.g., 'Baisakh to Shrawan' or '2082 Baisakh to Shrawan'). 
            Returns a dictionary with 'query', 'question', 'graph_keys', 'desc', and 'for_data'. 
            The 'for_data' list contains objects with 'forecast_date' (e.g., '2082-Shrawan') and 'forecast_sales'.
            If user ask for this/current/latest month then month is {latest_nepali_month} and calculate its forecast.
            If user ask upcomming/next/after-this/comming/another and such keywords then month is ({latest_nepali_month}+1) and calculate its forecast accordingly.
            Dont give multiple redundant answer.
            """
        ),
    )
]

# initializaing memory and agents
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)


@app.route("/assistant", methods=["POST"])
def assistant():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Query parameter is required"}), 400

    query = data["query"]
    logger.info(f"Received query: {query}")

    try:
        response = agent.run(
            f"Use the CSV Reader tool to get the forecasted sales for {query}. "
            f"If the query specifies a range (e.g., 'Baisakh to Shrawan'), return sales for all months in the range. "
            f"Return the result as plain text with each entry in the format: 'Forecast Date: YYYY-Month\nForecast Sales: value', separated by newlines."
            # f"If user ask for this/current/latest month then month is {latest_nepali_month} and calculate its forecast."
            # f"If user ask upcomming/next/after-this/comming/another and such keywords then month is ({latest_nepali_month}+1) and calculate its forecast accordingly."
        )
        logger.info(f"Raw agent response: {response}")

        load_response = {
            "query": query,
            "question": f"Forecasted sales for {query}?",
            "graph_keys": [["forecast_date", "forecast_sales"]],
            "desc": "Data processed successfully.",
            "for_data": [],
        }

        entries = response.strip().split("\n\n")
        print("######################################################")
        print(entries)
        print("######################################################")
        for_data = []
        for entry in entries:
            lines = entry.strip().split("\n")
            date_match = (
                re.search(r"Forecast Date: (\d{4}-[A-Za-z]+)", lines[0])
                if lines
                else None
            )
            sales_match = (
                re.search(r"Forecast Sales: ([\d.]+)", lines[1])
                if len(lines) > 1
                else None
            )

            if date_match and sales_match:
                forecast_date = date_match.group(1)
                forecast_sales = float(sales_match.group(1))
                for_data.append(
                    {"forecast_date": forecast_date, "forecast_sales": forecast_sales}
                )

        if for_data:
            load_response["for_data"] = for_data
            load_response["desc"] = "The latest data you asked for is now available."
            # load_response["desc"] = SalesForecastManager.get_random_response
        else:
            tool_output_match = re.search(
                r"\{.*?'for_data':\s*\[.*?\]\s*.*?\}", response
            )
            if tool_output_match:
                tool_output_str = tool_output_match.group(0)
                try:
                    tool_output = eval(tool_output_str)
                    if "for_data" in tool_output and tool_output["for_data"]:
                        load_response["for_data"] = tool_output["for_data"]
                        load_response["desc"] = tool_output.get(
                            "desc", "Data processed successfully."
                        )
                    else:
                        load_response["desc"] = "No forecast data found for this range."
                except Exception as e:
                    logger.error(f"Error parsing tool output: {str(e)}")
                    load_response["desc"] = "Error parsing forecast data."
            else:
                load_response["desc"] = (
                    "Could not extract forecast sales from the response."
                )

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        load_response = {
            "query": query,
            "question": f"Forecasted sales for {query}?",
            "graph_keys": [],
            "desc": f"Error: {str(e)}",
            "for_data": [],
        }

    try:
        historical_data = pd.read_csv(forecast_manager.historical_data_path)
        historical_data["year"] = historical_data["bs_year_month"].str[:4]
        historical_data["month_num"] = historical_data["bs_year_month"].str[5:7]
        # mapping into month name from month number
        historical_data["month"] = historical_data["month_num"].map(
            month_number_to_name
        )
        # formatting output date as (2078-Shrawan, 2078-Bhadra, 2078-Asoj)
        historical_data["formatted_date"] = (
            historical_data["year"] + "-" + historical_data["month"]
        )
        # list of formatted historical dates
        date_list = historical_data["formatted_date"].tolist()
        # list of formatted historical sales
        sales_list = historical_data["sales"].astype(str).tolist()
        load_response["data"] = {"date": date_list, "sales": sales_list}
    except Exception as e:
        logger.error(f"Error loading historical data: {str(e)}")
        load_response["data"] = {"date": [], "sales": []}

    # dumping it inside the final json
    final_response = json.dumps(load_response)
    # info for the log
    logger.info(f"Final response: {final_response}")
    return final_response


if __name__ == "__main__":
    if (
        not os.path.exists(forecast_manager.model_path)
        or forecast_manager.forecast_df.empty
    ):
        forecast_manager.train_model()
    app.run(debug=True, host="0.0.0.0", port=5000)
