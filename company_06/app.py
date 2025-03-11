from flask import Flask, request, jsonify, Response
import pandas as pd
import os
import json
from dotenv import load_dotenv
from langchain_together import ChatTogether
from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from utils import query_csv

history_data = r"Dataset\merge_06.csv"

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
app = Flask(__name__)
# Initialize LLM from TOgether AI
llm = ChatTogether(
    model="meta-llama/Llama-3-70b-chat-hf",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
tools = [
    Tool(
        name="CSV Reader",
        func=query_csv,
        description=(
            "Extracts forecasted sales data from the dataset based on user queries."
            "Understands both numerical months (e.g., '2082-04') and Nepali month names (e.g., '2082 Baisakh') and relative months. "
            "If the user asks about relative dates like next|upcoming|previous|last|current, first calculate the present date and identify it accordingly. "
            "Supports retrieving specific sales or calculating the average sales over a given range. "
            "If the user asks about sales within a range of months, return data separately for all in-between dates. "
            "Read Nepali month names correctly and get their index properly."
            "Give the final answer as a whole in json which includes question, graph_keys, desc, for_data."
            "If it is asked to calculate an average sales between month_1 to month_n then calculate average using average library of python."
            "Donot give any description, If you dont know simply leave a blank but donot say anything."
            "Enclose all property in double quotes"
        
        ),
    )
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
)


@app.route("/assistant", methods=["POST"])
def assistant():
    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({"error": "Query parameter is required"}), 400

    query = data["query"]

    historical_data = pd.read_csv(history_data)
    historical_data["date"] = historical_data["date"].astype(str)
    historical_data[["year", "month"]] = historical_data["date"].str.split("-", expand=True)

    # Map number to nepali month names
    nepali_months = {
        "01": "Baisakh",
        "02": "Jestha",
        "03": "Ashadh",
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

    # Replacing month number to month name
    historical_data["nepali_date"] = (
        historical_data["year"] + "-" + historical_data["month"].map(nepali_months)
    )

    print(historical_data[["nepali_date"]])

    response = agent.run(query)
    print("*********************************************************************************")
    print("Type is:>> ", type(response))
    ("*********************************************************************************")

    load_response = json.loads(response)
    print("Type of load response is :", load_response)
  


    date_list = historical_data["nepali_date"].astype(str).tolist()
    sales_list = historical_data["sales"].astype(str).tolist()

# Add historical data to the response
    load_response["data"] = {
    "date": date_list,
    "sales": sales_list
    }

    final_response = json.dumps(load_response)

    return final_response

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)


# from flask import Flask, request, jsonify, Response
# import pandas as pd
# import os
# from dotenv import load_dotenv
# from langchain_together import ChatTogether
# from langchain.agents import initialize_agent, AgentType
# from langchain.agents import Tool
# from langchain.memory import ConversationBufferMemory
# from utils import query_csv
# import json

# history_data = r"Dataset\merge_06.csv"

# load_dotenv()
# TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
# app = Flask(__name__)

# # Initialize LLM from TOgether AI
# llm = ChatTogether(
#     model="meta-llama/Llama-3-70b-chat-hf",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )

# tools = [
#     Tool(
#         name="CSV Reader",
#         func=query_csv,
#         description=(
#             "Extracts forecasted sales data from the dataset based on user queries."
#             "Understands both numerical months (e.g., '2082-04') and Nepali month names (e.g., '2082 Baisakh') and relative months. "
#             "If the user asks about relative dates like next|upcoming|previous|last|current, first calculate the present date and identify it accordingly. "
#             "Supports retrieving specific sales or calculating the average sales over a given range. "
#             "If the user asks about sales within a range of months, return data separately for all in-between dates. "
#             "Read Nepali month names correctly and get their index properly."
#             "Give the final answer in json question, graph_keys, desc, for_data."
#             "If it is asked to calculate an average sales between month_1 to month_n then calculate average using average library of python."
#             "Donot give any description, If you dont know simply leave a blank but donot say anything."
#         ),
#     )
# ]

# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     memory=memory,
#     verbose=True,
# )

# @app.route("/assistant", methods=["POST"])
# def assistant():
#     data = request.get_json()

#     if not data or "query" not in data:
#         return jsonify({"error": "Query parameter is required"}), 400

#     query = data["query"]

#     historical_data = pd.read_csv(history_data)
#     historical_data["date"] = historical_data["date"].astype(str)
#     historical_data[["year", "month"]] = historical_data["date"].str.split("-", expand=True)

#     # Map number to Nepali month names
#     nepali_months = {
#         "01": "Baisakh",
#         "02": "Jestha",
#         "03": "Ashadh",
#         "04": "Shrawan",
#         "05": "Bhadra",
#         "06": "Ashoj",
#         "07": "Kartik",
#         "08": "Mangsir",
#         "09": "Poush",
#         "10": "Magh",
#         "11": "Falgun",
#         "12": "Chaitra",
#     }

#     historical_data["nepali_date"] = (
#         historical_data["year"] + "-" + historical_data["month"].map(nepali_months)
#     )

#     print(historical_data[["nepali_date"]])

#     response = agent.run(query)
#     json_response = jsonify(response)
#     # print("The response is", response)

#     # try:
#     #     response_json = json.loads(response)
#     # except json.JSONDecodeError:
#     #     response_json = {"response": response} 
     

#     # date_list = historical_data["nepali_date"].astype(str).tolist()
#     # sales_list = historical_data["sales"].astype(str).tolist()

#     # response_json["data"] = {
#     #     "date": date_list,
#     #     "sales": sales_list
#     # }

#     # json_response = json.dumps(response_json)

#     return json_response

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5000)
