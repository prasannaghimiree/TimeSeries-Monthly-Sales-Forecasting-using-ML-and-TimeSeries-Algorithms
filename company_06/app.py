# # # # # # Flask API
# # # # # from flask import Flask, jsonify, request, render_template
# # # # # from sklearn.preprocessing import StandardScaler
# # # # # import pandas as pd
# # # # # import numpy as np
# # # # # import joblib
# # # # # import plotly
# # # # # import plotly.graph_objs as go
# # # # # import json
# # # # # from datetime import datetime, timedelta

# # # # # app = Flask(__name__)

# # # # # model = joblib.load("new_model.pkl")
# # # # # scaler_X = StandardScaler()

# # # # # scaler_y = StandardScaler()
# # # # # df = pd.read_csv("merge_06.csv", parse_dates=["date"], index_col="date")
# # # # # df = df.sort_index().dropna()




# # # # # # def prepare_features(last_date, periods=6):
# # # # # #     future_dates = pd.date_range(
# # # # # #         start=last_date + pd.DateOffset(months=1), periods=periods, freq="MS"
# # # # # #     )
# # # # # #     future_df = pd.DataFrame(index=future_dates)

# # # # # #     future_df["year"] = future_df.index.year
# # # # # #     future_df["month"] = future_df.index.month
# # # # # #     future_df["quarter"] = future_df.index.quarter

# # # # # #     full_df = pd.concat([df, future_df], axis=0)
# # # # # #     forecast_df = full_df.copy()

# # # # # #     last_sales = df["sales"].iloc[-1]
# # # # # #     forecast_df.loc[future_dates[0], "lag_1"] = last_sales
# # # # # #     required_features = ['year', 'month', 'quarter', 'lag_1', 'lag_11', 'lag_12','lag_13',
# # # # # #                     'rolling_mean_3', 'rolling_mean_6']
# # # # # #     scaler_X.fit(df[required_features])


    

# # # # # #     for i, date in enumerate(future_dates):
# # # # # #         if i > 0:
# # # # # #             prev_date = future_dates[i - 1]
# # # # # #             forecast_df.loc[date, "lag_1"] = forecast_df.loc[prev_date, "sales"]
# # # # # #         try:
# # # # # #             forecast_df.loc[date, "lag_11"] = forecast_df.loc[
# # # # # #                 date - pd.DateOffset(months=11), "sales"
# # # # # #             ]
# # # # # #             forecast_df.loc[date, "lag_12"] = forecast_df.loc[
# # # # # #                 date - pd.DateOffset(months=12), "sales"
# # # # # #             ]
# # # # # #             forecast_df.loc[date, "lag_13"] = forecast_df.loc[
# # # # # #                 date - pd.DateOffset(months=13), "sales"
# # # # # #             ]
# # # # # #         except KeyError:

# # # # # #             forecast_df.loc[date, "lag_11"] = forecast_df["lag_11"].mean()
# # # # # #             forecast_df.loc[date, "lag_12"] = forecast_df["lag_12"].mean()
# # # # # #             forecast_df.loc[date, "lag_13"] = forecast_df["lag_13"].mean()
# # # # # #         available_data = forecast_df["sales"].dropna()
# # # # # #         if len(available_data) >= 3:
# # # # # #             forecast_df.loc[date, "rolling_mean_3"] = available_data[-3:].mean()
# # # # # #         if len(available_data) >= 6:
# # # # # #             forecast_df.loc[date, "rolling_mean_6"] = available_data[-6:].mean()

# # # # # #         X_future = forecast_df.loc[date, required_features].values.reshape(1, -1)
# # # # # #         X_scaled = scaler_X.transform(X_future)
# # # # # #         prediction = scaler_y.inverse_transform(model.predict(X_scaled).reshape(-1, 1))
# # # # # #         forecast_df.loc[date, "sales"] = prediction[0][0]

# # # # # #     return forecast_df.tail(periods)
# # # # # def prepare_features(last_date, periods=6):
# # # # #     future_dates = pd.date_range(
# # # # #         start=last_date + pd.DateOffset(months=1), periods=periods, freq="MS"
# # # # #     )
# # # # #     future_df = pd.DataFrame(index=future_dates)

# # # # #     future_df["year"] = future_df.index.year
# # # # #     future_df["month"] = future_df.index.month
# # # # #     future_df["quarter"] = future_df.index.quarter

# # # # #     required_features = ['year', 'month', 'quarter', 'lag_1', 'lag_11', 'lag_12', 'lag_13', 'rolling_mean_3', 'rolling_mean_6']
# # # # #     forecast_df = pd.concat([df, future_df], axis=0)
# # # # #     forecast_df[required_features] = np.nan

# # # # #     last_sales = df["sales"].iloc[-1]
# # # # #     forecast_df.loc[future_dates[0], "lag_1"] = last_sales

# # # # #     for i, date in enumerate(future_dates):
# # # # #         if i > 0:
# # # # #             prev_date = future_dates[i - 1]
# # # # #             forecast_df.loc[date, "lag_1"] = forecast_df.loc[prev_date, "sales"]
    

# # # # #         for lag in [11, 12, 13]:
# # # # #             offset = pd.DateOffset(months=lag)
# # # # #             if (date - offset) in forecast_df.index:
# # # # #                 forecast_df.loc[date, f"lag_{lag}"] = forecast_df.loc[date - offset, "sales"]
# # # # #             else:
# # # # #                 forecast_df.loc[date, f"lag_{lag}"] = forecast_df[f"lag_{lag}"].mean()

# # # # #         available_data = forecast_df["sales"].dropna()
# # # # #         if len(available_data) >= 3:
# # # # #             forecast_df.loc[date, "rolling_mean_3"] = available_data[-3:].mean()
# # # # #         if len(available_data) >= 6:
# # # # #             forecast_df.loc[date, "rolling_mean_6"] = available_data[-6:].mean()

# # # # #         X_future = forecast_df.loc[date, required_features].values.reshape(1, -1)
# # # # #         X_scaled = scaler_X.transform(X_future)
# # # # #         prediction = scaler_y.inverse_transform(model.predict(X_scaled).reshape(-1, 1))
# # # # #         forecast_df.loc[date, "sales"] = prediction[0][0]

# # # # #     return forecast_df.tail(periods)


# # # # # @app.route("/predict", methods=["POST", "GET"])
# # # # # def predict():
# # # # #     try:

# # # # #         periods = request.json.get("periods", 6)

# # # # #         last_date = df.index[-1]
# # # # #         forecast = prepare_features(last_date, periods)

# # # # #         historical = df.reset_index().to_dict("records")
# # # # #         forecast_data = (
# # # # #             forecast.reset_index().rename(columns={"index": "date"}).to_dict("records")
# # # # #         )

# # # # #         return jsonify(
# # # # #             {
# # # # #                 "historical": historical,
# # # # #                 "forecast": forecast_data,
# # # # #                 "metadata": {
# # # # #                     "last_training_date": str(df.index[-1]),
# # # # #                     "forecast_start": str(forecast.index[0]),
# # # # #                     "forecast_periods": periods,
# # # # #                 },
# # # # #             }
# # # # #         )


# # # # #     except Exception as e:
# # # # #         return jsonify({"error": str(e)}), 500

# # # # # @app.route("/", methods=["GET"])
# # # # # def welcome():
# # # # #     return "<h1> Hello i am flask</h1>"




# # # # # if __name__ == "__main__":
# # # # #     app.run(debug=True, port=5000)


# # # # from flask import Flask, jsonify, request, render_template
# # # # from sklearn.preprocessing import StandardScaler
# # # # import pandas as pd
# # # # import numpy as np
# # # # import joblib
# # # # import matplotlib.pyplot as plt
# # # # from datetime import datetime
# # # # import io
# # # # import base64

# # # # app = Flask(__name__)

# # # # model = joblib.load("new_model.pkl")
# # # # scaler_X = StandardScaler()
# # # # scaler_y = StandardScaler()
# # # # df = pd.read_csv("merge_06.csv", parse_dates=["date"], index_col="date")
# # # # df = df.sort_index().dropna()

# # # # required_features = ['year', 'month', 'quarter', 'lag_1', 'lag_11', 'lag_12', 'lag_13', 'rolling_mean_3', 'rolling_mean_6']
# # # # scaler_X.fit(df[required_features])

# # # # def prepare_features(last_date, periods=6):
# # # #     future_dates = pd.date_range(
# # # #         start=last_date + pd.DateOffset(months=1), periods=periods, freq="MS"
# # # #     )
# # # #     future_df = pd.DataFrame(index=future_dates)
# # # #     future_df["year"] = future_df.index.year
# # # #     future_df["month"] = future_df.index.month
# # # #     future_df["quarter"] = future_df.index.quarter

# # # #     forecast_df = pd.concat([df, future_df], axis=0)
# # # #     forecast_df[required_features] = np.nan

# # # #     last_sales = df["sales"].iloc[-1]
# # # #     forecast_df.loc[future_dates[0], "lag_1"] = last_sales

# # # #     for i, date in enumerate(future_dates):
# # # #         if i > 0:
# # # #             prev_date = future_dates[i - 1]
# # # #             forecast_df.loc[date, "lag_1"] = forecast_df.loc[prev_date, "sales"]

# # # #         for lag in [11, 12, 13]:
# # # #             offset = pd.DateOffset(months=lag)
# # # #             if (date - offset) in forecast_df.index:
# # # #                 forecast_df.loc[date, f"lag_{lag}"] = forecast_df.loc[date - offset, "sales"]
# # # #             else:
# # # #                 forecast_df.loc[date, f"lag_{lag}"] = forecast_df[f"lag_{lag}"].mean()

# # # #         available_data = forecast_df["sales"].dropna()
# # # #         if len(available_data) >= 3:
# # # #             forecast_df.loc[date, "rolling_mean_3"] = available_data[-3:].mean()
# # # #         if len(available_data) >= 6:
# # # #             forecast_df.loc[date, "rolling_mean_6"] = available_data[-6:].mean()

# # # #         X_future = forecast_df.loc[date, required_features].values.reshape(1, -1)
# # # #         X_scaled = scaler_X.transform(X_future)
# # # #         prediction = scaler_y.inverse_transform(model.predict(X_scaled).reshape(-1, 1))
# # # #         forecast_df.loc[date, "sales"] = prediction[0][0]

# # # #     return forecast_df.tail(periods)


# # # # @app.route("/", methods=["GET"])
# # # # def welcome():
# # # #     return render_template("index.html")


# # # # @app.route("/predict", methods=["GET"])
# # # # def predict():
# # # #     try:
# # # #         periods = 6
# # # #         last_date = df.index[-1]
# # # #         forecast = prepare_features(last_date, periods)
# # # #         forecast_json = {str(date.date()): round(sales, 2) for date, sales in zip(forecast.index, forecast["sales"])}

# # # #         return jsonify(forecast_json)

# # # #     except Exception as e:
# # # #         return jsonify({"error": str(e)}), 500


# # # # @app.route("/plot", methods=["GET"])
# # # # def plot():
# # # #     periods = 6
# # # #     last_date = df.index[-1]
# # # #     forecast = prepare_features(last_date, periods)

# # # #     plt.figure(figsize=(10, 6))
# # # #     plt.plot(df.index, df["sales"], label="Historical Sales", color="blue")
# # # #     plt.plot(forecast.index, forecast["sales"], label="Forecasted Sales", color="red", linestyle="dashed")
# # # #     plt.title("Sales Forecast")
# # # #     plt.xlabel("Date")
# # # #     plt.ylabel("Sales")
# # # #     plt.legend()

   

# # # #     img = io.BytesIO()
# # # #     plt.savefig(img, format="png")
# # # #     img.seek(0)
# # # #     img_base64 = base64.b64encode(img.read()).decode("utf-8")
# # # #     plt.close()

# # # #     return render_template("plot.html", img_base64=img_base64)


# # # # if __name__ == "__main__":
# # # #     app.run(debug=True, port=5000)

# # # from flask import Flask, jsonify, request, render_template
# # # import pandas as pd
# # # import numpy as np
# # # import joblib
# # # import matplotlib.pyplot as plt
# # # from sklearn.preprocessing import StandardScaler
# # # import os
# # # from datetime import datetime
# # # import io
# # # import base64

# # # app = Flask(__name__)

# # # # Load Model and Data
# # # model = joblib.load("new_model.pkl")
# # # df = pd.read_csv("merge_06.csv", parse_dates=["date"], index_col="date")
# # # df = df.sort_index().dropna()

# # # scaler_X = StandardScaler()
# # # scaler_y = StandardScaler()

# # # required_features = ['year', 'month', 'quarter', 'lag_1', 'lag_11', 'lag_12', 'lag_13', 'rolling_mean_3', 'rolling_mean_6']

# # # scaler_X.fit(df[required_features])
# # # scaler_y.fit(df[['sales']])

# # # def prepare_features(last_date, periods=6):
# # #     future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='MS')
# # #     forecast_df = pd.DataFrame(index=future_dates)
# # #     forecast_df["year"] = future_dates.year
# # #     forecast_df["month"] = future_dates.month
# # #     forecast_df["quarter"] = future_dates.quarter

# # #     forecast_df["lag_1"] = df["sales"].iloc[-1]

# # #     for i, date in enumerate(future_dates):
# # #         if i > 0:
# # #             prev_date = future_dates[i - 1]
# # #             forecast_df.loc[date, "lag_1"] = forecast_df.loc[prev_date, "sales"]

# # #         for lag in [11, 12, 13]:
# # #             offset = pd.DateOffset(months=lag)
# # #             if (date - offset) in df.index:
# # #                 forecast_df.loc[date, f"lag_{lag}"] = df.loc[date - offset, "sales"]
# # #             else:
# # #                 forecast_df.loc[date, f"lag_{lag}"] = df[f"sales"].mean()

# # #         available_data = df["sales"].dropna()
# # #         if len(available_data) >= 3:
# # #             forecast_df.loc[date, "rolling_mean_3"] = available_data[-3:].mean()
# # #         if len(available_data) >= 6:
# # #             forecast_df.loc[date, "rolling_mean_6"] = available_data[-6:].mean()

# # #         X = forecast_df.loc[date, required_features].values.reshape(1, -1)
# # #         X_scaled = scaler_X.transform(X)
# # #         prediction = model.predict(X_scaled)
# # #         forecast_df.loc[date, "sales"] = scaler_y.inverse_transform(prediction.reshape(-1, 1))[0][0]

# # #     return forecast_df


# # # @app.route("/", methods=["GET"])
# # # def home():
# # #     return render_template("index.html")


# # # @app.route("/predict", methods=["POST"])
# # # def predict():
# # #     try:
# # #         periods = request.json.get("periods", 6)
# # #         last_date = df.index[-1]
# # #         forecast = prepare_features(last_date, periods)

# # #         forecast_json = {
# # #             str(date.strftime("%Y-%m-%d")): round(sales, 2)
# # #             for date, sales in zip(forecast.index, forecast["sales"])
# # #         }

# # #         # Plotting Forecast
# # #         img = io.BytesIO()
# # #         plt.figure(figsize=(10, 5))
# # #         plt.plot(df.index, df["sales"], label="Historical Sales")
# # #         plt.plot(forecast.index, forecast["sales"], label="Forecasted Sales", linestyle="dashed", color="red")
# # #         plt.xlabel("Date")
# # #         plt.ylabel("Sales")
# # #         plt.title("Sales Forecast")
# # #         plt.legend()
# # #         plt.grid(True)
# # #         plt.savefig(img, format='png')
# # #         img.seek(0)
# # #         plot_url = base64.b64encode(img.getvalue()).decode()

# # #         return render_template("result.html", forecast=forecast_json, plot_url=plot_url)

# # #     except Exception as e:
# # #         return jsonify({"error": str(e)}), 500


# # # if __name__ == "__main__":
# # #     app.run(debug=True, port=5000)

# # # Importing necessary libraries
# # from flask import Flask, jsonify, request, render_template
# # from sklearn.preprocessing import StandardScaler
# # import pandas as pd
# # import numpy as np
# # import joblib
# # import matplotlib.pyplot as plt
# # import json
# # from datetime import datetime, timedelta

# # app = Flask(__name__)

# # # Loading the model and scaler
# # model = joblib.load("new_model.pkl")
# # scaler_X = StandardScaler()
# # scaler_y = StandardScaler()

# # # Loading data
# # df = pd.read_csv("merge_06.csv", parse_dates=["date"], index_col="date")
# # df = df.sort_index().dropna()

# # # Function to prepare features for forecasting
# # def prepare_features(last_date, periods=6):
# #     future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq="MS")
# #     future_df = pd.DataFrame(index=future_dates)

# #     future_df["year"] = future_df.index.year
# #     future_df["month"] = future_df.index.month
# #     future_df["quarter"] = future_df.index.quarter

# #     required_features = ['year', 'month', 'quarter', 'lag_1', 'lag_11', 'lag_12', 'lag_13', 'rolling_mean_3', 'rolling_mean_6']
# #     forecast_df = pd.concat([df, future_df], axis=0)
# #     forecast_df[required_features] = np.nan

# #     last_sales = df["sales"].iloc[-1]
# #     forecast_df.loc[future_dates[0], "lag_1"] = last_sales

# #     # Populate lag features and rolling means
# #     for i, date in enumerate(future_dates):
# #         if i > 0:
# #             prev_date = future_dates[i - 1]
# #             forecast_df.loc[date, "lag_1"] = forecast_df.loc[prev_date, "sales"]

# #         for lag in [11, 12, 13]:
# #             offset = pd.DateOffset(months=lag)
# #             if (date - offset) in forecast_df.index:
# #                 forecast_df.loc[date, f"lag_{lag}"] = forecast_df.loc[date - offset, "sales"]
# #             else:
# #                 forecast_df.loc[date, f"lag_{lag}"] = forecast_df[f"lag_{lag}"].mean()

# #         available_data = forecast_df["sales"].dropna()
# #         if len(available_data) >= 3:
# #             forecast_df.loc[date, "rolling_mean_3"] = available_data[-3:].mean()
# #         if len(available_data) >= 6:
# #             forecast_df.loc[date, "rolling_mean_6"] = available_data[-6:].mean()

# #         # Prepare data for prediction
# #         X_future = forecast_df.loc[date, required_features].values.reshape(1, -1)
# #         if not np.isnan(X_future).any():
# #             X_scaled = scaler_X.transform(X_future)
# #             prediction = scaler_y.inverse_transform(model.predict(X_scaled).reshape(-1, 1))
# #             forecast_df.loc[date, "sales"] = prediction[0][0]

# #     return forecast_df.tail(periods)

# # @app.route("/predict", methods=["POST", "GET"])
# # def predict():
# #     try:
# #         periods = request.json.get("periods", 6)

# #         last_date = df.index[-1]
# #         forecast = prepare_features(last_date, periods)

# #         # Prepare data for JSON response
# #         forecast_data = forecast.reset_index().rename(columns={"index": "date"}).set_index("date")["sales"].to_dict()

# #         return jsonify(forecast_data)
    


# #     except Exception as e:
# #         return jsonify({"error": str(e)}), 500

# # @app.route("/plot", methods=["GET"])
# # def plot_forecast():
# #     try:
# #         periods = 6
# #         last_date = df.index[-1]
# #         forecast = prepare_features(last_date, periods)

        
# #         plt.figure(figsize=(10, 6))
# #         plt.plot(df.index, df['sales'], label='Historical Sales', color='blue')
# #         plt.plot(forecast.index, forecast['sales'], label='Forecasted Sales', color='red')
# #         plt.title('Sales Forecast')
# #         plt.xlabel('Date')
# #         plt.ylabel('Sales')
# #         plt.legend()
        
# #         # Save the plot as an image file
# #         plot_file = "forecast_plot.png"
# #         plt.savefig(plot_file)
# #         plt.close()

# #         return render_template("plot.html", plot_url=plot_file)

# #     except Exception as e:
# #         return jsonify({"error": str(e)}), 500

# # @app.route("/", methods=["GET"])
# # def welcome():
# #     return "<h1> Hello, I am Flask API for Sales Forecasting </h1>"

# # if __name__ == '__main__':
# #     app.run(debug=True)


# from flask import Flask, jsonify, request, render_template
# from sklearn.preprocessing import StandardScaler
# import pandas as pd
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# import json
# from datetime import datetime, timedelta

# app = Flask(__name__)

# # Loading the model and scaler
# model = joblib.load("new_model.pkl")
# scaler_X = StandardScaler()
# scaler_y = StandardScaler()

# # Loading data
# df = pd.read_csv("merge_06.csv", parse_dates=["date"], index_col="date")
# df = df.sort_index().dropna()

# # Function to prepare features for forecasting
# def prepare_features(last_date, periods=6):
#     future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq="MS")
#     future_df = pd.DataFrame(index=future_dates)

#     future_df["year"] = future_df.index.year
#     future_df["month"] = future_df.index.month
#     future_df["quarter"] = future_df.index.quarter

#     required_features = ['year', 'month', 'quarter', 'lag_1', 'lag_11', 'lag_12', 'lag_13', 'rolling_mean_3', 'rolling_mean_6']
#     forecast_df = pd.concat([df, future_df], axis=0)
#     forecast_df[required_features] = np.nan

#     last_sales = df["sales"].iloc[-1]
#     forecast_df.loc[future_dates[0], "lag_1"] = last_sales

#     # Populate lag features and rolling means
#     for i, date in enumerate(future_dates):
#         if i > 0:
#             prev_date = future_dates[i - 1]
#             forecast_df.loc[date, "lag_1"] = forecast_df.loc[prev_date, "sales"]

#         for lag in [11, 12, 13]:
#             offset = pd.DateOffset(months=lag)
#             if (date - offset) in forecast_df.index:
#                 forecast_df.loc[date, f"lag_{lag}"] = forecast_df.loc[date - offset, "sales"]
#             else:
#                 forecast_df.loc[date, f"lag_{lag}"] = forecast_df[f"lag_{lag}"].mean()

#         available_data = forecast_df["sales"].dropna()
#         if len(available_data) >= 3:
#             forecast_df.loc[date, "rolling_mean_3"] = available_data[-3:].mean()
#         if len(available_data) >= 6:
#             forecast_df.loc[date, "rolling_mean_6"] = available_data[-6:].mean()

#         # Print the prepared features to debug
#         print(f"Prepared features for {date}:")
#         print(forecast_df.loc[date, required_features])

#         # Prepare data for prediction
#         X_future = forecast_df.loc[date, required_features].values.reshape(1, -1)
#         if not np.isnan(X_future).any():
#             X_scaled = scaler_X.transform(X_future)
#             prediction = scaler_y.inverse_transform(model.predict(X_scaled).reshape(-1, 1))
#             forecast_df.loc[date, "sales"] = prediction[0][0]
#         else:
#             print(f"Warning: Missing or invalid features for {date}, skipping prediction.")
#             forecast_df.loc[date, "sales"] = np.nan

#     return forecast_df.tail(periods)

# @app.route("/predict", methods=["POST", "GET"])
# def predict():
#     try:
#         periods = request.json.get("periods", 6)

#         last_date = df.index[-1]
#         forecast = prepare_features(last_date, periods)

#         # Prepare data for JSON response
#         forecast_data = forecast.reset_index().rename(columns={"index": "date"}).set_index("date")["sales"].to_dict()

#         # Convert the keys (dates) to strings before returning as JSON
#         forecast_data = {str(date): value for date, value in forecast_data.items()}

#         return jsonify(forecast_data)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route("/plot", methods=["GET"])
# def plot_forecast():
#     try:
#         periods = 6
#         last_date = df.index[-1]
#         forecast = prepare_features(last_date, periods)

#         plt.figure(figsize=(10, 6))
#         plt.plot(df.index, df['sales'], label='Historical Sales', color='blue')
#         plt.plot(forecast.index, forecast['sales'], label='Forecasted Sales', color='red')
#         plt.title('Sales Forecast')
#         plt.xlabel('Date')
#         plt.ylabel('Sales')
#         plt.legend()

#         # Save the plot as an image file
#         plot_file = "forecast_plot.png"
#         plt.savefig(plot_file)
#         plt.close()

#         return render_template("plot.html", plot_url=plot_file)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route("/", methods=["GET"])
# def welcome():
#     return "<h1> Hello, I am Flask API for Sales Forecasting </h1>"

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, jsonify, Response
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import plotly.express as px
import plotly.io as pio
from datetime import datetime

app = Flask(__name__)

# Load data and models
df = pd.read_csv("merge_06.csv", parse_dates=["date"], index_col="date")
df = df.sort_index().dropna()

model = joblib.load("new_model.pkl")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

def generate_forecast():
    # Forecasting logic
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='MS')
    
    future_df = pd.DataFrame(index=future_dates)
    future_df['year'] = future_df.index.year
    future_df['month'] = future_df.index.month
    future_df['quarter'] = future_df.index.quarter
    
    full_df = pd.concat([df, future_df])
    required_features = ['year', 'month', 'quarter', 'lag_1', 'lag_11', 
                        'lag_12', 'lag_13', 'rolling_mean_3', 'rolling_mean_6']
    
    forecast_df = full_df.copy()
    forecast_df.loc[future_dates[0], 'lag_1'] = df['sales'].iloc[-1]
    
    for i, date in enumerate(future_dates):
        if i > 0:
            prev_date = future_dates[i-1]
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
    
    return forecast_df.loc[future_dates, 'sales']

@app.route('/forecast', methods=['GET'])
def get_forecast():
    forecast = generate_forecast()
    result = {date.strftime('%Y-%m-%d'): round(value, 2) 
             for date, value in forecast.items()}
    return jsonify(result)

@app.route('/plot/matplotlib', methods=['GET'])
def matplotlib_plot():
    forecast = generate_forecast()
    
    plt.figure(figsize=(12, 6))
    forecast.plot(kind='line', marker='o', title='6-Month Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.grid(True)
    plt.tight_layout()
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()
    
    return Response(img_buffer.getvalue(), mimetype='image/png')

@app.route('/plot/plotly', methods=['GET'])
def plotly_visualization():
    forecast = generate_forecast().reset_index()
    forecast.columns = ['Date', 'Sales']
    
    fig = px.line(forecast, x='Date', y='Sales', 
                 title='6-Month Sales Forecast',
                 markers=True, template='plotly_dark')
    fig.update_layout(hovermode='x unified')
    
    return pio.to_html(fig, full_html=False)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)