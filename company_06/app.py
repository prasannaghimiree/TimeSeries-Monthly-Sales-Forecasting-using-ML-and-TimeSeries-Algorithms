# Flask API
from flask import Flask, jsonify, request, render_template
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
import plotly
import plotly.graph_objs as go
import json
from datetime import datetime, timedelta

app = Flask(__name__)

model = joblib.load("new_model.pkl")
scaler_X = StandardScaler()
scaler_y = StandardScaler()
df = pd.read_csv("merge_06.csv", parse_dates=["date"], index_col="date")
df = df.sort_index().dropna()


def prepare_features(last_date, periods=6):
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1), periods=periods, freq="MS"
    )
    future_df = pd.DataFrame(index=future_dates)

    future_df["year"] = future_df.index.year
    future_df["month"] = future_df.index.month
    future_df["quarter"] = future_df.index.quarter

    full_df = pd.concat([df, future_df], axis=0)
    forecast_df = full_df.copy()

    last_sales = df["sales"].iloc[-1]
    forecast_df.loc[future_dates[0], "lag_1"] = last_sales



    for i, date in enumerate(future_dates):
        if i > 0:
            prev_date = future_dates[i - 1]
            forecast_df.loc[date, "lag_1"] = forecast_df.loc[prev_date, "sales"]
        try:
            forecast_df.loc[date, "lag_11"] = forecast_df.loc[
                date - pd.DateOffset(months=11), "sales"
            ]
            forecast_df.loc[date, "lag_12"] = forecast_df.loc[
                date - pd.DateOffset(months=12), "sales"
            ]
        except KeyError:

            forecast_df.loc[date, "lag_11"] = forecast_df["lag_11"].mean()
            forecast_df.loc[date, "lag_12"] = forecast_df["lag_12"].mean()
        available_data = forecast_df["sales"].dropna()
        if len(available_data) >= 3:
            forecast_df.loc[date, "rolling_mean_3"] = available_data[-3:].mean()
        if len(available_data) >= 6:
            forecast_df.loc[date, "rolling_mean_6"] = available_data[-6:].mean()

        X_future = forecast_df.loc[date, model.feature_names_in_].values.reshape(1, -1)
        X_scaled = scaler_X.transform(X_future)
        prediction = scaler_y.inverse_transform(model.predict(X_scaled).reshape(-1, 1))
        forecast_df.loc[date, "sales"] = prediction[0][0]

    return forecast_df.tail(periods)


@app.route("/predict", methods=["POST"])
def predict():
    try:

        periods = request.json.get("periods", 6)

        last_date = df.index[-1]
        forecast = prepare_features(last_date, periods)

        historical = df.reset_index().to_dict("records")
        forecast_data = (
            forecast.reset_index().rename(columns={"index": "date"}).to_dict("records")
        )

        return jsonify(
            {
                "historical": historical,
                "forecast": forecast_data,
                "metadata": {
                    "last_training_date": str(df.index[-1]),
                    "forecast_start": str(forecast.index[0]),
                    "forecast_periods": periods,
                },
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/graph")
def show_graph():

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["sales"], mode="lines+markers", name="Historical Data"
        )
    )

    last_date = df.index[-1]
    forecast = prepare_features(last_date, 6)

    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["sales"],
            mode="lines+markers",
            name="Forecast",
            line=dict(dash="dot"),
        )
    )

    graph_json = fig.to_json()

    return render_template("graph.html", graphJSON=graph_json)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
