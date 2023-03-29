# Importing prerequisite liabraries
import dash
from dash import dcc
from dash import html
import datetime as dt
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from model_lstm import main

app = dash.Dash(__name__)

server = app.server

#adding a head file for styling css
app.head = [html.Link(rel="stylesheet", href="assets/styles.css")]

#adding layout of the app
app.layout = html.Div([
    #Child Division 1
    html.Div(
    [
        html.P("Welcome to the Stock Dash App!", className="start"),
        html.Div([
            # stock code input
            dcc.Input(placeholder="Enter stock code",
            type="text", value="", id="stockCode"),
            html.Button('Submit', id="submitCode")
        ]),
        html.Div([
            # Date range picker input
            dcc.DatePickerRange(start_date_placeholder_text="Start Date",
            end_date=dt.date.today(),
            max_date_allowed = dt.date.today(),
            id="datePicker")
        ]),
        html.Div([
            # Stock price button
            html.Button("Stock Price", id="stockPrice"),
            # Indicators button
            html.Button("Indicators", id="indicatorButton"),
            # Number of days of forecast input
            dcc.Input(placeholder="Number of days",
            value="", type="number", min="1", id="forecastDays"),
            # Forecast button
            html.Button("Forecast", id="forecastButton")
        ]),
    ],
    className="nav"),
    #Child Division 2
    html.Div(
          [
            html.Div(
                  [  # Logo
                    # Company Name
                    html.H1(id="companyName")
                  ],
                className="header"),
            html.Div( #Description
              id="description", className="decription_ticker"),
            html.Div([
                # Stock price plot
            ], id="graphs-content"),
            html.Div([
                # Indicator plot
            ], id="main-content"),
            html.Div([
                # Forecast plot
            ], id="forecast-content")
          ],
        className="content")
    ]
,className="container")


@app.callback(
    dash.Output("companyName", "children"),
    dash.Input("submitCode","n_clicks"),
    dash.State("stockCode","value")
    )
def update_data(n_clicks, value):
    if n_clicks is None or value=="":
        return
    else:
        ticker = yf.Ticker(value)
        inf = ticker.info
        #df = pd.DataFrame().from_dict(inf, orient="index").T
        return inf['longName']

@app.callback(
    dash.Output("graphs-content", "children"),
    dash.Input("stockPrice", "n_clicks"),
    dash.State("datePicker", "start_date"),
    dash.State("datePicker", "end_date"),
    dash.State("stockCode", "value")
)
def get_stock_price(n_clicks, start_date, end_date, value):
    if n_clicks is None or value == "":
        return
    else:
        df = yf.download(value, start = start_date, end= end_date)
        df.reset_index(inplace=True)
        print(df)
        fig = get_stock_price_fig(df)
        return dcc.Graph(figure = fig)

def get_stock_price_fig(df):
    fig = px.line(df, x = df['Date'], y = [df['Open'], df['Close']], 
    title="Closing and Opening Price Vs Date", width=800, height=400)
    return fig


@app.callback(
    dash.Output("main-content", "children"),
    dash.Input("indicatorButton", "n_clicks"),
    dash.State("datePicker", "start_date"),
    dash.State("datePicker", "end_date"),
    dash.State("stockCode", "value")
)
def get_stock_price_more(n_clicks, start_date, end_date, value):
    if n_clicks is None or value == "":
        return
    else:
        df = yf.download(value, start = start_date, end= end_date)
        df.reset_index(inplace=True)
        fig = get_more(df)
        return dcc.Graph(figure = fig)

def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df, x = df['Date'], y = df['EWA_20'], 
                     title="Exponential Moving Average Vs Date", width=800, height=400)
    fig.update_traces(mode = 'lines+markers')
    return fig


@app.callback(
    dash.Output("forecast-content", "children"),
    dash.Input("forecastButton", "n_clicks"),
    #dash.State("datePicker", "start_date"),
    #dash.State("datePicker", "end_date"),
    dash.State("stockCode", "value"),
    dash.State("forecastDays", "value")
)
def forecast(n_clicks, stock_code, days):
    if n_clicks is None or stock_code == "":
        return
    else:
        predictions = main(stock_code, days, "forecast")
        date_array = [dt.date.today() + dt.timedelta(days=x) for x in range(1, days+1)]
        fig = plot_forecast(predictions, date_array)
        return dcc.Graph(figure=fig)

def plot_forecast(predictions, date_array):
    df = pd.DataFrame(dict(
        Date = date_array,
        Forecast = predictions
    ))
    fig = px.line(df, x=df['Date'], y=df['Forecast'],
                  title="Future Forecast Vs Date", width=800, height=400)
    fig.update_traces(mode='lines+markers')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)