from os import name
import dash
import dash_core_components as dcc   
import dash_html_components as html 
from dash.dependencies import Input, Output
from numpy.core.arrayprint import FloatingFormat
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import pandas as pd
pd.options.mode.chained_assignment = None 
from dash.exceptions import PreventUpdate
import numpy as np
from datetime import date, timedelta
from arch import arch_model
from arch.__future__ import reindexing
from functions import *

app = dash.Dash(external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

app.layout = html.Div([

    html.Div([    
       
        html.Div([
            html.H1('Dashboard',style={'text-align':'center'}, className = "start"),


            html.H6("Time Period",style={'color':'white'}),
            dcc.Dropdown(id="time_period", options=[
                {'label': '6 Months', 'value': '6m'},
                {'label': '1 year', 'value': '1y'},
                {'label': '3 years', 'value': '3y'},
                {'label': '5 years', 'value': '5y'},
            ], placeholder='Time Period', value='1y'),
            html.Br(),
            html.H6("Technical Indicators",style={'color':'white'}),
            dcc.Dropdown(id="indicators", options=[
                {'label': 'RSI', 'value': 'RSI'},
                {'label': 'SMA', 'value': 'SMA'},
                {'label': 'EMA', 'value': 'EMA'},
                {'label': 'MACD', 'value': 'MACD'},
                {'label': 'Bollinger Bands', 'value': 'BB'}
            ],placeholder='Indicator', value='BB' ),
            html.Br(),
            html.H6("Returns",style={'color':'white'}),
            dcc.Dropdown(id="returns", options=[
                {'label': 'Daily Returns', 'value': 'Daily Returns'},
                {'label': 'Cumulative Returns', 'value': 'Cumulative Returns'}
            ],placeholder='Returns', value='Daily Returns'),

            ]),

    ], className="Navigation"),

    html.Br(),html.Br(),

    html.Div([
        html.Div([
            dcc.Dropdown(id="dropdown_tickers", options=[
                {"label":"HDFC Bank Limited", "value":"HDFCBANK.NS"},
                {"label":"ICICI Bank Limited", "value":"ICICIBANK.NS"},
                {"label":"RBL Bank Limited", "value":"RBLBANK.NS"},
                {"label":"Equitas Small Finance Bank Limited", "value":"EQUITASBNK.NS"},
                {"label":"DCB Bank Limited", "value":"DCBBANK.NS"},

                {"label":"Maruti Suzuki India Limited", "value":"MARUTI.NS"},
                {"label":"Tata Motors Limited ", "value":"TATAMOTORS.NS"},
                {"label":"Escorts Limited", "value":"ESCORTS.NS"},
                {"label":"Atul Auto Limited", "value":"ATULAUTO.NS"},
                {"label":"Force Motors Limited", "value":"FORCEMOT.BO"},

                {"label":"Tata Chemicals Limited", "value":"TATACHEM.NS"},
                {"label":"Pidilite Industries Limited", "value":"PIDILITIND.NS"},
                {"label":"Deepak Nitrite Limited", "value":"DEEPAKNTR.NS"},
                {"label":"Navin Fluorine International Limited", "value":"NAVINFLUOR.NS"},
                {"label":"Valiant Organics Limited", "value":"VALIANTORG.NS"},

                {"label":"Avenue Supermarts Limited ", "value":"DMART.NS"},
                {"label":"Trent Limited", "value":"TRENT.NS"},
                {"label":"V-Mart Retail Limited", "value":"VMART.NS"},
                {"label":"Future Retail Limited", "value":"FRETAIL.NS"},
                {"label":"Shoppers Stop Limited", "value":"SHOPERSTOP.NS"},

                {"label":"Zomato Limited", "value":"ZOMATO.NS"},
                {"label":"G R Infraprojects Limited", "value":"GRINFRA.NS"},
                {"label":"Dodla Dairy Limited", "value":"DODLA.NS"},
                {"label":"India Pesticides Limited ", "value":"IPL.NS"},
                {"label":"Times Green Energy (India) Lim", "value":"TIMESGREEN.BO"},

                {"label":"DLF Limited", "value":"DLF.NS"},
                {"label":"Godrej Properties Limited", "value":"GODREJPROP.NS"},
                {"label":"Oberoi Realty Limited", "value":"OBEROIRLTY.NS"},
                {"label":"Sunteck Realty Limited ", "value":"SUNTECK.NS"},
                {"label":"Nirlon Limited", "value":"NIRLON.BO"},
            ], placeholder='Select Stock', value="HDFCBANK.NS"),

            html.Div([], id="c_graph"), 
            html.Div([], id="graphs"),
            html.Br(),
            html.H5('Closing Prices Simulations',style={'text-align':'center'}),
            html.Div([], id="gbm_graph"), 
            html.Div("Daily Volatility: ",style={'width': '90%', 'display': 'inline-block','text-align':'right'}),
            html.Div(id="sd_gm",style={'width': '10%', 'display': 'inline-block','text-align':'center'}),
            html.H5('Volatility Simulation',style={'text-align':'center'}),
            html.Div([], id="garch_graph"),
            html.H4('Risk Ratios',style={'text-align':'center'}),
            html.Div([
                html.Div([
                    html.H6("Alpha (NIFTY 50)"),
                    html.Div(id="a_val"),
                    ],style={'width': '49%', 'display': 'inline-block'}),
                html.Div([
                    html.H6("Beta (NIFTY 50)"),
                    html.Div(id="b_val"),
                    ],style={'width': '49%', 'display': 'inline-block'}),
            ]), 
            html.Div([   
                html.Div([
                    html.H6("Sharpe Ratio"),
                    html.Div(id="sr_val"),
                    ],style={'width': '49%', 'display': 'inline-block'}),
                html.Div([
                    html.H6("Sortino Ratio"),
                    html.Div(id="sor_val"),
                    ],style={'width': '49%', 'display': 'inline-block'}),
            ]),
            html.Div([
                html.H6("Standard Deviation"),
                html.Div(id="sd_val"),
                ]),
        ], id="main-content"), 
    ],className="Panel1"),

    html.Div([

        html.Div([
            dcc.Dropdown(id="dropdown_tickers2", options=[
                {"label":"HDFC Bank Limited", "value":"HDFCBANK.NS"},
                {"label":"ICICI Bank Limited", "value":"ICICIBANK.NS"},
                {"label":"RBL Bank Limited", "value":"RBLBANK.NS"},
                {"label":"Equitas Small Finance Bank Limited", "value":"EQUITASBNK.NS"},
                {"label":"DCB Bank Limited", "value":"DCBBANK.NS"},

                {"label":"Maruti Suzuki India Limited", "value":"MARUTI.NS"},
                {"label":"Tata Motors Limited ", "value":"TATAMOTORS.NS"},
                {"label":"Escorts Limited", "value":"ESCORTS.NS"},
                {"label":"Atul Auto Limited", "value":"ATULAUTO.NS"},

                {"label":"Tata Chemicals Limited", "value":"TATACHEM.NS"},
                {"label":"Pidilite Industries Limited", "value":"PIDILITIND.NS"},
                {"label":"Deepak Nitrite Limited", "value":"DEEPAKNTR.NS"},
                {"label":"Navin Fluorine International Limited", "value":"NAVINFLUOR.NS"},
                {"label":"Valiant Organics Limited", "value":"VALIANTORG.NS"},

                {"label":"Avenue Supermarts Limited ", "value":"DMART.NS"},
                {"label":"Trent Limited", "value":"TRENT.NS"},
                {"label":"V-Mart Retail Limited", "value":"VMART.NS"},
                {"label":"Future Retail Limited", "value":"FRETAIL.NS"},
                {"label":"Shoppers Stop Limited", "value":"SHOPERSTOP.NS"},

                {"label":"Zomato Limited", "value":"ZOMATO.NS"},
                {"label":"G R Infraprojects Limited", "value":"GRINFRA.NS"},
                {"label":"Dodla Dairy Limited", "value":"DODLA.NS"},
                {"label":"India Pesticides Limited ", "value":"IPL.NS"},
                {"label":"Times Green Energy (India) Lim", "value":"TIMESGREEN.BO"},

                {"label":"DLF Limited", "value":"DLF.NS"},
                {"label":"Godrej Properties Limited", "value":"GODREJPROP.NS"},
                {"label":"Oberoi Realty Limited", "value":"OBEROIRLTY.NS"},
                {"label":"Sunteck Realty Limited ", "value":"SUNTECK.NS"},
                {"label":"Nirlon Limited", "value":"NIRLON.BO"},
            ], placeholder='Select Stock'),

            html.Div([], id="c_graph2"), 
            html.Div([], id="graphs2"),
            html.Br(),
            html.H5('Closing Prices Simulations',style={'text-align':'center'}),
            html.Div([], id="gbm_graph2"), 
            html.Div("Daily Volatility: ",style={'width': '90%', 'display': 'inline-block','text-align':'right'}),
            html.Div(id="sd_gm2",style={'width': '10%', 'display': 'inline-block','text-align':'center'}),
            html.H5('Volatility Simulation',style={'text-align':'center'}),
            html.Div([], id="garch_graph2"), 
            html.H4('Risk Ratios',style={'text-align':'center'}),
            html.Div([
                html.Div([
                    html.H6("Alpha (NIFTY 50)"),
                    html.Div(id="a_val2"),
                    ],style={'width': '49%', 'display': 'inline-block'}),
                html.Div([
                    html.H6("Beta (NIFTY 50)"),
                    html.Div(id="b_val2"),
                    ],style={'width': '49%', 'display': 'inline-block'}),
            ]), 
            html.Div([   
                html.Div([
                    html.H6("Sharpe Ratio"),
                    html.Div(id="sr_val2"),
                    ],style={'width': '49%', 'display': 'inline-block'}),
                html.Div([
                    html.H6("Sortino Ratio"),
                    html.Div(id="sor_val2"),
                    ],style={'width': '49%', 'display': 'inline-block'}),
            ]),
            html.Div([
                html.H6("Standard Deviation"),
                html.Div(id="sd_val2"),
                ]),
        ], id="main-content2"),
    ], className="Panel2"),  

],className="container")


@app.callback(
            [Output("c_graph", "children")],
            [Output("graphs", "children")],
            [Output("a_val", "children")],
            [Output("b_val", "children")],
            [Output("sr_val", "children")],
            [Output("sor_val", "children")],
            [Output("sd_val", "children")],
            [Output("sd_gm", "children")],
            [Output("gbm_graph", "children")],
            [Output("garch_graph", "children")],
            [Input("time_period", "value")],
            [Input("dropdown_tickers", "value")],
            [Input("indicators", "value")],
            [Input("returns", "value")],
)

def stock_prices(v, v2, v3, v4):
    if v2 == None:
        raise PreventUpdate

    df = yf.download(v2)
    # df = pd.read_csv('data.csv')
    df = df.tail(1800)
    df.reset_index(inplace=True)

    if v=='6m':
        time_period = 126
    elif v=='1y':
        time_period = 252
    elif v=='3y':
        time_period = 756
    elif v=='5y':
        time_period = 1800

    # Alpha & Beta Ratio
    beta_r = yf.download("^NSEI")
    # beta_r = pd.read_csv('b.csv')
    beta_r = beta_r.tail(time_period)
    df_data = df.tail(time_period)
    beta_r.reset_index(inplace=True)
    
    beta_r = beta_r[["Date", 'Adj Close']]
    beta_r.columns = ['Date', "NIFTY"]
    beta_r = pd.merge(beta_r, df_data[['Date', 'Adj Close']], how='inner', on='Date')
    beta_r.columns = ['Date', 'NIFTY', 'Stock']

    beta_r[['Stock Returns','NIFTY Returns']] = beta_r[['Stock','NIFTY']]/\
        beta_r[['Stock','NIFTY']].shift(1) -1
    beta_r.dropna(inplace=True)

    cov = np.cov(beta_r["Stock Returns"],beta_r["NIFTY Returns"])
    Beta_Ratio = cov[0,1]/cov[1,1]
    Alpha_Ratio = np.mean(beta_r["Stock Returns"]) - Beta_Ratio*np.mean(beta_r["NIFTY Returns"])

    Alpha_Ratio = round(Alpha_Ratio*12,4)
    Beta_Ratio = round(Beta_Ratio,2)

    # Standard Deviation
    SD = round(df_data['Adj Close'].std(),2)

    # Sharpe & Sortino Ratio
    df_data['Normalized Returns'] = df_data['Adj Close']/df_data.iloc[0]['Adj Close']
    df_data['Daily Normalized Returns'] = df_data['Normalized Returns'].pct_change(1)
    Sharpe_Ratio = round((df_data['Daily Normalized Returns'].mean()/df_data['Daily Normalized Returns'].std())*(252**0.5),2)

    down_returns = df_data.loc[df_data['Daily Normalized Returns'] < 0]
    down_SD = down_returns['Daily Normalized Returns'].std()
    Sortino_Ratio = round((df_data['Daily Normalized Returns'].mean()/down_SD)*(252**0.5),2)

    # Plotting over the time period's data
    MACD(df_data)
    RSI(df_data)
    BB(df_data)
    df_data['SMA'] = SMA(df_data)
    df_data['EMA'] = EMA(df_data)

    fig = get_stock_price_fig(df_data,v3,v4)
    current = df_data.iloc[-1][2]
    yesterday = df_data.iloc[-2][2]

    # Change graph
    fig1 = go.Figure(go.Indicator(
            mode="number+delta",
            value=current,
            delta={'reference': yesterday, 'relative': True,'valueformat':'.2%'}))
    fig1.update_traces(delta_font={'size':15},number_font = {'size':40})
    fig1.update_layout(height=100, margin=dict(b=10,t=20,l=100),)
    if current >= yesterday:
            fig1.update_traces(delta_increasing_color='green')
    elif current < yesterday:
            fig1.update_traces(delta_decreasing_color='red')

    df = df[['Date','Adj Close']]
    # GBM Model
    fig2, sd_gm = GBM(df.tail(30))

    # GARCH Model
    fig3 = GARCH(df.tail(756))

    return [dcc.Graph(figure=fig1,config={'displayModeBar': False}),
            dcc.Graph(figure=fig,config={'displayModeBar': False}),
            Alpha_Ratio,
            Beta_Ratio,
            Sharpe_Ratio,
            Sortino_Ratio,
            SD,
            round(sd_gm,4),
            dcc.Graph(figure=fig2,config={'displayModeBar': False}),
            dcc.Graph(figure=fig3,config={'displayModeBar': False}),]


@app.callback(
            [Output("c_graph2", "children")],
            [Output("graphs2", "children")],
            [Output("a_val2", "children")],
            [Output("b_val2", "children")],
            [Output("sr_val2", "children")],
            [Output("sor_val2", "children")],
            [Output("sd_val2", "children")],
            [Output("sd_gm2", "children")],
            [Output("gbm_graph2", "children")],
            [Output("garch_graph2", "children")],
            [Input("time_period", "value")],
            [Input("dropdown_tickers2", "value")],
            [Input("indicators", "value")],
            [Input("returns", "value")],
)

def stock_prices2(v, v2, v3, v4):
    if v2 == None:
        raise PreventUpdate

    df2 = yf.download(v2)
    # df2 = pd.read_csv('data.csv')
    df2 = df2.tail(1800)
    df2.reset_index(inplace=True)

    if v=='6m':
        time_period = 126
    elif v=='1y':
        time_period = 252
    elif v=='3y':
        time_period = 756
    elif v=='5y':
        time_period = 1800

    # Alpha & Beta Ratio
    beta_r2 = yf.download("^NSEI")
    # beta_r2 = pd.read_csv('b.csv')
    beta_r2 = beta_r2.tail(time_period)
    df_data = df2.tail(time_period)
    beta_r2.reset_index(inplace=True)
    
    beta_r2 = beta_r2[["Date", 'Adj Close']]
    beta_r2.columns = ['Date', "NIFTY"]
    beta_r2 = pd.merge(beta_r2, df_data[['Date', 'Adj Close']], how='inner', on='Date')
    beta_r2.columns = ['Date', 'NIFTY', 'Stock']

    beta_r2[['Stock Returns','NIFTY Returns']] = beta_r2[['Stock','NIFTY']]/\
        beta_r2[['Stock','NIFTY']].shift(1) -1
    beta_r2.dropna(inplace=True)

    cov = np.cov(beta_r2["Stock Returns"],beta_r2["NIFTY Returns"])
    Beta_Ratio = cov[0,1]/cov[1,1]
    Alpha_Ratio = np.mean(beta_r2["Stock Returns"]) - Beta_Ratio*np.mean(beta_r2["NIFTY Returns"])

    Alpha_Ratio = round(Alpha_Ratio*12,4)
    Beta_Ratio = round(Beta_Ratio,2)

    # Standard Deviation
    SD = round(df_data['Adj Close'].std(),2)

    # Sharpe & Sortino Ratio
    df_data['Normalized Returns'] = df_data['Adj Close']/df_data.iloc[0]['Adj Close']
    df_data['Daily Normalized Returns'] = df_data['Normalized Returns'].pct_change(1)
    Sharpe_Ratio = round((df_data['Daily Normalized Returns'].mean()/df_data['Daily Normalized Returns'].std())*(252**0.5),2)

    down_returns = df_data.loc[df_data['Daily Normalized Returns'] < 0]
    down_SD = down_returns['Daily Normalized Returns'].std()
    Sortino_Ratio = round((df_data['Daily Normalized Returns'].mean()/down_SD)*(252**0.5),2)

    # Plotting over the time period's data
    df_data = df_data.tail(time_period)
    MACD(df_data)
    RSI(df_data)
    BB(df_data)
    df_data['SMA'] = SMA(df_data)
    df_data['EMA'] = EMA(df_data)

    fig = get_stock_price_fig(df_data,v3,v4)
    current = df_data.iloc[-1][2]
    yesterday = df_data.iloc[-2][2]

    # Change graph
    fig1 = go.Figure(go.Indicator(
            mode="number+delta",
            value=current,
            delta={'reference': yesterday, 'relative': True,'valueformat':'.2%'}))
    fig1.update_traces(delta_font={'size':15},number_font = {'size':40})
    fig1.update_layout(height=100, margin=dict(b=10,t=10,l=100),)
    if current >= yesterday:
            fig1.update_traces(delta_increasing_color='green')
    elif current < yesterday:
            fig1.update_traces(delta_decreasing_color='red')

    # GBM Model
    fig2, sd_gm2 = GBM(df2.tail(30))

    # GARCH Model
    fig3 = GARCH(df2.tail(756))

    return [dcc.Graph(figure=fig1,config={'displayModeBar': False}),
            dcc.Graph(figure=fig,config={'displayModeBar': False}),
            Alpha_Ratio,
            Beta_Ratio,
            Sharpe_Ratio,
            Sortino_Ratio,
            SD,
            round(sd_gm2,4),
            dcc.Graph(figure=fig2,config={'displayModeBar': False}),
            dcc.Graph(figure=fig3,config={'displayModeBar': False}),]
app.run_server(debug=True)