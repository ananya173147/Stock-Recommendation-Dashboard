import dash
import dash_core_components as dcc   
import dash_html_components as html 
from dash.dependencies import Input, Output
from dash_html_components.Div import Div
import yfinance as yf
import pandas as pd
pd.options.mode.chained_assignment = None 
from dash.exceptions import PreventUpdate
from datetime import date, datetime
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
                {'label': 'Bollinger Bands', 'value': 'Bollinger Bands'}
            ],placeholder='Indicator', value='Bollinger Bands' ),
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
            ], placeholder='Select Stock'),

            html.Div([], id="c_graph"), 
            html.Div([], id="graphs"),
            html.Br(),
            html.H4('Past Trend vs. Future Projections',style={'text-align':'center'}),
            html.H5('Closing Prices',style={'text-align':'center'}),
            html.Div([], id="gbm_graph"),
            html.Br(), 
            html.H5('Daily Volatility (%)',style={'text-align':'center'}),
            html.Div([], id="garch_graph"),
            html.Br(),
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
            html.H4('Past Trend vs. Future Projections',style={'text-align':'center'}),
            html.H5('Closing Prices',style={'text-align':'center'}),
            html.Div([], id="gbm_graph2"), 
            html.Br(),
            html.H5('Daily Volatility (%)',style={'text-align':'center'}),
            html.Div([], id="garch_graph2"),
            html.Br(), 
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
    html.Br(),
    html.Div([
        html.H3('Interpretation',style={'text-align':'center'}),
        html.H5('Technical indicators'),
        html.Li('Bollinger Bands is a measure of volatility. High volatility is signified by wide bands while low volatility is signified by narrow bands. Generally, high volatility is followed by low volatility'),
        html.Li('RSI or Relative Strength Index, is a measure to evaluate overbought and oversold conditions.'),
        html.Li('SMA or Simple Moving Average using 50 day (fast) and 200 day (slow) lines - short term going above long term is bullish trend. Short term going below long term is bearish'),
        html.Li('EMA or Exponential Moving Average gives higher significance to recent price data'),
        html.Li('MACD or Moving Average Convergence Divergence signifies no trend reversal unless there are crossovers. The market is bullish when signal line crosses above blue line, bearish when signal line crosses below blue line'),
        
        html.H5('Risk ratios'),
        html.Li('Alpha: Return performance as compared to benchmark of market'),
        html.Li('Beta: Relative price movement of a stock to go up and down as compared to the market trend'),
        html.Li('Sharpe Ratio: Returns generated per unit of risk - the higher the better'),
        html.Li('Sortino Ratio: Returns as compared to only downside risk'),
    ])

    ],className="Panels"),

],className="container")

beta_r = N50()

@app.callback(
            [Output("c_graph", "children")],
            [Output("graphs", "children")],
            [Output("a_val", "children")],
            [Output("b_val", "children")],
            [Output("sr_val", "children")],
            [Output("sor_val", "children")],
            [Output("sd_val", "children")],
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

    if os.path.exists(v2+'.csv'):
        df = pd.read_csv(v2+'.csv')
        now = datetime.now()
        today345pm = now.replace(hour=15, minute=45, second=0, microsecond=0)
        if df['Date'].iloc[-1]!=date.today().isoformat() and date.today().isoweekday() in range(1,6) and now>today345pm:
            df = yf.download(v2,start='2016-01-01')
            df.reset_index(inplace=True)
            df.to_csv(v2+'.csv')
    else:
        df = yf.download(v2,start='2016-01-01')
        df.reset_index(inplace=True)
        df.to_csv(v2+'.csv')
    
    df = df.tail(1800)
    df['Date']= pd.to_datetime(df['Date'])

    if v=='6m':
        time_period = 126
    elif v=='1y':
        time_period = 252
    elif v=='3y':
        time_period = 756
    elif v=='5y':
        time_period = 1800

    # Alpha & Beta Ratio
    beta_r = pd.read_csv('benchmark.csv')
    beta_r = beta_r.tail(time_period)
    df_data = df.tail(time_period)
    Alpha_Ratio, Beta_Ratio = alpha_beta(beta_r, df_data)

    # Standard Deviation
    SD = round(df_data['Adj Close'].std(),2)

    # Sharpe & Sortino Ratio
    Sharpe_Ratio, Sortino_Ratio = sharpe_sortino(df_data)

    # Plotting over the time period's data
    MACD(df)
    RSI(df)
    BB(df)
    df['SMA_50'] = SMA(df, 50)
    df['SMA_200'] = SMA(df, 200)
    df['EMA'] = EMA(df)

    fig = get_stock_price_fig(df.tail(time_period),v3,v4)
    current = df_data.iloc[-1][2]
    yesterday = df_data.iloc[-2][2]

    # Change graph
    fig1 = change_graph(current,yesterday)

    df = df[['Date','Adj Close']]

    # GBM Model
    fig2= gbm(df.tail(30))

    # GARCH Model
    fig3 = garch(df.tail(30))

    return [dcc.Graph(figure=fig1,config={'displayModeBar': False}),
            dcc.Graph(figure=fig,config={'displayModeBar': False}),
            Alpha_Ratio,
            Beta_Ratio,
            Sharpe_Ratio,
            Sortino_Ratio,
            SD,
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

    if os.path.exists(v2+'.csv'):
        df2 = pd.read_csv(v2+'.csv')
        now = datetime.now()
        today345pm = now.replace(hour=15, minute=45, second=0, microsecond=0)
        if df2['Date'].iloc[-1]!=date.today().isoformat() and date.today().isoweekday() in range(1,6) and now>today345pm:
            df2 = yf.download(v2,start='2016-01-01')
            df2.reset_index(inplace=True)
            df2.to_csv(v2+'.csv')
    else:
        df2 = yf.download(v2,start='2016-01-01')
        df2.reset_index(inplace=True)
        df2.to_csv(v2+'.csv')

    df2 = df2.tail(1800)
    df2['Date']= pd.to_datetime(df2['Date'])

    if v=='6m':
        time_period = 126
    elif v=='1y':
        time_period = 252
    elif v=='3y':
        time_period = 756
    elif v=='5y':
        time_period = 1800

    # Alpha & Beta Ratio
    beta_r2 = pd.read_csv('benchmark.csv')
    beta_r2 = beta_r2.tail(time_period)
    df_data = df2.tail(time_period)
    Alpha_Ratio, Beta_Ratio = alpha_beta(beta_r2, df_data)

    # Standard Deviation
    SD = round(df_data['Adj Close'].std(),2)

    # Sharpe & Sortino Ratio
    Sharpe_Ratio, Sortino_Ratio = sharpe_sortino(df_data)

    # Plotting over the time period's data
    MACD(df2)
    RSI(df2)
    BB(df2)
    df2['SMA_50'] = SMA(df2, 50)
    df2['SMA_200'] = SMA(df2, 200)
    df2['EMA'] = EMA(df2)

    fig = get_stock_price_fig(df2.tail(time_period),v3,v4)
    current = df2.iloc[-1][2]
    yesterday = df2.iloc[-2][2]

    # Change graph
    fig1 = change_graph(current,yesterday)

    df2 = df2[['Date','Adj Close']]

    # GBM Model
    fig2= gbm(df2.tail(30))

    # GARCH Model
    fig3 = garch(df2.tail(30))

    return [dcc.Graph(figure=fig1,config={'displayModeBar': False}),
            dcc.Graph(figure=fig,config={'displayModeBar': False}),
            Alpha_Ratio,
            Beta_Ratio,
            Sharpe_Ratio,
            Sortino_Ratio,
            SD,
            dcc.Graph(figure=fig2,config={'displayModeBar': False}),
            dcc.Graph(figure=fig3,config={'displayModeBar': False}),]
app.run_server(debug=True)