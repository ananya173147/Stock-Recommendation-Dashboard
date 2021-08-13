from os import name
import dash
import dash_core_components as dcc   
import dash_html_components as html 
from dash.dependencies import Input, Output
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import pandas as pd
pd.options.mode.chained_assignment = None 
from dash.exceptions import PreventUpdate
import numpy as np
from datetime import date, timedelta
from sklearn import preprocessing

## Functions for calculating SMA, EMA, MACD, RSI
def SMA(data, period = 100, column = 'Adj Close'):
        return data[column].rolling(window=period).mean()

def EMA(data, period = 20, column = 'Adj Close'):
        return data[column].ewm(span=period, adjust = False).mean()

def MACD(data, period_long = 26, period_short = 12, period_signal = 9, column = 'Adj Close'):
        shortEMA = EMA(data, period_short, column=column)
        longEMA = EMA(data, period_long, column=column)
        data['MACD'] = shortEMA - longEMA
        data['Signal_Line'] = EMA(data, period_signal, column = 'MACD')
        return data

def RSI(data, period = 14, column = 'Adj Close'):
        delta = data[column].diff(1)
        delta = delta[1:]
        up = delta.copy()
        down = delta.copy()
        up[up<0] = 0
        down[down>0] = 0
        data['up'] = up
        data['down'] = down
        avg_gain = SMA(data, period, column = 'up')
        avg_loss = abs(SMA(data, period, column = 'down'))
        RS = avg_gain/avg_loss
        RSI = 100.0 - (100.0/(1.0+RS))
        data['RSI'] = RSI
        return data

def BB(data):
        data['TP'] = (data['Adj Close'] + data['Low'] + data['High'])/3
        data['std'] = data['TP'].rolling(20).std(ddof=0)
        data['MA-TP'] = data['TP'].rolling(20).mean()
        data['BOLU'] = data['MA-TP'] + 2*data['std']
        data['BOLD'] = data['MA-TP'] - 2*data['std']
        return data

def get_stock_price_fig(df,v2,v3):

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.039,
    row_width=[0.1,0.2,0.1, 0.3],subplot_titles=("", "", "", ""))

    fig.add_trace(go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Adj Close'],showlegend = False, name = 'Price'),row=1,col=1)

    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'],opacity=0.5,showlegend = False, name = 'Volume'),
    row = 2, col= 1)

    # Indicators
    if v2=='RSI':
        fig.add_trace(go.Scatter(x = df['Date'], y=df['RSI'], mode="lines", name = 'RSI'),
        row = 3, col= 1)
        fig.layout.xaxis.showgrid=False
    elif v2=='SMA':
        fig.add_trace(go.Scatter(x = df['Date'], y=df['SMA'], mode="lines", name = 'SMA'),
        row = 3, col= 1)
        fig.layout.xaxis.showgrid=False
    elif v2=='EMA':
        fig.add_trace(go.Scatter(x = df['Date'], y=df['EMA'], mode="lines", name = 'EMA'),
        row = 3, col= 1)
        fig.layout.xaxis.showgrid=False
    elif v2=='MACD':
        fig.add_trace(go.Scatter(x = df['Date'], y=df['MACD'], mode="lines",name = 'MACD'),
        row = 3, col= 1)
        fig.add_trace(go.Scatter(x = df['Date'], y=df['Signal_Line'], mode="lines",name='Signal_Line'),
        row = 3, col= 1)
        fig.layout.xaxis.showgrid=False
    elif v2=='BB':
        fig.add_trace(go.Scatter(x = df['Date'], y=df['Adj Close'], mode="lines",line=dict(color='rgb(31, 119, 180)'),name = 'Close'),
        row = 3, col= 1) 
        fig.add_trace(go.Scatter(x = df['Date'], y=df['BOLU'],mode="lines", line=dict(width=0.5), marker=dict(color="#89BCFD"),showlegend=False,name = 'Upper Band'),
        row = 3, col= 1)
        fig.add_trace(go.Scatter(x = df['Date'], y=df['BOLD'], mode="lines",line=dict(width=0.5),marker=dict(color="#89BCFD"),showlegend=False,fillcolor='rgba(228, 240, 255, 0.5)',fill='tonexty',name = 'Lower Band'),
        row = 3, col= 1)
        fig.layout.xaxis.showgrid=False         

    # Returns
    if v3=="Daily Returns":
        rets = df['Adj Close'] / df['Adj Close'].shift(1) - 1
        fig.add_trace(go.Scatter(x = df['Date'], y=rets, mode="lines", name = 'Daily Return'),
        row = 4, col= 1,)
        fig.layout.xaxis.showgrid=False
    elif v3=="Cumulative Returns":
        rets = df['Adj Close'] / df
        ['Adj Close'].shift(1) - 1
        cum_rets = (rets + 1).cumprod()
        fig.add_trace(go.Scatter(x = df['Date'], y=cum_rets, mode="lines", name = 'Cumulative Returns'),
        row = 4, col=1)
        fig.layout.xaxis.showgrid=False

    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(margin=dict(b=0,t=0,l=0,r=0),plot_bgcolor='#F3F6FA',width=1000, height=600, 
                      xaxis_showticklabels=True, xaxis4_showticklabels=False, xaxis3_showgrid = False, xaxis4_showgrid = False)
    fig.layout.xaxis.showgrid=False
    return fig


## Function for applying Geometric Brownian Model 
def GBM(df):

    end_date = date.today().isoformat()   
    #start_date = (date.today()-timedelta(days=30)).isoformat()
    pred_end_date = (date.today()+timedelta(days=30)).isoformat()
    
    df = df[['Date','Adj Close']].reset_index(drop=True)

    returns = (df.loc[1:,'Adj Close'] - \
        df.shift(1).loc[1:,'Adj Close'])/\
        df.shift(1).loc[1:,'Adj Close']

    # Assigning Parameters
    S = df.loc[df.shape[0]-1,'Adj Close']
    dt = 1
    trading_days = pd.date_range(start=pd.to_datetime(end_date,format='%Y-%m-%d') + 
                    pd.Timedelta('1 days'),
                    end=pd.to_datetime(pred_end_date,format='%Y-%m-%d')).to_series().map(lambda k:
                    1 if k.isoweekday() in range(1,6) else 0).sum()
    T = trading_days
    N = T/dt
    t = np.arange(1,int(N)+1)
    mu = np.mean(returns)
    sd = np.std(returns)
    pred_no = 4
    b = {str(k): np.random.normal(0,1,int(N)) for k in range(1, pred_no+1)}
    W = {str(k): b[str(k)].cumsum() for k in range(1, pred_no+1)}

    # Drift & Diffusion 
    drift = (mu - 0.5 * sd**2) * t
    diffusion = {str(k): sd*W[str(k)] for k in range(1, pred_no+1)}
    #print(drift, diffusion)

    # Prediction Values
    Pred = np.array([S*np.exp(drift+diffusion[str(k)]) for k in range(1, pred_no+1)]) 
    Pred = np.hstack((np.array([[S] for k in range(pred_no)]), Pred))
    #print(Pred)

    fig = go.Figure()
    for i in range(pred_no):
        fig.add_trace(go.Scatter(mode="lines",x=pd.date_range(start=df['Date'].max(),
                        end = pred_end_date, freq='D').map(lambda k:
                        k if k.isoweekday() in range(1,6) else np.nan).dropna(),
                        y=Pred[i,:],name='GBM '+str(i),
                        text=["Daily Volatility: "+str(sd)],
                        textposition="bottom center"))
        fig.layout.xaxis.showgrid=False   
        fig.update_layout(margin=dict(b=0,t=0,l=0,r=0),plot_bgcolor='#F3F6FA')

    return fig

app = dash.Dash(external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

app.layout = html.Div([

    html.Div([    
       
        html.Div([
            html.H1('Dashboard',style={'text-align':'center'}, className = "start"),
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

            ], placeholder='Select Stock', value='HDFCBANK.NS'),
            dcc.Dropdown(id="indicators", options=[
                {'label': 'RSI', 'value': 'RSI'},
                {'label': 'SMA', 'value': 'SMA'},
                {'label': 'EMA', 'value': 'EMA'},
                {'label': 'MACD', 'value': 'MACD'},
                {'label': 'Bollinger Bands', 'value': 'BB'}
            ],placeholder='Indicator', value='BB' ),

            dcc.Dropdown(id="Returns", options=[
                {'label': 'Daily Returns', 'value': 'Daily Returns'},
                {'label': 'Cumulative Returns', 'value': 'Cumulative Returns'}
            ],placeholder='Returns', value='Daily Returns'),
            ]),

    ], className="Navigation"),

    html.Br(),html.Br(),
    html.Div([
        html.Div([
            html.Div([], id="c_graph"), 
            html.Div([], id="graphs"),
            html.Br(),
            html.H4('Risk Ratios (Over the last <5 years)',style={'text-align':'center'}),

            html.Div([
                html.H5("Standard Deviation",style={'width': '39%', 'display': 'inline-block'}),
                html.Div(id="sd_val",style={'width': '20%', 'display': 'inline-block'}),
            ],style={'width': '60%', 'display': 'inline-block'}),
            html.Div([
                html.H5("Beta",style={'width': '50%', 'display': 'inline-block'}),
                html.Div(id="b_val",style={'width': '9%', 'display': 'inline-block'}),
            ],style={'width': '39%', 'display': 'inline-block'}),
            html.Div([
                html.H5("Sharpe Ratio",style={'width': '39%', 'display': 'inline-block'}),
                html.Div(id="sr_val",style={'width': '24%', 'display': 'inline-block'}),
            ],style={'width': '60%', 'display': 'inline-block'}),
            html.Div([
                html.H5("Alpha",style={'width': '50%', 'display': 'inline-block'}),
                html.Div(id="a_val",style={'width': '24%', 'display': 'inline-block'}),
            ],style={'width': '39%', 'display': 'inline-block'}),
            html.Br(),
            html.H4('Prediction of prices over the next month',style={'text-align':'center'}),
            html.Div([], id="gbm_graph"), 
        ], id="main-content")           

    ],className="content")

],className="container")



@app.callback(
            [Output("c_graph", "children")],
            [Output("graphs", "children")],
            [Output("sd_val", "children")],
            [Output("b_val", "children")],
            [Output("sr_val", "children")],
            [Output("a_val", "children")],
            [Output("gbm_graph", "children")],
            [Input("dropdown_tickers", "value")],
            [Input("indicators", "value")],
            [Input("Returns", "value")],
)

def stock_prices(v, v2, v3):
    if v == None:
        raise PreventUpdate

    df = yf.download(v)
    # df.to_csv("data.csv")
    # df = pd.read_csv("data.csv")
    df.reset_index(inplace=True)
    df = df.tail(1800)

    # Standard Deviation
    SD = round(df['Adj Close'].std(),2)

    # Beta & Alpha Ratio
    beta_r = yf.download("^NSEI")
    # #beta_r.to_csv("b.csv")
    # beta_r = pd.read_csv("b.csv")
    beta_r.reset_index(inplace=True)
    beta_r = beta_r[["Date", 'Adj Close']]
    beta_r.columns = ['Date', "NIFTY"]
    beta_r = beta_r.tail(1800)
    beta_r = pd.merge(beta_r, df[['Date', 'Adj Close']], how='inner', on='Date')
    beta_r.columns = ['Date', 'NIFTY', 'Stock']

    beta_r[['Stock Returns','NIFTY Returns']] = beta_r[['Stock','NIFTY']]/\
        beta_r[['Stock','NIFTY']].shift(1) -1
    beta_r.dropna(inplace=True)

    cov = np.cov(beta_r["Stock Returns"],beta_r["NIFTY Returns"])
    Beta_Ratio = cov[0,1]/cov[1,1]
    Alpha_Ratio = np.mean(beta_r["Stock Returns"]) - Beta_Ratio*np.mean(beta_r["NIFTY Returns"])

    Alpha_Ratio = round(Alpha_Ratio*12,4)
    Beta_Ratio = round(Beta_Ratio,2)

    # Sharpe Ratio
    df['Normalized Returns'] = df['Adj Close']/df.iloc[0]['Adj Close']
    df['Daily Normalized Returns'] = df['Normalized Returns'].pct_change(1)
    Sharpe_Ratio = round((df['Daily Normalized Returns'].mean()/df['Daily Normalized Returns'].std())*(252**0.5),2)

    # Plotting over the last year's data
    df = df.tail(365)
    MACD(df)
    RSI(df)
    BB(df)
    df['SMA'] = SMA(df)
    df['EMA'] = EMA(df)

    fig = get_stock_price_fig(df,v2,v3)
    current = df.iloc[-1][2]
    yesterday = df.iloc[-2][2]

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

    # GBM Model
    df = df.tail(30)
    fig2 = GBM(df)

    return [dcc.Graph(figure=fig1,config={'displayModeBar': False}),
            dcc.Graph(figure=fig,config={'displayModeBar': False}),
            SD,
            Beta_Ratio,
            Sharpe_Ratio,
            Alpha_Ratio,
            dcc.Graph(figure=fig2,config={'displayModeBar': False}),]


app.run_server(debug=True)