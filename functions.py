import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import pandas as pd
pd.options.mode.chained_assignment = None 
import numpy as np
from datetime import date, timedelta, datetime
from arch import arch_model
from arch.__future__ import reindexing
import statsmodels.graphics.tsaplots as sgt
import matplotlib.pyplot as plt
from functions import *

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

## Function for plotting Stock Prices, Volume, Indicators & Returns
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
                fig.add_trace(go.Scatter(x = df['Date'], y=df['RSI'], mode="lines", name = 'RSI',
                marker=dict(color='rgb(31, 119, 180)'), showlegend = False),row = 3, col= 1)
                fig.layout.xaxis.showgrid=False
        elif v2=='SMA':
                fig.add_trace(go.Scatter(x = df['Date'], y=df['SMA'], mode="lines", name = 'SMA', 
                showlegend = False, marker=dict(color='rgb(31, 119, 180)')),row = 3, col= 1)
                fig.layout.xaxis.showgrid=False
        elif v2=='EMA':
                fig.add_trace(go.Scatter(x = df['Date'], y=df['EMA'], mode="lines", name = 'EMA', 
                showlegend = False, marker=dict(color='rgb(31, 119, 180)')),row = 3, col= 1)
                fig.layout.xaxis.showgrid=False
        elif v2=='MACD':
                fig.add_trace(go.Scatter(x = df['Date'], y=df['MACD'], mode="lines",name = 'MACD', 
                showlegend = False, marker=dict(color='rgb(31, 119, 180)')),row = 3, col= 1)
                fig.add_trace(go.Scatter(x = df['Date'], y=df['Signal_Line'], mode="lines",name='Signal_Line', 
                showlegend = False, marker=dict(color='#ff3333')),row = 3, col= 1)
                fig.layout.xaxis.showgrid=False
        elif v2=='BB':
                fig.add_trace(go.Scatter(x = df['Date'], y=df['Adj Close'], mode="lines",
                line=dict(color='rgb(31, 119, 180)'),name = 'Close',showlegend = False),row = 3, col= 1) 
                fig.add_trace(go.Scatter(x = df['Date'], y=df['BOLU'],mode="lines", line=dict(width=0.5), 
                marker=dict(color="#89BCFD"),showlegend=False,name = 'Upper Band'),row = 3, col= 1)
                fig.add_trace(go.Scatter(x = df['Date'], y=df['BOLD'], mode="lines",line=dict(width=0.5),
                marker=dict(color="#89BCFD"),showlegend=False,fillcolor='rgba(228, 240, 255, 0.5)',fill='tonexty',name = 'Lower Band'),row = 3, col= 1)
                fig.layout.xaxis.showgrid=False        

        # Returns
        if v3=="Daily Returns":
                rets = df['Adj Close'] / df['Adj Close'].shift(1) - 1
                fig.add_trace(go.Scatter(x = df['Date'], y=rets, mode="lines", showlegend = False, name = 'Daily Return', line=dict(color='#FF4136')),
                row = 4, col= 1,)
                fig.layout.xaxis.showgrid=False
        elif v3=="Cumulative Returns":
                rets = df['Adj Close'] / df['Adj Close'].shift(1) - 1
                cum_rets = (rets + 1).cumprod()
                fig.add_trace(go.Scatter(x = df['Date'], y=cum_rets, mode="lines", showlegend = False, name = 'Cumulative Returns', line=dict(color='#FF4136')),
                row = 4, col=1)
                fig.layout.xaxis.showgrid=False

        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.update_layout(margin=dict(b=0,t=0,l=0,r=0),plot_bgcolor='#ebf3ff',width=500, height=600, 
                        xaxis_showticklabels=True, xaxis4_showticklabels=False, xaxis3_showgrid = False, xaxis4_showgrid = False)
        fig.layout.xaxis.showgrid=False
        return fig

## Function for simulation of prices using Geometric Brownian Modeling 
def GBM(df):

        end_date = date.today().isoformat()   
        pred_end_date = (date.today()+timedelta(days=30)).isoformat()
        
        df = df.reset_index(drop=True)

        returns = (df.loc[1:,'Adj Close'] - df.shift(1).loc[1:,'Adj Close'])/df.shift(1).loc[1:,'Adj Close']

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
        pred_no = 20
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
                fig.add_trace(go.Scatter(mode="lines",showlegend = False,
                                x = df['Date'], y = df['Adj Close'],name = 'Close'))
                fig.add_trace(go.Scatter(mode="lines",showlegend = False,
                                x=pd.date_range(start=df['Date'].max(),
                                end = pred_end_date, freq='D').map(lambda k:
                                k if k.isoweekday() in range(1,6) else np.nan).dropna(),
                                y=Pred[i,:],name='GBM '+str(i)))
                fig.layout.xaxis.showgrid=False   
                fig.update_layout(margin=dict(b=0,t=0,l=0,r=0),plot_bgcolor='#ebf3ff',width=500, height=300)

        return fig, sd


## Function for forecasting volatility using GARCH
def GARCH(df):
        pred_end_date = (date.today()+timedelta(days=30)).isoformat()
        df = df.reset_index(drop = True)
        df['Date']= pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df['returns'] = df['Adj Close'].pct_change(1).mul(100)
        df['vola'] = df['returns'].abs()
        train_df = df.head(630)
        test_df = df.tail(126)

        garch_df = pd.DataFrame(df['returns'].shift(1).loc[df.index])
        garch_df.at[train_df.index, 'returns'] = train_df['returns']

        model = arch_model(garch_df['returns'][1:], p = 1, o = 0, q = 1, vol = "GARCH") 
        model_results = model.fit(last_obs = np.datetime64(test_df.index[0]), update_freq = 5)
        #model_results.summary()

        predictions_df = test_df.copy()
        predictions_df["Predictions"] = model_results.forecast().residual_variance.loc[test_df.index]
        # print(predictions_df['Predictions'])

        forecasts = model_results.forecast(horizon=30, start=test_df.index[-1], method='simulation')
        forecasts = forecasts.residual_variance.T

        fig = go.Figure()
        fig.add_trace(go.Scatter(mode='lines', showlegend=False,
                        x = pd.date_range(start=test_df.index[-30],end=test_df.index[-1]),
                        y = predictions_df['vola'],name='Volatility'))
        fig.add_trace(go.Scatter(mode='lines', showlegend=False,
                        x = pd.date_range(start=test_df.index[-1],end=pd.to_datetime(pred_end_date,format='%Y-%m-%d')),
                        y=forecasts[test_df.index[-1]],name='Forecast'))
        fig.layout.xaxis.showgrid=False   
        fig.update_layout(margin=dict(b=0,t=0,l=0,r=0),plot_bgcolor='#ebf3ff',width=500, height=300)

        return fig