# src/ou_model.py

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arbitragelab.optimal_mean_reversion as omr
import statsmodels.api as sm

def download_log_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)
    for ticker in tickers.split():
        data["Close", ticker] = np.log(data["Close", ticker])
    return data["Close"][tickers.split()]

def train_ou_model(data, discount_rate=[0.05, 0.05], transaction_cost=[0.02, 0.02], stop_loss=0.2):
    model = omr.OrnsteinUhlenbeck()
    model.fit(data, data_frequency="D", discount_rate=discount_rate,
              transaction_cost=transaction_cost, stop_loss=stop_loss)
    return model

def print_z_scores(model):
    desc = model.description()
    vol = desc['volatility']
    mean = model.theta
    entry = model.optimal_entry_level()
    exit = model.optimal_liquidation_level()
    print(f"Entry: {(entry - mean) / vol:.2f}σ, Exit: {(exit - mean) / vol:.2f}σ")

def regression_ou_params(spread_series):
    X_t = spread_series.shift(1).iloc[1:]
    Y = spread_series.iloc[1:]
    X_t = sm.add_constant(X_t)
    model = sm.OLS(Y, X_t).fit()
    beta1 = model.params[1]
    beta0 = model.params[0]
    mu = 1 - beta1
    theta = beta0 / mu
    return mu, theta
