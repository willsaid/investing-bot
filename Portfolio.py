""" Computes a portfolios Cumulative Return, Average Daily Returns, Risk, and Sharpe Ratio
Optimizes your portfolio with stock covariance

todo
fix init start date end date
show before and after better in legend
improve printing output, make code pretty, etc
"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import scipy.optimize as spo
import time

import QuoteHistory



class Portfolio(object):

    def __init__(self, start_value, symbols, allocs, start_date, end_date, df=None):
        self.start_value = start_value
        self.start_date = start_date
        self.end_date = end_date
        self.allocs = allocs
        self.symbols = symbols
        if df is None:
            df = self.get_data(symbols)
        self.df = df
        self.port_val = self.get_port_val(df, allocs, start_value)
        self.daily_returns = self.daily_returns(self.port_val)

    def get_data(self, symbols):
        """Read stock data (adjusted close) for given symbols from CSV files."""
        df = QuoteHistory.get_data('^GSPC', self.start_date, self.end_date)
        df = df.rename(columns={'Adj Close': '^GSPC'}).drop('Volume', axis=1).dropna()

        for symbol in symbols:
            try:
                df_temp = QuoteHistory.get_data(symbol, self.start_date, self.end_date)
                df_temp = df_temp.drop('Volume', axis=1)
                df[symbol] = df_temp
            except Exception:
                print('Failed to determine for {}'.format(symbol))

        df = df.drop('^GSPC', axis=1) # get rid of first stock
        return df


    def get_port_val(self, df, allocs, start_value):
        # print("raw df")
        # print(df)
        df = df / df.iloc[0]
        # print("normalized")
        # print('here')
        # print(df)
        # print(allocs)
        df = df * allocs
        # print("with allocs")
        # print(df)
        df = df * start_value
        # print("with start values")
        # print(df)
        df = df.sum(axis = 1)
        # print("port value, after summing")
        # print(df)
        return df

    def daily_returns(self, df):
        returns = df.copy()
        returns[1:] = (df[1:] / df[:-1].values) - 1
        returns.ix[0] = 0
        return returns



    # MAIN FUNCTIONS

    def cumulative_return(self):
        return self.port_val[-1] / self.port_val[0] - 1


    def avg_daily_returns(self):
        # print("daily returns")
        # print(self.daily_returns)
        return self.daily_returns.mean()


    def risk(self):
        return self.daily_returns.std()


    def sharpe_ratio(self):
        """
        the annualized risk-adjusted return
        assuming daily frequency, and an annual risk-free amount of 2%
        Sharpe Ratio = sqrt(252) * mean(daily returns - daily risk free)
                       -------------------------------------------------
                       stddev(daily returns)
        SR should be around 0.5-3
        """
        # annual_free = 1.02 # 2 percent annual
        # daily_risk_free = annual_free ** (1. / 252) - 1
        sqrt = np.sqrt(252)
        mean = (self.daily_returns - 0).mean()
        stddev = self.daily_returns.std()
        return sqrt * mean / stddev


    def ending_value(self):
        return self.port_val[-1]

    def sharpe_optimizer(self, allocs):
        """ function being minimized """

        # x = [0.06613882 0.93386119 0.        ], y = 3.5938127680415937
        daily_returns = Portfolio(1.0, self.symbols, allocs, self.start_date, self.end_date, self.df).daily_returns

        y = np.sqrt(252) * daily_returns.mean() / daily_returns.std()

        print("x = {}, y = {}".format(allocs, y))

        # return negative because minimizing a neg sharpe is maximizing it
        return y * -1


    def optimizer(self):

        Xguess = self.allocs
        bounds = len(self.symbols) * [(0, 1)]
        min_result = spo.minimize(self.sharpe_optimizer, Xguess, method='SLSQP',\
                bounds = bounds,\
                constraints = ({ 'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs) }),\
                options={'disp':True})
        print('Minima found at')
        print("x = {}, y = {}".format(min_result.x, min_result.fun))

        # plot guess
        guess = Portfolio(1.0, self.symbols, Xguess, self.start_date, self.end_date, self.df).port_val
        plt.plot(guess)

        # plot optimized
        optimized = Portfolio(1.0, self.symbols, min_result.x, self.start_date, self.end_date, self.df).port_val
        plt.plot(optimized)

        plt.show()


















#
