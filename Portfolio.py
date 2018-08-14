""" Computes a portfolio's Cumulative Return, Average Daily Returns, Risk, and Sharpe Ratio """

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import scipy.optimize as spo
import time
import Keys

class Portfolio(object):

    def __init__(self, start_value, start_date, end_date, symbols, allocs, df=None):
        self.start_value = start_value
        self.start_date = start_date
        self.end_date = end_date
        self.allocs = allocs
        self.symbols = symbols
        if df is None:
            dates = pd.date_range(start_date, end_date)
            df = self.get_data(symbols, dates)
        self.df = df
        self.port_val = self.get_port_val(df, allocs, start_value)
        self.daily_returns = self.daily_returns(self.port_val)

    def get_csv(self, symbol, key):
        """ Gets daily data from alphavantage API for past 100 days"""
        df = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={}&apikey={}&outputsize=full&datatype=csv'.format(symbol, key),\
                            index_col='timestamp', parse_dates=True, usecols=['timestamp', 'adjusted_close'], na_values=['NaN'])
        df = df.iloc[::-1] # reverses for date ordering
        return df

    def get_data(self, symbols, dates):
        """Read stock data (adjusted close) for given symbols from CSV files."""
        apis = [Keys.key]
        api_count = 0 # 0-5
        df = self.get_csv('^GSPC', apis[0])['2017-01-01': '2017-12-31']
        df.dropna()
        api_count += 1
        for symbol in symbols:
            print(symbol)
            time.sleep(61)
            #
            # print('a')
            # if api_count == 4:
            #     api_count = 0
            #     print('b')
            #     if api_index == 7:
            #         api_index = 0
            #         print('c')
            #     else:
            #         api_index += 1
            #         print('d')
            # else:
            #     api_count += 1
            #     print('e')

            # print(apis[0])

            df_temp = self.get_csv(symbol, apis[0])
            df_temp = df_temp.rename(columns={'adjusted_close': symbol})
            df[symbol] = df_temp['2017-01-01': '2017-12-31']
        df = df.loc[:, symbols[0]:symbols[-1]]
        print('FINAL DF:')
        print(df)
        self.df = df
        return df

    def get_port_val(self, df, allocs, start_value):
        # print("raw df")
        # print(df)
        df = df / df.iloc[0]
        # print("normalized")
        # print(df)
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

# x = [0.00000000e+00, 9.58494965e-16, 1.32725190e-14, 3.69272282e-02,\
#  1.53696400e-14, 9.66007166e-15, 1.67921551e-01, 1.47104562e-01,\
#  4.82097641e-02, 1.30556244e-01, 0.00000000e+00, 6.76189218e-15,\
#  5.14947084e-02, 9.44793291e-02, 1.67976329e-14, 4.39450312e-02,\
#  3.36186639e-02, 0.00000000e+00, 1.15473200e-02, 1.57644599e-14,\
#  7.43988871e-15, 4.71039436e-15, 0.00000000e+00, 7.14087672e-02,\
#  0.00000000e+00, 1.52262395e-01, 0.00000000e+00, 0.00000000e+00,\
#  0.00000000e+00, 1.05244363e-02]
    def sharpe_optimizer(self, allocs):
        """ function being minimized """

        # x = [0.06613882 0.93386119 0.        ], y = 3.5938127680415937
        daily_returns = Portfolio(1.0, self.start_date, self.end_date, self.symbols, allocs, self.df).daily_returns

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
        guess = Portfolio(1.0, self.start_date, self.end_date, self.symbols, Xguess, self.df).port_val
        plt.plot(guess)

        # plot optimized
        optimized = Portfolio(1.0, self.start_date, self.end_date, self.symbols, min_result.x, self.df).port_val
        plt.plot(optimized)

        plt.show()









def test_run():
    port = Portfolio(1, '2017-01-01', '2017-12-31', ['GOOGL', 'AAPL', 'AMZN'], [.3, .3, .4])
    # port = Portfolio(1, '2017-01-01', '2017-12-31', \
    #             ['WBA', 'CSCO', 'PG', 'UNH', 'AXP', 'PFE', 'BA', 'MCD', 'V', 'JNJ', 'VZ', 'DIS', 'KO', 'WMT', 'TRV', 'AAPL', 'CAT', 'UTX', 'NKE', 'MSFT',\
    #              'CVX', 'INTC', 'MRK', 'MMM', 'XOM', 'HD', 'GS', 'IBM', 'JPM', 'DWDP'],\
    #             30 * [1./30])

    print("Sharpe:")
    print(port.sharpe_ratio())
    print("Volatility:")
    print(port.risk())
    print("Avg Daily Return:")
    print(port.avg_daily_returns())
    print("Cumulative returns:")
    print(port.cumulative_return())
    print("Ideal Allocations for Sharpe:")
    print(port.optimizer())

if __name__ == "__main__":
    test_run()












#
