""" Computes a Portfolio's Cumulative Return, Average Daily Returns, Risk, and Sharpe Ratio.

Optimizes a portfolio using stock covariance for volatility minimization
and cumulative return maximization with stochastic gradient descent
to minimize the negative Sharpe Ratio function for various X parameters
representing the different percentage allocations for each Stock.

Note that this minimizes the Sharpe Ratio, not cumulative returns.
This is because optimizing for cumulative returns is trivial;
merely invest 100% in the one stock that has increased the most!
Instead, this minimizes risk as well, which also makes it much more useful
for the future since securities tend to maintain the same levels of volatility.

Imagine two companies competing in the same sector. Perhaps one company does better,
the other tends to do worse. This is a negative covariance.
In this case, it would be possible to attain the returns of both of the stocks
with nearly zero risk, as the volatilities of each company become cancelled out
if allocations are set evenly to 50% and 50%.
This is the big picture of how this optimizes for risk mitigation in addition to cumulative returns.

See 'Examples.py' for example usage.

"""


import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import scipy.optimize as spo
import time

# local
import QuoteHistory



class Portfolio(object):

    def __init__(self, start_value, symbols, allocs, start_date=None, end_date=None, fresh=False, days=None, df=None):
        """start_value: starting value of portfolio, like 1000.50
        allocs: percentage allocations for each stock, like [0.3, 0.4, 0.3]
        start_date, end_date: like '2018-12-13'
        fresh: if True, will fetch entirely new data for each stock
        df: dataframe used ONLY within optimizer function below.
        """
        self.start_value = start_value
        self.start_date = start_date
        self.end_date = end_date
        self.allocs = allocs
        self.fresh = fresh
        self.symbols = symbols
        self.days = days
        if df is None:
            df = self.get_data(symbols)
        self.df = df
        self.port_val = self.get_port_val(df, allocs, start_value)
        self.daily_returns = self.daily_returns(self.port_val)

    def get_data(self, symbols):
        """Read stock data (adjusted close) for given symbols from CSV files.
        Returns dataframe of all stocks inner joined on ^GSPC's 'Date' index_col
        """
        # set up dataframe with S&P 500
        df = QuoteHistory.get_data('^GSPC', self.fresh, self.start_date, self.end_date, self.days)
        df = df.rename(columns={'Adj Close': '^GSPC'}).drop('Volume', axis=1).dropna()
        for symbol in symbols:
            try:
                df_temp = QuoteHistory.get_data(symbol, self.fresh, self.start_date, self.end_date, self.days)
                df_temp = df_temp.drop('Volume', axis=1) # bc we only want adjusted close
                df[symbol] = df_temp # adds symbol to DF
            except Exception:
                print('Failed to determine for {}'.format(symbol))

        df = df.drop('^GSPC', axis=1) # gets rid of first stock used for indexing
        return df



    def debug(self):
        """ Prints out the analysis of the portfolio """
        # Optimize the portfolio
        x, y = self.optimizer()
        optimized = Portfolio(1.0, self.symbols, x, self.start_date, self.end_date, self.fresh, self.days, self.df)

        print('Minima found at\nx = {}, y = {}'.format(x, y))
        print("\nOptimized Allocations for Maxmimum Risk-Adjusted Returns:")

        # sort X and its accompanying symbols, descending
        sorted_indices = x.argsort()
        x = x[sorted_indices[::-1]]
        symbols = np.array(self.symbols)
        symbols = symbols[sorted_indices[::-1]]

        # print out each symbol with its percentage allocation
        for i in range(0, len(x)):
            symbol = symbols[i]
            percent = round((x[i] * 100), 2)
            print('{}: {}%'.format(symbol, percent))

        # print stats for before and after
        print("\nBefore optimization:")
        self.print_stats(self)
        print("\nAfter optimization:")
        self.print_stats(optimized)

        # plot before and after
        ax = self.port_val.plot(title='Portfolio Optimization Results', label='Before')
        optimized.port_val.plot(label='After')
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Total Value")
        ax.legend(loc='best')
        plt.show()

    def print_stats(self, portfolio):
        """ prints out a portfolio's stats. called from self.debug() """
        sharpe = round(portfolio.sharpe_ratio(portfolio.daily_returns), 2)
        volatility = round(portfolio.risk(), 5)
        dr = round(portfolio.avg_daily_returns() * 100, 4)
        cr = round(portfolio.cumulative_return() * 100, 2)
        print("Sharpe Ratio: {}".format(sharpe))
        print("Volatility: {}".format(volatility))
        print("Average Daily Returns: {}%".format(dr))
        print("Cumulative Returns: {}%".format(cr))



    def get_port_val(self, df, allocs, start_value):
        """ returns dataframe of portfolio's total value day-by-day """
        df /= df.iloc[0]
        df *= allocs
        df *= start_value
        df = df.sum(axis = 1)
        return df


    def daily_returns(self, df):
        """ returns daily returns df,
        where 0.01 on a specific day represents 1% gained on that day
        """
        returns = df.copy()
        returns[1:] = (df[1:] / df[:-1].values) - 1
        returns.ix[0] = 0
        return returns


    def cumulative_return(self):
        """ return on the portfolio from start to finish,
        like 1.5 representing +50% in returns
        """
        return self.port_val[-1] / self.port_val[0] - 1


    def avg_daily_returns(self):
        """ average of daily returns """
        return self.daily_returns.mean()


    def risk(self):
        """ volatility of a stock measured by standard deviation """
        return self.daily_returns.std()


    def sharpe_ratio(self, daily_returns):
        """
        the annualized risk-adjusted return.
        assumes daily frequency, and an annual risk-free amount of 2%.
        Sharpe Ratio = sqrt(252) * mean(daily returns - daily risk free)
                       -------------------------------------------------
                       stddev(daily returns)
        SR should be around 0.5-3.
        Less than 1 is risky, 1-3 is good, more than 3 is amazing.
        """
        # daily_risk_free = 1.02 ** (1. / 252) - 1 # 2 percent annual, current T-Bill
        sqrt = np.sqrt(252)
        mean = daily_returns.mean()
        stddev = daily_returns.std()
        return sqrt * mean / stddev


    def ending_value(self):
        return self.port_val[-1]


    def sharpe_optimizer(self, allocs):
        """ function being minimized
        Takes `allocs`, a list of percentage allocations like [0.6, 0.4] as input
         to determine gradient descent of the sharpe ratio function
        """
        daily_returns = Portfolio(1.0, self.symbols, allocs, self.start_date, self.end_date, self.fresh, self.days, self.df).daily_returns

        # get sharpe ratio with new allocations
        y = self.sharpe_ratio(daily_returns)

        # Example: x = [0.06613882 0.93386119 0.000000], y = 3.5938127680415937
        print("x = {}, y = {}".format(allocs, y))

        # negative because MINIMIZING a NEGATIVE sharpe is the same as maximizing it
        return y * -1


    def optimizer(self):
        """ Minimizes the sharpe_optimizer() function using gradient descent.
        """
        Xguess = self.allocs # initial guess for X
        bounds = len(self.symbols) * [(0, 1)] # keep allocations between 0 and 1
        # constrains sum to equal 1.0 (100%)
        min_result = spo.minimize(self.sharpe_optimizer, Xguess, method='SLSQP',\
                bounds = bounds,\
                constraints = ({ 'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs) }),\
                options={'disp':True})

        return min_result.x, min_result.fun
