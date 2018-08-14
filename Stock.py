""" Models a Stock with various indicators,
with the primary feature of deciding whether to Buy, Sell, or Hold

TODO:
sells negative?
also use outputsize=full and tail the last year for more info than just 100 days
portfolio- also optimize dow jones 2017
add support for inputting whole portfolio to determine covariance
loop thru portfolio and sp500 to see which to buy and sell
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from datetime import datetime
import warnings
import Keys

class Stock(object):
    """ Evaluates a given Stock with various indicators
    and suggests to buy, sell, or hold using technical analysis.
    Technical analysis assumes the price represents all information, public and private.
    Note that technical analysis may not be enough to make a decision;
    fundamental analysis of a company's earnings, dividends,
    balance sheet, growth potential, and common sense are necessary.
    """


    # SETUP
    #
    def __init__(self, symbol, shares, avg_paid):
        """ Inputs:
        Symbol, Current Shares, Avg Price Paid
        """
        self.buys = 0
        self.sells = 0
        self.symbol = symbol
        self.daily = self.get_data(symbol)
        self.sp = self.get_data('^GSPC')
        warnings.filterwarnings(action="ignore", module="sklearn", message="internal gelsd")
        self.shares = shares
        self.avg_paid = avg_paid


    def get_data(self, symbol):
        """ Gets daily data from alphavantage API for past 100 days"""
        df = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={}&apikey={}&datatype=csv'.format(symbol, Keys.key),\
                            index_col='timestamp', parse_dates=True, usecols=['timestamp', 'open', 'close', 'adjusted_close', 'volume'], na_values=['NaN'])
        df = df.dropna()
        df = df.iloc[::-1] # reverses for date ordering
        return df





    def buy_or_sell(self):
        """ Returns Buy, Sell, or Hold, with % certainty
        Example: returns 'Buy 93%'

        Calculates indicators for several time periods and sees what they say.
        For example, calculates all indicators for both the past week, month, and year,
        and see if they all inidcate a "Buy".

        Points Algorithm:
        If an indicator supports a Buy, increment self.Buy , and vice versa.
        I will then subtract Sell points from Buy points to see the certainty in either direction.

        Also charts SMA with Bollinger Bands, MACD histogram, and scatter with SP500
        """
        print('\n' + self.symbol + ':\n')
        self.plot()

        self.predict()
        self.check_sma()
        self.check_bollinger()
        self.volume()
        self.rsi()
        self.sharpe()
        self.extrema()
        self.net_gains()
        self.beta_and_alpha()

        self.decision()
        plt.show()

    def decision(self):
        print('\nBuying Points: {}'.format(self.buys))
        print('Selling Points: {}'.format(self.sells))

        if self.buys > self.sells and self.sells * 2 < self.buys:
            decision = 'BUY'
        elif self.sells > self.buys and self.buys * 2 < self.sells:
            decision = 'SELL'
        else:
            decision = 'HOLD'
        print('\nFINAL DECISION: \n{}'.format(decision))

    # plot the data
    #
    def plot(self):
        # Adjusted Close
        adjusted_close = self.normalize(self.daily['adjusted_close'])
        ax = adjusted_close.plot(title=self.symbol + ' 100 Days', label=self.symbol, color='cyan', linewidth=5)
        self.ax = ax # used by other plot like lin reg
        # Simple Moving Average
        self.sma(normalize=True).plot(label="SMA", alpha=0.5)

        # Bollinger Bands
        upper, lower = self.bollinger_bands(normalize=True)
        upper.plot(label='upper', ax=ax, alpha=0.5)
        lower.plot(label='lower', ax=ax, alpha=0.5)

        # S & P 500 ETF (the market)
        sp_normalized = self.normalize(self.sp['adjusted_close'])
        sp_normalized.plot(label='S&P 500', alpha=0.5)

        # Add axis labels and legend
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Price")
        ax.legend(loc='upper left')

        # Plot Volume on a different Range
        ax2 = ax.twinx()
        ax2.plot(self.daily['volume'], color='gray', alpha=0.3)
        ax2.set_ylabel("Volume")
        ax2.legend(loc='lower right')













    # INDICATOR HELPERS
    #
    def sma(self, normalize=False, window=20):
        """ Simple Moving Average
        Determines the simple moving average, with a 20-day window by default.
        Returns the SMA as a <class 'pandas.core.series.Series'>
        """
        adj_close = self.daily['adjusted_close']
        if normalize: adj_close = self.normalize(adj_close)
        sma = adj_close.rolling(window).mean()
        # print(sma)
        return sma

    def macd(self):
        """ macd
        Also charts MACD
        """
        return self.ema()

    def ema(self, normalize=False, span=20):
        """ exponential moving average """
        values = self.daily['adjusted_close']
        if normalize: values = self.normalize(values)
        return pd.Series.ewm(values, span=span).mean()

    def rolling_std(self, window=20, normalize=False):
        """Return rolling standard deviation of given values, using specified window size."""
        values = self.daily['adjusted_close']
        if normalize: values = self.normalize(values)
        return values.rolling(window).std()

    def bollinger_bands(self, normalize=False):
        """ Bollinger Bands
        Given SMA, returns the Series of upper and lower Bollinger Bands
        """
        sma = self.sma(normalize=normalize)
        rstd = self.rolling_std(normalize=normalize)
        upper_band = sma + rstd
        lower_band = sma - rstd
        return upper_band, lower_band

    def normalize(self, df):
        """normalizes data by dividing by first row"""
        return df / df.ix[0,:]

    def daily_returns(self, df):
        """Compute and return the daily return values."""
        daily_returns = df.copy()
        daily_returns[1:] = (df[1:] / df[:-1].values) - 1
        daily_returns.ix[0] = 0
        return daily_returns









    # INDICATORS
    #
    def predict(self):
        """Predicts tomorrow's price using Machine Learning algorithms
        KNN and Linear Regression

        Points: 1 point for each percentage gained or lost
        """
        # format data
        df = self.normalize(self.daily)
        x = df.index.astype(np.int64).values.reshape(-1, 1)
        y = self.normalize(df[['adjusted_close']]).values

        one_day_time = 86400000000000
        x_tomorrow = x[-1] + one_day_time
        x_incl_tomorrow = np.append(x, [x_tomorrow], axis=0)
        dates = pd.to_datetime(x_incl_tomorrow.reshape(-1))

        # average the predictions
        lin_reg = self.linear_regression(x, y, x_tomorrow, x_incl_tomorrow, dates)
        knn = self.knn(x, y, x_tomorrow, x_incl_tomorrow, dates)
        tomorrow = [(lin_reg + knn) / 2]
        today = [df['adjusted_close'][-1]]

        # plot dotted line connecting stock today with tomorrow's prediction
        predicting_line = np.append(today, tomorrow, axis=0)
        self.ax.plot(dates[-2:], predicting_line, color='cyan', dashes=([1, 1, 1, 1]))
        self.ax.plot(pd.to_datetime(x_tomorrow), tomorrow, marker='o', markersize=3, color="cyan")


    def linear_regression(self, x, y, x_tomorrow, x_incl_tomorrow, dates):
        """ Checks Trend
        determines whether Linearly Regressing up, down or flat within time period.

        Points: buys + 2 for up, sells + 2 for down
        """
        regr = linear_model.LinearRegression()
        regr.fit(x, y)

        # Get Coefficients
        m = regr.coef_[0]
        b = regr.intercept_

        tomorrow_normalized = regr.predict([x_tomorrow])[0][0]
        tomorrow = tomorrow_normalized * self.daily['adjusted_close'][0]
        today = self.daily['adjusted_close'][-1]
        percent_gain = ((tomorrow / today) - 1) * 100

        # plot Lin Reg
        self.ax.plot(dates, regr.predict(x_incl_tomorrow), color='k', label='Linear Regression', dashes=([2, 2, 10, 2]))
        self.ax.legend(loc='best')

        # update points
        if m > 0:
            # trend is upwards
            print('Positive Regression Trend: buys + 2')
            self.buys += 2
        else:
            print('Negative Regression Trend: sells + 2')
            self.sells += 2

        if percent_gain > 0:
            print('Linear Regression: Price will be up {0} % tomorrow, buys + 1, predicted close is {1}'.format(percent_gain, tomorrow))
            self.buys += int(round(percent_gain, 0))
        else:
            print('Linear Regression: Price will be down {0} % tomorrow, sells + 1, predicted close is {1}'.format(percent_gain, tomorrow))
            self.sells += int(round(percent_gain, 0))

        return tomorrow_normalized

    def knn(self, x, y, x_tomorrow, x_incl_tomorrow, dates):
        """ predicts future price using knn

        Points:
            if future price (next day/week/year, depending on input)
            is higher than current, add 3 points, and an additional 3 for each
            std dev from current level
        """
        neigh = KNeighborsRegressor(n_neighbors=5)
        neigh.fit(x, y)

        tomorrow_normalized = neigh.predict([x_tomorrow])[0][0]
        tomorrow = tomorrow_normalized * self.daily['adjusted_close'][0]
        today = self.daily['adjusted_close'][-1]
        percent_gain = ((tomorrow / today) - 1) * 100
        percent_gain_int = abs(int(round(percent_gain, 0)))
        self.ax.plot(dates, neigh.predict(x_incl_tomorrow),label='KNN', dashes=([1, 1, 5, 1]))
        self.ax.legend(loc='best')

        if percent_gain > 0:
            print('KNN: Price will be up {0} % tomorrow, buys + {1}, predicted close is {2}'.format(percent_gain, percent_gain_int, tomorrow))
            self.buys += percent_gain_int
        else:
            print('KNN: Price will be down {0} % tomorrow, sells + {1}, predicted close is {2}'.format(percent_gain, percent_gain_int, tomorrow))
            self.sells += percent_gain_int

        return tomorrow_normalized


    def check_sma(self):
        """ Checks Simple Moving Average

        Points: buys + 2 for below, sells + 2 for above
        """
        sma = self.sma()
        if self.daily['adjusted_close'][-1] < sma[-1]:
            print('Below SMA: buys ++')
            self.buys += 1
        else:
            print('Above SMA: sells ++')
            self.sells += 1

    def check_bollinger(self):
        """ Checks Bollinger Bands
        Checks for crossovers INTO a band, or the current price's relation to the bands.

        Points:
            + 5 for crossover into TODO
            + 3 for just being outside band
        """
        upper, lower = self.bollinger_bands()
        if self.daily['adjusted_close'][-1] > upper[-1]:
            print('Above upper bollinger: sells ++')
            self.sells += 1
        elif self.daily['adjusted_close'][-1] < lower[-1]:
            print('Below lower bollinger: buys ++')
            self.buys += 1

    def volume(self):
        """ Volume
        Determines whether volume has been trending up or down recently.
        Higher volume supports a trend.

        Points:
            if more than 1 std, add 2 to each
            if more than avg, add 1 to each
            if less than avg, subtract 1, etc etc
        """
        vol = self.daily['volume']
        sma = vol.rolling(20).mean()
        std = vol.rolling(20).std()
        upper = sma + std
        lower = sma - std

        if vol[-1] > upper[-1]:
            print('volume > 1 STD above sma: buys++, sells++')
            self.sells += 1
            self.buys += 1
        else:
            print('Volume in normal levels. Upper Limit: {0}, Current: {1}'.format(upper[-1], vol[-1]))

    def rsi(self, days=14):
        """ relevant strength index oscillator

        Points:
            if more than 70%, sell + 3
            if less than 30%, buy + 3
        """
        daily_returns = self.daily_returns(self.daily[['adjusted_close']]).tail(14)
        green_days = daily_returns[daily_returns > 0].dropna()
        red_days = daily_returns[daily_returns < 0].dropna()

        # RS= avg of Up Days - avg of Down Days. Note im adding bc red_days is negative.
        RS = (green_days.sum() / days + red_days.sum() / days)['adjusted_close']
        rsi = 100 - (100 / (1 + RS))

        if rsi < 0.3:
            print('RSI is under 30%: {0} buys ++'.format(rsi))
            self.buys += 1
        elif rsi > 0.7:
            print('RSI over 70%: {0} sells ++'.format(rsi))
            self.sells += 1
        else:
            print('RSI is normal(between 0.3-0.7): {}'.format(rsi))


    def sharpe(self):
        """ sharpe ratio

        Points:
            if sr < 1, add 1 to sell
            if >1, add 2 to buy
            if >2, add 3 to buy
        """
        annual_free = 1.02 # 2 percent annual
        daily_risk_free = annual_free ** (1. / 252) - 1
        sqrt = np.sqrt(252)
        dr = self.daily_returns(self.daily['adjusted_close'])
        mean = (dr - daily_risk_free).mean()
        stddev = dr.std()
        sharpe =  sqrt * mean / stddev
        rounded = int(round(sharpe, 0))

        if sharpe < 1:
            print('Sharpe Ratio too low: {} sells++'.format(sharpe))
        elif sharpe < 2:
            print('Sharpe Ratio is {}, buys++'.format(sharpe))
        elif sharpe < 3:
            print('Good Sharpe Ratio of {}, buys + 2'.format(sharpe))
        else:
            print('Very good Sharpe Ratio of {0}, buys + {1}'.format(sharpe, rounded))


    def extrema(self):
        """ checks for extrema: 100 day high/low

        Points:
            If its the min/max of the specified time frame, add 2
            if its within 4 of the min/max, add 1
        """
        df = self.daily['adjusted_close']
        min = df.min()
        max = df.max()
        if df[-1] == min:
            print('Local Minimum: Buys++')
            self.buys += 1
        elif df[-1] == max:
            print('Local Maximum: Sells++')
            self.sells += 1
        else:
            print('Not a local extrema.')

    def net_gains(self):
        """ Net Gains from this stock, if i have been holding.
        This is very important!
        "The first rule is to never lose. And the second rule is to never
        forget about that first rule." - Warren Buffett

        Points:
            Add 5 points to sell if i have made money, subtract 5 if ive lost money.
            Buying doenst really matter here
        """
        if self.shares == 0:
            print('No shares owned.')
        else:
            price = self.daily['adjusted_close'][-1]
            if self.avg_paid < price:
                penalty = int(round(self.sells / 4.0, 0))
                print('NET LOSS: AVOID SELLING! sells - {}'.format(penalty))
                self.sells -= penalty
            else:
                print('No net loss.')


    def beta_and_alpha(self):
        """ checks the Beta and Alpha of the stock with the market (sp500) '

        Also charts scatter plot.

        Points:
            Positive Alpha: + 2 to buy
            buy + int(Beta) if sp500 has upward trend, - if market is going down.
            Im assuming market is going up.
        """
        # make scatter plot
        sp_temp = self.daily_returns(self.sp.rename(columns={'adjusted_close': '^GSPC'}))
        symbol_temp = self.daily_returns(self.daily.rename(columns={'adjusted_close': self.symbol}))
        joined = sp_temp.merge(symbol_temp, on='timestamp')

        # beta and alpha
        beta, alpha = np.polyfit(joined["^GSPC"], joined[self.symbol], 1)

        if alpha > 0:
            self.buys += 1
            print('Alpha > 0: buys++ {}'.format(alpha))

        # assuming favorable market conditions. else, it would be -=
        if beta > 1:
            self.buys += 1
            print('Beta > 1: buys++ {}'.format(beta))

        # finish plotting scatter
        joined.plot(kind = 'scatter', x='^GSPC', y=self.symbol)
        plt.plot(joined["^GSPC"], beta * joined['^GSPC'] + alpha, '-', color='r', label='Correlation')
        # print('joined ^GSPC:')
        # print(joined["^GSPC"])

        # plot expected beta (slope) of 1 and alpha (y- int.) of zero
        plt.plot(joined["^GSPC"], 1 * joined['^GSPC'] + 0, '-', color='gray', label='Beta of 1')
        plt.plot(joined["^GSPC"], 0 * joined['^GSPC'] + 0, '-', color='gray', label='Alpha of 0')
        plt.legend(loc='best')




if __name__ == '__main__':
    # symbol = input('Symbol: ')
    # shares = input('Current Shares: ')
    # avg_paid = input('Avg Price Paid: ') if int(shares) > 0 else 0
    #
    # Stock(symbol, shares, avg_paid).buy_or_sell()

    Stock('AAPL', 0, 0).buy_or_sell()
    Stock('AMZN', 0, 0).buy_or_sell()

    # dj = ['WBA', 'CSCO', 'PG', 'UNH', 'AXP', 'PFE', 'BA', 'MCD', 'V', 'JNJ', 'VZ', 'DIS', 'KO', 'WMT', 'TRV', 'AAPL', 'CAT', 'UTX', 'NKE', 'MSFT',\
    #  'CVX', 'INTC', 'MRK', 'MMM', 'XOM', 'HD', 'GS', 'IBM', 'JPM', 'DWDP']

    # dj = ['AAPL', 'CAT', 'UTX', 'NKE', 'MSFT',\
    #     'CVX', 'INTC', 'MRK', 'MMM', 'XOM', 'HD', 'GS', 'IBM', 'JPM', 'DWDP']
    #
    # for stock in dj:
    #     Stock(stock, 0, 0).buy_or_sell()

#
