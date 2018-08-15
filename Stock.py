""" Models a Stock with various indicators,
with the primary feature of deciding whether to Buy, Sell, or Hold a Stock

TODO:
automatic trading
clean code, pretty output, round doubles
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
import warnings

import QuoteHistory

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
    def __init__(self, symbol, shares, avg_paid, plot=True):
        """ Inputs:
        Symbol, Current Shares, Avg Price Paid
        """
        self.buys = 0
        self.sells = 0
        self.shares = shares
        self.symbol = symbol
        self.will_plot = plot
        self.daily = QuoteHistory.get_data(symbol)
        self.sp = QuoteHistory.get_data('^GSPC')
        warnings.filterwarnings(action="ignore", module="sklearn", message="internal gelsd")
        self.avg_paid = avg_paid
        self.debug = '\n' + self.symbol + ':\n'


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
        # print('\n' + self.symbol + ':\n')
        if self.will_plot:
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

        print(self.debug)

        if self.will_plot:
            plt.show()

    def decision(self):
        # print('\nBuying Points: {}'.format(self.buys))
        # print('Selling Points: {}'.format(self.sells))
        self.debug += '\nBuying Points: {}'.format(self.buys)
        self.debug += '\nSelling Points: {}'.format(self.sells)
        if self.buys > self.sells and self.sells * 2 < self.buys:
            decision = 'BUY'
        elif self.sells > self.buys and self.buys * 2 < self.sells:
            decision = 'SELL'
        else:
            decision = 'HOLD'
        # print('\nFINAL DECISION: \n{}'.format(decision))
        self.debug += '\nFINAL DECISION: \n{}'.format(decision)
        self.buying_certainty = self.buys - self.sells

    # plot the data
    #
    def plot(self):
        # Adjusted Close
        Adj_Close = self.normalize(self.daily['Adj Close'])
        ax = Adj_Close.plot(title=self.symbol, label=self.symbol, color='cyan', linewidth=5)
        self.ax = ax # used by other plot like lin reg
        # Simple Moving Average
        self.sma(normalize=True).plot(label="SMA", alpha=0.5)

        # Bollinger Bands
        upper, lower = self.bollinger_bands(normalize=True)
        upper.plot(label='upper', ax=ax, alpha=0.5)
        lower.plot(label='lower', ax=ax, alpha=0.5)

        # S & P 500 ETF (the market)
        sp_normalized = self.normalize(self.sp['Adj Close'])
        sp_normalized.plot(label='S&P 500', alpha=0.5)

        # Add axis labels and legend
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Price")
        ax.legend(loc='upper left')

        # Plot Volume on a different Range
        ax2 = ax.twinx()
        ax2.plot(self.daily['Volume'], color='gray', alpha=0.3)
        ax2.set_ylabel("Volume")
        ax2.legend(loc='lower right')













    # INDICATOR HELPERS
    #
    def sma(self, normalize=False, window=20):
        """ Simple Moving Average
        Determines the simple moving average, with a 20-day window by default.
        Returns the SMA as a <class 'pandas.core.series.Series'>
        """
        adj_close = self.daily['Adj Close']
        if normalize: adj_close = self.normalize(adj_close)
        sma = adj_close.rolling(window).mean()
        return sma

    def macd(self):
        """ macd
        Also charts MACD
        """
        return self.ema()

    def ema(self, normalize=False, span=20):
        """ exponential moving average """
        values = self.daily['Adj Close']
        if normalize: values = self.normalize(values)
        return pd.Series.ewm(values, span=span).mean()

    def rolling_std(self, window=20, normalize=False):
        """Return rolling standard deviation of given values, using specified window size."""
        values = self.daily['Adj Close']
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
        y = self.normalize(df[['Adj Close']]).values

        one_day_time = 86400000000000
        x_tomorrow = x[-1] + one_day_time
        x_incl_tomorrow = np.append(x, [x_tomorrow], axis=0)
        dates = pd.to_datetime(x_incl_tomorrow.reshape(-1))

        # average the predictions
        lin_reg = self.linear_regression(x, y, x_tomorrow, x_incl_tomorrow, dates)
        knn = self.knn(x, y, x_tomorrow, x_incl_tomorrow, dates)
        tomorrow_norm = [(lin_reg + knn) / 2]
        today_norm = [df['Adj Close'][-1]]
        tomorrow = round((tomorrow_norm[0] * self.daily['Adj Close'][0]), 2)
        today = self.daily['Adj Close'][-1]
        # self.debug += '\nExpected price (mean of ML models): {}'.format(tomorrow[0] * self.daily['Adj Close'][0])
        percent_gain = round((((tomorrow / today) - 1) * 100), 2)
        percent_gain_int = abs(int(round(percent_gain, 0)))
        # self.debug += '\nExpected price gain: {} %'.format(percent_gain_int)

        if percent_gain > 0:
            self.debug += '\nExpected price gain: {} %, buys + {}, predicted close is {}'.format(percent_gain, percent_gain_int, tomorrow)
            self.buys += percent_gain_int
        else:
            self.debug += '\nExpected price gain: n {} %, sells + {}, predicted close is {}'.format(percent_gain, percent_gain_int, tomorrow)
            self.sells += percent_gain_int

        # plot dotted line connecting stock today with tomorrow's prediction
        predicting_line = np.append(today_norm, tomorrow_norm, axis=0)

        if self.will_plot:
            self.ax.plot(dates[-2:], predicting_line, color='cyan', dashes=([1, 1, 1, 1]))
            self.ax.plot(pd.to_datetime(x_tomorrow), tomorrow_norm, marker='o', markersize=3, color="cyan")


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
        tomorrow = round((tomorrow_normalized * self.daily['Adj Close'][0]), 2)
        today = self.daily['Adj Close'][-1]
        percent_gain = round(((tomorrow / today) - 1) * 100, 2)
        percent_gain_int = abs(int(round(percent_gain, 0)))

        # plot Lin Reg
        if self.will_plot:
            self.ax.plot(dates, regr.predict(x_incl_tomorrow), color='k', label='Linear Regression', dashes=([2, 2, 10, 2]))
            self.ax.legend(loc='best')

        # update points
        if m > 0:
            # trend is upwards
            # print('Positive Regression Trend: buys + 3')
            self.debug += '\nPositive Regression Trend: buys + 3'
            self.buys += 3
        else:
            # print('Negative Regression Trend: sells + 3')
            self.debug += '\nNegative Regression Trend: sells + 3'
            self.sells += 3

        if percent_gain > 0:
            # print('Regression: Price will be up {} %, buys + {}, predicted close is {}'.format(percent_gain, percent_gain_int, tomorrow))
            self.debug += '\nRegression: Price will be up {} %, predicted close is {}'.format(percent_gain, tomorrow)
        else:
            # print('Regression: Price will be down {} %, sells + {}, predicted close is {}'.format(percent_gain, percent_gain_int, tomorrow))
            self.debug += '\nRegression: Price will be down {} %, predicted close is {}'.format(percent_gain, tomorrow)

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
        tomorrow = round((tomorrow_normalized * self.daily['Adj Close'][0]), 2)
        today = self.daily['Adj Close'][-1]
        percent_gain = round(((tomorrow / today) - 1) * 100, 2)
        percent_gain_int = abs(int(round(percent_gain, 0)))

        if self.will_plot:
            self.ax.plot(dates, neigh.predict(x_incl_tomorrow),label='KNN', dashes=([1, 1, 5, 1]))
            self.ax.legend(loc='best')

        if percent_gain > 0:
            # print('KNN: Price will be up {0} % tomorrow, buys + {1}, predicted close is {2}'.format(percent_gain, percent_gain_int, tomorrow))
            self.debug += '\nKNN: Price will be up {} % tomorrow, predicted close is {}'.format(percent_gain, tomorrow)
        else:
            # print('KNN: Price will be down {0} % tomorrow, sells + {1}, predicted close is {2}'.format(percent_gain, percent_gain_int, tomorrow))
            self.debug += '\nKNN: Price will be down {} % tomorrow, predicted close is {}'.format(percent_gain, tomorrow)

        return tomorrow_normalized


    def check_sma(self):
        """ Checks Simple Moving Average

        Points: buys + 2 for below, sells + 2 for above
        """
        sma = self.sma()
        if self.daily['Adj Close'][-1] < sma[-1]:
            # print('Below SMA: buys + 1')
            self.debug += '\nBelow SMA: buys + 1'
            self.buys += 1
        else:
            # print('Above SMA: sells + 1')
            self.debug += '\nAbove SMA: sells + 1'
            self.sells += 1

    def check_bollinger(self):
        """ Checks Bollinger Bands
        Checks for crossovers INTO a band, or the current price's relation to the bands.

        Points:
            + 5 for crossover into TODO
            + 3 for just being outside band
        """
        upper, lower = self.bollinger_bands()
        if self.daily['Adj Close'][-1] > upper[-1]:
            # print('Above upper bollinger: sells + 1')
            self.debug += '\nAbove upper bollinger: sells + 1'
            self.sells += 1
        elif self.daily['Adj Close'][-1] < lower[-1]:
            # print('Below lower bollinger: buys + 1')
            self.debug += '\nBelow lower bollinger: buys + 1'
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
        vol = self.daily['Volume']
        sma = vol.rolling(20).mean()
        std = vol.rolling(20).std()
        upper = sma + std
        lower = sma - std

        if vol[-1] > upper[-1]:
            # print('Volume > 1 STD above sma: buys + 1, sells + 1')
            self.debug += '\nVolume > 1 STD above sma: buys + 1, sells + 1'
            self.sells += 1
            self.buys += 1
        else:
            # print('Volume in normal levels. Upper Limit: {0}, Current: {1}'.format(upper[-1], vol[-1]))
            self.debug += '\nVolume in normal levels'

    def rsi(self, days=14):
        """ relevant strength index oscillator

        Points:
            if more than 70%, sell + 3
            if less than 30%, buy + 3
        """
        daily_returns = self.daily_returns(self.daily[['Adj Close']]).tail(14)
        green_days = daily_returns[daily_returns > 0].dropna()
        red_days = daily_returns[daily_returns < 0].dropna()

        # RS= avg of Up Days - avg of Down Days. Note im adding bc red_days is negative.
        RS = (green_days.sum() / days + red_days.sum() / days)['Adj Close']
        rsi = 100 - (100 / (1 + RS))
        rsi = round(rsi, 2)
        if rsi < 0.3:
            # print('RSI is under 30%: {0} buys ++'.format(rsi))
            self.debug += '\nRSI is under 30%: {0} buys ++'.format(rsi)
            self.buys += 1
        elif rsi > 0.7:
            # print('RSI over 70%: {0} sells ++'.format(rsi))
            self.debug += '\nRSI over 70%: {0} sells ++'.format(rsi)
            self.sells += 1
        else:
            # print('RSI is normal(between 0.3-0.7): {}'.format(rsi))
            self.debug += '\nRSI is normal(between 0.3-0.7): {}'.format(rsi)


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
        dr = self.daily_returns(self.daily['Adj Close'])
        mean = (dr - daily_risk_free).mean()
        stddev = dr.std()
        sharpe =  sqrt * mean / stddev
        sharpe = round(sharpe, 1)
        rounded = int(round(sharpe, 0))

        if sharpe < 1:
            # print('Sharpe Ratio too low: {} sells++'.format(sharpe))
            self.debug += '\nSharpe Ratio too low: {} sells++'.format(sharpe)
            self.sells += 1
        elif sharpe < 2:
            # print('Sharpe Ratio is {}, buys++'.format(sharpe))
            self.debug += '\nSharpe Ratio is {}, buys++'.format(sharpe)
            self.buys += 1
        elif sharpe < 3:
            # print('Good Sharpe Ratio of {}, buys + 2'.format(sharpe))
            self.debug += '\nGood Sharpe Ratio of {}, buys + 2'.format(sharpe)
            self.buys += 2
        else:
            # print('Very good Sharpe Ratio of {0}, buys + 3'.format(sharpe))
            self.debug += '\nVery good Sharpe Ratio of {0}, buys + 3'.format(sharpe)
            self.buys += 3

    def extrema(self):
        """ checks for extrema: 100 day high/low

        Points:
            If its the min/max of the specified time frame, add 2
            if its within 4 of the min/max, add 1
        """
        df = self.daily['Adj Close']
        min = df.min()
        max = df.max()
        if df[-1] == min:
            # print('Local Minimum: Buys++')
            self.debug += '\nLocal Minimum: Buys++'
            self.buys += 1
        elif df[-1] == max:
            # print('Local Maximum: Sells++')
            self.debug += '\nLocal Maximum: Sells++'
            self.sells += 1
        else:
            # print('Not a local extrema.')
            self.debug += '\nNot a local extrema.'

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
            # print('No shares owned.')
            self.debug += '\nNo shares owned.'
        else:
            price = self.daily['Adj Close'][-1]
            gains = self.shares * (price - self.avg_paid)
            percent = (price / avg_paid - 1) * 100
            gains = round(gains, 2)
            percent = round(percent, 2)
            if gains < 0:
                penalty = self.sells - int(round(self.sells / 3.0, 0))
                # print('NET LOSS: AVOID SELLING! sells - {}'.format(penalty))
                self.debug += '\nNET LOSS: {}, {}%, AVOID SELLING! sells - {}'.format(gains, percent, penalty)
                self.sells -= penalty
            else:
                # print('No net loss.')
                self.debug += '\nNet gains: ${}, {}%'.format(gains, percent)


    def beta_and_alpha(self):
        """ checks the Beta and Alpha of the stock with the market (sp500) '

        Also charts scatter plot.

        Points:
            Positive Alpha: + 2 to buy
            buy + int(Beta) if sp500 has upward trend, - if market is going down.
            Im assuming market is going up.
        """
        # make scatter plot
        sp_temp = self.daily_returns(self.sp.rename(columns={'Adj Close': '^GSPC'}))
        symbol_temp = self.daily_returns(self.daily.rename(columns={'Adj Close': self.symbol}))
        joined = sp_temp.merge(symbol_temp, on='Date')

        # beta and alpha
        beta, alpha = np.polyfit(joined["^GSPC"], joined[self.symbol], 1)
        beta = round(beta, 3)
        alpha = round(alpha, 5)
        if alpha > 0:
            self.buys += 1
            # print('Alpha > 0: buys++ {}'.format(alpha))
            self.debug += '\nAlpha > 0: buys++ {}'.format(alpha)
        else:
            # print('Alpha < 0: {}'.format(alpha))
            self.debug += '\nAlpha < 0: {}'.format(alpha)

        # assuming favorable market conditions. else, it would be -=
        if beta > 1:
            self.buys += 1
            # print('Beta > 1: buys++ {}'.format(beta))
            self.debug += '\nBeta > 1: buys++ {}'.format(beta)
        else:
            # print('Beta < 1: {}'.format(beta))
            self.debug += '\nBeta < 1: {}'.format(beta)

        # finish plotting scatter
        if self.will_plot:
            joined.plot(kind = 'scatter', x='^GSPC', y=self.symbol)
            plt.plot(joined["^GSPC"], beta * joined['^GSPC'] + alpha, '-', color='r', label='Correlation')

            # plot expected beta (slope) of 1 and alpha (y- int.) of zero
            plt.plot(joined["^GSPC"], 1 * joined['^GSPC'] + 0, '-', color='gray', label='Beta of 1')
            plt.plot(joined["^GSPC"], 0 * joined['^GSPC'] + 0, '-', color='gray', label='Alpha of 0')
            plt.legend(loc='best')




if __name__ == '__main__':
    symbol = input('Symbol: ')
    shares = int(input('Current Shares: '))
    avg_paid = int(input('Avg Price Paid: ')) if int(shares) > 0 else 0

    Stock(symbol, shares, avg_paid).buy_or_sell()
