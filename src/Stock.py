""" Models a Stock with various indicators,
with the primary feature of deciding whether to Buy, Sell, or Hold a Stock.

See 'Examples.py' for example usage.

TODO:
Set up automatic trading
Show date range in plot title
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
import warnings

# local
from src import QuoteHistory


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
    def __init__(self, symbol, shares, avg_paid, fresh=False, fresh_sp=False, plot=True, days=140):
        """ Sample Inputs:
        Symbol 'SBUX', Current Shares 0, Avg Price Paid 50.10
        Fresh = True fetches new data
        """
        self.buys = 0
        self.sells = 0
        self.shares = shares
        self.symbol = symbol
        self.will_plot = plot
        self.daily = QuoteHistory.get_data(symbol, fresh, days=days)
        self.sp = QuoteHistory.get_data('^IXIC', fresh_sp, days=days) # now the nasdaq
        warnings.filterwarnings(action="ignore", module="sklearn", message="internal gelsd")
        self.avg_paid = avg_paid
        self.debug = '\n' + self.symbol + ':\n'



    def buy_or_sell(self, debug=True):
        """ Determines whether to Buy, Sell, or Hold
        along with an explanation and several charts.

        If an indicator supports a Buy, increment self.Buys, and vice versa.
        I will then subtract Sell points from Buy points to see the certainty in either direction.
        """
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
        if debug: print(self.debug)
        if self.will_plot:
            plt.show()


    def decision(self):
        self.debug += '\nBuying Points: {}'.format(self.buys)
        self.debug += '\nSelling Points: {}'.format(self.sells)
        if self.buys > self.sells and self.sells * 2 < self.buys:
            decision = 'BUY'
        elif self.sells > self.buys and self.buys * 2 < self.sells:
            decision = 'SELL'
        else:
            decision = 'HOLD'
        self.debug += '\nFINAL DECISION: \n{}'.format(decision)
        # certainty to just BUY, used for s&p traversals.
        self.buying_certainty = self.buys - self.sells
        # percent certainty for a sell or buy
        # is difference divided by sum of sell and buys.
        # for example, certainty of 1 sell/9 buys =
        # (abs(1-9)) / (1 + 9) = 8 / 10 = 0.80 = 80% certainty (to sell, in this case)
        percent_certainty = abs(self.buys - self.sells) / (self.buys + self.sells)
        if decision == 'HOLD':
            percent_certainty = 1 - percent_certainty
        percent_certainty = round(percent_certainty * 100, 2)
        self.debug += '\nCertainty: {}%'.format(percent_certainty)


    def plot(self):
        # Adjusted Close
        Adj_Close = self.normalize(self.daily['Adj Close'])
        ax = Adj_Close.plot(title=self.symbol, label=self.symbol, color='cyan', linewidth=5)
        self.ax = ax # used by other plots like lin reg
        # Simple Moving Average
        self.sma(normalize=True).plot(label="SMA", alpha=0.5)

        # Bollinger Bands
        upper, lower = self.bollinger_bands(normalize=True)
        upper.plot(label='upper', ax=ax, alpha=0.5)
        lower.plot(label='lower', ax=ax, alpha=0.5)

        # S&P 500 (the market)
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













    # HELPERS
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

    def ema(self, normalize=False, span=20):
        """ exponential moving average """
        values = self.daily['Adj Close']
        if normalize: values = self.normalize(values)
        return pd.Series.ewm(values, span=span).mean()

    def rolling_std(self, window=20, normalize=False):
        """Returns rolling standard deviation of given values, using specified window size."""
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
        return df / df.ix[0, :]

    def daily_returns(self, df):
        """Compute and return the daily return values."""
        daily_returns = df.copy()
        daily_returns[1:] = (df[1:] / df[:-1].values) - 1
        daily_returns.ix[0] = 0
        return daily_returns









    # INDICATORS
    #
    def predict(self):
        """Predicts tomorrow's price using ML algorithms KNN and Linear Regression
        """
        # format data
        df = self.normalize(self.daily)
        x = df.index.astype(np.int64).values.reshape(-1, 1)
        y = self.normalize(df[['Adj Close']]).values

        # format time
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
        percent_gain = round((((tomorrow / today) - 1) * 100), 2)
        percent_gain_int = abs(int(round(percent_gain, 0)))

        if percent_gain > 0:
            self.debug += '\nExpected price gain: {} %, buys + {}, predicted close is {}'.format(percent_gain, percent_gain_int, tomorrow)
            self.buys += percent_gain_int
        else:
            self.debug += '\nExpected price gain: {} %, sells + {}, predicted close is {}'.format(percent_gain, percent_gain_int, tomorrow)
            self.sells += percent_gain_int

        # plots dotted line connecting stock today with tomorrow's prediction
        predicting_line = np.append(today_norm, tomorrow_norm, axis=0)

        if self.will_plot:
            self.ax.plot(dates[-2:], predicting_line, color='cyan', dashes=([1, 1, 1, 1]))
            self.ax.plot(pd.to_datetime(x_tomorrow), tomorrow_norm, marker='o', markersize=3, color="cyan")


    def linear_regression(self, x, y, x_tomorrow, x_incl_tomorrow, dates):
        """ Checks Trend
        determines whether Linearly Regressing up, down or flat within time period.
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
            self.debug += '\nPositive Regression Trend: buys + 3'
            self.buys += 3
        else:
            self.debug += '\nNegative Regression Trend: sells + 3'
            self.sells += 3

        if percent_gain > 0:
            self.debug += '\nRegression: Price will be up {} %, predicted close is {}'.format(percent_gain, tomorrow)
        else:
            self.debug += '\nRegression: Price will be down {} %, predicted close is {}'.format(percent_gain, tomorrow)

        return tomorrow_normalized


    def knn(self, x, y, x_tomorrow, x_incl_tomorrow, dates):
        """ predicts future price using knn """
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
            self.debug += '\nKNN: Price will be up {} %, predicted close is {}'.format(percent_gain, tomorrow)
        else:
            self.debug += '\nKNN: Price will be down {} %, predicted close is {}'.format(percent_gain, tomorrow)

        return tomorrow_normalized


    def check_sma(self):
        """ Checks Simple Moving Average
        """
        sma = self.sma()
        if self.daily['Adj Close'][-1] < sma[-1]:
            self.debug += '\nBelow SMA: buys + 1'
            self.buys += 1
        else:
            self.debug += '\nAbove SMA: sells + 1'
            self.sells += 1

    def check_bollinger(self):
        """ Checks Bollinger Bands
        Checks for crossovers INTO a band, or the current price's relation to the bands.
        """
        upper, lower = self.bollinger_bands()
        if self.daily['Adj Close'][-1] > upper[-1]:
            self.debug += '\nAbove upper bollinger: sells + 1'
            self.sells += 1
        elif self.daily['Adj Close'][-1] < lower[-1]:
            self.debug += '\nBelow lower bollinger: buys + 1'
            self.buys += 1

    def volume(self):
        """ Volume
        Determines whether volume has been trending up or down recently.
        Higher volume supports a trend.
        """
        vol = self.daily['Volume']
        sma = vol.rolling(20).mean()
        std = vol.rolling(20).std()
        upper = sma + std
        lower = sma - std

        if vol[-1] > upper[-1]:
            self.debug += '\nVolume > 1 STD above sma: buys + 1 and sells + 1'
            self.sells += 1
            self.buys += 1
        else:
            self.debug += '\nVolume in normal levels'

    def rsi(self, days=14):
        """ relevant strength index oscillator
        """
        daily_returns = self.daily_returns(self.daily[['Adj Close']]).tail(14)
        green_days = daily_returns[daily_returns > 0].dropna()
        red_days = daily_returns[daily_returns < 0].dropna()

        # RS= avg of Up Days - avg of Down Days. Note im adding bc red_days is negative.
        RS = (green_days.sum() / days + red_days.sum() / days)['Adj Close']
        rsi = 100 - (100 / (1 + RS))
        rsi = round(rsi, 2)
        if rsi < 0.3:
            self.debug += '\nRSI is under 30%: {0} buys + 1'.format(rsi)
            self.buys += 1
        elif rsi > 0.7:
            self.debug += '\nRSI over 70%: {0} sells + 1'.format(rsi)
            self.sells += 1
        else:
            self.debug += '\nRSI is normal(between 0.3-0.7): {}'.format(rsi)


    def sharpe(self):
        """ sharpe ratio
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
            self.debug += '\nSharpe Ratio too low: {} sells + 1'.format(sharpe)
            self.sells += 1
        elif sharpe < 2:
            self.debug += '\nSharpe Ratio is {}, buys + 1'.format(sharpe)
            self.buys += 1
        elif sharpe < 3:
            self.debug += '\nGood Sharpe Ratio of {}, buys + 2'.format(sharpe)
            self.buys += 2
        else:
            self.debug += '\nVery good Sharpe Ratio of {0}, buys + 3'.format(sharpe)
            self.buys += 3

    def extrema(self):
        """ checks for extrema: 100 day high/low
        """
        df = self.daily['Adj Close']
        min = df.min()
        max = df.max()
        if df[-1] == min:
            self.debug += '\nLocal Minimum: Buys + 1'
            self.buys += 1
        elif df[-1] == max:
            self.debug += '\nLocal Maximum: Sells + 1'
            self.sells += 1
        else:
            self.debug += '\nNot a local extrema.'

    def net_gains(self):
        """ Net Gains from this stock, if i have been holding.
        This is very important!
        "The first rule is to never lose. And the second rule is to never
        forget about that first rule." - Warren Buffett
        """
        if self.shares == 0:
            self.debug += '\nNo shares owned.'
        else:
            price = self.daily['Adj Close'][-1]
            gains = self.shares * (price - self.avg_paid)
            percent = (price / self.avg_paid - 1) * 100
            gains = round(gains, 2)
            percent = round(percent, 2)
            if gains < 0:
                penalty = self.sells - int(round(self.sells / 3.0, 0))
                self.debug += '\nNET LOSS: {}, {}%, AVOID SELLING! sells - {}'.format(gains, percent, penalty)
                self.sells -= penalty
            else:
                self.debug += '\nNet gains: ${}, {}%'.format(gains, percent)


    def beta_and_alpha(self):
        """ checks the Beta and Alpha of the stock with the market (sp500) '

        Also charts scatter plot.
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
            self.debug += '\nAlpha > 0: buys + {}'.format(alpha)
        else:
            self.debug += '\nAlpha < 0: {}'.format(alpha)

        # assuming favorable market conditions. else, it would be sells + 1.
        if beta > 1:
            self.buys += 1
            self.debug += '\nBeta > 1: buys + {}'.format(beta)
        else:
            self.debug += '\nBeta < 1: {}'.format(beta)

        # finish plotting scatter
        if self.will_plot:
            ax = joined.plot(title=self.symbol + ' vs The Market', kind = 'scatter', x='^GSPC', y=self.symbol)
            ax.set_xlabel("S&P 500")
            plt.plot(joined["^GSPC"], beta * joined['^GSPC'] + alpha, '-', color='r', label='Correlation')

            # plot expected beta (slope) of 1 and alpha (y- int.) of zero
            plt.plot(joined["^GSPC"], 1 * joined['^GSPC'] + 0, '-', color='gray', label='Beta of 1')
            plt.plot(joined["^GSPC"], 0 * joined['^GSPC'] + 0, '-', color='gray', label='Alpha of 0')
            plt.legend(loc='best')
