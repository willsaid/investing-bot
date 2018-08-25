""" Example usages of the Stock and Portfolio classes.

Usage: 'python3 Examples.py'
"""

import pandas as pd
import numpy as np

# local
from src import Stock, Portfolio



def best_in_sp500(fresh):
    """ Finds the top 20 stocks to buy right now in the S&P 500 """
    stocks = []
    df = pd.read_csv('indices/sp_members.csv')
    for sym in df['Symbol']:
        print(sym)
        try:
            x = Stock.Stock(sym,  0, 0, fresh, plot=False)
            x.buy_or_sell(debug=False)
            stocks.append(x)
        except Exception:
            print('Failed to determine for {}'.format(sym))

    stocks.sort(key=lambda x: x.buying_certainty, reverse=True)
    print('\n\n\nSORTED STOCKS:\n\n\n')
    sorted = stocks[:20]
    for x in sorted:
        print(x.debug)

def sp_over_time(fresh):
    """ performs stock_over_time() on all stocks in the S&P 500, then ranks the top 20 best stocks to buy
    given historical in multiple time scales like 5 years, 1 year, and 100 days
    """
    stocks = []
    df = pd.read_csv('indices/sp_members.csv')
    fresh_copy = fresh
    fresh_sp = fresh
    for sym in df['Symbol']:
        print(sym)
        fresh_copy = fresh
        try:
            day_lengths = [x * (7./5) for x in [365*5, 365, 100]] # * 7/5 for just trading days
            stock_intervals = [] # the list of, say, only AAPL's 5 year, 1 year, and 100 day analysis
            for days_length in day_lengths:
                x = Stock.Stock(sym, 0, 0, fresh_copy, fresh_sp=fresh_sp, plot=False, days=days_length)
                x.buy_or_sell(debug=False)
                fresh_copy = False # so it doesn't change existing csv data of 5 years
                fresh_sp = False # so it doesn't fetch GSPC (s&p 500) again each time
                stock_intervals.append(x)
            stocks.append(stock_intervals)
        except Exception:
            print('Failed to determine for {}'.format(sym))

    # sort the stock list of interval lists by the mean buying certainty of each stock's intervals
    stocks.sort(key=lambda x: np.array([y.buying_certainty for y in x]).mean(), reverse=True)
    print('\n\n\nSORTED STOCKS:\n\n\n')
    sorted = stocks[:20] # only show top 20
    for stock_interval in sorted:
        for stock in stock_interval:
            print(stock.debug)

def optimized_dowjones(fresh):
    """ The best possible allocations of all 30 stocks in the Dow Jones Index over the past year """
    stocks = pd.read_csv('indices/dow_members.csv')['Symbol'].values
    guess_allocations = 30 * [1./30] # even distribution of 1/30 for each stock
    port = Portfolio.Portfolio(1, stocks, guess_allocations, fresh=fresh, days=365).debug()


def custom_portfolio(fresh):
    """ Example portfolio optimization and analysis of 4 stocks over the past year """
    Portfolio.Portfolio(1, ['AAPL', 'GOOGL', 'AMZN', 'FB'], [.25, .25, .25, .25], fresh=fresh, days=365).debug()


def single_stock(fresh):
    """ Analyzes and predicts future prices for a stock given performance in the past 100 trading days """
    symbol = input('\nSymbol: ')
    shares = int(input('Current Shares: '))
    avg_paid = int(input('Avg Price Paid: ')) if int(shares) > 0 else 0

    Stock.Stock(symbol, shares, avg_paid, fresh).buy_or_sell()


def stock_over_time(fresh):
    """ Same as single_stock(), but incorporates several intervals of data:
    Past 5 trading years, past 1 trading year, and past 100 trading days
    """
    symbol = input('\nSymbol: ')
    shares = int(input('Current Shares: '))
    avg_paid = int(input('Avg Price Paid: ')) if int(shares) > 0 else 0

    # past 5 years, 1 year, 100 trading days, 1 trading week
    day_lengths = [x * (7./5) for x in [365*5, 365, 100]] # for just trading days
    for days_length in day_lengths:
        Stock.Stock(symbol, shares, avg_paid, fresh, days=days_length).buy_or_sell()
        fresh = False # so it doesn't change existing csv data of 5 years

if __name__ == '__main__':
    option = input('Choose example:\n  \'b\': Best Stocks to Buy in S&P 500\n  \'o\': Optimized Dow Jones Portfolio\n  \'s\': Single Stock Prediction\n  \'c\': Custom Portfolio Analysis\n  \'t\': Stock Analysis over Time\n  \'sp\': Best S&P Stocks over Time\n')
    fresh = input('Need Fresh data? (Default to yes) [\'y\' or \'n\']\n') == 'y'

    if option == 'b':
        best_in_sp500(fresh)
    elif option == 'o':
        optimized_dowjones(fresh)
    elif option == 's':
        single_stock(fresh)
    elif option == 'c':
        custom_portfolio(fresh)
    elif option == 't':
        stock_over_time(fresh)
    elif option == 'sp':
        sp_over_time(fresh)
#
