""" Example usages of the Stock and Portfolio classes.

Usage: 'python3 Examples.py'
"""

import pandas as pd

# local
import Stock
import Portfolio


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


def optimized_dowjones(fresh):
    """ The best possible allocations of stocks in the Dow Jones Index from the past year """
    stocks = pd.read_csv('indices/dow_members.csv')['Symbol'].values
    guess_allocations = 30 * [1./30] # even distribution of 1/30 for each stock
    port = Portfolio.Portfolio(1, stocks, guess_allocations, '2017-08-16', '2018-08-16', fresh).debug()


def custom_portfolio(fresh):
    """ Example portfolio optimization and analysis
    of Apple and Google from Aug 16, 2017 to Aug 16, 2018.
    """
    Portfolio.Portfolio(1, ['AAPL', 'GOOGL', 'AMZN'], [.34, .33, .33], '2017-08-16', '2018-08-16', fresh).debug()


def single_stock(fresh):
    symbol = input('\nSymbol: ')
    shares = int(input('Current Shares: '))
    avg_paid = int(input('Avg Price Paid: ')) if int(shares) > 0 else 0

    Stock.Stock(symbol, shares, avg_paid, fresh).buy_or_sell()



if __name__ == '__main__':
    option = input('Choose example:\n  \'b\': Best Stocks to Buy in S&P 500\n  \'o\': Optimized Dow Jones Portfolio\n  \'s\': Single Stock Prediction\n  \'c\': Custom Portfolio Analysis\n')
    fresh = input('Need Fresh data? (Default to yes) [\'y\' or \'n\']\n') == 'y'

    if option == 'b':
        best_in_sp500(fresh)
    elif option == 'o':
        optimized_dowjones(fresh)
    elif option == 's':
        single_stock(fresh)
    elif option == 'c':
        custom_portfolio(fresh)
#
