
import pandas as pd

import Stock
import Portfolio

def best_in_sp500():
    """ Finds the top 20 stocks to buy right now in the S&P 500 """
    stocks = []
    df = pd.read_csv('indices/sp_members.csv')
    for sym in df['Symbol']:
        sym.replace(".", "-") # BRK.B -> BRK-B
        try:
            x = Stock.Stock(sym,  0, 0, plot=False)
            x.buy_or_sell()
            stocks.append(x)
        except Exception:
            print('Failed to determine for {}'.format(sym))

    stocks.sort(key=lambda x: x.buying_certainty, reverse=True)
    print('\n\n\nSORTED STOCKS:\n\n\n')
    sorted = stocks[:20]
    for x in sorted:
        print(x.debug)


def optimized_dowjones():
    """ The best possible allocations of stocks in the Dow Jones from the past year """
    # stocks = pd.read_csv('indices/dow_members.csv')['Symbol'].values
    # port = Portfolio.Portfolio(1, stocks, 30 * [1./30], '2017-01-01', '2017-12-31')
    port = Portfolio.Portfolio(1, ['GOOGL', 'AAPL', 'AMZN'], [1., 0., 0.], '2017-01-01', '2017-12-31')
    print("Sharpe:")
    print(port.sharpe_ratio())
    print("Volatility:")
    print(port.risk())
    print("Avg Daily Return:")
    print(port.avg_daily_returns())
    print("Cumulative returns:")
    print(port.cumulative_return())
    print("Ideal Allocations for Sharpe:")
    port.optimizer()

if __name__ == '__main__':
    optimized_dowjones()


#
