""" Fetches Quote History from Yahoo Finance by setting a cookie,
downloading the parsed CSV to the data/ directory,
and returning the pandas dataframe for the quote
"""

import re
from urllib.request import urlopen, Request, URLError
import calendar
import datetime
import getopt
import sys
import time
import pandas as pd
import os


crumble_link = 'https://finance.yahoo.com/quote/{0}/history?p={0}'
crumble_regex = r'CrumbStore":{"crumb":"(.*?)"}'
cookie_regex = r'set-cookie: (.*?); '
quote_link = 'https://query1.finance.yahoo.com/v7/finance/download/{}?period1={}&period2={}&interval=1d&events={}&crumb={}'


def get_crumble_and_cookie(symbol):
    link = crumble_link.format(symbol)
    response = urlopen(link)
    match = re.search(cookie_regex, str(response.info()))
    cookie_str = match.group(1)
    text = response.read().decode("utf-8")
    match = re.search(crumble_regex, text)
    if match is not None:
        crumble_str = match.group(1)
        return crumble_str , cookie_str


def download_quote(symbol, date_from, date_to, events):
    time_stamp_from = calendar.timegm(datetime.datetime.strptime(date_from, "%Y-%m-%d").timetuple())
    next_day = datetime.datetime.strptime(date_to, "%Y-%m-%d") + datetime.timedelta(days=1)
    time_stamp_to = calendar.timegm(next_day.timetuple())
    attempts = 0
    while attempts < 5:
        if get_crumble_and_cookie(symbol) is None: return None
        crumble_str, cookie_str = get_crumble_and_cookie(symbol)
        link = quote_link.format(symbol, time_stamp_from, time_stamp_to, events,crumble_str)
        r = Request(link, headers={'Cookie': cookie_str})
        try:
            response = urlopen(r)
            text = response.read()
            print ("{} downloaded".format(symbol))
            return text
        except URLError:
            print ("{} failed at attempt # {}".format(symbol, attempts))
            attempts += 1
            time.sleep(2 * attempts)
    return b''


def get_data(symbol_val, is_fresh, start=None, end=None, days=None):
    """ Returns pandas dataframe of Date, Adj Close, and Volume from Yahoo Finance, or None if not available.
    End date can be assumed to be today.
    Start date is automatically 140 days ago, or about 100 market days.
    Days ex: 365 for the past year.
    """
    symbol_val = symbol_val.replace('.', '-') # BRK.B -> BRK-B
    event_val = "history" # historical data
    output_val = "data/" + symbol_val + '.csv' # file destination

    # set begin and end date time strings
    if start is None and end is None:
        from_val, to_val = get_date(days)
    elif end is None:
        from_val = start
        to_val = get_date_string(datetime.datetime.now())
    else:
        from_val = start
        to_val = end

    # use old data if present and if is_fresh
    if not is_fresh and os.path.isfile(output_val):
        try:
            return pd.read_csv(output_val, index_col='Date', parse_dates=True, usecols=['Date', 'Adj Close', 'Volume'], na_values=['NaN']).dropna()[from_val: to_val]
        except Exception:
            print('Failed to read from {}. Now fetching {} fresh from network.'.format(output_val, symbol_val))

    # Download data from Yahoo
    print ("downloading {}".format(symbol_val))
    csv = download_quote(symbol_val, from_val, to_val, event_val)
    if csv is not None:
        with open(output_val, 'wb') as f:
            f.write(csv)
        print ("{} written to {}".format(symbol_val, output_val))
        return pd.read_csv(output_val, index_col='Date', parse_dates=True, usecols=['Date', 'Adj Close', 'Volume'], na_values=['NaN']).dropna()[from_val: to_val]


def get_date(days):
    """ Returns starting and ending date (like '2018-08-15') given amount of days to go back """
    if days is None: days = 140 # about 100 trading days
    now = datetime.datetime.now()
    past = now - datetime.timedelta(days=days)
    return get_date_string(past), get_date_string(now)

def get_date_string(datetime):
    return '{}-{}-{}'.format(datetime.year, datetime.month, datetime.day)
