## Downloaded from
## https://stackoverflow.com/questions/44044263/yahoo-finance-historical-data-downloader-url-is-not-working
## Modified for Python 3
## Added --event=history|div|split   default = history
## changed so "to:date" is included in the returned results
## usage: download_quote(symbol, date_from, date_to, events).decode('utf-8')

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
        #print link
        r = Request(link, headers={'Cookie': cookie_str})

        try:
            response = urlopen(r)
            text = response.read()
            print ("{} downloaded".format(symbol))
            return text
        except URLError:
            print ("{} failed at attempt # {}".format(symbol, attempts))
            attempts += 1
            time.sleep(2*attempts)
    return b''


def get_data(symbol_val, start=None, end=None):
    """ Returns pandas dataframe of Date, Adj Close, and Volume from Yahoo Finance API, or None if not available.
    End date can be assumed to be today. Start date is automatically 140 days ago.
    """
    event_val = "history"
    output_val = "data/" + symbol_val + '.csv'

    if start is None and end is None:
        from_val, to_val = get_date(140)
    elif end is None:
        from_val = start
        now = datetime.datetime.now()
        to_val = '{}-{}-{}'.format(now.year, now.month, now.day)
    else:
        from_val = start
        to_val = end

    # if os.path.isfile(output_val):
    #      df = pd.read_csv(output_val, index_col='Date', parse_dates=True, usecols=['Date', 'Adj Close', 'Volume'], na_values=['NaN'])
    #      df = df.dropna()
    #      return df

    print ("downloading {}".format(symbol_val))
    csv = download_quote(symbol_val, from_val, to_val, event_val)
    if csv is not None:
        with open(output_val, 'wb') as f:
            f.write(csv)
        print ("{} written to {}".format(symbol_val, output_val))
        df = pd.read_csv(output_val, index_col='Date', parse_dates=True, usecols=['Date', 'Adj Close', 'Volume'], na_values=['NaN'])
        df = df.dropna()
        return df

def get_date(days):
    """ Returns starting and ending date (like '2018-08-15') given amount of days to go back """
    now = datetime.datetime.now()
    past = now - datetime.timedelta(days=days)
    now_string = '{}-{}-{}'.format(now.year, now.month, now.day)
    past_string = '{}-{}-{}'.format(past.year, past.month, past.day)
    return past_string, now_string
