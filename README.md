# investing-bot
### Predictive model for the quantitative analysis of stocks using machine learning AI

## Usage 
##### Setup
Make sure you have <a href="https://www.python.org/downloads/">Python 3 installed</a>.
You may also need to install some libraries with:
```
pip3 install urllib.request scikit-learn matplotlib pandas scipy
```

##### Run
Inside the directory you downloaded Investing Bot, run Examples.py:
```
python3 Examples.py
```
Examples.py will walk you through example usages of tbe Stock and Portfolio classes, like: 
- Analyzing and forecasting a single stock as a buy, sell, or holding oppurtunity
- Scanning all 500 stocks in the S&P 500 index, ranking the top 20 stocks to buy right now
- Optimizing the allocation percentages of stocks in your custom Portfolio to maximize risk-adjusted returns (Sharpe Ratio)
- Maxmizing the Sharpe Ratio of all 30 stocks in the Dow Jones Industrial Index by determining optimal percentage allocations

These will also plot charts to display the data.

<img src="/screenshots/fb.png" width="250px">
<img src="/screenshots/aapl1.png" width="250px">
<img src="/screenshots/aapl2.png" width="250px">
<img src="/screenshots/dow1.png" width="250px">
<img src="/screenshots/dow2.png" width="250px">
