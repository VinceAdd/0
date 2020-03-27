# 0

ptf_optimizer_v1.0.py

inputs

# period to use in yahoo finance data extraction
yf_period = "5y"
# output directory
output_path = "C:/"
# list of tickers to look for
tickers = ["DAI.DE", "BNP.PA"]


Calculates the weights to maximize the ptf return, minimize the standard deviation, maximize the Sharpe Ratio, minimize the VaR @ 95 and 99 % for a given period of time.
Correlation matrix is also given.
