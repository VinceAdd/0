# 0

ptf_optimizer_v1.0.py

<strong> inputs <strong />

<strong>  >> period to use in yahoo finance data extraction <strong /><br />
yf_period = "5y"  <br />
<strong> >> output directory  <strong /><br />
output_path = "C:/"  <br />
<strong> >> list of tickers to look for  <strong /><br />
tickers = ["DAI.DE", "BNP.PA"]  <br />

Goal :
Calculates the weights to maximize the ptf return, minimize the standard deviation, maximize the Sharpe Ratio, minimize the VaR @ 95 and 99 % for a given period of time.
Correlation matrix is also given.
