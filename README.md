# 0

ptf_optimizer_v1.0.py

<strong> inputs <strong />

<strong>  >> period to use in yahoo finance data extraction <strong /><br />
yf_period = "5y"  --> valid period : 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max  <br />
<strong> >> output directory  <strong /><br />
output_path = os.path.dirname(os.path.realpath(__file__)) <br />
<strong> >> list of tickers to look for  <strong /><br />
tickers = ["DAI.DE", "BNP.PA"]  <br />

Goal :
Calculates the weights to maximize the portfolio return, minimize the standard deviation, maximize the Sharpe Ratio, minimize the Value At Risk @ 95 and 99 % for a given period of time.
Correlation matrix is also given.
