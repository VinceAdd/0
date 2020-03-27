# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:22:25 2020

@author: vincent
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

import math
from scipy.optimize import minimize
from scipy.stats import norm

import yfinance as yf


# -- inputs -- #
# period to use in yahoo finance
yf_period = "5y"
# output directory
output_path = "C:/"
# list of tickers to look for
tickers = ["DAI.DE", "BNP.PA"]


def main(yf_period, tickers):

    filenames = []
    err_list = []
    no_err_list = []
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dataframes = []
    final_dfs = []
    ### updated : ase when error occurs on loading process
    for tick in tickers:
        print("loading " + str(tick))
        temp = yf.download(tick, period=yf_period)
        if len(temp) < 1:
            err_list.append(tick)
            continue
        dataframes.append(temp)

    for tick in tickers:
        if tick not in err_list:
            no_err_list.append(tick)
    tickers = no_err_list
    print(
        "nb of err : "
        + str(len(err_list))
        + ", remaining "
        + str(len(tickers))
        + " tickers"
    )
    ### updated : ase when error occurs on loading process

    expected_return = []  # Historical average return
    annual_avg_exp_return = []
    des = pd.DataFrame()
    for df in dataframes:
        df.dropna(inplace=True)
        df["return"] = df["Adj Close"].pct_change(periods=1, fill_method="ffill")
        expected_return = np.append(expected_return, np.mean(df["return"]))
        annual_avg_exp_return = np.append(
            annual_avg_exp_return, np.mean(df["return"]) * 252
        )
        des = des.append(df.describe())
        final_dfs.append(df)
    dfs = pd.concat(final_dfs, axis=1, join="outer")
    dfs.drop(
        ["Open", "High", "Low", "Close", "Adj Close", "Volume"], axis=1, inplace=True
    )
    dfs.dropna(inplace=True)

    cov_mat = np.array(dfs.cov())
    corr_mat = np.array(dfs.corr())

    corr = pd.DataFrame(corr_mat, index=tickers, columns=tickers)

    n = len(dfs.columns)
    equal_weight = 1 / n  #  x0 for optimization
    weights = np.array([equal_weight] * n)
    w_mat = (1 + dfs).cumprod(axis=0).shift() * weights
    w_mat.iloc[0] = weights

    ### -- OPTIMIZATION -- ###

    cons = [{"type": "eq", "fun": constraint1}]
    bnds = [(0.0, 1.0)] * len(weights)

    res_max_return = minimize(
        max_ptf_return,
        x0=weights,
        args=annual_avg_exp_return,
        constraints=cons,
        bounds=bnds,
    )
    res_min_std = minimize(
        min_ptf_std, x0=weights, args=cov_mat, constraints=cons, bounds=bnds
    )
    res_max_sharpe = minimize(
        max_sharpe_ratio,
        x0=weights,
        args=(annual_avg_exp_return, cov_mat),
        constraints=cons,
        bounds=bnds,
    )
    res_min_VaR95 = minimize(
        min_ptf_VaR,
        x0=weights,
        args=(annual_avg_exp_return, cov_mat, 0.95),
        constraints=cons,
        bounds=bnds,
    )
    res_min_VaR99 = minimize(
        min_ptf_VaR,
        x0=weights,
        args=(annual_avg_exp_return, cov_mat, 0.99),
        constraints=cons,
        bounds=bnds,
    )

    results = {
        "max_return": res_max_return,
        "min_std": res_min_std,
        "max_sharpe": res_max_sharpe,
        "min_VaR95": res_min_VaR95,
        "min_VaR99": res_min_VaR99,
    }

    for k, v in results.items():
        showresult(
            k,
            v,
            tickers,
            filenames,
            annual_avg_exp_return,
            cov_mat,
            dfs,
            weights,
            corr,
            output_path,
        )

    ### -- FUNCTIONS -- ###


# function to calculate the ptf return
def calcPftReturn(w, µ):
    ptfReturn = np.matmul(w, µ)
    return ptfReturn


# function to calculate the ptf std
def calcPftStd(w, cov):
    ptfStd = math.sqrt(
        np.matmul(np.matmul(w, cov), w)  # matrix product / multiplication
    ) * math.sqrt(
        252
    )  # annually
    return ptfStd


# function to calculate ptf sharpe ratio
def calcPtfSharpeRatio(ptfReturn, ptfStd, rf=0.0):
    return (ptfReturn - rf) / ptfStd


# function to calculate ptf VaR         # -- parametric
def calcVaR(µ, std, z):
    return -µ + (std * norm.ppf(z))


# MAXIMIZE the ptf_return = Max(V(w) * V(µ))
def max_ptf_return(w, µ):
    return -1 * calcPftReturn(w, µ)


# MINIMIZE the ptf_std
def min_ptf_std(w, cov):
    return calcPftStd(w, cov)


# MAXIMIZE the SharpeRatio
def max_sharpe_ratio(w, µ, cov):
    ptfReturn = calcPftReturn(w, µ)
    ptfStd = calcPftStd(w, cov)
    SharpeRatio = calcPtfSharpeRatio(ptfReturn, ptfStd)
    return -1 * SharpeRatio


# MINIMIZE the ptf VaR
def min_ptf_VaR(w, µ, cov, z):
    ptfReturn = calcPftReturn(w, µ)
    ptfStd = calcPftStd(w, cov)
    VaR = calcVaR(ptfReturn, ptfStd, z)
    return VaR


def constraint1(w):
    return w.sum() - 1.0  # weights must equal to 1, but constraint must equal 0.

    ### -- RESULT FETCHER -- ###


def showresult(
    key,
    res,
    tickers,
    filenames,
    annual_avg_exp_return,
    cov_mat,
    dfs,
    weights,
    corr,
    output_path,
):
    print(res.message)
    print(res.success)
    res_rounded = np.round(res.x, 2)
    if tickers:
        res_table = pd.DataFrame(
            res_rounded.reshape(-1, len(res_rounded)), columns=tickers
        )
        print(res_table.T)
    else:
        res_table = pd.DataFrame(
            res_rounded.reshape(-1, len(res_rounded)), columns=filenames
        )
        print(res_table.T)
    res_table = res_table.T.rename(columns={0: "weights"})
    res_table.to_excel
    ptfReturn = calcPftReturn(res.x, annual_avg_exp_return)
    ptfStd = calcPftStd(res.x, cov_mat)
    SharpeRatio = calcPtfSharpeRatio(ptfReturn, ptfStd)
    VaR95 = calcVaR(ptfReturn, ptfStd, 0.95)
    VaR99 = calcVaR(ptfReturn, ptfStd, 0.99)
    Daily_VaR95 = VaR95 / math.sqrt(252)
    Daily_VaR99 = VaR99 / math.sqrt(252)
    print("optimized ptf return : " + str(np.round(ptfReturn, 4)))
    print("optimized ptf std dev : " + str(np.round(ptfStd, 4)))
    print("optimized ptf sharpe ratio : " + str(np.round(SharpeRatio, 4)))
    print("Annual 99 VaR : " + str(VaR99 * 100) + " %")
    print("Daily 99 VaR : " + str(VaR99 / math.sqrt(252) * 100) + " %")
    print("Annual 99 VaR for a 10k ptf: " + str(VaR99 * 10000))
    print("Daily 99 VaR for a 10k ptf: " + str(VaR99 / math.sqrt(252) * 10000))
    print("Annual 95 VaR : " + str(VaR95 * 100) + " %")
    print("Daily 95 VaR : " + str(VaR95 / math.sqrt(252) * 100) + " %")
    print("Annual 95 VaR for a 10k ptf: " + str(VaR95 * 10000))
    print("Daily 95 VaR for a 10k ptf: " + str(VaR95 / math.sqrt(252) * 10000))

    result = {
        "ptfReturn": ptfReturn,
        "ptfStd": ptfStd,
        "SharpeRatio": SharpeRatio,
        "Annual VaR95": VaR95,
        "Annual VaR99": VaR99,
        "Annual VaR95 10k ptf": VaR95 * 10000,
        "Annual VaR99 10k ptf": VaR99 * 10000,
        "Daily VaR95": Daily_VaR95,
        "Daily VaR99": Daily_VaR95,
        "Daily VaR95 10k ptf": Daily_VaR95 * 10000,
        "Daily VaR99 10k ptf": Daily_VaR99 * 10000,
    }
    global_result = pd.DataFrame.from_dict(result, orient="index")

    with pd.ExcelWriter(output_path + "result_" + str(key) + ".xlsx") as writer:
        res_table.to_excel(writer, sheet_name="Result table")
        corr.to_excel(writer, sheet_name="Correl table")
        global_result.to_excel(writer, sheet_name="Ptf result")


if __name__ == "__main__":

    print("Starting @ %s" % datetime.now())

    main(yf_period, tickers)
