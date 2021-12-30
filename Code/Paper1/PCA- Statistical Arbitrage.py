# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 10:14:15 2018

@author: xinzeng
"""

import pandas as pd
import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime

data1 = pd.read_csv("data_aggregate.csv")
column_list = data1.columns.values.tolist()
#print(column_list)

# initialize the needed variables
delay = 252
factor = 15
k = 8.4
# k=4.2
window = 60
sbo = 2
sso = 2
sbc = 0.75
ssc = 0.5
r = 0.00
tran_cost = 0
leverage = 1
start_val = 100

# PCA Method
# find return
def find_Return(price):
    ret = (price - price.shift(1))/price
    ret = ret.drop(ret.index[0])
    # fill the nan values with 0
    ret = ret.fillna(value = 0)
    return ret

#print(find_Return(data1[column_list[1]]))

# Using PCA to find factors we need. fac_num total number of factors, delay total number of days we use
def find_Factor(ret, delay, fac_num):
    # standardize the return
    mean = ret.mean(axis = 0)
    std = ret.std(axis = 0)
    std_ret = (ret - mean)/std
    
    #PCA process
    pca = PCA(n_components = fac_num)
    pca.fit(std_ret[0:delay])
    weight = pd.DataFrame(pca.components_)
    weight.columns = std.index
    weight = weight/std
    factor_ret = pd.DataFrame(np.dot(ret, weight.transpose()),index = ret.index)
    return factor_ret, weight

def find_Residue(ret,ret_factorret):
    #storing the residues
    res = pd.DataFrame(columns = ret.columns, index = ret.index)
    coef = pd.DataFrame(columns = ret.columns, index = range(15))
    ols = LinearRegression()
    for i in ret.columns:
        ols.fit(ret_factorret, ret[i])
        res[i] = ret[i]-ols.intercept_-np.dot(ret_factorret, ols.coef_)
        coef[i] = ols.coef_
    return res,coef

def find_Target_sscore(res, k):
    cum_res = res.cumsum()
    m = pd.Series(index = cum_res.columns)
    sigma_eq = pd.Series(index = cum_res.columns)
    for i in cum_res.columns:
        b = cum_res[i].autocorr()
        print(b)
        try:
            k_0 = -math.log(b) * 252
        except ValueError:
            k_0 = 0
        if k_0 > k:
            temp = (cum_res[i]-cum_res[i].shift(1)* b)[1:]
            a = temp.mean()
            cosi =temp - a
            m[i] = a/(1-b)
            sigma_eq[i]=math.sqrt(cosi.var()/(1-b*b))
    m = m.dropna()
    m = m - m.mean()
    sigma_eq = sigma_eq.dropna()
    s_score = -m/sigma_eq
    return s_score

def find_Sharpe_Ratio(pnl,r):
    mean = math.log(pnl.iloc[len(pnl.index)-1]/pnl.iloc[0])/(len(pnl.index)) * 252.0
    print(mean)
    ret = find_Return(pnl)
    std = ret.std() * math.sqrt(252)
    return (mean - r)/std

def find_Maximum_Drawdown(pnl):
    ret = find_Return(pnl)
    r = ret.add(1).cumprod()
    dd = r.div(r.cummax()).sub(1)
    mdd = dd.min()
    end = dd.argmin()
    start = r.loc[:end].argmax()
    return mdd

def find_Cumulative_Return(pnl):
    return (pnl.iloc[len(pnl.index)-1] - pnl.iloc[0]) / pnl.iloc[0]


# get the dataframe of return
returns = pd.DataFrame()
for i in range(1, len(column_list)):
    returns[column_list[i]] = find_Return(data1[column_list[i]])
# get the fator details
PCA_factor, weight = find_Factor(returns, delay, factor)
# get the residual details
residual = find_Residue(returns, PCA_factor)

data2 = data1.drop(["Date"], axis = 1)
position_stock = pd.DataFrame(0,columns = data2.columns, index = ['stock']+list(range(15)))
position_stock_before = pd.Series(0, index = data2.columns)
pnl = pd.Series(start_val, index = data2.index[delay-1:])

#len(data1.index)-1
for t in range(delay-1,len(data1.index)-1):
    price_t = data2[(t-window):(t+1)]
    # get the dataframe of return
    ret_t = find_Return(price_t)
    # get the fator details
    ret_factorret_t = PCA_factor[(t-window+1):(t+1)]
    # get the residual details
    res_t, coef_t = find_Residue(ret_t,ret_factorret_t)
    target = find_Target_sscore(res_t, k)
    print(target)
    # find the strategy for this time period:
    for i in position_stock.columns:
        if not i in target.index :
            if position_stock[i]['stock'] != 0:
                position_stock[i] = 0
        else:
            if position_stock[i]['stock'] == 0:
                if target[i] < -sbo:
                    position_stock[i]['stock'] = leverage
                    position_stock[i][1:] = -leverage * coef_t[i]
                elif target[i] > sso:
                    position_stock[i]['stock'] = - leverage
                    position_stock[i][1:] = leverage * coef_t[i]
            elif position_stock[i][0] >0 and target[i] > -ssc:
                position_stock[i] = 0
            elif position_stock[i][0] <0 and target[i] < sbc:
                position_stock[i] = 0
    # calculate the pnl for the next period
#    dps_t = dps_original.iloc[t+1]
    pri_t = data2.iloc[t+1]
#    temp = (dps_t/pri_t).fillna(0)
    position_stock_temp = pd.Series(0,index = data2.columns)
    fac_sum = position_stock.sum(axis = 1)[1:]
    for i in weight.columns:
        position_stock_temp = sum(weight[i] * fac_sum)
    position_stock_temp = position_stock_temp + position_stock.iloc[0]
    change = sum(abs(position_stock_temp - position_stock_before))
    position_stock_before = position_stock_temp
#    pnl.iloc[t-delay + 2] = pnl.iloc[t-delay + 1] * ( 1 + r /252.0) + np.dot(position_stock.loc['stock'], ret_original.iloc[t]+temp) + np.dot(position_stock.sum(axis = 1)[1:], ret_factorret.iloc[t]) - position_stock.sum().sum() * r /252.0 - change * tran_cost
    pnl.iloc[t-delay + 2] = pnl.iloc[t-delay + 1] * ( 1 + r /252.0) + np.dot(position_stock.loc['stock'], returns.iloc[t]) + np.dot(position_stock.sum(axis = 1)[1:], PCA_factor.iloc[t]) - position_stock.sum().sum() * r /252.0 - change * tran_cost
    print(pnl.iloc[t-delay + 2])

sharpe_ratio = find_Sharpe_Ratio(pnl,r)
maximum_drawdowns = find_Maximum_Drawdown(pnl)
cumulative_return = find_Cumulative_Return(pnl)


plt.plot(list(range(3390)),pnl)
x_value = []
for element in data1["Date"][delay-1:]:
    x_value.append(datetime.datetime.strptime(element, '%Y-%m-%d'))
plt.plot(x_value,list(pnl))    
    
    