# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 10:59:29 2018

@author: xinzeng
"""

import pandas as pd
import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# initialize the needed variables
delay = 252
factor = 15
k = 8.4
# k=4.2
window = 60
sbo = 1.25
sso = 1.25
sbc = 0.75
ssc = 0.5
r = 0.00
tran_cost = 0.0005
leverage = 1
start_val = 100

data1 = pd.read_csv("data_aggregate.csv")
column_list = data1.columns.values.tolist()
# store the end point in each ETF
ETF_list = [2, 11, 16, 20, 27, 33, 39, 47, 55, 76, 90, 106, 126, 132, 143]
data_2 = data1.drop(["Date"], axis = 1)

def find_Return(price):
    ret = (price - price.shift(1))/price
    ret = ret.drop(ret.index[0])
    # fill the nan values with 0
    ret = ret.fillna(value = 0)
    return ret

def find_Residue(ret1,ret_factorret1):
    #storing the residues
    ret = ret1.to_frame()
    ret_factorret = ret_factorret1.to_frame()
    res = pd.DataFrame(columns = ret.columns, index = ret.index)
    coef = pd.DataFrame(columns = ret.columns, index = list(range(1)))
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

Index = pd.DataFrame()
Index["Date"] = data1["Date"]
for i in range(len(ETF_list)):
    datas = []
    if i == 0:
        begin = 1
    else:
        begin = ETF_list[i-1]
    end = ETF_list[i]
    for j in range(len(data1["Date"])):
        m = 0
        for k in range(begin, end+1):
            m += data1[column_list[k]][j]
        datas.append(m/(end-begin+1))
    Index[str(i)] = datas

Index2 = Index.drop(["Date"], axis = 1)
return_stock = find_Return(data_2)
return_ETF = find_Return(Index2)

position_stock = pd.DataFrame(0,columns = data_2.columns, index = ['stock']+list(range(1)))
position_stock_before = pd.Series(0, index = data_2.columns)
pnl = pd.Series(start_val, index = data_2.index[delay-1:])

ETF_list1 = [2, 11, 16, 20, 27, 33, 39, 47, 55, 76, 90, 106, 126, 132, 143]

for t in range(delay-1, len(data1.index)-1):
    residuals = pd.DataFrame()
    coefs = pd.DataFrame()
    for j in range(1, len(column_list)):
        for o in range(0, len(ETF_list1)):
            if j <= ETF_list1[o]:
                need = o
                break
        price_t = data_2[column_list[j]][(t-window):(t+1)]
        ret_t = find_Return(price_t)
        price2_t = Index2[str(need)][(t-window):(t+1)]
        ret2_t = find_Return(price2_t)
    # get the residual details
        res_t, coef_t = find_Residue(ret_t,ret2_t)
        residuals[column_list[j]] = res_t[column_list[j]]
        coefs[column_list[j]] = coef_t[column_list[j]]
    target = find_Target_sscore(residuals, k)
    for i in position_stock.columns:
        if not i in target.index :
            if position_stock[i]['stock'] != 0:
                position_stock[i] = 0
        else:
            if position_stock[i]['stock'] == 0:
                if target[i] < -sbo:
                    position_stock[i]['stock'] = leverage
                    position_stock[i][1:] = -leverage * coefs[i]
                elif target[i] > sso:
                    position_stock[i]['stock'] = - leverage
                    position_stock[i][1:] = leverage * coefs[i]
            elif position_stock[i][0] >0 and target[i] > -ssc:
                position_stock[i] = 0
            elif position_stock[i][0] <0 and target[i] < sbc:
                position_stock[i] = 0
    #calculate the pnl for the next period
    pri_t = data_2.iloc[t+1]
    position_stock_temp = pd.Series(0,index = data_2.columns)
    fac_sum = position_stock.sum(axis = 1)[1:]
    for i in data_2.columns:
        position_stock_temp = np.mean(fac_sum)
    position_stock_temp = position_stock_temp + position_stock.iloc[0]
    change = sum(abs(position_stock_temp - position_stock_before))
    position_stock_before = position_stock_temp
#    pnl.iloc[t-delay + 2] = pnl.iloc[t-delay + 1] * ( 1 + r /252.0) + np.dot(position_stock.loc['stock'], ret_original.iloc[t]+temp) + np.dot(position_stock.sum(axis = 1)[1:], ret_factorret.iloc[t]) - position_stock.sum().sum() * r /252.0 - change * tran_cost
    pnl.iloc[t-delay + 2] = pnl.iloc[t-delay + 1] * ( 1 + r /252.0) + np.dot(position_stock.loc['stock'], return_stock.iloc[t]) - position_stock.sum().sum() * r /252.0 - change * tran_cost
    print(pnl.iloc[t-delay + 2])

x_value = []
for element in data1["Date"][delay-1:]:
    x_value.append(datetime.datetime.strptime(element, '%Y-%m-%d'))
plt.plot(x_value,pnl)          
    
    
    
    
