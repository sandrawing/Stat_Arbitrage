# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:06:22 2018

@author: xinzeng
"""

import pandas as pd
import numpy as np
import pylab
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import statsmodels.tsa.stattools as ts
import statsmodels.tools as st

data1 = pd.read_csv("C:\\Users\\xinzeng\\Desktop\\Internship\\data_long.csv")
interval = 3

index1 = []
index2 = [0]
months = []
index11 = 0
for i in range(0, len(data1["date"])):
    l1 = data1["date"][i].split(" ")
    l2 = l1[0].split("/")
    year = int(l2[0])
    month = int(l2[1])
    months.append(month)
    if i > 0 and months[i] != months[i-1]:
        index11 += 1
        index1.append(index11)
        index2.append(i)
    else:
        index1.append(index11)
    
data1["index"] = index1

close = []
for i in range(0, len(index2)):
    if i!=0 and i%3 == 0:
        close.append(index2[i])

# num_index >= interval
def get_ratio(num_index):
    index_num = num_index - num_index%3
    begin = index2[index_num-interval]
    end = index2[index_num]
    x_value = data1["ldce_price"][begin:end]
    y_value = data1["ppdce_price"][begin:end]
    result1 = smf.OLS(y_value, x_value).fit()
    x_value = st.add_constant(x_value)
    result2 = smf.OLS(y_value, x_value).fit()
    return [round(result1.params[0],1), round(result2.params[1],1)]


quantile1 = [0]*index2[interval]
quantile2 = [0]*index2[interval]
quantile3 = [0]*index2[interval]

for i in range(index2[interval], len(data1["date"])):
#    begin = i - (index2[data1["index"][i]]-index2[data1["index"][i]-3])
    begin = i - index2[interval]
    ratio = get_ratio(data1["index"][i])[1]
    residual = pd.DataFrame()
    residual["resi"] = data1["ppdce_price"][begin:i] - ratio*data1["ldce_price"][begin:i]
    quantile1.append(residual["resi"].quantile(0.05))
    quantile2.append(residual["resi"].quantile(0.5))
    quantile3.append(residual["resi"].quantile(0.95))


position = [0]*index2[interval]
return1 = [0]*index2[interval]
cumu_return = [1]*index2[interval]
current_value = index2[interval]
cumu_return1 = 1
num_trading = 0
index_trading = []
index_close = []
for i in range(index2[interval], len(data1["date"])):
    ratio = get_ratio(data1["index"][i])[1]
    spread = data1["ppdce_price"][i] - ratio*data1["ldce_price"][i]
    if (i+1) in close:
        po = -sum(position)
        position.append(po)
        if po == -1:
            index_close.append(i)
        elif po == 1:
            index_close.append(i)
    elif sum(position) == 1:
        if spread >= quantile2[i]:
            position.append(-1)
            index_close.append(i)
        else:
            position.append(0)
    elif sum(position) == -1:
        if spread <= quantile2[i]:
            position.append(1)
            index_close.append(i)
        else:
            position.append(0)
    else:
        if spread < quantile1[i]:
            position.append(1)
            num_trading += 1
            index_trading.append(i)
        elif spread > quantile3[i]:
            position.append(-1)
            num_trading += 1
            index_trading.append(i)
        else:
            position.append(0)

print(num_trading)
print(index_trading)
print(index_close)

def find_max_index(status,j):
    index1 = j-1
    while index1 >= 0:
        if position[index1] == status:
            return index1
        else:
            index1 -= 1

return1 = [0]*index2[interval]
return2 = [0]*index2[interval]
value = [1]*index2[interval]
value2 = [1]*index2[interval]
for i in range(index2[interval], len(data1["date"])):
    tempt_position = position[:i]
    ratio = get_ratio(data1["index"][i])[1]
    spread = data1["ppdce_price"][i] - ratio*data1["ldce_price"][i]
    if sum(tempt_position) == 0:
        return1.append(0)
        return2.append(0)
        value.append(value[i-1])
        value2.append(value2[i-1])
    elif sum(tempt_position) == 1:
        point = find_max_index(1,i)
        new_y1 = ratio*data1["ldce_price"][i]
        new_ratio = get_ratio(data1["index"][point])[1]
        new_y2 = data1["ldce_price"][point]*new_ratio
        numo = data1["ppdce_price"][i]-data1["ppdce_price"][point] + new_y2 - new_y1
        deno = new_y2 + data1["ppdce_price"][point]
        return1.append(numo/deno)
        valuei = value[point]*(1+return1[i])
        return2.append(3*numo/deno)
        value2.append(value2[point]*(1+return2[i]))
        value.append(valuei)
    else:
        point = find_max_index(-1,i)
        new_y1 = ratio*data1["ldce_price"][i]
        new_ratio = get_ratio(data1["index"][point])[1]
        new_y2 = data1["ldce_price"][point]*new_ratio
        numo = data1["ppdce_price"][i]-data1["ppdce_price"][point] + new_y2 - new_y1
        deno = new_y2 + data1["ppdce_price"][point]
        return1.append(-numo/deno)
        valuei = value[point]*(1+return1[i])
        return2.append(-3*numo/deno)
        value2.append(value2[point]*(1+return2[i]))
        value.append(valuei)         


drawdown = 0
for i in range(0, len(value)):
    d = (max(value[:i+1])-value[i])/max(value[:i+1])
    if d > drawdown:
        drawdown = d

print(drawdown)
'''
l = []
for i in range(0, len(value)):
    l.append(i)
pylab.plot(l[index2[interval]:],value[index2[interval]:])
pylab.show()

l_date = []
l_date2 = []
for element in index_trading:
    print(value[element])
    l_date.append(data1["date"][element])
for element in index_close:
    print(value[element])
    l_date2.append(data1["date"][element])

for i in range(1, len(index_trading)):
    returni = (value[index_trading[i]] - value[index_trading[i-1]])/value[index_trading[i-1]]
    print(returni)
''' 
spread1 = []    
for element in index_trading:
    new_ratio1 = get_ratio(data1["index"][element])[1]
    print(new_ratio1)
    spread1.append(data1["ppdce_price"][element]-new_ratio1*data1["ldce_price"][element])
    if data1["ppdec_volume"][element] <= data1["ldce_volume"][element]/new_ratio1:
        print(data1["ppdec_volume"][element])
    else:
        print(data1["ldce_volume"][element]/new_ratio1)

spread2 = []
for element in index_close:
    new_ratio1 = get_ratio(data1["index"][element])[1]
    spread2.append(data1["ppdce_price"][element]-new_ratio1*data1["ldce_price"][element])
    if data1["ppdec_volume"][element] <= data1["ldce_volume"][element]/new_ratio1:
        print(data1["ppdec_volume"][element])
    else:
        print(data1["ldce_volume"][element]/new_ratio1)

