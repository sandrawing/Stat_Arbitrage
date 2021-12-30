# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 13:13:59 2018

@author: xinzeng
"""

import pandas as pd
import pylab
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import statsmodels.tsa.stattools as ts
import statsmodels.tools as st

data1 = pd.read_csv("C:\\Users\\xinzeng\\Desktop\\Internship\\data_new3.csv")

def get_date_index(data,date):
    for i in range(0, len(data["date"])):
        if date in data["date"][i]:
            return i

index1 = get_date_index(data1,"2018-04-09")
print(index1)

begin = 384
end = int((len(data1["date"])-begin)/2+begin)
x_value = data1["pp_price"][begin:end]
y_value = data1["INE_price"][begin:end]
result1 = smf.OLS(y_value, x_value).fit()
x_value = st.add_constant(x_value)
result2 = smf.OLS(y_value, x_value).fit()
ratio1 = round(result1.params[0],1)
ratio2 = round(result2.params[1],1)

def get_quantile(data, index, quant):
    week_data = data["price spread"][begin:index]
    quantile1 = week_data.quantile(quant)
    return quantile1

spread = []
for i in range(0, len(data1["date"])):
    spread.append(data1["INE_price"][i]-data1["pp_price"][i]*ratio1)
data1["price spread"] = spread

index3 = end
position = [0]*index3
return1 = [0]*index3
cumu_return = [1]*index3
current_value = [0]*index3
cumu_return1 = 1
num_trading = 0
index_trading = []
index_close = []
for i in range(end, len(data1["date"])):
    if sum(position) == 1:
        if data1["price spread"][i] >= get_quantile(data1, i, 0.5):
            position.append(-1)
            index_close.append(i)
        else:
            position.append(0)
    elif sum(position) == -1:
        if data1["price spread"][i] <= get_quantile(data1, i, 0.5):
            position.append(1)
            index_close.append(i)
        else:
            position.append(0)
    else:
        if data1["price spread"][i] < get_quantile(data1, i, 0.05):
            position.append(1)
            num_trading += 1
            index_trading.append(i)
        elif data1["price spread"][i] > get_quantile(data1, i, 0.95):
            position.append(-1)
            num_trading += 1
            index_trading.append(i)
        else:
            position.append(0)

#print(num_trading)
#print(index_trading)
#print(index_close)

def find_max_index(status,j):
    index1 = j-1
    while index1 >= 0:
        if position[index1] == status:
            return index1
        else:
            index1 -= 1

return1 = [0]*index3
return2 = [0]*index3
value = [1]*index3
value2 = [1]*index3
for i in range(index3, len(data1["date"])):
    tempt_position = position[:i]
    if sum(tempt_position) == 0:
        return1.append(0)
        return2.append(0)
        value.append(value[i-1])
        value2.append(value2[i-1])
    elif sum(tempt_position) == 1:
        point = find_max_index(1,i)
        new_y1 = data1["pp_price"][i]*ratio1
        new_y2 = data1["pp_price"][point]*ratio1
        numo = data1["INE_price"][i]-data1["INE_price"][point] + new_y2 - new_y1
        deno = new_y2 + data1["INE_price"][point]
        return1.append(numo/deno)
        valuei = value[point]*(1+return1[i])
        return2.append(3*numo/deno)
        value2.append(value2[point]*(1+return2[i]))
        value.append(valuei)
    else:
        point = find_max_index(-1,i)
        new_y1 = data1["pp_price"][i]*ratio1
        new_y2 = data1["pp_price"][point]*ratio1
        numo = data1["INE_price"][i]-data1["INE_price"][point] + new_y2 - new_y1
        deno = new_y2 + data1["INE_price"][point]
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



l = []
for i in range(0, len(value)):
    l.append(i)
pylab.plot(l[index3:],value[index3:])
pylab.show()
pylab.plot(l[index3:],value2[index3:])
pylab.show()

l_date = []
l_date2 = []
for element in index_trading:
#    print(value[element])
    l_date.append(data1["date"][element])
#    print(data1["date"][element])
    print(data1["price spread"][element])
for element in index_close:
    print(value[element])
    l_date2.append(data1["date"][element])
#    print(data1["date"][element])
#    print(data1["price spread"][element])
for i in range(1, len(index_trading)):
    returni = (value[index_trading[i]] - value[index_trading[i-1]])/value[index_trading[i-1]]
    print(returni)

#print(l_date)
#print(l_date2)

#for element in index_trading:
#    if data1["INE_volume"][element] <= data1["pp_volume"][element]/ratio1:
#        print(data1["INE_volume"][element])
#    else:
#        print(data1["pp_volume"][element]/ratio1)
#
#for element in index_close:
#    if data1["INE_volume"][element] <= data1["pp_volume"][element]/ratio1:
#        print(data1["INE_volume"][element])
#    else:
#        print(data1["pp_volume"][element]/ratio1)

l = []
for i in range(0, len(value)):
    l.append(i)
pylab.plot(l[index3:],value[index3:])
pylab.show()
pylab.plot(l[index3:],data1["price spread"][index3:])
pylab.show()
pylab.plot(l[begin:index3],data1["price spread"][begin:index3])
pylab.show()
