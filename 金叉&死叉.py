from matplotlib import colors
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.core.algorithms import mode
import tushare as ts
import csv

import matplotlib.pyplot as plt
# data = ts.get_k_data('000001', start='1900-01-01')
# with open('平安股票', mode='w', encoding='utf-8') as f:
#     f.write(data)

# data.to_csv('平安股票.csv')
df = pd.read_csv('平安股票.csv')
df.drop(labels='Unnamed: 0', inplace=True, axis=1)
df['date'] = pd.to_datetime(df['date'])

df.set_index('date', inplace=True)
# print(df.info())
Ex5 = df['close'].rolling(5).mean()
Ex30 = df['close'].rolling(30).mean()
pa = DataFrame({'M5': Ex5[40:300], 'M30': Ex30[40:300]})
# pa.plot()
# plt.plot(Ex5[50:300], label='M5')
# plt.plot(Ex30[50:300], label='M30')
# plt.legend()
# plt.show()
s1 = Ex5[30:] < Ex30[30:]
# print(s1)
s2 = Ex5[30:] >= Ex30[30:]
df_ = df[30:]
dead_date = df_.loc[s1 & s2.shift(1)].index
# print(dead_date)
gold_date = df_.loc[~(s1 | (s2.shift(1)))].index
# print(gold_date)
sr1 = Series(data=1, index=gold_date)
sr2 = Series(data=0, index=dead_date)
# print(sr2)
sr = sr1.append(sr2)
sr.sort_index(inplace=True)
# print(sr)
new_sr = sr['2010':'2020']

money = 100000
pre_price = money
sum = 0
# print(df.index[0])
# for i in range(len(new_sr)):
#     if(sr[i] == 1):
#         hand = pre_price//(df.loc[new_sr.index[i]]['open']*100)
#         sum = hand*100
#         pre_price -= sum*df.loc[new_sr.index[i]]['open']
#     else:
#         pre_price += sum*df.loc[df.index[i]]['open']
#         sum = 0
# lastmoney = df['open'][-1]*sum

# award = lastmoney+pre_price-money
# print(award)
plt.scatter(dead_date[10:50], gold_date[10:50], colors=['red', 'green'])
plt.show()
