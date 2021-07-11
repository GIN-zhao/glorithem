import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.core.algorithms import mode
from pandas.core.indexes.datetimes import DatetimeIndex
import tushare as ts
import csv
# data = ts.get_k_data(code='600519', start='1900-01-01')

# with open('茅台股票.txt', mode='w', encoding='utf-8') as fs:

#     fs.write(data)
# print(data)
# data.to_csv('茅台股票.csv')
df = pd.read_csv('../茅台股票.csv')
df = df.drop(labels='Unnamed: 0', axis=1)
df['date'] = pd.to_datetime(df['date'])
# print(df['date'])
date_time = DatetimeIndex(df['date'])
print(date_time.weekday)
print('-'*30)


df.set_index('date', inplace=True)
# df.drop(['date'])
# print(df['date'].dtype)
judge = (df['close']-df['open'])/df['open'] > 0.03
judge2 = (df['close']-df['close'].shift(1))/df['close'].shift(1) < -0.02
# print(judge.head().loc[1])
# print(judge.dtype)
temp = df.loc[judge2].values.astype(int)
print(temp)
new_df = df['2010':'2020']
test_month = new_df.resample('M')
# print('test_month=', test_month)

df_month = new_df.resample('M').first()
df_year = new_df.resample('A').last()
pre_sum = df_month['close'].sum()*100
post_sum = df_year['close'].sum()*100*12
print(post_sum-pre_sum)
# print(df_month)
