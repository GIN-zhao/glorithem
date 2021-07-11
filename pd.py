from numpy.core.arrayprint import printoptions
from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from pandas.io.formats.format import DataFrameFormatter
import tushare as ts
from sklearn import preprocessing as ps
from sklearn.feature_extraction.text import CountVectorizer as cvz
from sklearn.feature_extraction.text import TfidfVectorizer as Tvz
from sklearn.preprocessing import StandardScaler
# # s1 = pd.Series([1, 2, 3])
# s1 = pd.Series(np.random.randint(80, 100, 5), index=[
#                'one', 'two', 'thre', 'four', 'five'])
# data = {
#     'one': {'balana': 1.42, 'apple': 1.12},
#     'two': {'balana': 5.42},
# }
# s1 = pd.Series(data)
# print(s1.index)
# print(s1.values)
# s1.name = 'grade'
# s1.index.name = 'fruit'
# s1 = pd.DataFrame(data, index=['one', 'two'])
# s1 = pd.DataFrame(data)

# s1=

# print(s1)
# df1 = pd.DataFrame(np.arange(12).reshape(4, 3), index=[
#                    'a', 'b', 'c', 'd'], columns=list("ABC"))
# df2 = pd.DataFrame(np.arange(9).reshape(3, 3), index=[
#                    'a', 'd', 'f'], columns=list("ABD"))
# print(df1)
# print(df2)
# print(df1.loc('a').max())
# a = np.arange(10)
# print(a.sum())
# ps1 = pd.Series(np.random.randn(5, 4))
# print(ps1)
# df3 = df1+df2
# print(df3)
# s = Series([1, 2, 3])
# s2 = Series(np.arange(5))
# s3 = Series(data)
# print(s3.one['balana'])
# index = ['maths', 'Chinese', 'English']
# cls = ['groce']
# df = DataFrame(data=np.random.randint(
#     1, 100, size=(3, 1)), index=index, columns=cls)
# # 'love','this','maths','Chinese','English'
# print(df)
# print(df.columns)
# print(df.index)
# print(df.shape)
# cls = ['A', 'B', 'C', 'D']

# index = ['a', 'b', 'c', 'd', 'e']
# df = DataFrame(data=np.random.randint(
#     1, 100, size=(5, 4)), index=index, columns=cls)
# # print(df)
# print(df['A'])
# # print(df[['A', 'B']])
# print(df.loc['a'])
# df['A':'C']
# print(df.loc['a':'c', 'A':'C'])

# data = [['LEDIV', '32990'], ['Printer', '5990'],
#         ['Split AC', '32050'], ['Micoware', '12670']]

# temp = DataFrame(data=data, columns=['产品名称', '费用'])
# print(temp)
# print(temp.loc[1:3, '产品名称'])
# tmp2 = DataFrame(np.random.randint(0, 0, size=(3, 3)))
# data = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
# tmp2 = DataFrame(data=data)
# print(tmp2.replace(0, 1))
# print(tmp2)
# ---------------------------------------------------------------------
data = {"price": [234, 236, 346, 653, 980, 325]}

price_frame = DataFrame(data)
t1 = ps.scale(price_frame)
s_data1 = DataFrame(t1, columns=['price'])


data2 = DataFrame(data)
transfer = StandardScaler()
t2 = transfer.fit_transform(data2)
s_data2 = DataFrame(t2, columns=['price'])


print(s_data1)
print('-'*50)
print(s_data2)
# --------------------------------------------------------------------
# cmts = ['nice product', 'good time',
#         'you belong to me', 'love', 'love story']
# count_vect = cvz()
# x_train_counts = count_vect.fit_transform(cmts)

# word_frame = DataFrame(x_train_counts.toarray())
# print(word_frame)

# word_dict = dict((v, k) for k, v in count_vect.vocabulary_.items())
# # word_list = count_vect.get_feature_names()
# # print(type(word_list))
# # print(type(word_dict))
# print('------------------------------------')
# # print(word_dict)
# print('-------------------------------------')
# word_frame = word_frame.rename(columns=word_dict)
# print(word_frame['love'].sum())
# print(word_frame.columns.size)
# print(word_frame)
# ----------------------------------------------------------------------------
# tvz = Tvz()
# test = tvz.fit_transform(cmts)
# # print(tvz)
# print('-'*30)
# word_frame = DataFrame(test.toarray())
# print(word_frame)
# print('-'*30)
# word_dict = dict((v, k) for k, v in tvz.vocabulary_.items())
# word_frame = word_frame.rename(columns=word_dict)
# print(word_frame)
