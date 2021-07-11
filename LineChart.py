import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
a = range(1, 16)
mean = 5
sigma = 10
# %matplotlib inline
b = np.random.normal(mean, sigma, 15).astype(int)
# # plt.plot(a, b)
# # plt.show()
# plt.plot(a, b, color='Red')
# plt.show()
one = [34, 45, 33, 45, 49]
two = [12, 13, 6, 14, 10]
three = [67, 78, 90, 75, 85]
sales = DataFrame({'one': one, 'two': two, 'three': three})
# sales.plot(xticks=range(1, 5), yticks=range(0, 100, 20))
# plt.show()
# plt.bar(a, b)
# sales.plot(kind='bar')
prescribe = [3, 4, 5, 7, 12]  # 指定
label = ['A', 'B', 'C', 'D', 'E']
# plt.pie(a, labels=['AA', 'BB', 'CC', 'DD', 'EE'])
# sales.plot(kind='pie')
# plt.pie(prescribe, labels=label)
depict_data = np.random.normal(mean, sigma, 500)
depict_data1 = np.sort(np.random.normal(20, 5, 50).astype(int))
portray_data2 = np.sort(np.random.normal(20, 5, 50).astype(int))
# depicktt_data2= np.random.normal(20, 5, 50).astype(int)
# plt.hist(depict_data)
plt.scatter(depict_data1, portray_data2)

plt.show()
