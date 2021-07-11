from numpy.core.numeric import outer
import pandas as pd
import numpy as np
from pandas import DataFrame
# from pandas import
df1 = DataFrame(data=np.random.randint(4, 20, size=(5, 4)),
                columns=['A', 'B', 'C', 'D'])
df2 = DataFrame(data=np.random.randint(4, 20, size=(5, 4)),
                columns=['A', 'B', 'C', 'E'])
# print(df1)
table1 = pd.concat((df1, df2), axis=0)
table1.fillna(table1['D'].mean(), inplace=True)
# print(table1)
tab2 = df1.append(df2)
# print(tab2)
tab3 = pd.merge(df1, df2, on=['A', 'B', 'C'])
print(tab3)
