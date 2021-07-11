import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn
# ----------------创建数据-------------------------
mean = 25
sigma = 10
infer_data = np.random.normal(mean, sigma, 500).astype(int)
inferiority_data = np.random.normal(mean+5, sigma-4, 500).astype(int)
inference_data = np.random.normal(mean-4, sigma+3, 500).astype(int)
justice_data = DataFrame(
    {"A": infer_data, "B": inferiority_data, "C": inference_data})

data = ['Delhi', 'Pune', 'Ajmer']
# print(justice_data)
justice_data['city'] = np.random.choice(data, 500, p=[0.5, 0.2, 0.3]).tolist()
print(justice_data)

justice_data['gender'] = np.random.choice(
    ['Male', 'Female'], 500, p=[0.6, 0.4]).tolist()

# -------------------------绘图---------------------------------------------------------------------
# seaborn.displot(infer_data, bins=10,)
seaborn.jointplot(inferiority_data, inference_data, kind='kde')
plt.show()
