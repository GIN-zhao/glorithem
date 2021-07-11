import numpy as np
from numpy.core.arrayprint import printoptions
from numpy.core.fromnumeric import ravel, size

# a = np.arange(10)
# print(a)
# print(a.dtype)


# class Person:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age


# b = np.array([Person('lisi', 23), Person('zhaowu', 24)])
# print(b)
# print(type(b[0].name))
# print(b.size)
# print(b.itemsize)
# a2 = np.random.randint(0, 10, size=(4, 6))
# print(a2)
# print(a2[1:3])
# print(a2[[0, 2, 3]])
# print(a2[0, 1])
# print(a2[[0, 1], [1, 2]])
# print(a2[1:3, 2:4])
# print(a2[:, 1:3])

temp = np.arange(24).reshape((4, 6))
# print(temp < 10)
# print(temp[temp < 10])
# temp[(temp > 4) & (temp < 10)] = 1
# print(temp)
# ret = np.where(temp < 11, 0, 1)
# print(ret)
# a2 = temp*1.628
# print(a2.round(2))
# temp2 = np.random.randint(-2, 8, size=(4, 6))
# temp2 = temp2*2.123
# a3 = temp2+temp
# print(a3.round(2))
# a1=np.random.randint((4,))
# a1 = np.arange(10)
# print(a1)
# a2 = a1.view()
# print(a2)
# a2[5] = 10
# print(a1)

# a3 = a1.copy()
# print(a3)
# a3[5] = 24234
# print(a1)
# ravel
# flatten
# a1=np.arange(10)
# a1 = np.random.uniform(-10, 10, size=(3, 5))
# print(np.abs(a1))
# print(np.sqrt(abs(a1)))
# a = np.random.randint(1, 6, size=(3, 3, 3))
# print(a2)
# print(a2[1, 1][1])
# print(a2[1, 1, 1])
# a = np.arange(1, 10).reshape(3, 3)
# a = np.random.randint(2, 10, size=(3, 3))
# print(a)
# test = np.sum(a, axis=1)
# print(test)
# test = np.prod(a, axis=0)
# test = np.mean(a, axis=0)
# print(a)
# print(test)
# test = np.min(a, axis=0)
# test = np.max(a, axis=0)
# test = np.argmin(a, axis=1)
# test = np.median(a, axis=1)
# print(test)
# a2 = a[a < 5]
# a2 = a < 5
# print(a2)
# test = np.any(a2, axis=1)
# test2 = np.all(a2, axis=1)
# test = np.sort(a, axis=0)
# test = np.argsort(a, axis=0)
# test = -np.sort(-a, axis=1)
# a = np.sort(a, axis=1)
# print(a)
# index = np.argsort(-a, axis=1)
# test = np.take(a, index, axis=1)
# test = np.linspace(0, 1, 10)
# print(test.round(2))
a = np.random.randint(2, 10, size=(10, 5))
# test = np.apply_along_axis(
#     lambda x: x[(x != x.max()) & (x != x.min())].mean(), axis=1, arr=a)
# print(a)
# test = np.unique(a, return_counts=True)
# print(test)
