'''author:zjk'''

#导入模块
import pandas as pd
from math import e  # 引入自然数e
import numpy as np  # 科学计算库
from scipy.optimize import leastsq  # 引入最小二乘法算法
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

#nt的求解
def func(params, t):
    m, p, q = params
    fz = (p * (p + q) ** 2) * e ** (-(p + q) * t)  # 分子的计算
    fm = (p + q * e ** (-(p + q) * t)) ** 2  # 分母的计算
    nt = m * fz / fm  # nt值
    return nt

# 误差函数函数
def error(params, t, y):
    return func(params, t) - y

df = pd.read_csv('zbag.csv')
df = df[df.columns[4:24]]
xi = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20])
It = []
for i in range(0,193):
    yi = df.iloc[i,:]
    yi = yi.values
    p0 = [100, 0.3, 0.3]
    params = leastsq(error, p0, args=(xi, yi))     # 把error函数中除了p0以外的参数打包到args中
    params = params[0]
    m, p, q = params     # 读取结果
    y_hat = []
    for t in xi:
        y = func(params, t)
        y_hat.append(y)

    #绘图 只显示出前十张
    picture_name = 'Bass model' +  str(i)
    fig = plt.figure()
    plt.plot(yi, color='r', label='true')
    plt.plot(y_hat, color='b', label='predict')
    plt.title(picture_name)
    plt.legend()
    if i <= 10:
        plt.show()

    #归一化
    sum = 0
    for hat in y_hat:
        sum = sum + hat
    y_hat = y_hat/sum     #归一化之后的结果
    y_hat = list(y_hat)
    It.append(y_hat)

df2 = pd.DataFrame(It)

#聚类算法实现
columns_name = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10','11','12','13','14','15','16','17','18','19','20']
columns_name = pd.Series(columns_name) #将其转化为数组类型
df2 = df2.rename(columns=columns_name)

X = df2[columns_name] #X是标签
apriori = int(input('输入要聚类的个数：'))
km = KMeans(n_clusters=apriori).fit(X)
print(km.labels_) #分类的结果

df2['cluster'] = km.labels_
#print(df2.sort_values('cluster'))
#可以按照cluster切分，然后利用subplot绘图

#每个类别的均值
df_mean = df2.groupby('cluster').mean()
df_mean = df_mean.T
df_mean.plot()
plt.show()