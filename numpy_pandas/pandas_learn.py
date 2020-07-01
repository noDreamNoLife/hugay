import pandas as pd
from pandas import Series, DataFrame
import numpy as np

# Series
obj = Series([1, 2, 3, 4, 5])
print(obj)
print(obj.index)
print(obj.values)

# 自定义索引
obj = Series([1, 2, 3, 4, 5], index=(1, 2, 3, 4, 5))
print(obj)

# series还可以看成字典的形式
data = {'a': 1, 'b': 2, 'c': 3}
obj = Series(data)
print(obj)
keys = ['a','c']
obj_1 = Series(data,index=keys)
print(obj_1)

# series中简单的缺失值判断与处理
data = {'a':None,'b': 2, 'c': 3}
obj = Series(data)
print(pd.isnull(obj))
print((obj.isnull()))

# 自定义索引名
data = {'xiaohu':None,'xiaodai': 2, 'xiaowang': 3}
obj = Series(data)
obj.name = 'xingminghemingzi'
obj.index.name = 'xingming'
print(obj)


# DataFrame
data = {
    '篮球':['詹姆斯','杜兰特'],
    '足球':['C罗','梅西'],
    '乒乓球':['马龙','张继科'],
}
df = DataFrame(data)
print(df)
print(data['篮球'])

dates = pd.date_range('20200627',periods=5)
print(dates)
df = DataFrame(np.random.rand(5,3),index=dates,columns=['A','B','C'])
print(df)
print(df.loc['2020-06-27':'2020-06-28',['A','B']])
print(df.at['2020-06-29','A'])
print(df.head(2))
print(df.tail(2))


# pandas中的重新索引reindex
obj = Series([1.1, 2.2, 3.3,], index=['a','b','c'])
print(obj)
obj_reindex = obj.reindex(['a','b','c','d','e'])
print(obj_reindex)
obj_2 = obj.reindex(['a','b','c','d','e'],fill_value=1.0)
print(obj_2)
obj_3 = obj.reindex(['a','b','c','d','e'],method='ffill') # 前向填充，bfill为反向填充
print(obj_3)

# 算数运算与数据对齐
s1 = Series([1.1,2.2,3.3],index=['a','b','c'])
s2 = Series([-1.1,-2.2,-3.0,4.4],index=['a','b','c','d'])
s3 = s1+s2
print(s3)

d1 = DataFrame(np.arange(9).reshape((3,3)),index=[1,2,3],columns=list('abc'))
d2 = DataFrame(np.arange(12).reshape((4,3)),index=[1,2,3,4],columns=list('cde'))
d3 = d1+d2
print(d3)
d3 = d1.add(d2,fill_value=0)
print(d3)


# Dataframe与series之间的运算与排序
df1 = DataFrame(np.arange(12).reshape((4,3)),columns=list('abc'),index=[1,2,3,4])
s1 = Series(df1.loc[1])
print(df1)
print(s1)
dele = df1-s1 #广播相减
print(dele)

s2 = Series(np.arange(3),index=['c','d','e'])
add1 = df1+s2 # 不同索引会合并
print(add1)

s2 = Series([3,1,2],index=['c','d','e'])
s1 = s2.sort_values()
print(s1)
s1 = s2.sort_index()
print(s1)

df1 = DataFrame(np.arange(8).reshape((2,4)),columns=['d','b','a','c'],index=[2,1])
df2 = df1.sort_index()
print(df2)
df2 = df1.sort_index(axis=1)
print(df2)
df2 = df1.sort_values(by='d')
print(df2)