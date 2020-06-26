import numpy as np

# numpy其实就是一个多维的数组对象
# 创建一个numpy的数组
data = [1, 2, 3, 4, 5]
n = np.array(data * 10)
print(data)
print(n)

# 每一个numpy的数组都有两个比较常用的属性，分别是shape和dtype
print(n.shape)  # 获取numpy数组的的维度
print(n.dtype)  # 获取到数组的类型

# 嵌套序列 这是由一组等长的列表组成的列表
arr = [[1, 2, 3, 4], [1, 2, 3, 4]]
arr2 = np.array(arr)
print(arr2)
print(arr2.ndim)  # 打印出当前数组的维度
print(arr2.shape)

# numpy对数据类型的判断

arr = [['1', '2', 3, 4], [5, 6, 7, 8]]
arr2 = np.array(arr)
print(arr2)
print(arr2.dtype)  # unicode类型

arr = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(arr)
print(arr2)
print(arr2.dtype)  # int类型

arr = [[1.1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(arr)
print(arr2)
print(arr2.dtype)  # 当成员中有一个类型为float类型时，numpy对数据类型的判断会推断成float类型

# numpy对指定长度的数组进行创建
arr = np.zeros(10)  # 创建长度为10的全为0的一个数组
print(arr)

arr = np.ones((2, 3))  # 创建一个全为1的一个数组
print(arr)

arr = np.arange(10)  # arrange相当于range的数组版本
print(arr)

# 进行数据类型的转换
arr = np.array([1.1, 1.5, 1.8, -2.5, -2.8])
print(arr)
print(arr.dtype)
arr2 = arr.astype(np.int32)  # astype会将其转换为指定要转换的数据类型
print(arr2)
print(arr2.dtype)

# 当数组不用在循环的条件下一个个的计算，就可以通过矢量化的运算进行批量的计算
arr1 = np.array([[1, 2, 3, 4, ], [5, 6, 7, 8]])
arr2 = np.array([[5, 6, 7, 8, ], [9, 6, 7, 8]])
print(arr1 + arr2)  # 加减乘除都可以进行类似的计算

# numpy中的索引和切片与比较操作
arr = np.arange(10)
print(arr[1])
print(arr[4:])
arr[0:4] = 11
print(arr)

# 二维的索引
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(arr[0, 1])

names = np.array(['xiaoming', 'xiaohong', 'xiaohu'])
print(names == 'xiaoming')
print((names == 'xiaoming') & (names == 'xiaohu'))  # 进行比较操作，与操作
print((names == 'xiaoming') | (names == 'xiaohu'))  # 进行比较操作，或操作

# 花式索引
arr = np.arange(32).reshape((8, 4))
print(arr)
print(arr[[1, 3, 5, 7]])
print(arr[[1, 3, 5, 7], [0, 3, 2, 1]])

# 数组的转置和轴对换
arr = np.arange(15).reshape((3, 5))
print(arr)
print(arr.transpose())
print(arr.T)

# 条件逻辑转数组，np.where 的使用
x_arr = np.array([1.1, 1.2, 1.3])
y_arr = np.array([2.1, 2.2, 2.3])
condition = np.array([True, False, True])
print(np.where(condition, x_arr, y_arr))

arr = np.random.randn(4, 4)
print(arr)
arr_1 = np.where(arr > 0, 2, -2)
print(arr_1)
arr_2 = np.where(arr > 0, 2, arr)
print(arr_2)

# numpy中的数学运算与排序
arr = np.random.randn(4, 4)
print(arr)
print(np.mean(arr))
print(np.sum(arr))
print(np.std(arr))
# 在一个维度上进行运算
print(arr.mean(axis=1))

arr = np.random.randn(4)
print(arr)
arr.sort()
print(arr)

# numpy的文件操作
arr = np.arange(10)
np.save('save_file_name', arr)
print(np.load('save_file_name.npy'))
# 保存为压缩格式文件
arr = np.arange(10)
np.savez('save_file_name_zip', a=arr)
print(np.load('save_file_name_zip.npz')['a'])

arr = np.arange(10)
np.savetxt('save_file_name_txt', arr, delimiter=',')  # 保存成文本文件，以逗号为分隔符保存
print(np.loadtxt('save_file_name_txt.txt'),delimiter=',')

# 线性代数
x= np.array([[1,2,3],[4,5,6]])
y = np.array([[1,2],[3,4],[5,6]])
print(x.dot(y))