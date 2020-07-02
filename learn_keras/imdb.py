import keras
import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt

# 处理的数据集一共是50000条，分别有25000条评论用于训练与测试

# 1. 导入数据,其中参数num_words表示保留了10000个高频词汇，低频的词汇就不保留了
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(train_data[0])
print(train_labels[0])


# 2. 将整数序列编码为二进制矩阵
def vectorize_sequences(sequences, dimension=10000):
    # （创建一个形状为 (len(sequences), dimension) 的零矩阵）
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # （将 results[i] 的指定索引设为 1）
    return results


# （将训练数据向量化）
x_train = vectorize_sequences(train_data)
# （将测试数据向量化）
x_test = vectorize_sequences(test_data)

print(x_train[0])

# 将标签也向量化
y_train = np.asarray(train_labels.astype('float32'))
y_test = np.asarray(test_labels.astype('float32'))

# 3. 构建网络
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))


# 4. 编译模型
model.compile(
    optimizer=optimizers.RMSprop(lr=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 5. 留出验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 6.训练模型,这里会返回一个
history =model.fit(
    partial_x_train,
    partial_y_train,
    batch_size=512,
    epochs=20,
    validation_data=(x_val,y_val)
)


# 7. 使用matplotlib画出损失和准确率
history_dict = history.history
loss_value = history_dict['loss']
val_loss_value = history_dict['val_loss']

epochs = range(1,len(loss_value)+1)
plt.plot(epochs,loss_value,'bo',label = 'train_loss')
plt.plot(epochs,val_loss_value,'b',label = 'val_loss')
plt.title('train and validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()