import keras
import numpy as np
from keras.datasets import boston_housing
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt

# 回归问题，预测房价的中位数，其中有404个训练数据，102个测试用例

# 1. 加载数据
(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)

# 2. 数据处理，因为输入数据的特征取值范围不同，存在一些取值范围时0-1，一些事0-100，所以需要对数据进行标准化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data/= std

# 这里对测试数据也进行标准版，但是只能使用测试数据的均值和标准差，不能再测试数据上运算得到任何结果
test_data-=mean
test_data/=std

# 3. 构建模型,这里因为要将模型多次实例化，所以定义了一个函数
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model

# 4. 因为数据集较少，使用k折交叉验证的方法，来验证方法
# 首先定义要分几折，这里分的是4折
k=4
num_val_samples = len(train_data)//4
num_epoch = 500
all_scores = []
all_mae_histories = []

for i in range(k):
    # 首先准备验证数据，即第k个分区的数据
    print('proposing fold # ',i)
    val_data = train_data[num_val_samples*i:num_val_samples*(i+1)]
    val_targets = train_targets[num_val_samples*i:num_val_samples*(i+1)]

    # 然后准备剩余k-1个分区的训练数据
    partial_train_data = np.concatenate(
        [train_data[:num_val_samples*i],
         train_data[num_val_samples*(i+1):]],
        axis=0
    )
    partial_train_targets = np.concatenate(
        [train_targets[:num_val_samples*i],
         train_targets[num_val_samples*(i+1):]],
        axis=0
    )

    # 最后训练模型,这里的verbose设置为0，是控制日志的输出，0为不输出，静默模式
    model = build_model()
    history = model.fit(partial_train_data,partial_train_targets,
                            validation_data=(val_data,val_targets),
                            batch_size=1,epochs=num_epoch,verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
    average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epoch)]

#绘制图像
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# 5.最后测试训练模型
model = build_model()
model.fit(train_data,train_targets,batch_size=16,epochs=80,verbose=0)
test_mse,test_mae = model.evaluate(test_data,test_targets)
print(test_mae)