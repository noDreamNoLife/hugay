from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 微调网络，指的是可以解冻已经训练好的的网络的一部分层，让它们与自己定义的全连接层一起训练
# 一般解冻的层是最高的几层，因为这些层次更专业的特征，训练这些特征更容易得到更好的效果

## 首先先加载VGG16的模型，看一下其结构
conv_base_vgg16 = VGG16(
    weights='imagenet',  # 该参数是模型初始化的权重检查点
    include_top=False,  # 这个是表示是否包含其全连接层
    input_shape=(150, 150, 3)
)

print(conv_base_vgg16.summary())

## 使用数据增强的特征提取
## 在这个vgg16的网络基础上，加上我们的全连接层
model = models.Sequential()
model.add(conv_base_vgg16)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

conv_base_vgg16.trainable = True

set_trainable = False
for layer in conv_base_vgg16.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

# 在上述设置，完成后，就可以对数据进行数据增强，然后训练全连接层中的参数
# 使用数据增强的方法来防止过拟合
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # 这是将像素值变到0-1之间
    rotation_range=40,  # 这是图像随机旋转的角度范围
    width_shift_range=0.2,  # 这是图像在水平方向上平移的范围，这是相对于总宽度的比值
    height_shift_range=0.2,  # 这是图像在垂直方向上平移的范围，这是相对于总高度的比值
    shear_range=0.2,  # 这是随机错切变换的角度
    zoom_range=0.2,  # 这是图像随机缩放的范围
    horizontal_flip=True,  # 随机将一半的图像水平翻转
    fill_mode='nearest'  # 填充新像素的方法，这里采用最近填充

)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    directory='./Data/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
validation_generator = test_datagen.flow_from_directory(
    directory='./Data/validation',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

# 训练模型，保持模型
history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50
)

model.save('cats_and_dogs_small_4.h5')

# 绘制图像，观察损失与准确率
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='train_acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

## 让曲线看起来更加平滑
def smooth_curve(points, factor=0.8):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

plt.plot(epochs,
         smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs,
         smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,
         smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs,
         smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()