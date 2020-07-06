from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

## 在一个小的训练集上，从头开始训练一个模型
# 这次处理的问题是一个猫狗图像二分类的问题，使用一个很简单的卷积神经网络进行测试

# 定义模型
model = models.Sequential()
model.add(layers.Conv2D(32, kernel_size=(3, 3),
                        activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])

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

# 训练模型，并保存模型
history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50
)

model.save('cats_and_dogs_small_2.h5')

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
