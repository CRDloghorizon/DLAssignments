import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import to_categorical
import keras.layers as nn
from keras import losses, Sequential
from keras import optimizers
import matplotlib.pyplot as plt
import math
import random

def draw_digit(position, image, title):
    plt.subplot(*position)
    plt.imshow(image.reshape(-1, 28), cmap='gray_r')
    plt.axis('off')
    plt.title(title)

def draw_batch(batch_size, dit_x, dit_y):
    r_index = random.sample(range(len(dit_y)), batch_size)
    images, labels = dit_x[r_index], dit_y[r_index]
    image_num = images.shape[0]
    row = math.ceil(image_num ** 0.5)
    column = row
    plt.figure(figsize=(row,column))
    for i in range(row):
        for j in range(column):
            index = i*column+j
            if index < image_num:
                position = (row, column, index+1)
                image = images[index]
                title = "label:%d" % (labels[index])
                draw_digit(position, image, title)



if __name__ == "__main__":
    # Tuple of Numpy arrays (x_train, y_train), (x_test, y_test)
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    draw_batch(25, train_x, train_y)
    #plt.show()
    #plt.close()

    train_x = train_x.astype('float32')
    train_x /= 255
    train_x = train_x.reshape(-1, 28, 28, 1)
    train_y = to_categorical(train_y, 10)
    test_x = test_x.astype('float32')
    test_x /= 255
    test_x = test_x.reshape(-1, 28, 28, 1)
    test_y = to_categorical(test_y, 10)

    model = Sequential()
    # layer 1: conv2d cout=32,k=3,s=1,p=0,relu (26 26 32)
    model.add(nn.Conv2D(32, kernel_size=(3,3), strides=(1,1), input_shape=(28, 28, 1)))
    model.add(nn.Activation('relu'))
    # layer 2: maxpool, k=2 (13 13 32)
    model.add(nn.MaxPool2D(pool_size=(2,2)))
    # layer 3: conv2d cout=48,k=3,s=1,p=1,relu (13 13 48)
    model.add(nn.ZeroPadding2D(padding=(1,1)))
    model.add(nn.Conv2D(48, kernel_size=(3,3), strides=(1,1)))
    model.add(nn.Activation('relu'))
    # BN + maxpool (6 6 48)
    model.add(nn.BatchNormalization(epsilon=1e-6))
    model.add(nn.MaxPool2D(pool_size=(2,2)))
    # layer 4: conv2d cout=64,k=1,s=1,p=1,relu (7 7 64)
    model.add(nn.ZeroPadding2D(padding=(1, 1)))
    model.add(nn.Conv2D(64, kernel_size=(2, 2), strides=(1, 1)))
    model.add(nn.Activation('relu'))
    # BN + maxpool (3 3 64)
    model.add(nn.BatchNormalization(epsilon=1e-6))
    model.add(nn.MaxPool2D(pool_size=(2, 2)))
    # dropout + flatten
    model.add(nn.Dropout(0.25))
    model.add(nn.Flatten())
    # Fully connected layer
    model.add(nn.Dense(576, activation='relu'))
    model.add(nn.Dense(10, activation='softmax'))
    model.summary()

    opt = optimizers.Adadelta()
    # optimizer 'sgd' or pass lr/.. by new object sgd; similar for loss
    model.compile(loss=losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
    batch_size = 100
    epochs = 12
    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=0, validation_split=0.2)

    model.save('cnn_mnist_t1.pth')

    # verbose => print info type
    loss, accuracy = model.evaluate(test_x, test_y, verbose=1)
    loss_t, accuracy_t = model.evaluate(train_x, train_y, verbose=1)
    print('test loss:%.4f  accuracy:%.4f' % (loss, accuracy))
    print('train loss:%.4f  accuracy:%.4f' % (loss_t, accuracy_t))

    # 99.11%  99.66%




