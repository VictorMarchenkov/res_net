__author__ = 'victor'
__created__ = '12.02.2018'

from keras.layers import Input, Conv1D, Conv2D, BatchNormalization, add
from keras.models import Model
from keras.callbacks import TensorBoard
from keras import backend as K
import os
import numpy as np
import matplotlib.pyplot as plt
import random

ROW = 256
COL = 256
filter = (3, 3)
im_shape = (ROW, COL, 1)
n_epoch = 5
batch = 10

FOLDER = "C:\\Users\\victor.SMISLAB\\Desktop\\train_all\\"
list_files = os.listdir(FOLDER)
ARRAY_LENGTH = len(list_files)


data = np.memmap('./data.npy', dtype='float32', mode='r', shape=(ARRAY_LENGTH, ROW, COL))
print(len(data))
plt.figure(figsize=(8, 6))
# n = 3
# for i in range(n):
#     ax = plt.subplot(2, 3, i+1)  # using i+1 since 0 is deprecated in future matplotlib
#     plt.imshow(random.choice(data), cmap=plt.cm.gray)
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

DIV = int(0.8*len(data))



data_noisy = np.memmap('./data_noisy.npy', dtype='float32', mode='r', shape=(ARRAY_LENGTH, ROW, COL))

# plt.figure(figsize=(8, 6))
# n = 3
# for i in range(n):
#     ax = plt.subplot(2, 3, i+1)  # using i+1 since 0 is deprecated in future matplotlib
#     plt.imshow(random.choice(data_noisy), cmap=plt.cm.gray)
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()


data = data.reshape(len(data), COL, ROW, 1)
data_noisy = data_noisy.reshape(len(data_noisy), COL, ROW, 1)

x_pos_noisy = data_noisy[:DIV]
x_noisy_test = data_noisy[DIV:]

x_pos_train = data[:DIV]
x_pos_test = data[DIV:]



def residual_block():
    input_img = Input(shape=im_shape)
    y = Conv2D(64, filter, dilation_rate=(1, 1), padding='same')(input_img)

    shortcut1 = input_img
    y = Conv2D(64, filter, activation='relu', dilation_rate=(2, 2), padding='same')(y)
    y = Conv2D(64, filter, activation='relu', dilation_rate=(3, 3), padding='same')(y)
    y = add([shortcut1, y])

    y = Conv2D(64, filter, activation='relu', dilation_rate=(4, 4), padding='same')(y)

    shortcut2 = y
    y = Conv2D(64, filter,activation='relu', dilation_rate=(3, 3), padding='same')(y)
    y = Conv2D(64, filter,activation='relu', dilation_rate=(2, 2), padding='same')(y)
    y = add([shortcut2, y])

    y = Conv2D(1, filter, activation='relu', dilation_rate=(1, 1), padding='same')(y)

    # identity shortcuts used directly when the input and output are of the same dimensions


    denoiser = Model(input_img, y)
    print(denoiser.summary())
    denoiser.compile(optimizer='adam', loss='mean_squared_error')
    denoiser.fit(x_pos_noisy, x_pos_train,
                    epochs=n_epoch,
                    batch_size=batch,
                    shuffle=True,
                    validation_data=(x_noisy_test, x_pos_test),
                    callbacks=[TensorBoard(log_dir='./tmp/tb', histogram_freq=0, write_graph=True, write_images=True)]
                    )
    denoiser.save('{}_{}_autoencoder.h5'.format(n_epoch, batch))
residual_block()

