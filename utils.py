import keras
import keras.backend as K
from keras.datasets import mnist
from keras.models import load_model
import numpy as np
import scipy.io as sio
# from pathlib2 import Path
from collections import namedtuple


def get_data(activation, isTrain, fake_size):
    # Returns two namedtuples, with MNIST training and testing data
    #   trn.X is training data
    #   trn.y is trainiing class, with numbers from 0 to 9
    #   trn.Y is training class, but coded as a 10-dim vector with one entry set to 1
    # similarly for tst
    # (X_train, _), (X_test, _) = mnist.load_data(path="./datasets/mnist.npz")
    (X_train, _), (X_test, _) = mnist.load_data(path="/cluster/home/it_stu150/lzc/IB_GAN_ZC/datasets/mnist.npz")
    # (X_train, _), (X_test, _) = mnist.load_data(path="C:\\Users\\leezh\\Desktop\\L_ZhiCheng\\上海交通大学\\屠老师实验室\\Codes\\IB_GAN_ZC\\datasets\\mnist.npz")
    #sample size, test_set=10000, train_set=60000, 60000 too large for memory

    X_train = np.reshape(X_train,[-1,28*28*1])
    X_test = np.reshape(X_test,[-1,28*28*1])

    X_train = X_train / 127.5 - 1.
    X_test = X_test / 127.5 - 1.

    y_train = np.ones((X_train.shape[0],1)).astype(int)

    Y_train = keras.utils.np_utils.to_categorical(y_train, 2)

    Dataset = namedtuple('Dataset',['X','y','Y'])
    trn = Dataset(X_train, y_train, Y_train)

    if not isTrain: # means not training phase, modified the test set for MI calculations
        gen_imgs = fake_datasets(activation, fake_size)
        fake_y = np.zeros((fake_size,)).astype(int)
        y_test = np.ones((X_test.shape[0]-fake_size,)).astype(int)
        fake_real_tstX = np.concatenate([X_test[:X_test.shape[0]-fake_size], gen_imgs], axis=0)
        fake_real_tsty = np.concatenate([y_test, fake_y], axis=0)
        tst = Dataset(fake_real_tstX , fake_real_tsty, _)

    else:
        tst = Dataset(X_test , _, _)

    del X_train, y_train, Y_train, X_test

    return trn,tst

def fake_datasets(activation, fake_size):

    # generator = load_model("/cluster/home/it_stu150/lzc/IB_GAN_ZC/models/generator_"+ self.activation + ".h5")
    generator = load_model("models/generator_"+ activation + ".h5")
    # generator = load_model("C:\\Users\\leezh\\Desktop\\L_ZhiCheng\\上海交通大学\\屠老师实验室\\Codes\\IB_GAN_ZC\\models\\generator.h5")
    noise = np.random.normal(0, 1, (fake_size,100))
    gen_imgs = generator.predict(noise)

    return gen_imgs