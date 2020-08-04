import argparse
import datetime
import numpy as np
import tensorflow as tf
# import tensorflow_addons as tfa
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import optimizers
import pandas as pd
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from utils import mnist_reader
import numpy as np
from sklearn.model_selection import train_test_split
from models import *
from augmentation_utils import *
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint,TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorboard

def parse_option():
    parser = argparse.ArgumentParser('Arguments for image classification on Fashion MNIST dataset')
    # training params
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size for training'
                        )
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate for training'
                        )
    parser.add_argument('--epoch', type=int, default=100,
                        help='number of epochs for training'
                        )
    parser.add_argument('--model', type=str, default='SmallNet',
                        help='model to use, choose from ("SmallNet", "BigNet")'
                        )
    parser.add_argument('--logdir', type=str, default='summary',
                        help='to log summary'
                        )
    parser.add_argument('--use_data_aug',type=bool,default=True,
                        help='data augmentation'
                        )
    parser.add_argument('--data_aug', type=str, default='simple',
                        help='data augmentatin strategy to use, choose from ("simple","random_erasing")'
                        )
    parser.add_argument('--inference_only', type=bool,default=False,
                        help='read checkpoint to do inference'
                        )
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer to use, choose from ("adam", "sgd")'
                        )
    parser.add_argument('--use_dropout', type=bool,default=True,
                        help='dropout usage'
                        )
    parser.add_argument('--use_bn', type=bool,default=False,
                        help='batch normalization'
                        )
    parser.add_argument('--l2_lambda', type=float, default=0.0001,
                        help='penalty for l2 regularization'
                        )
    parser.add_argument('--dropout_p', type=float, default=0.02,
                        help='probability for dropout'
                        )
    parser.add_argument('--train_val_split', type=bool,default=True,
                        help='split training data into train and val data'
                        )
    parser.add_argument('--standardize_data', type=bool,default=True,
                        help='subtract by mean and divide by variance'
                        )
    parser.add_argument('--smooth_labels', type=bool,default=True,
                        help='smoothing of labels for noisy annotations'
                        )
    parser.add_argument('--write_summary', type=bool,default=True,
                        help='write summary for tensorboard'
                        )
    parser.add_argument('--draw_figures', action='store_true',
                        help='produce figures for the projections'
                        )

    args = parser.parse_args()
    return args


def build_model(use_bn,lr,use_dropout=False):
    model = BigNet(10,use_dropout,use_bn)
    adam = optimizers.Adam(lr=lr, decay=1e-6)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model2(l2_lambda,dropout_p,use_bn=out,lr=out,use_dropout=True):
    model = BigNet(10,use_dropout,dropout_p,l2_lambda,use_bn)
    adam = optimizers.Adam(lr=lr, decay=1e-6)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    args = parse_option()
    print(args)


    x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    x_train=x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test=x_test.reshape(x_test.shape[0], 28, 28 ,1)
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train=keras.utils.to_categorical(y_train)
    y_test=keras.utils.to_categorical(y_test)
    num_classes = 10


    print("Grid search for batch_size,batch norm and learning rate")
    model = KerasClassifier(build_fn=build_model,,epochs=40,verbose=1)
    batch_size = [32,64,128]
    lr = [0.01,0.001]
    use_bn = [True,False]
    param_grid = dict(batch_size=batch_size, lr=lr,use_bn=use_bn)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(x_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    # print("Grid search for l2_lambda and dropout_p")
    # model = KerasClassifier(build_fn=build_model2,batch_size = out,epochs=50,verbose=0)
    # l2_lambda = [0.0001,0.00001,0.000001]
    # dropout_p = [0.1,0.2,0.3]
    # param_grid = dict(l2_lambda=l2_lambda,dropout_p=dropout_p)
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    # grid_result = grid.fit(x_train, y_train)
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))



if __name__ == '__main__':
    main()












