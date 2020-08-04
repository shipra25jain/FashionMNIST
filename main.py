import argparse
import datetime
import numpy as np
import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.keras import optimizers
import pandas as pd
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from utils import mnist_reader
import numpy as np
from sklearn.model_selection import train_test_split
from models import *
from custom_utils import *
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint,TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
import tensorboard
from tensorflow.keras.losses import CategoricalCrossentropy
import time
import matplotlib.pyplot as plt

def parse_option():
    parser = argparse.ArgumentParser('Arguments for image classification on Fashion MNIST dataset')
    # training params
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size for training'
                        )
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate for training'
                        )
    parser.add_argument('--epochs', type=int, default=150,
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
    parser.add_argument('--ckpt', type=str, default=None,
                        help='checkpoint path for trained model'
                        )
    parser.add_argument('--inference_only', type=bool,default=False,
                        help='read checkpoint to do inference'
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

    args = parser.parse_args()
    return args


def main():
    args = parse_option()
    print(args)

    x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    x_test, y_test_labels = mnist_reader.load_mnist('data/fashion', kind='t10k')
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    x_train=x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test=x_test.reshape(x_test.shape[0], 28, 28 ,1)
    # x_train = x_train / 255.0
    # x_test = x_test / 255.0
    y_train=keras.utils.to_categorical(y_train)
    y_test=keras.utils.to_categorical(y_test_labels)
    num_classes = 10
    if(args.train_val_split):
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, shuffle=True)
    if(args.model == 'SmallNet'):
    	model = SmallNet(num_classes,args.use_dropout,args.dropout_p,args.l2_lambda,args.use_bn)
    else :
    	model = BigNet(num_classes,args.use_dropout,args.dropout_p,args.l2_lambda,args.use_bn)
    adam = optimizers.Adam(lr=args.lr)
    if(args.smooth_labels):
        loss = CategoricalCrossentropy(label_smoothing=0.1)
    else:
        loss = CategoricalCrossentropy()
    model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
    print(model.summary())
    
    
    val_datagen = ImageDataGenerator(
     		rescale= 1/255.0,
            featurewise_center=args.standardize_data, 
            featurewise_std_normalization=args.standardize_data 
            )
    test_datagen = ImageDataGenerator(
     		rescale= 1/255.0,
            featurewise_center=args.standardize_data, 
            featurewise_std_normalization=args.standardize_data
            )
    if(args.standardize_data):
        val_datagen.fit(x_train)
        test_datagen.fit(x_train)

    if(args.inference_only):
        if(args.ckpt == None):
            print("please provide trained model checkpoint for inference")
            return
        checkpoint_path = args.ckpt
        print(checkpoint_path)
        model.load_weights(checkpoint_path,by_name=True)
        start_time = time.time()
        predictions = np.argmax(model.predict_generator(test_datagen.flow(x_test,y_test,batch_size=100,shuffle=False)),axis=1)
        # _, test_acc = model.evaluate_generator(test_datagen.flow(x_test,y_test,batch_size=100,shuffle=False))
        print("--- %s seconds ---" % (time.time() - start_time))
        # print(test_acc)
        plot_confusion_matrix(predictions,y_test_labels)
        return

    filepath = "weights_best.hdf5"

    if(args.train_val_split):
        training_callbacks = [
             EarlyStopping(
                monitor='val_accuracy',
                patience=12       
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=8
            ),
            ModelCheckpoint(
                filepath, monitor='val_accuracy', 
                verbose=1,
                save_best_only=True,
                mode='max'
            )
        ]
    else:
        training_callbacks = [
        ReduceLROnPlateau(
                monitor='loss',
                factor=0.1,
                patience=8
            )
        ]

    if(args.write_summary):
        training_callbacks.append(
            TensorBoard(
            log_dir=args.logdir
        ))

    train_datagen = train_data_generator(args)
    
    if(args.standardize_data):
        train_datagen.fit(x_train)

    #model.fit(x_train, y_train, batch_size=args.batch_size, epochs=100, validation_data =(x_valid, y_valid))
    if(args.train_val_split):
        model.fit_generator(train_datagen.flow(x_train,y_train,batch_size=args.batch_size), epochs=args.epochs, validation_data = val_datagen.flow(x_valid, y_valid,batch_size=100),callbacks = training_callbacks)
    else:
        model.fit_generator(train_datagen.flow(x_train,y_train,batch_size=args.batch_size), epochs=args.epochs,callbacks = training_callbacks)

    test_loss, test_acc = model.evaluate_generator(test_datagen.flow(x_test,y_test,batch_size=1))
    print("Accuracy for last checkpoint : ",test_acc)
    model.save_weights('weights_last.hdf5')
    if(args.train_val_split):
        model.load_weights('weights_best.hdf5')
        test_loss, test_acc = model.evaluate_generator(test_datagen.flow(x_test,y_test,batch_size=1))
        print("Accuracy for best checkpoint : ",test_acc)

if __name__ == '__main__':
    main()












