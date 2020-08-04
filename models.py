import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation,MaxPooling2D,Flatten, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout,Activation
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import Sequential


def SmallNet(num_classes,use_dropout,dropout_p=0.2, l2_lambda=0.0,use_bn=False):
	model = Sequential()
	n_channels = [32,64,128]
	l2 = regularizers.l2(l2_lambda)
	for i in range(3):
		if(i==0):
			model.add(Conv2D(n_channels[i], kernel_size=(3, 3), padding='same', kernel_regularizer=l2,input_shape=(28,28,1)))
		else:
			model.add(Conv2D(n_channels[i], kernel_size=(3, 3), padding='same', kernel_regularizer=l2))
		if(use_bn):
			model.add(BatchNormalization())
		model.add(Activation(relu))
		model.add(MaxPooling2D(pool_size=(2,2)))
		if(use_dropout):
			model.add(Dropout(dropout_p))
	model.add(Flatten())
	model.add(Dense(256,activation='relu',kernel_regularizer=l2))
	if(use_dropout):
		model.add(Dropout(dropout_p))
	model.add(Dense(128,activation='relu',kernel_regularizer=l2))
	if(use_dropout):
		model.add(Dropout(dropout_p))
	model.add(Dense(10,activation='softmax'))
	return model



def BigNet(num_classes,use_dropout,dropout_p=0.2, l2_lambda=0.0,use_bn=False):
	model = Sequential()
	n_channels = [32,64,128]
	l2 = regularizers.l2(l2_lambda)
	for i in range(3):
		if(i==0):
			model.add(Conv2D(n_channels[i], kernel_size=(3, 3), padding='same', kernel_regularizer=l2,input_shape=(28,28,1)))
		else:
			model.add(Conv2D(n_channels[i], kernel_size=(3, 3), padding='same', kernel_regularizer=l2))
		if(use_bn):
			model.add(BatchNormalization())
		model.add(Activation(relu))
		model.add(Conv2D(n_channels[i], kernel_size=(3, 3), padding='same', kernel_regularizer=l2))
		if(use_bn):
			model.add(BatchNormalization())
		model.add(Activation(relu))
		model.add(MaxPooling2D(pool_size=(2,2)))
		if(use_dropout):
			model.add(Dropout(dropout_p))
	model.add(Flatten())
	model.add(Dense(256,activation='relu',kernel_regularizer=l2))
	if(use_dropout):
		model.add(Dropout(dropout_p))
	model.add(Dense(128,activation='relu',kernel_regularizer=l2))
	if(use_dropout):
		model.add(Dropout(dropout_p))
	model.add(Dense(num_classes,activation='softmax',kernel_regularizer=l2))
	return model







