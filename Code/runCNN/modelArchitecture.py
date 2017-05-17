
from keras.layers import Input, Dense, Flatten, Activation
from keras.models import Model
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam
from keras.models import Model
print('training new model')


def new_architecture(num_dim):
	Bottom_dense=True #set to false if you dont want fully connected layers

	inputs = Input(shape =(30,360,num_dim), name = 'main_input' )
	x =ZeroPadding2D((1,1), name ='zero_1') (inputs)
	x = Convolution2D(2, (3, 3), activation='relu', name = 'conv_1')(x)
	#x =ZeroPadding2D((1,1),  name = 'zero_2')(x)
	#x =Convolution2D(32, (3, 3), activation='relu', name = 'conv_2')(x)
	x= MaxPooling2D((2,2), strides=(2,2), name = 'max_1')(x)

	#x =ZeroPadding2D((1,1), name= 'zero_3')(x)
	#x= Convolution2D(32, (3, 3,), activation='relu', name = 'conv_3')(x)
	#x= ZeroPadding2D((1,1),  name = 'zero_4')(x)
	#x=Convolution2D(64, (3, 3), activation='relu', name = 'conv_4')(x)
	#x=MaxPooling2D((2,2), strides=(2,2), name= 'max_2')(x)

	#x= ZeroPadding2D((1,1),  name = 'zero_5')(x)
	#x= Convolution2D(16, (3, 3,), activation='relu', name = 'conv_5')(x)
	#x= ZeroPadding2D((1,1),  name = 'zero_6')(x)
	#x=Convolution2D(16, (3, 3), activation='relu', name = 'conv_6')(x)
	#x=MaxPooling2D((2,2), strides=(2,2), name= 'max_3')(x)

	# x= ZeroPadding2D((1,1),  name = 'zero_7')(x)
	# x= Convolution2D(16, (3, 3,), activation='relu', name = 'conv_7')(x)
	# x= ZeroPadding2D((1,1),  name = 'zero_8')(x)
	# x=Convolution2D(16, (3, 3), activation='relu', name = 'conv_8')(x)
	# x=MaxPooling2D((2,3), strides=(2,2), name= 'max_4')(x)
	# inputs = Input(shape =(30,360,num_dim), name = 'main_input' )
	# x =ZeroPadding2D((1,1), name ='zero_1') (inputs)
	# x = Convolution2D(16, (3, 3), activation='relu', name = 'conv_1')(x)
	# x =ZeroPadding2D((1,1),  name = 'zero_2')(x)
	# x =Convolution2D(32, (3, 3), activation='relu', name = 'conv_2')(x)
	# x= MaxPooling2D((2,2), strides=(2,2), name = 'max_1')(x)
	# x =ZeroPadding2D((1,1), name= 'zero_3')(x)
	# x= Convolution2D(32,( 3, 3), activation='relu', name = 'conv_3')(x)
	# x= ZeroPadding2D((1,1),  name = 'zero_4')(x)
	# x=Convolution2D(64, (3, 3), activation='relu', name = 'conv_4')(x)
	# x=MaxPooling2D((2,2), strides=(2,2), name= 'max_2')(x)

	x =Flatten(name = 'flatten_1')(x)
	if Bottom_dense == True :
		x=Dense(128, activation='relu', name = 'dense_1')(x)
		x=Dropout(0.5, name = 'drop_1')(x)
		x= Dense(128, activation='relu', name = 'dense_2')(x)
		x= Dropout(0.5, name = 'drop_2')(x)
		#x= Dense(32, activation='relu', name = 'dense_3')(x)
		#x = Dropout(0.5, name = 'drop_3')(x)
		#x=Dense(32, activation='relu', name = 'dense_4')(x)
		#x= Dropout(0.5, name = 'drop_4')(x)
		x=Dense(361, activation='softmax', name = 'dense_5')(x)
	model = Model(inputs, x, name= 'CNN_1_2_2_hidden_128_drop')
	model.compile(loss='categorical_crossentropy', optimizer=adam(lr=1e-4), metrics=['accuracy'])
	return model
