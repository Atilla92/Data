


import time
import glob, os, os.path
import h5py
import numpy as np

import matplotlib.pylab as plt

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import cv2
from keras.preprocessing.image import img_to_array
from vis.utils import utils
from vis.visualization import visualize_cam

from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, AveragePooling2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam
from keras.models import Model
import tensorflow as tf
pathFiles = "/home/atilla/Documents/DeepLearning/Model/"
os.chdir(pathFiles)
from keras.models import load_model
#model = load_model('/home/atilla/Documents/DeepLearning/Model/modelRGB.hdf5')

def global_average_pooling(x):
    return K.mean(x, axis = (2, 3))

def global_average_pooling_shape(input_shape):
    return input_shape[0:2]



predefine_model = True
if predefine_model == True:
	Bottom_dense=True
# 7. Define model architecture
#model = Sequential()
#inputs = Input(shape =(30,360,2), name = 'main_input' )
	inputs = Input(shape =(30,360,3), name = 'main_input' )
	x = Lambda(lambda x: x[:,:,:,:2], output_shape = (30,360,2))(inputs)
	x =ZeroPadding2D((1,1), name ='zero_1', data_format = 'channels_last') (x)
	x = Convolution2D(8, (3, 3), activation='relu', name = 'conv_1')(x)
	# x =ZeroPadding2D((1,1),  name = 'zero_2')(x)
	# x =Conv2D(16, (3, 3), activation='relu', name = 'conv_2')(x)
	x= MaxPooling2D((2,2), strides=(2,2), name = 'max_1')(x)
	x =Flatten(name = 'flatten_1')(x)
	#x = Lambda(global_average_pooling, output_shape=global_average_pooling_shape)(x)

	if Bottom_dense == True :
	    x =Dense(128, activation='relu', name = 'dense_1')(x)
	    x=Dropout(0.5, name = 'drop_1')(x)
	    x = Dense(128, activation='relu', name = 'dense_2')(x)
	    x = Dropout(0.5, name ='drop_2')(x)
	#     x = Dense(32, activation='relu', name = 'dense_3')(x)
	#     x = Dense(32, activation='relu', name = 'dense_5')(x)
	    #x = Dropout(0.5, name = 'drop_3')(x)
	    x=Dense(361, activation='softmax', name = 'dense_4')(x)
	#     x=Dense(361, name = 'dense_3')(x)
	#     x= Activation ('softmax',name = 'activation')(x)

	model = Model(inputs, x, name= 'simple_model')
	model.compile(loss='categorical_crossentropy', optimizer=adam(lr=1e-4), metrics=['accuracy'])
	#model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

	model.summary()


predefine_model_2 = False
if predefine_model_2 == True:
	model = Sequential()



	model.add(Lambda(lambda x: x[:,:,:,:2], input_shape = (30,360,3),output_shape = (30,360,2)))
	model.add(ZeroPadding2D((1,1), name ='zero_1', data_format = 'channels_last'))
	model.add(Convolution2D(8, (3, 3), activation='relu', name = 'conv_1'))
	model.add(MaxPooling2D((2,2), strides=(2,2), name = 'max_1'))
	#model.add(Lambda(global_average_pooling, output_shape=global_average_pooling_shape))
	model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
	model.add(Dense(361, activation='softmax', name = 'dense_4'))

	model.compile(loss='categorical_crossentropy', optimizer=adam(lr=1e-4), metrics=['accuracy'])
	#model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

	model.summary()




model.load_weights('/home/atilla/Documents/DeepLearning/Model/Compare/Simple/2017-05-18-11:10:18/weights.hdf5')
#model.load_weights('/home/atilla/Documents/DeepLearning/Model/weightsRGB.hdf5')
#model.summary()


pathFiles = "/home/atilla/Documents/DeepLearning/Test/"
os.chdir(pathFiles)
file_name = 'V4_'
detail = 'compact_all_rand'
files = [file for file in glob.glob("*.hdf5") if file_name+detail in file]
print (files)
f = h5py.File(files[0])
dataset_x = f['data_x'][...]
dataset_y_raw = f['data_y_raw'][...]
length = len(dataset_y_raw)
length_train = int(round(0.8*length))
length_test = int(round(0.2*length))
#train_data_x = dataset_x[0:length_train]
train_data_y_raw = dataset_y_raw[0: length_train]
test_data_y_raw = dataset_y_raw[length_train: length_train+length_test+1]
train_data_y_deg = np.round(train_data_y_raw*180/np.pi+180)
test_data_y_deg = np.round(test_data_y_raw*180/np.pi+180)


layer_name = 'conv_1'
layer_idx = [idx for idx, layer in enumerate (model.layers) if layer.name == layer_name][0]



#hello

from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from vis.visualization import visualize_saliency

# image_paths = [
#     "http://www.tigerfdn.com/wp-content/uploads/2016/05/How-Much-Does-A-Tiger-Weigh.jpg",
#     "http://www.slate.com/content/dam/slate/articles/health_and_science/wild_things/2013/10/131025_WILD_AdeliePenguin.jpg.CROP.promo-mediumlarge.jpg",
#     "https://www.kshs.org/cool2/graphics/dumbbell1lg.jpg",
#     "http://tampaspeedboatadventures.com/wp-content/uploads/2010/10/DSC07011.jpg",
#     "http://ichef-1.bbci.co.uk/news/660/cpsprodpb/1C24/production/_85540270_85540265.jpg"
# ]

# heatmaps = []
# for path in image_paths:
#     seed_img = utils.load_img(path, target_size=(30, 360))
#     print(seed_img.shape, 'shape image')
#     x = np.expand_dims(img_to_array(seed_img), axis=0)
#     print(x.shape, 'shape x 1')
#     x = preprocess_input(x)
#     print(x.shape, 'shape x 2')
#     pred_class = np.argmax(model.predict(x))

#     # Here we are asking it to show attention such that prob of `pred_class` is maximized.
#     heatmap = visualize_saliency(model, layer_idx, [pred_class], seed_img)
#     heatmaps.append(heatmap)







train_data_rgb =np.array([np.transpose(np.vstack(
                [dataset_x[1],np.zeros([1,dataset_x.shape[2], dataset_x.shape[3]])]),[1,2,0])])
img =train_data_rgb[0]



print(img.shape, train_data_rgb.shape)
#plt.imshow(img, aspect ='auto')
heatmaps = []
predicted_class = np.argmax(model.predict(train_data_rgb))
#pred_cate = to_categorical(pred_class, 361)
#print(pred_class, 'error is here')


#heatmap = visualize_cam(model, layer_idx, [pred_class],img, text='pred_class')



#### Start new vis cam model
#from keras.applications.vgg16 import (VGG16, preprocess_input, decode_predictions)


def grad_cam(input_model, image, category_index, layer_name):
    model = Sequential()
    model.add(input_model)

    nb_classes = 361
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    model.add(Lambda(target_layer,
                     output_shape = target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)
    print(loss, 'loss')
    conv_output =  [l for l in model.layers[0].layers if l.name is layer_name][0].output
    print(conv_output, 'conv output', model.layers[-1])
    grads = normalize(K.gradients(loss, conv_output)[0])
    #iterate = K.function([img, K.learning_phase()], [loss, grads])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])
    output, grads_val = gradient_function([image, 0])
 #    output, grads_val = output[0, :], grads_val[0, :, :, :]

    loss_value, grads_value = iterate([img[0], 1])

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


#cam, heatmap = grad_cam(model, img, predicted_class, 'conv_1')
cam, heatmap = grad_cam(model, train_data_rgb, predicted_class, 'conv_1')