import time
import glob, os, os.path
import h5py
import numpy as np

#import matplotlib.pylab as plt

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'


from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam, Adadelta
from keras.models import Model
from computationsModel import * 
from modelArchitectureDarioA import new_architecture
from keras.layers import Input, Dense, Flatten, Activation
from keras.models import Model
from keras.models import load_model
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint, TensorBoard




# Define which files and where they are located
# path_architecture ='/home/ajv/Desktop/DroneAI/Model/'
# pathFiles_dataset = "/home/ajv/Desktop/DroneAI/Model/"
# file_name = 'V4_'
# detail = 'compact_all_rand'
path_architecture ='/home/ajv/Desktop/DroneAI/Model/'
pathFiles_dataset = "/home/ajv/Desktop/DroneAI/Model/"
file_name = 'V4_'
detail = 'compact_all_rand'



# Define size training and test data set
train_size = 0.8 # percentage of dataset
test_size = 0.2 # percentage of dataset
batch_size = 12
epochs = 70
print('number of epochs:', epochs)
# x and y dimensions of input images
shapex, shapey = 360, 30



#5 Load new model or create new
new_model = True	# Make a new model architecture
eva_model = True	# Evaluate model
save_model = True # Save model
rgb_model = False # True for 3 dimensional inpput
save_predictions = True # Save predictions


# Load new model


# name_model = 'modelDario'
# name_weights = 'weightsDario'
# path_model = '/home/ajv/Desktop/DroneAI/Model/Dario/'

# #Load old model 
# if new_model == False:
# 	name_model_load= 'modelRGB_small2017-04-21-21:37:16.hdf5'
# 	name_weights_load = 'weightsRGB_small'
# 	path_model_load = '/home/ajv/Desktop/DroneAI/Model/'
# 	print('loading model'+ name_model_load)

# # Save model
# path_save = '/home/ajv/Desktop/DroneAI/Model/Dario/'
# 4 Define paramters model
name_model = 'modelA2'
name_weights = 'weightsA2'
path_model = '/home/ajv/Desktop/DroneAI/Model/Dario/'

#Load old model 
if new_model == False:
	name_model_load= 'modelRGB_small2017-04-21-21:37:16.hdf5'
	name_weights_load = 'weightsRGB_small'
	path_model_load = '/home/ajv/Desktop/DroneAI/Model/Dario/'
	print('loading model'+ name_model_load)


# Save model
path_save = '/home/ajv/Desktop/DroneAI/Model/Dario/Compare/'

os.chdir(path_save)
file_new  =  time.strftime('%F-%T')
if not os.path.exists(file_new):
    os.makedirs(file_new)
# 4 Define paramters model


##############################################################################
#1 Load data for training and testing model

dataset_x, dataset_y, dataset_y_raw = load_data(pathFiles_dataset,file_name, detail)

#2 Split data into training ad testing set

train_data_x, train_data_y, train_data_y_raw, train_data_y_deg,test_data_x, test_data_y, test_data_y_raw, test_data_y_deg = split_data(
										dataset_x, dataset_y, dataset_y_raw, train_size, test_size)


# Reformat data, dim last, posible 3 D

train_data_x_ch_last, test_data_x_ch_last, num_dim = reformat_data(rgb_model, train_data_x, test_data_x)


# Load model
if new_model == True:
	os.chdir(path_architecture)
	model = new_architecture(num_dim)

if new_model == False:
	print('loading model:',name_model_load )
	os.chdir(path_model_load)
	model = load_model(path_model_load+name_model_load)
model.summary()



#############Save and Run #################

# Checkerpoints, call_back, weights
from keras.callbacks import CSVLogger
checkpointer = ModelCheckpoint(filepath = path_save+file_new+'/'+'weights.hdf5', verbose = 1, save_best_only = True)
#tb_CallBack = TensorBoard(log_dir=path_save+ file_new+'/'+'tensorBoard', histogram_freq=0, write_graph=True, write_images=True)
csv_log = CSVLogger(path_save + file_new+ '/'+ 'logfile' , separator = ',', append = False)
# 9. Fit model on training data


# Evaluate model
if eva_model == True:
	model.fit(train_data_x_ch_last, train_data_y,
	        	  batch_size=batch_size, epochs=epochs, verbose=1, validation_split = 0.2,  callbacks=[checkpointer, csv_log] )
	score = model.evaluate(test_data_x_ch_last, test_data_y, verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
	print(model.metrics_names, score)



	
# Save model
if save_model == True:
	os.chdir(path_save)
	name_new_model =file_name+ name_model+ '.hdf5'#+time.strftime('%F-%T')+'.hdf5'
	name_weights = file_name + name_weights+'.hdf5' #time.strftime('%F-%T')+'.hdf5'
	print('evaluating and saving new model', name_new_model)
	model.save(path_save+file_new+'/'+name_new_model)
	os.chdir(path_save + file_new +'/')
	f = h5py.File('score')
	f.create_dataset('score', data = score, dtype = 'f', chunks = True)


	#model.save()


##### Predictions ####


y_pred_max,y_pred_max_train,  y_pred_train, y_pred = predict_model(test_data_x_ch_last, train_data_x_ch_last, train_data_y_deg, test_data_y_deg, model)

print('predictions executed')

if save_predictions == True:
	os.chdir(path_save + file_new +'/')
	f = h5py.File(file_name +'predictionsA2.hdf5', "w")
	f.create_dataset('train_predict_max', data= y_pred_max_train, dtype = 'f', chunks = True)
	f.create_dataset('test_predict_max', data= y_pred_max, dtype = 'f', chunks = True)
	f.create_dataset('train_predict', data= y_pred_train, dtype = 'f', chunks = True)
	f.create_dataset('test_predict', data= y_pred, dtype = 'f', chunks = True)


