import time
import glob, os, os.path
import h5py
import numpy as np

#import matplotlib.pylab as plt

import os
os.environ['KERAS_BACKEND'] = 'theano'


###########Load Keras Models ############################
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import adam, Adadelta
from keras.models import Model
from computationsModel import * 
from modelArchitecture import new_architecture
from keras.models import load_model
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import CSVLogger


################# Location and name files ###############
#0. Define which files and where they are located
path_architecture ='/home/ajv/Desktop/DroneAI/Model/' #Path to model architecture file
pathFiles_dataset = "/home/ajv/Desktop/DroneAI/Model/" # Path to training data
file_name = 'V4_' # Version of training data
detail = 'compact_all_rand' # define name data file, randomized or non randomized


###################### Size dataset, Epochs, Batch ################################
# Define size training and test data set
train_size = 0.8 # percentage of dataset
test_size = 0.2 # percentage of dataset
batch_size = 12
epochs = 100
print('number of epochs:', epochs)
# x and y dimensions of input images
shapex, shapey = 360, 30


######################### Set Load or create new model #############################
#1 Load new model or create new
new_model = True	# Make a new model architecture, set False to load old model
eva_model = True	# Evaluate model, set false to load old weights
save_model = True # Save model
rgb_model = False # True for 3 dimensional inpput
save_predictions = True # Save predictions of model output



##################### Location and name for Saving model #####################################
# # Define where to save model, weights and name files

name_model = 'modelDario'
name_weights = 'weightsDario'
path_save = '/home/ajv/Desktop/DroneAI/Model/Dario/April/'

############################ Import training data ##################################################
# Load data for training and testing model
dataset_x, dataset_y, dataset_y_raw = load_data(pathFiles_dataset,file_name, detail)

#2 Split data into training ad testing set
train_data_x, train_data_y, train_data_y_raw, train_data_y_deg,test_data_x, test_data_y, test_data_y_raw, test_data_y_deg = split_data(
										dataset_x, dataset_y, dataset_y_raw, train_size, test_size)

# Reformat data, dim last, posible 3 D
train_data_x_ch_last, test_data_x_ch_last, num_dim = reformat_data(rgb_model, train_data_x, test_data_x)


############################ Load or Create new model ####################################################################

#Load old model 
if new_model == False:
	name_model_load= 'modelRGB_small2017-04-21-21:37:16.hdf5'
	name_weights_load = 'weightsRGB_small'
	path_model_load = '/home/atilla/Documents/DeepLearning/Model/Carlo/'
	print('loading model'+ name_model_load)

# Create new model
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

checkpointer = ModelCheckpoint(filepath = path_save+name_weights+time.strftime('%F-%T')+'.hdf5', verbose = 1, save_best_only = True)

## If you want to use Tesnorboard callback, need to use tensorflow instead, Keras backend with tensorflow
#tb_CallBack = TensorBoard(log_dir=path_save+'tensorBoard_'+ time.strftime('%F-%T'), histogram_freq=0, write_graph=True, write_images=True)

csv_log = CSVLogger(path_save + 'logfile' + time.strftime('%F-%T'), separator = ',', append = False)



# Evaluate model
if eva_model == True:
	model.fit(train_data_x_ch_last, train_data_y,
	        	  batch_size=batch_size, epochs=epochs, verbose=1, validation_split = 0.2,  callbacks=[checkpointer, csv_log] )
	score = model.evaluate(test_data_x_ch_last, test_data_y, verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

##########################SAVE MODEL#################################	
# Save model
if save_model == True:
	os.chdir(path_save)
	name_new_model =file_name+ name_model+time.strftime('%F-%T')+'.hdf5'
	name_weights = file_name + name_weights+time.strftime('%F-%T')+'.hdf5'
	print('evaluating and saving new model', name_new_model)
	model.save(path_save+name_new_model)
#######################################################################


##### Evaluate and store Predictions ##################################

# Predictions of evaluated model
y_pred_max,y_pred_max_train,  y_pred_train, y_pred = predict_model(test_data_x_ch_last, train_data_x_ch_last, train_data_y_deg, test_data_y_deg, model)
print('predictions executed')

if save_predictions == True:
	os.chdir(path_save)
	f = h5py.File(file_name + "predictions"+ name_new_model+'.hdf5', "w")
	f.create_dataset('train_predict_max', data= y_pred_max_train, dtype = 'f', chunks = True)
	f.create_dataset('test_predict_max', data= y_pred_max, dtype = 'f', chunks = True)
	f.create_dataset('train_predict', data= y_pred_train, dtype = 'f', chunks = True)
	f.create_dataset('test_predict', data= y_pred, dtype = 'f', chunks = True)


