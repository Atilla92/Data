import os, glob
os.environ['KERAS_BACKEND'] = 'theano'
import h5py
import numpy as np 

import matplotlib.pyplot as plt
from keras.models import load_model
from computationMakePlots import *
import keras.backend as K


pathFiles = "/home/atilla/Documents/DeepLearning/Model/Compare/Simple/"
os.chdir(pathFiles)




font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 24}

plt.rc('font', **font)



# pathFiles = "/home/atilla/Documents/DeepLearning/Model/Compare/"
# os.chdir(pathFiles)
# os.environ['KERAS_BACKEND'] = 'tensorflow'
# import tensorflow as tf

# #files = find_files(pathFiles)
# files = [file for file in glob.glob("2017*") ]#if 'VersionBig_' in file]
# print (files)

# data_accuracy, data_parameters, data_loss, data_name = extract_info(files, pathFiles)





#files = find_files(pathFiles)
files = [file for file in glob.glob("2017*") ]#if 'VersionBig_' in file]
print (files)
#files = files[0,2,3,4,5]
data_accuracy, data_parameters, data_loss, data_name = extract_info(files, pathFiles)
print(data_name, 'data naaaaamesssssss')

plt.show()
store_data = False
if store_data == True:
	pathFiles = "/home/atilla/Documents/DeepLearning/Model/Compare/CNN"
	os.chdir(pathFiles)
	dt = h5py.special_dtype(vlen=bytes)
	f = h5py.File('DNN4_data.hdf5')
	f.create_dataset('accuracy', data = data_accuracy, dtype = 'f', chunks = True )
	f.create_dataset ('parameters', data = data_parameters, dtype = 'f', chunks = True)
	f.create_dataset ('loss', data = data_loss, dtype = 'f', chunks = True)
	f.create_dataset ('name', data = data_name, dtype = dt)


scatter_final = False
if scatter_final == True:
	names = ['2 hidden + drop','1 hidden', '2 hidden (64)', '0 hidden', '2 hidden (64)+ drop','2 hidden', '1 hidden + drop']
	names = ['DNN - 1 hidden (128)', 'CNN - 2 hidden (128)', 'DNN - 2 Hidden (64)', 'DNN - 0 Hidden', 'DNN - 4 Hidden (32)', 'DNN - 2 Hidden (128)']
	colors = np.arange(len(names))
	colors2 = ['g', 'b', 'r', 'k', 'm', 'y', 'g', 'b', 'r', 'k', 'm', 'y','g' ]
	colours = ['g', 'b', 'r', 'y', 'm', 'k', 'b', 'b']
	colours3 = ['g', 'b', 'r', 'y', 'm', 'k']

	plt.figure(1)
	x= np.array(data_accuracy)
	#y= np.divide(np.array(data_parameters),10**6)
	y = np.log10(np.array(data_parameters))
	size_s = 150
	for i in range(len(colours3)):
		plt.scatter(x[i], y[i], c =colours[i], marker = 'o', label = data_name[i], linewidth = 0, s=  size_s)
		k= 6+i
		if k<=10:

			plt.scatter(x[k], y[k], c =colours[i], marker = 's', label = data_name[k], linewidth = 0, s=  size_s)

	plt.xlabel('Test Accuracy [-]')
	plt.ylabel('Parameters [log10]')
	plt.title('Comparisson DNNs & CNN')
	plt.legend(scatterpoints = 1, fontsize = 16, loc = 'upper left')
	plt.show()


#------------------------- Make Plots-------------------
scatter_plot =False
if scatter_plot == True:
	size_s = 150
	x= np.array(data_accuracy)
	y= np.divide(np.array(data_parameters),10**6)
	#y = np.log10(np.array(data_parameters))
	print(x.shape)

	plt.figure(1)
	size_s = 150
	# plot_a = plt.scatter(x[0], y[0], c =colors2[2], marker = 's', label = names[0], linewidth = 0, s=  size_s)
	# plot_b = plt.scatter(x[1], y[1], c =colors2[1], marker = 'o', label = names[1], linewidth = 0, s= size_s)
	# plot_c = plt.scatter(x[2], y[2], c =colors2[2], marker = 'o', label = names[2], linewidth = 2, s= size_s)
	# plot_d = plt.scatter(x[3], y[3], c =colors2[0], marker = 'o', label = names[3], linewidth = 0, s= size_s)
	# plot_e = plt.scatter(x[4], y[4], c =colors2[2], marker = 's', label = names[4], linewidth = 2, s= size_s)
	# plot_f = plt.scatter(x[5], y[5], c =colors2[2], marker = 'o', label = names[5], linewidth = 0, s= size_s)
	# plot_g = plt.scatter(x[6], y[6], c =colors2[1], marker = 's', label = names[6], linewidth = 0, s= size_s)

	plot_a = plt.scatter(x[0], y[0], c =colours[0], marker = 'o', label = names[0], linewidth = 0, s=  size_s)
	plot_b = plt.scatter(x[1], y[1], c =colours[1], marker = 'o', label = names[1], linewidth = 0, s= size_s)
	plot_c = plt.scatter(x[2], y[2], c =colours[2], marker = 's', label = names[2], linewidth = 0, s= size_s)
	plot_d = plt.scatter(x[3], y[3], c =colours[3], marker = 'o', label = names[3], linewidth = 0, s= size_s)
	plot_e = plt.scatter(x[4], y[4], c =colours[4], marker = 's', label = names[4], linewidth = 0, s= size_s)
	plot_f = plt.scatter(x[5], y[5], c =colours[5], marker = 'o', label = names[5], linewidth = 0, s= size_s)



	#plot_g = plt.scatter(x[6], y[6], c =colours[6], marker = 's', label = names[6], linewidth = 0, s= size_s)


	plt.xlabel('Test Accuracy [-]')
	plt.ylabel('Parameters [10^6]')
	plt.title('Comparisson DNNs & CNN')
	plt.legend(scatterpoints = 1, fontsize = 16, loc = 'lower right')
	plt.show()


