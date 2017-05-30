
import glob, os
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import h5py


def size_model(model): # Compute number of params in a model (the actual number of floats)
    return sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])
#extract logfile 


def extract_info(files, pathFiles, make_plots, make_plots_loss):
	extract_score = True
	label_size = 30
	title_size = 38
	data_scores = []
	data_name = []
	data_loss = []
	data_accuracy= []
	data_parameters = []
	for a, i in enumerate(files):
		model_name = 'Whatever'
		path_log = pathFiles +i+'/'
		os.chdir(path_log)
		print (path_log)
		print i
		
		
		# Import model and extrac scores:
		modelfile = [file for file in glob.glob('V4_model*')]

		if extract_score == True:
			scorefile = [file for file in glob.glob('score*')]
			for score_file in scorefile:
				f= h5py.File(score_file)
				scores = f['score'][...]
				print(scores[1], 'scores')

		if modelfile != []:
			model = load_model(modelfile[0])
			x = model.summary()
			parameters = size_model(model)
			#print(x.type)
			config_model = model.get_config()
			model_name = config_model['name']
			model_name = str(model_name)
			print(model_name)
		#print (modelfile)
		

		# Import Logg and create plots:
		logfile =[file for file in glob.glob("log*")][0]
		data= np.genfromtxt(logfile, delimiter = ',')
		#print (data [1:, :])
		f = open(logfile, 'r')
		epoch = data [1:,0]
		accuracy_training = data [1:,1]
		loss_training = data [1:, 2]
		accuracy_validation = data [1:,3]
		loss_validation = data [1:, 4]
		print type(a) 
		if make_plots == True: #and a!=1:
			print('plotting')
			colours = ['g', 'b', 'r', 'y', 'm', 'k', 'b', 'b']
			plt.figure(1)
			plt.plot(epoch, accuracy_training*100, label ='training ', linewidth = 2.5)#+model_name, linewidth = 2.5)
			plt.plot(epoch, accuracy_validation*100, label ='validation ', linewidth =2.5) #model_name, linewidth = 2.5)#, color = colours[a])
			plt.xlabel('Epochs', fontsize = label_size)
			plt.ylabel('Accuracy [%]', fontsize =label_size)
			plt.title('Accuracy', fontsize = title_size)
			plt.axis([0,70,0,100])
			plt.legend(fontsize = 20, loc = 'lower right')
			plt.show()
		if make_plots_loss == True:
			plt.figure(2)
			plt.plot(epoch, loss_training, label= 'training_'+ model_name, linewidth = 2.5)
			plt.plot(epoch, loss_validation, label ='validation '+model_name, linewidth = 2.5) #, color =  colours[a])
			plt.title('Loss Model'+ str(i))
			plt.xlabel('Epochs', fontsize = label_size)
			plt.ylabel('Loss', fontsize = label_size)
			plt.title('Loss', fontsize = title_size)
			plt.axis([0,70,0,20])
			plt.legend( fontsize = 20, loc = 'lower right')
			plt.show()
		data_parameters.append(parameters)
		data_accuracy.append(scores[1])
		data_loss.append(scores[0])
		data_name.append(model_name)


		

		# for line in f:
		# 	line = line.strip()
			#data= np.genfromtxt(line, delimiter = ',')
		f.close
	return data_accuracy, data_parameters, data_loss, data_name