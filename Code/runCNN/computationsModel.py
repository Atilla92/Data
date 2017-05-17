import glob, os, os.path
import h5py
import numpy as np

from sklearn.metrics import classification_report

def load_data(pathFiles_dataset,file_name, detail):
	os.chdir(pathFiles_dataset)
	np.random.seed(123)
	files = [file for file in glob.glob("*.hdf5") if file_name+detail in file]
	print(files)
	f = h5py.File(files[0])

	#1 Load files
	dataset_x = f['data_x'][...]
	dataset_y = f['data_y'][...]
	dataset_y_raw = f['data_y_raw'][...]

	print (dataset_x.shape, dataset_y.shape,dataset_y_raw.shape, 'shape imported dataset')

	return dataset_x, dataset_y, dataset_y_raw


def split_data(dataset_x,dataset_y, dataset_y_raw, train_size, test_size):
	length = len(dataset_x)
	length_train = int(round(train_size*length))
	length_test = int(round(test_size*length))

	train_data_x = dataset_x[0:length_train]
	train_data_y = dataset_y[0: length_train]
	train_data_y_raw = dataset_y_raw[0: length_train]
	test_data_x = dataset_x[length_train: length_train+length_test+1]
	test_data_y = dataset_y[length_train: length_train+length_test+1]
	test_data_y_raw = dataset_y_raw[length_train: length_train+length_test+1]
	train_data_y_deg = np.round(train_data_y_raw*180/np.pi+180)
	test_data_y_deg = np.round(test_data_y_raw*180/np.pi+180)

	print(train_data_x.shape, train_data_y.shape,train_data_y_raw.shape, 'train data')
	print(test_data_x.shape, test_data_y.shape,test_data_y_raw.shape, 'test data')
	print(train_data_y_deg.shape, test_data_y_deg.shape, 'degree train and test')
	return train_data_x, train_data_y, train_data_y_raw, train_data_y_deg, test_data_x, test_data_y, test_data_y_raw, test_data_y_deg


def reformat_data(rgb_model, train_data_x, test_data_x):
# set to True for 3 dim, to false for 2 dim, 
	if rgb_model == False: 
		print('only 2 dimensions input')
		train_data_x_t = np.transpose (train_data_x, [0,2,3,1])
		test_data_x_t = np.transpose (test_data_x, [0,2,3,1])
		print(train_data_x_t.shape,test_data_x_t.shape)
		num_dim= 2
		return train_data_x_t,test_data_x_t, num_dim
	# 3.1 Reformat to RGB 

	if rgb_model == True:
		print('3 channels input, last dim filled with zeros')
		train_data_rgb =np.array([np.transpose(np.vstack(
		        [i,np.zeros([1,train_data_x.shape[2], train_data_x.shape[3]])]),[1,2,0]) for i in train_data_x])
		print (np.array(train_data_rgb.shape),'train data RGB')
		test_data_rgb =np.array([np.transpose(np.vstack(
		        [i,np.zeros([1,test_data_x.shape[2], test_data_x.shape[3]])]),[1,2,0]) for i in test_data_x])
		print (np.array(test_data_rgb.shape), 'test data RGB')
		num_dim = 3
		return train_data_rgb, test_data_rgb, num_dim


def predict_model(test_data_x_t, train_data_x_t, train_data_y_deg, test_data_y_deg, model):
	y_pred = model.predict((test_data_x_t))
	y_pred_max = np.argmax(y_pred, axis=1)
	y_pred_train = model.predict((train_data_x_t))
	y_pred_max_train = np.argmax(y_pred_train, axis=1)
	#predictions_test = classification_report(test_data_y_deg, y_pred_max)
	#predictions_train = classification_report(train_data_y_deg, y_pred_max_train)
	return y_pred_max, y_pred_max_train, y_pred_train, y_pred
