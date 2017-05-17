
import time
import glob, os, os.path
import h5py
import numpy as np
from safeFiles import *
from dataPreprocessing import *
from computations import *

safe_Files = True
pathFiles = "/home/atilla/Documents/DeepLearning/Test/"
os.chdir(pathFiles)



files = [file for file in glob.glob("*.hdf5") if 'VersionBig_' in file]
print(files)

###############################################################################
# For newer datasets, set gaps_status to true and you will also get the three gaps data,
# still need to extraxt training set and storing it properly for the gaps
#train_x, train_y, gaps_y, mat2 = extract_files(files, gaps_status= True)
gaps_status= False
start_time = time.time()
train_x, train_y, __ = extract_files(files, gaps_status)

end_time = time.time()-start_time
print('finished', end_time)
##########################################################################
#Normalize input data and categorize output data
train_norm = train_x
train_norm_x = normalize_data(train_norm)
train_y_2= train_y
train_cate_y = categorize_data(train_y_2)


#########################################################################
#Randomize data

# Still need to fix this
if gaps_status== True:

	gaps_y_2 = gaps_y
	gaps_cate_y = categorize_data(gaps_y_2)
	gaps_cate_y = np.sum(gaps_cate_y, axis = 0)
	gaps_cate_y = gaps_cate_y.reshape(1, len(gaps_cate_y))
	train_norm_rand_x, train_x_rand, train_cate_rand_y, train_y_rand, gaps_cate_rand , gaps_y_rand= randomize_data(train_norm_x, train_x, train_cate_y, train_y, gaps_cate_y, gaps_y, gaps_status)

elif gaps_status == False:

	train_norm_rand_x, train_x_rand, train_cate_rand_y, train_y_rand= randomize_data(train_norm_x, train_x, train_cate_y, train_y,__, __, gaps_status)
#train_norm_rand_x2, gaps_cate_rand_y = randomize_data(train_norm_x, gaps_cate_y)
 

 ################################################################################
#Store datasets, one file for the randomized version, one for the non randomized version
# Gaps cannot be stored yet, missing info in training data, and randomization is not finished 
if safe_Files == True:
	file_name = 'V4_'
	f = h5py.File(file_name + "compact_all_rand.hdf5", "w")
	f.create_dataset('data_x', data= train_norm_rand_x, dtype = 'f', chunks = True)
	f.create_dataset('data_y', data = train_cate_rand_y, dtype = 'f', chunks = True)
	f.create_dataset('data_y_raw', data = train_y_rand, dtype = 'f', chunks = True )
	f.create_dataset('data_x_raw', data= train_x_rand, dtype = 'f', chunks = True)
	if gaps_status == True:
		f.create_dataset('gaps_y', data = gaps_cate_rand, dtype = 'f', chunks = True )
		f.create_dataset('gaps_y_raw', data = gaps_y_rand, dtype = 'f', chunks = True )


	f = h5py.File(file_name + "compact_non_random.hdf5", "w")
	f.create_dataset('data_x', data= train_norm_x, dtype = 'f', chunks = True)
	f.create_dataset('data_y', data = train_cate_y, dtype = 'f', chunks = True)
	f.create_dataset('data_y_raw', data = train_y, dtype = 'f', chunks = True )
	f.create_dataset('data_x_raw', data= train_x, dtype = 'f', chunks = True)
	if gaps_status == True:
		f.create_dataset('gaps_y_raw', data = gaps_y, dtype = 'f', chunks = True )	
		f.create_dataset('gaps_y', data = gaps_cate_y, dtype = 'f', chunks = True )





####################################################################
#Sanity checks
# import matplotlib.pylab as plt
# import numpy as np
# bins = np.linspace(-np.pi, np.pi, 360)
# plt.hist(train_y, bins)

# plt.show()



# ##########################
# fov_x  = np.pi #half FOV along azimuth in radians
# fov_y = np.pi/4 # half FOV along elevation in radians
# res_x = 360 #amount of pixels along azimuth
# res_y = 30 # amount of pixels along elevation

# phis   = np.linspace(-fov_x, fov_x, res_x)
# thetas = np.linspace(-fov_y, fov_y, res_y)


# layout = np.array([[phi, theta] 
#                    for theta in thetas
#                    for phi in phis])
# plt.figure()
# ofx = train_x[0][0][...].flatten()

# ofy = train_x[0][1][...].flatten()

# plt.quiver(layout[:,0], layout[:,1], 
#            ofx, ofy)
# #plt.show()

# #plt.close()