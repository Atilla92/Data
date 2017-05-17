
import numpy as np
import matplotlib.pylab as plt
import time 
import pickle
from numpy import linalg as LA
from random import randint
from graphics import *
from computations import *
from intersection import sphere_line_intersection
import glob, os, os.path
from safeFiles import *
from runfiles import *
#############################################################
# Initial restrictions on obstacles
numObj = 10
minRadius = 0.5
maxRadius = 5
maxDis= 10
area = 10
#Safety factor for intersection between obstalces
SF1 = 1.5


# Set boundaries velocity and heading
maxVel = 20
maxHeading = np.pi*2
SF2 = 1.5 # safety factor

# Set eye parameters:
fov_x  = np.pi #half FOV along azimuth in radians
fov_y = np.pi/4 # half FOV along elevation in radians
res_x = 360 #amount of pixels along azimuth
res_y = 30# amount of pixels along elevation
############################################################################
############################################################################


num_random_param = 10
pathFiles = "/home/atilla/Documents/DeepLearning/Test/"
group_name = 'trial_'
# Do one sequence
## 1 Create Obstacles

num_scenes = 20
file_name = 'VersionSmall_'

start_time = time.time()
#########################################################################################
for i in range(num_scenes):
	#test =  'mytestfile2.hdf5'
	test = file_name+time.strftime('%F-%T')+'.hdf5'

	create_samples(maxDis, maxRadius, numObj, SF1, fov_x, fov_y, res_x, res_y,
					pathFiles, test, maxVel, maxHeading, area, SF2, num_random_param, group_name)

############################################################################
end_time = time.time()-start_time
print('finished', end_time)

############################################################################
f = h5py.File(test)
listf = [key for key in f.keys() if group_name in key]

#plt.show()

###########################################################################
# obstacles = rand_obstacle(maxDis,maxRadius, numObj, SF1)
# u, a, b , layout= unit_vectors(fov_x, fov_y, res_x, res_y)
# ############################################################################

# files= find_files(pathFiles)

# #print(obstacles.shape)

# save_obstacles(test, files, np.array(obstacles),res_x, res_y, numObj, fov_x, fov_y )

# ###########################################################################
# samples = rand_samples(maxVel, maxHeading, obstacles, area, SF2, numSamples)

# #print( len(samples))


# ###########################################################################
# 	# velocity = rand_velocity(maxVel)
# 	# heading = rand_heading(maxHeading)
# 	# position = rand_position(obstacles, area, SF2)
# ################################################################################

# num=1
# for sam in samples:
# 	position = sam[0]
# 	velocity = sam[1]
# 	heading = sam[2]
# 	obstacles_rot = object_rotation(heading, obstacles)
# 	velocity_rot = velocity_rotation(heading, velocity)
# 	intersection_points = intersection_obstacles(u, obstacles_rot, position )
# 	ofx, ofy = optic_flow(u, intersection_points, a,b, velocity_rot)
# 	#print(ofx.shape)
# 	heading_new = delanauy_heading(position, obstacles, heading)
# 	save_results(num, test, ofx, ofy, heading_new,position, velocity,heading )
# 	num=num+1



# plt.quiver(layout[:,0], layout[:,1], 
#            ofx, ofy, scale =5)
# plt.show()



# xOF = [f[x]['ofx'][:] for x in listf]
# print(listf,np.array(xOF).shape, np.array(xOF[1]).shape)

#### cuidado q trayout nombre lo tienes que camvbiar en varios sitios
#####  need to store layout as well i suppose

####### SANITY CHECK ######

# print(obstacles, obstacles_rot.shape, velocity, velocity_rot,u.shape , a.shape, len(intersection_points), 
# 	len(ofx), len(ofy),
# 			heading_new)



# display_vectors(u.T, u.T)
#display_vectors(u.T, b.T)
#plt.show()

# plt.quiver(layout[:,0], layout[:,1], 
#            ofx, ofy, scale =5)
# plt.show()

# plt.plot(np.array(intersection_points))
# plt.show()
# plt.close()