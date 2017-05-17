
import numpy as np
from random import randint
from numpy import linalg as LA
from scipy.spatial import ConvexHull
import matplotlib.path as mltp
import matplotlib.pylab as plt
from intersection import sphere_line_intersection 
from scipy.spatial import Delaunay
from graphics import *

def rand_obstacle(maxDis,maxRadius, numObj, SF1):
	s =[]
	s.append(np.array([randint(-maxDis, maxDis), randint(-maxDis, maxDis),0, np.random.sample()*maxRadius]))

	stopObj =False
	i = 1
	while stopObj == False: 
	    intersect = False
	    s_2 = np.array([randint(-maxDis, maxDis), randint(-maxDis, maxDis),0, np.random.sample()*maxRadius])
	    for num in s:
	        if LA.norm([num[0:2]-s_2[0:2]])< (num[3]+s_2[3])+SF1:
	            intersect = True

	    if intersect == False:
	        s.append(s_2)
	        i = i+1
	        if i>numObj-1:
	            stopObj = True
	obstacle =s
	return obstacle

def rand_velocity(maxVel):
	velocity = [randint(-maxVel, maxVel),randint(-maxVel,maxVel),0]
	velocity = np.divide(velocity, LA.norm(velocity))
	return velocity

def rand_heading(maxHeading):
	heading = np.random.sample(1)*maxHeading
	return heading

def rand_position(s, area, SF2):
	# Convec hull of points
	s1 =np.array(s)
	s1 = s1[:,0:2]
	hull = ConvexHull(s1)
	path =  mltp.Path(s1[hull.vertices])
	# Randomize position, check whether it lies within convex hull of obstacles, and whether it does not
	#intersect with objects
	stopPos = False
	while stopPos == False:
	    intersect = True
	    pos = np.array([randint(0,area),randint(0,area),0])
	    if path.contains_point(pos) ==True:
	       
	        for num in s:                
	            if LA.norm([num[0:2] -pos[0:2]])< (num[3]+SF2) and intersect== True:
	                position = pos
	                intersect = False
	     
	        if intersect == True:
	            position = pos
	            stopPos = True
	return position


def object_rotation(heading, obstacle ):
	obstacles = np.array(obstacle) 
	alpha= -heading[0] # negative rotation about z axis
	rot = np.array([np.cos(alpha), -np.sin(alpha), 0,np.sin(alpha), np.cos(alpha),0, 0,0,1])
	rot = rot.reshape(3,3)
	mat_rot= np.array([[np.dot(rot, i)] for i in obstacles[:,0:3]])
	[r,d,c] = mat_rot.shape
	obstacles_rot = mat_rot.reshape(r,c)
	obstacles[:,0:3]=obstacles_rot
	return obstacles

def velocity_rotation(heading, velocity):
	#  Velocity rotation
	alpha= -heading[0]
	rot = np.array([np.cos(alpha), -np.sin(alpha), 0,np.sin(alpha), np.cos(alpha),0, 0,0,1])
	rot = rot.reshape(3,3)
	vel_rot = np.dot(rot, velocity)
	return vel_rot

def unit_vectors(fov_x, fov_y, res_x, res_y):
	phis   = np.linspace(-fov_x, fov_x, res_x)
	thetas = np.linspace(-fov_y, fov_y, res_y)
	layout = np.array([[phi, theta] 
                   for theta in thetas
                   for phi in phis])
	# Compute unit vectors for each pixel
	u = np.array([[ np.cos(theta) * np.cos(phi),
                np.cos(theta) * np.sin(phi),
                np.sin(theta)] for [phi, theta] in layout])
	# Compute a: lateral vector of the base (a, b, u)
	z = np.array([0, 0, 1])
	a = np.cross(z[np.newaxis], u)
	a /= np.linalg.norm(a, axis=1)[:,np.newaxis]
	# Compute b: vertical vector of the base (a, b, u)
	b = np.cross(u, a)
	return u,a, b, layout 

def display_vectors(X, V):
    fig, axarr = plt.subplots(nrows=1, 
                              ncols=2, 
                              figsize=[10,5])
    # Top view
    for [ax, title, xlabel, ylabel, xx, yy, uu, vv] in zip(axarr, 
                                                   ['Top view', 'Side view'],
                                                   ['X', 'X'],
                                                   ['Y', 'Z'],
                                                   [X[0], X[0]],
                                                   [X[1], X[2]],
                                                   [V[0], V[0]],
                                                   [V[1], V[2]]):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.quiver(xx, yy, uu, vv)

    fig.tight_layout()


def intersection_obstacles(unit, obstacle, position, ):

	D = np.array([[sphere_line_intersection(position, u,  obs[0:3], obs[3]) for u in unit] for obs in obstacle]).T
	Df = []
	for d in D:
	    if np.sum(d) > 0:
	        Df.append(d[np.min(np.nonzero(d))])
	    else:
	        Df.append(0)
	return Df

def optic_flow(unit, D_vec,a,b,v):
	mat = np.column_stack([unit, D_vec])
	D_vec = np.array(D_vec)
	p = []
	#print(mat, 'mat')
	for i in mat:
	    if i[3]!= 0:       
	        p.append( - (v - np.dot(i[0:3], v[np.newaxis,:].T) * i[0:3]) / (i[3])) 
	        #print(p)
	    if i[3]== 0:
	        p.append([0,0,0])
	p = np.array(p)
	# # Project 3d optic flow vector on (a, b)
	ofx = np.sum(p*a, axis=1)
	ofy = np.sum(p*b, axis=1)
	return ofx, ofy

def delanauy_heading(position, obstacles):
	# Obstacles, radius, and position
	points = np.array(obstacles)[:,0:2]
	p = position[0:2]
	r = np.array(obstacles) [:,3]
	# Determine Delanauy triangles, simplices and respective points
	tri = Delaunay(points)
	median = points[tri.simplices]
	# Determine in which triangle point is located
	triangle_index = tri.find_simplex(p)
	# Extract properties of this triangle, radius, points
	points_index = points[tri.simplices[triangle_index]]
	radius_index = r[tri.simplices[triangle_index]]
	# Add last first element again for loop
	rads= np.hstack([radius_index, radius_index[0]])
	base = np.vstack([points_index, points_index[0]])
	# Estimate median of triangle in which the drone is located
	# Median with raidus taken into acount, and median without radius taken into account 
	median =np.array([ [
                     base[i], #0 first point [x,y]
                     base[i+1], #1 next point [x,y]
                    LA.norm(   #2 gap with radius scalar
                        np.add(base[i],
                            np.multiply(rads[i],
                                    np.divide(([base[i+1]-base[i]]),
                                             LA.norm([base[i+1]-base[i]]))))
                            -
                        np.add(base[i+1],
                        np.multiply(-rads[i+1],
                                    np.divide(([base[i+1]-base[i]]),
                                             LA.norm([base[i+1]-base[i]]))))),
                    LA.norm([base[i]- base[i+1]]), #3 gap without radius, scalar
                    (np.add(base[i], # 4 midpoint with radius
                            np.multiply(rads[i],
                                    np.divide(([base[i+1]-base[i]]),
                                             LA.norm([base[i+1]-base[i]]))))
                                +
                        np.add(base[i+1],
                        np.multiply(-rads[i+1],
                                    np.divide(([base[i+1]-base[i]]),
                                             LA.norm([base[i+1]-base[i]])))))/2,
                    (base[i]+base[i+1])/2 #5 midpoint without radius
            
                  
                   ]

                  for i in range(len(points_index)) ])
	return median

def one_gap(median, heading,p):
	winner_index =median[np.argmax(median[:,2])]
	p = p[0:2]
	dx_dy = winner_index[4]-p
	y1 = dx_dy[0]
	psi = np.arctan2(y1[1],y1[0])
	if psi<0:
	    psi = 2*np.pi+psi
	new_heading = psi-heading 

	while new_heading > 2*np.pi:
		new_heading = new_heading-2*np.pi
	while new_heading < -2*np.pi:
		new_heading = new_heading+2*np.pi

	return new_heading

def angle_gaps(median, heading,p):
	p = p[0:2]
	gaps = [med-p for med in median[:,4]]
	psi =np.array([np.arctan2(g[0][1],g[0][0]) for g in gaps])
	for a,p in enumerate(psi):
	     if p<0:
	        psi[a]=2*np.pi+p
	gaps_heading = np.subtract(psi, heading)
	return gaps_heading

def convert_heading(new_heading):
	#ew_heading=np.array([5, 0 , 2])
	for a,h in enumerate(new_heading):
		if h>=np.pi:
			new_heading[a] = h -2*np.pi
		if h<= -np.pi:
			new_heading[a] = h +2*np.pi
		#else:
		#	new_heading[a] = h
	return new_heading
