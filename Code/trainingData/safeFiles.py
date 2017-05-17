import h5py
from computations import*
from profilestats import profile
import glob, os, os.path

def find_files(pathFiles):
    os.chdir(pathFiles)
    files = [file for file in glob.glob("*.hdf5")]
    return files

def save_obstacles(test, files,obstacles, Resolutionx, Resolutiony, numberObj, FOVx,FOVy, group_name):

    exist = False
    # check whether file already exists, if exist overwrite, else creates a new file
    for s in files:
        if test == s:
            exist = True
            print('exists')
            #f.close()
            f = h5py.File(test)
            if 'obstacles' in f:
                r= obstacles.shape
                f['obstacles'].resize(r)
                f['obstacles'][...] = obstacles
                data= f['obstacles']

            else:    
                data = f.create_dataset('obstacles', data =obstacles, dtype = 'f', chunks=True )

    # if file does not exist create a new file
    if exist == False:
        f =h5py.File(test, 'w')
        data = f.create_dataset('obstacles', data=obstacles,dtype='f', chunks = True)

    # Store parameters
    data.attrs['Resolution']= [Resolutionx, Resolutiony]
    data.attrs['NumberObjects']=numberObj
    data.attrs['FOV']=[FOVx,FOVy] 
    data.attrs['Nomenclature']= group_name   
    f.close()


def rand_samples(maxVel, maxHeading, obstacles, area, SF2, numSamples):
    velocity = [rand_velocity(maxVel) for i in range(numSamples)]
    heading = [rand_heading(maxHeading) for i in range(numSamples)]
    position = [rand_position(obstacles, area, SF2) for i in range(numSamples)]
    samples = np.array([[pos, vel, head] 
                   for pos in position
                   for vel in velocity
                    for head in heading])
    return samples
def save_results(num, test, ofx, ofy, new_heading,position, velocity,heading, group_name, heading_gaps  ):
    f =h5py.File(test)
    name = group_name+str(num)
    if name in f:
        grp = f[name]
    else:
        grp = f.create_group(name)
    if "ofx" in grp:
        r = ofx.shape
        grp['ofx'].resize(r)
        grp['ofx'][...]=ofx
    else:
        grp.create_dataset("ofx",data= ofx,dtype='f', chunks = True)
    if "ofy" in grp:
        r = ofy.shape
        grp['ofy'].resize(r)
        grp['ofy'][...]=ofy
    else:
        grp.create_dataset("ofy",data= ofy, dtype='f', chunks = True)
    if 'new_heading' in grp:
        grp['new_heading'][...]=new_heading
    else:
        grp.create_dataset("new_heading", data =new_heading )
    if 'gaps' in grp:
        grp['gaps'][...]=new_heading
    else:
        grp.create_dataset("gaps", data =heading_gaps )
    grp.attrs['position']=position
    grp.attrs['velocity']= velocity
    grp.attrs['heading']=heading
    f.close()

