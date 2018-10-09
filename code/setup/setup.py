import os

def notes_home_dir():
    noteshomedir = '/global/u1/z/zzhang13/DES_Galaxy_Cluster/'
    if not os.path.exists(noteshomedir):
        raise Exception('something is very wrong: %s does not exist'%noteshomedir)
    return noteshomedir

def code_home_dir():
    codehomedir = os.path.join(notes_home_dir(), 'code/')
    return codehomedir

def data_home_dir():
    datahomedir = os.path.join(notes_home_dir(), 'data/')
    if not os.path.exists(datahomedir):
        os.makedirs(datahomedir)
    return datahomedir

def image_home_dir():
    imghomedir = os.path.join(notes_home_dir(), 'img/')
    if not os.path.exists(imghomedir):
        os.makedirs(imghomedir)
    return imghomedir
    
def sdss_filter_dir():
    filterdir = os.path.join(data_home_dir(), 'sdss_filters/')
    if not os.path.exists(filterdir):
        os.makedirs(filterdir)
    return filterdir

#for  0.34 < z < 0.90
def buzzard_particles_highz_dir():
    buzz_highzdir = '/global/cscratch1/sd/zzhang13/BCC/Chinchilla/Herd/Chinchilla-0/Lb2600/pixlc/'
    if not os.path.exists(buzz_highzdir):
	raise Exception('something is very wrong %s does not exist'%buzz_highzdir)
    return buzz_highzdir
    
#for z < 0.34
def buzzard_particles_lowz_dir():
    buzz_lowzdir = '/project/projectdirs/des/jderose/BCC/Chinchilla/Herd/Chinchilla-0/Lb1050/output/pixlc/'
    if not os.path.exists(buzz_lowzdir):
	raise Exception('something is very wrong %s does not exist'%buzz_lowzdir)
    return buzz_lowzdir
    
if __name__ == '__main__':
    sdssfilterdir = sdss_filter_dir()
    datahomedir = data_home_dir()
    imghomedir = image_home_dir()

    print "data home directory:", datahomedir
    print "SDSS filter directory:", sdssfilterdir
    
    import py_compile
    py_compile.compile(os.path.join(code_home_dir(),'setup/setup.py'))
