import os

def home_dir():
    homedir = '/home/zzbenjamin94/Desktop/Astronomy/Research/DES_Galaxy_Cluster/'
    if not os.path.exists(homedir):
        raise Exception('something is very wrong: %s does not exist'%homedir)
    return homedir

def tools_home_dir():
    toolshomedir = os.path.join(home_dir(), 'tools/')
    return toolshomedir

def data_home_dir():
    datahomedir = os.path.join(home_dir(), 'data/')
    if not os.path.exists(datahomedir):
        os.makedirs(datahomedir)
    return datahomedir

def bigdata_home_dir():
    #uses the /data dir in your office workstation with ~800 GB of space
    bigdata_dir = '/data/DES/Cluster/'
    if not os.path.exists(bigdata_dir):
        os.makedirs(bigdata_dir)
    return bigdata_dir

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


    
if __name__ == '__main__':
    homedir = home_dir()
    toolsdir = tools_home_dir()
    sdssfilterdir = sdss_filter_dir()
    datahomedir = data_home_dir()

    print "home_dir:", homedir
    print "tools_home_dir:", toolsdir
    print "data home directory:", datahomedir
    
    import py_compile
    py_compile.compile(os.path.join(tools_home_dir(),'setup/setup.py'))
