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
