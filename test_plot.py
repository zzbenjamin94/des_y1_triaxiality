import numpy as np

# import pyplot and set some parameters to make plots prettier
import matplotlib
print "Default backend is: %s" % (matplotlib.get_backend(),)
matplotlib.use('TkAgg')
from code.plot_utils import plot_pretty
#plot_pretty()
import matplotlib.pyplot as plt
print "Current backend is: %s" % (matplotlib.get_backend(),)
x=np.linspace(0,3*np.pi,500)
plt.plot(x, np.sin(x**2))
plt.savefig('test.png')
plt.show()

