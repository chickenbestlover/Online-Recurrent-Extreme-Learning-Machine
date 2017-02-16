from suite import MySuite
from matplotlib import pyplot as plt
from numpy import *

mysuite = MySuite()
exps = mysuite.get_exps()

print 'lowest offset for experiment normal:', mysuite.get_value(exps[0], 0, 'offset', 'min')
print 'lowest offset for experiment highstd:', mysuite.get_value(exps[1], 0, 'offset', 'min')

# plot results
plt.plot(mysuite.get_history(exps[0], 0, 'offset'), linewidth=2, color='blue', label='std=1')
plt.plot(mysuite.get_history(exps[1], 0, 'offset'), linewidth=2, color='red', label='std=5')
plt.xlabel('iterations')
plt.ylabel('offset')
plt.title('sample mean offset to true mean')
plt.legend()
plt.show()