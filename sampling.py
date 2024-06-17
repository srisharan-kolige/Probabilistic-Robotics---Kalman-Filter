import math
import numpy as np
import matplotlib.pyplot as plt

xmean = 8.37563452
xvar = 6.70050761
xstn = math.sqrt(xvar)
nx = 50
samp = np.random.normal(loc=xmean, scale=xstn, size=nx)
y = np.zeros(nx)

ax = plt.subplot(111)
ax.set_xlim(0, 16)
ax.set_ylim(-1, 1)
ax.axhline(y=0, color='k', lw=0.5)
plt.plot(samp, y, '.')
plt.show()