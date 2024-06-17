import math
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


def Kalman_filter(mu_prev, sigma_prev):
    mu_bar = A_t.dot(mu_prev)
    sigma_bar = A_t.dot(sigma_prev).dot(A_t.transpose()) + R_t
    z_t = np.array(np.random.normal(loc=mu_bar[0], scale=np.array([[math.sqrt(sigma_bar[0][0]), math.sqrt(sigma_bar[1][1])]])))
    K_t = sigma_bar.dot(C_t.transpose()).dot(inv(C_t.dot(sigma_bar).dot(C_t.transpose()) + Q_t))
    mu = mu_bar + K_t.dot(z_t - C_t.dot(mu_bar))
    sigma = (np.identity(2) - K_t.dot(C_t)).dot(sigma_bar)
    return mu_bar, mu, sigma


mu_w = 0
var_w = 1
A_t = np.array([[1, 1], [0, 1]])
G = np.array([[0.5], [1]])
R_t = var_w * G.dot(G.transpose())
C_t = np.array([[1, 0]])
Q_t = np.array([[8]])
mu_0 = np.array([[0, 0], [0, 0]])
sigma_0 = np.array([[0, 0], [0, 0]])
belx = []
bely = []
posx = []
posy = []

for t in range(31):
    if t == 0:
        mu_bel = mu_0
        mu_t = mu_0
        sigma_t = sigma_0
        z = np.array([[0, 0]])
    else:
        mu_bel, mu_t, sigma_t = Kalman_filter(mu_t, sigma_t)
    belx.append(mu_bel[0][0])
    posx.append(mu_t[0][0])
    bely.append(mu_bel[0][1])
    posy.append(mu_t[0][1])

ax = plt.subplot(111)
ax.axvline(x=0, color='k', lw=0.3)
ax.axhline(y=0, color='k', lw=0.3)
ax.set_xlabel('x position')
ax.set_ylabel('y position')
plt.plot(belx, bely, 'c.-')
plt.plot(posx, posy, 'r.-')
plt.title("Predicted and True X-Y Position of the Sail Boat")
plt.show()