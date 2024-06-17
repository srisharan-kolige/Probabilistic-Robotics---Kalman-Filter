import math
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


def Kalman_filter(mu_prev, sigma_prev, p_fail):
    mu_bar = A_t.dot(mu_prev)
    sigma_bar = A_t.dot(sigma_prev).dot(A_t.transpose()) + R_t
    if np.random.choice(a=[0, 1], size=None, p=[1-p_fail, p_fail]):
        # randomly generate z_t
        z_t = np.array(np.random.normal(loc=mu_bar[0][0], scale=math.sqrt(sigma_bar[0][0]), size=None))
        K_t = sigma_bar.dot(C_t.transpose()).dot(inv(C_t.dot(sigma_bar).dot(C_t.transpose()) + Q_t))
    else:
        # set K_t to zero (no measurement)
        z_t = np.array([[0]])
        K_t = np.array([[0], [0]])
    mu = mu_bar + K_t.dot(z_t - C_t.dot(mu_bar))
    sigma = (np.identity(2) - K_t.dot(C_t)).dot(sigma_bar)
    return mu_bar, mu, sigma
    


p_gps_fail = float(input("Enter p_gps-fail: "))  # input any float from 0 to 1
mu_w = 0
var_w = 1
A_t = np.array([[1, 1], [0, 1]])
G = np.array([[0.5], [1]])
R_t = var_w * G.dot(G.transpose())
C_t = np.array([[1, 0]])
Q_t = np.array([[8]])
mu_0 = np.array([[0], [0]])
sigma_0 = np.array([[0, 0], [0, 0]])
bel = []
pos = []

for t in range(21):
    if t == 0:
        mu_bel = mu_0
        mu_t = mu_0
        sigma_t = sigma_0
    elif t != 20:
        mu_bel, mu_t, sigma_t = Kalman_filter(mu_t, sigma_t, p_gps_fail)  # measurement acquired according to p_gps_fail
    else:
        mu_bel, mu_t, sigma_t = Kalman_filter(mu_t, sigma_t, 0)  # need measurement at t = 20 to observe error
    bel.append(mu_bel[0][0])
    pos.append(mu_t[0][0])

ax = plt.subplot(111)
ax.set_xlim(-0, 20.5)
plt.xticks(np.arange(0, 21, 1.0))
ax.set_xlabel('t')
ax.set_ylabel('position')
ax.axhline(y=0, color='k', lw=0.3)
plt.plot(list(range(21)), bel, 'c.-')
plt.plot(list(range(21)), pos, 'r.-')
plt.title(r'$p_{gps-fail} = $' + str(p_gps_fail) + '\n' + 'Expected Error: ' + str(round(abs(pos[-1] - bel[-1]), 3)))
plt.show()