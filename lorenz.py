import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

x1, y1, z1 = 0, 1, 10

x2, y2, z2 = 0, 1.0001, 10  

rho, sigma, beta = 28, 10, 8/3


t0 = 0
tf = 100
dt = 0.008
t = np.arange(t0, tf + dt, dt)
n = len(t)

def EDOs(t, r):
    x, y, z = r
    return np.array([sigma*(y - x),
                     rho*x - y - x*z,
                     x*y - beta*z])


def RK4(t, r, f, dt):
    k1 = dt*f(t, r)
    k2 = dt*f(t + dt/2, r + k1/2)
    k3 = dt*f(t + dt/2, r + k2/2)
    k4 = dt*f(t + dt, r + k3)
    return r + (k1 + 2*k2 + 2*k3 + k4)/6

r1 = [x1, y1, z1]
evol1 = np.zeros((n, 3))
evol1[0,0], evol1[0,1], evol1[0,2] = r1[0], r1[1], r1[2]

for i in range(n - 1):
    evol1[i + 1] = RK4(t[i], [evol1[i,0], evol1[i,1], evol1[i,2]], EDOs, dt)


r2 = [x2, y2, z2]
evol2 = np.zeros((n, 3))
evol2[0,0], evol2[0,1], evol2[0,2] = r2[0], r2[1], r2[2]

for i in range(n - 1):
    evol2[i + 1] = RK4(t[i], [evol2[i,0], evol2[i,1], evol2[i,2]], EDOs, dt)


distances = np.sqrt(np.sum((evol1 - evol2)**2, axis=1))

fig = plt.figure('Атрактор Лоренца: ефект метелика', facecolor='k', figsize=(10, 9))
fig.tight_layout()


ax = fig.add_subplot(111, projection='3d')
ax.set(facecolor='k')
ax.set_axis_off()

def update(i):
    i_max = min(i, n-1)

    ax.clear()
    ax.set(facecolor='k')
    ax.set_axis_off()
    
    ax.view_init(-6, -56 + i / 2)
    
    ax.plot(evol1[:i_max,0], evol1[:i_max,1], evol1[:i_max,2], color='red', lw=0.9)
    
    ax.plot(evol2[:i_max,0], evol2[:i_max,1], evol2[:i_max,2], color='cyan', lw=0.9)
    
    if i_max > 0:
        ax.scatter(evol1[i_max-1,0], evol1[i_max-1,1], evol1[i_max-1,2], color='red', s=20)
        ax.scatter(evol2[i_max-1,0], evol2[i_max-1,1], evol2[i_max-1,2], color='cyan', s=20)
    
    current_distance = distances[min(i_max, len(distances)-1)]
    info_text = f"Початкова різниця: 0.0001   Поточна відстань: {current_distance:.4f}"
    ax.text2D(0.5, 0.02, info_text, transform=ax.transAxes, color='white',
              horizontalalignment='center', bbox=dict(facecolor='black', alpha=0.5))

ani = animation.FuncAnimation(fig, update, frames=np.arange(15000), interval=2, repeat=False)

plt.show()