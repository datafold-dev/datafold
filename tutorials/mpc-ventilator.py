#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

from scipy.optimize import minimize
from math import sin


# In[2]:


plt.rcParams["figure.figsize"] = (20,10)


# In[3]:


# arm radius
r_m = 0.5 # m
m_kg = 4
# arm inertia (dominated by point load at end)
theta = m_kg * r_m ** 2
Fg = m_kg * 9.81
# damper
k = 0.1

## disturbances / plant model
r_real_m = r_m + 0.1  # m
k_real = k + 0.1
angle_noise_std = np.deg2rad(2) # rad
angle_vel_noise_std = np.deg2rad(6) # rad/s
motor_noise_std = 2  # N

def system_dynamics(x, u):  
    x1, x2 = x
    
    dx_dt = [
        x2,
        r_m / theta * (Fg * sin(x1) - u) - k * x2
    ]
    
    return np.array(dx_dt)

def system_dynamics_plant(x, u):  
    x1, x2 = x
    
    # IMU noise
    x1 += np.random.normal(0, angle_noise_std)
    x2 += np.random.normal(0, angle_vel_noise_std)
    # Motor noise
    u += np.random.normal(0, motor_noise_std)
    
    dx_dt = [
        x2,
        (r_real_m) / theta * (Fg * sin(x1) - u) - (k_real) * x2
    ]
    
    return np.array(dx_dt)

assert np.allclose(system_dynamics([np.deg2rad(90), 0], Fg), [0,0])


# In[5]:

x0 = np.array([np.deg2rad(90), 0])

duration_s = 180

# this cant be too small for it to work
delT = 0.01
N = 4
n_steps = int(duration_s / delT)

opti_options = {"maxfun": 3}


# In[6]:


x = x0
xs = [x]
us = [0]

u_traj_init = np.ones(N)
u = None

for i in range(n_steps):
    if u is not None:
        # warm start using state similarity
        # initial value embedding
        u_traj_init = u_opt
    
    Q = [
        [4,0],
        [0,1]
    ]
    
    # this has to be very small because we weight N vs radians
    # i.e. if x and u are weighted the same, a relatively small
    # deviation in u of 1N is weighted the same as ~60°
    R = 0.00001
    
    def obj(u_traj):
        xk = x
        
        # single-shooting
        cost = 0
        for k in range(N):
            uk = u_traj[k]
            
            cost += 1/2 * (xk.T @ Q @ xk + uk * R * uk)
            
            xk1 = xk + delT * system_dynamics(xk, uk)
            xk = xk1
        
        return cost
    
    def jac(u_traj):
        return R * u_traj

    res = minimize(obj, u_traj_init, options=opti_options, method="L-BFGS-B")
    
    if not res.success and not "EVALUATIONS EXCEEDS LIMIT" in str(res.message):
        print("optimization failed ", i)
        print(res)
        break
    
    # use first control input
    u_opt = res.x
    u = u_opt[0]
    
    x = x + delT * system_dynamics_plant(x, u)
    
    if i % 100 == 0:
        print(f"In step {i}/{n_steps} u={u:.2f}N, x={x.T}")
    
    us.append(u)
    xs.append(x)

xs = np.array(xs)
us = np.array(us)


# ## Video Rendering

# In[8]:


from matplotlib import animation, rc
from IPython.display import HTML

fig, axes = plt.subplots(ncols=3)

axes[0].set_xlim((-2,2))
axes[0].set_ylim((-2,2))
axes[0].set_aspect("equal")
axes[0].set_facecolor("#729fcf")
axes[0].tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
axes[0].set_title("Visual Simulation")
axes[0].set_anchor("N")
plt.figtext(0.05,0,f"""
Zero-Mean Normal Noise (Std)
x1 (Angle): {np.rad2deg(angle_noise_std):.0f} °
x2 (Angle Velocity): {np.rad2deg(angle_vel_noise_std):.2f} °/s
u (Motor Force): {motor_noise_std:.0f} N
""", fontsize=12, transform=axes[0].transAxes)

axes[1].plot(np.rad2deg(xs[:,0]))
axes[1].set_title("Angle $x_1$ [°]")
oi = axes[1].axvline(0, color="r")

axes[2].plot(us)
axes[2].set_title("Control $u$ [N]")
oi2 = axes[2].axvline(0, color="r")

# center bearing
axes[0].add_artist(plt.Circle([0,0], radius=0.02, color="k"))

line_arm = axes[0].plot([],[], "k")
line_force = axes[0].plot([],[], "r")

def animate(k):
    state = xs[k]
    phi, dphi = state

    r = 0.5
    F = us[k]

    x = r * np.sin(phi)
    y = r * np.cos(phi)

    x_force = x + F * np.cos(phi)
    y_force = y - F * np.sin(phi)

    line_arm[0].set_data([0,x],[0,y])
    line_force[0].set_data([x, x_force],[y, y_force])
    oi.set_xdata(k)
    oi2.set_xdata(k)
    
    return (oi, oi2)

step = round(n_steps/100)
anim = animation.FuncAnimation(fig, animate, frames=np.arange(len(xs)//step) * step, interval=1000//30, blit=True)


# In[8]:


anim.save('ventilator-mpc.mp4', codec='h264')
