#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import scipy
from pydmd import DMDc

from datafold import DMDControl, TSCDataFrame

# In[2]:


def create_system(n, m):
    A = scipy.linalg.helmert(n, True)
    B = np.random.rand(n, n) - 0.5
    x0 = np.array([0.25] * n)
    u = np.random.randn(n, m - 1) - 0.5
    snapshots = [x0]
    for i in range(m - 1):
        snapshots.append(A.dot(snapshots[i]) + B.dot(u[:, i]))
    snapshots = np.array(snapshots).T
    return {"snapshots": snapshots, "u": u, "B": B, "A": A}


def solve_system(A, B, s0, u, dt=1):
    snapshots = [s0]
    m = u.shape[1]

    if dt != 1:
        A = scipy.linalg.fractional_matrix_power(A.copy(), dt)
        B = scipy.linalg.fractional_matrix_power(B.copy(), dt)

    for i in range(m):
        snapshots.append(A.dot(snapshots[i]) + B.dot(u[:, i]))

    return np.array(snapshots).T


# In[3]:


s = create_system(25, 10)
print(s["snapshots"].shape)


# In[4]:


X = TSCDataFrame.from_array(s["snapshots"].T)
U = TSCDataFrame.from_array(s["u"].T)


# In[5]:


X.shape


# In[6]:


dmdc = DMDc(svd_rank=-1)
dmdc.fit(s["snapshots"], s["u"])


dmdcdf = DMDControl()
dmdcdf.fit(X=X, U=U)

# In[8]:


plt.figure(figsize=(16, 6))

original = s["snapshots"].real
pydmd = dmdc.reconstructed_data().real
datafold = dmdcdf.reconstruct(X, U=U).to_numpy().T

plt.subplot(131)
plt.title("Original system")
plt.pcolor(original)
plt.colorbar()

plt.subplot(132)
plt.title("Reconstructed system PyDMD")
plt.pcolor(pydmd)
plt.colorbar()

plt.subplot(133)
plt.title("Reconstructed system datafold")
plt.pcolor(datafold.real)
plt.colorbar()

err_pydmd = np.max(np.sum((pydmd - original), axis=1))
err_datafold = np.max(np.sum((datafold - original), axis=1))

print(err_pydmd)
print(err_datafold)

# In[11]:


new_u = np.exp(s["u"])

U_new = TSCDataFrame.from_array(new_u.T)

plt.figure(figsize=(8, 6))

original = solve_system(s["A"], B=s["B"], s0=s["snapshots"][:, 0], u=new_u)
pydmd = dmdc.reconstructed_data().real
datafold = dmdcdf.reconstruct(X, U=U).to_numpy().T

plt.subplot(131)
plt.pcolor(original)
plt.colorbar()

plt.subplot(132)
plt.pcolor(dmdc.reconstructed_data(new_u).real)
plt.colorbar()

plt.subplot(133)
plt.pcolor(dmdcdf.reconstruct(X=X, U=U_new).to_numpy().real.T)
plt.colorbar()

err_pydmd = np.max(np.sum((pydmd - original), axis=1))
err_datafold = np.max(np.sum((datafold - original), axis=1))

print(err_pydmd)
print(err_datafold)

# In[15]:


dmdc.dmd_time["dt"] = 0.5
new_u = np.random.rand(s["u"].shape[0], dmdc.dynamics.shape[1] - 1)

U_new = TSCDataFrame.from_array(
    new_u.T, time_values=np.arange(0, 0.5 * new_u.shape[1], 0.5)
)

original = solve_system(s["A"], B=s["B"], s0=s["snapshots"][:, 0], u=new_u, dt=0.5)
pydmd = dmdc.reconstructed_data(new_u)
datafold = dmdcdf.reconstruct(X=X, U=U_new).to_numpy().T


plt.figure(figsize=(8, 6))

plt.subplot(231)
plt.title("Original")
plt.pcolor(original.real)
plt.colorbar()

plt.subplot(234)
plt.pcolor(original.imag)
plt.colorbar()

plt.subplot(232)
plt.title("PyDMD: new time sampling")

plt.pcolor(pydmd.real)
plt.colorbar()

plt.subplot(235)
plt.pcolor(pydmd.imag)
plt.colorbar()

plt.subplot(233)
plt.title("datafold: new time sampling")
plt.pcolor(datafold.real)
plt.colorbar()

plt.subplot(236)
plt.pcolor(datafold.imag)
plt.colorbar()

plt.figure(figsize=(8, 6))


dmdc_A = np.linalg.multi_dot(
    [dmdc.modes, np.diag(dmdc.eigs), np.linalg.pinv(dmdc.modes)]
)

plt.subplot(231)
plt.imshow(s["A"])

plt.subplot(234)
plt.imshow(s["B"])

plt.subplot(232)
plt.imshow(dmdc_A.real)

plt.subplot(235)
plt.imshow(dmdc.B)

plt.subplot(233)
plt.imshow(dmdcdf.sys_matrix_)

plt.subplot(236)
plt.imshow(dmdcdf.control_matrix_)


plt.show()
