#!/usr/bin/env python
# coding: utf-8

# In[1]:



import matplotlib.pyplot as plt
import numpy as np
import scipy

from datafold import DMDControl, TSCDataFrame

from pydmd import DMDc


# In[2]:


def create_system(n, m):
    A = scipy.linalg.helmert(n, True)
    B = np.random.rand(n, n)-.5
    x0 = np.array([0.25]*n)
    u = np.random.rand(n, m-1)-.5
    snapshots = [x0]
    for i in range(m-1):
        snapshots.append(A.dot(snapshots[i])+B.dot(u[:, i]))
    snapshots = np.array(snapshots).T
    return {'snapshots': snapshots, 'u': u, 'B': B, 'A': A}

def solve_system(A, B, s0, u):

    snapshots = [s0]

    m = u.shape[1]

    for i in range(m):
        snapshots.append(A.dot(snapshots[i]) + B.dot(u[:, i]))

    return np.array(snapshots).T


# In[3]:


s = create_system(25, 10)
print(s['snapshots'].shape)


# In[4]:


X = TSCDataFrame.from_array(s['snapshots'].T)
U = TSCDataFrame.from_array(s['u'].T)


# In[5]:


X.shape


# In[6]:


dmdc = DMDc(svd_rank=-1)
dmdc.fit(s['snapshots'], s['u'])





dmdcdf = DMDControl()
dmdcdf.fit(X=X, U=U);

# In[8]:


plt.figure(figsize=(16,6))

original = s['snapshots'].real
pydmd = dmdc.reconstructed_data().real
datafold = dmdcdf.reconstruct(X, U=U).to_numpy().T

plt.subplot(131)
plt.title('Original system')
plt.pcolor(original)
plt.colorbar()

plt.subplot(132)
plt.title('Reconstructed system PyDMD')
plt.pcolor(pydmd)
plt.colorbar()

plt.subplot(133)
plt.title('Reconstructed system datafold')
plt.pcolor(datafold)
plt.colorbar()

err_pydmd = np.linalg.norm(np.sum((pydmd - original), axis=1))
err_datafold = np.linalg.norm(np.sum((datafold - original), axis=1))

print(err_pydmd)
print(err_datafold)

# In[11]:


new_u = np.exp(s['u'])
U_new = TSCDataFrame.from_array(new_u.T)

plt.figure(figsize=(8,6))

pydmd = dmdc.reconstructed_data().real
datafold = dmdcdf.reconstruct(X, U=U).to_numpy().T

plt.subplot(121)
plt.pcolor(dmdc.reconstructed_data(new_u).real)
plt.colorbar()

plt.subplot(122)
plt.pcolor(dmdcdf.reconstruct(X=X, U=U_new).T)
plt.colorbar()


# In[15]:


dmdc.dmd_time['dt'] = .5
new_u = np.random.rand(s['u'].shape[0], dmdc.dynamics.shape[1]-1)

U_new = TSCDataFrame.from_array(new_u.T, time_values=np.arange(0, 0.5*new_u.shape[1], 0.5))

plt.figure(figsize=(8,6))
plt.subplot(121)
plt.title("PyDMD: new time sampling")
plt.pcolor(dmdc.reconstructed_data(new_u).real)
plt.colorbar()

plt.subplot(122)
plt.title("datafold: new time sampling")
plt.pcolor(dmdcdf.reconstruct(X=X, U=U_new).T)
plt.colorbar()
plt.show()


# In[ ]:




