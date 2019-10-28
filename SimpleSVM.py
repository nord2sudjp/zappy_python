#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(n_samples=150, centers=2,random_state=0, cluster_std=0.6)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');
plt.show()


# In[2]:


xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA',
                     alpha=0.4)

plt.xlim(-1, 3.5);


# In[3]:


from sklearn.svm import SVC # "Support vector classifier"
model = SVC(kernel='linear', C=1E10)
model.fit(X, y)


# In[4]:


def plot_svc_decision_function(model, ax=None, plot_support=True):
   if ax is None:
       ax = plt.gca()
   xlim = ax.get_xlim()
   ylim = ax.get_ylim()

   # create grid to evaluate model
   x = np.linspace(xlim[0], xlim[1], 30)
   y = np.linspace(ylim[0], ylim[1], 30)
   Y, X = np.meshgrid(y, x)
   xy = np.vstack([X.ravel(), Y.ravel()]).T
   P = model.decision_function(xy).reshape(X.shape)
       
       
   # plot decision boundary and margins
   ax.contour(X, Y, P, colors='k',
                     levels=[-1, 0, 1], alpha=0.5,
                     linestyles=['--', '-', '--'])

   # plot support vectors
   if plot_support:
       ax.scatter(model.support_vectors_[:, 0],
                         model.support_vectors_[:, 1],
                         s=300, linewidth=1, facecolors='none');
       ax.set_xlim(xlim)
       ax.set_ylim(ylim)


# In[5]:


plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model);


# In[6]:


x = np.linspace(0, 1, 30)
y = np.linspace(-1, 1, 30)
print(x)
print(y)


# In[7]:


Y, X = np.meshgrid(y, x)
print(Y)


# In[ ]:





# In[ ]:




