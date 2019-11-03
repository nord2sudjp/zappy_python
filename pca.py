#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# In[5]:


rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
#
# rand : 0～1の範囲の一様分布 からランダム値を取得
# randn : 標準正規分布（ガウス分布）からランダム値を取得
# np.dotで内積を取得。この時点で2x200
# 転置して200x2列のデータが出来上がる。
# 
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal'); # x軸, y軸ラベルの範囲を一致させる


# In[6]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)


# In[7]:


print(pca.components_) # 主成分


# In[8]:


print(pca.explained_variance_) # 寄与率


# In[9]:


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                           linewidth=2,
                           shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal');


# In[10]:


from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape


# In[11]:


pca = PCA(2)  # project from 64 to 2 dimensions
projected = pca.fit_transform(digits.data)
print(digits.data.shape)
print(projected.shape)


# In[12]:


plt.scatter(projected[:, 0], projected[:, 1], c=digits.target, edgecolor='none', alpha=0.5,cmap=plt.cm.get_cmap('Spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();


# In[13]:


pca = PCA().fit(digits.data)
print(pca.explained_variance_ratio_) 


# In[14]:


plt.plot(np.cumsum(pca.explained_variance_ratio_)) 
# 主成分の寄与率を積みあげて表表示する。
# 10成分を使えば、80%以上のデータを復元できる。
# 40でほぼすべてのデータを復元している→おそらく右と左の空白部と考えられる。
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# In[15]:


def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4),
                                     subplot_kw={'xticks':[], 'yticks':[]},
                                     gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
                          cmap='binary', interpolation='nearest',
                          clim=(0, 16))
plot_digits(digits.data)


# In[16]:


np.random.seed(42)
noisy = np.random.normal(digits.data, 4)
plot_digits(noisy)


# In[17]:


pca = PCA(0.50).fit(noisy)
pca.n_components_


# In[18]:


components = pca.transform(noisy)
filtered = pca.inverse_transform(components)
plot_digits(filtered)


# In[19]:


print(components)


# In[20]:


from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)


# 

# In[21]:


from sklearn.decomposition import PCA as RandomizedPCA
pca = RandomizedPCA(150)
pca.fit(faces.data)


# In[22]:


fig, axes = plt.subplots(3, 8, figsize=(9, 4),
                                 subplot_kw={'xticks':[], 'yticks':[]},
                                 gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(62, 47), cmap='bone') # PCAで次元削減した結果を表示する


# In[23]:


plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
# この結果によると80次元あればほぼ特徴量の90%を復元できる。
# 150だとおよそ95%の情報量を復元している。


# In[25]:


# PCAでイメージを取り扱うときのお約束
pca = RandomizedPCA(150).fit(faces.data)
components = pca.transform(faces.data)
projected = pca.inverse_transform(components)


# In[29]:


# Plot the results
fig, ax = plt.subplots(2, 20, figsize=(20, 2.5),
                               subplot_kw={'xticks':[], 'yticks':[]},
                               gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i in range(20):
    ax[0, i].imshow(faces.data[i].reshape(62, 47), cmap='binary_r')
    ax[1, i].imshow(projected[i].reshape(62, 47), cmap='binary_r')

ax[0, 0].set_ylabel('full-dim\ninput')
ax[1, 0].set_ylabel('150-dim\nreconstruction');


# In[23]:


# How to use numpy.random.normal.
s = [1,2,3]
print(np.random.normal(s,0.1))
# array([0.92786152, 1.92813434, 3.11822813])
print(np.random.normal(s,4))
# array([6.19127741, 0.9960995 , 6.13342335])

