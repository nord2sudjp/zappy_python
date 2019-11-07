#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np


# In[2]:


from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4,
                              cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50);


# In[3]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)


# In[4]:


plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


# In[32]:


from sklearn.metrics import pairwise_distances_argmin
n_clusters = 4
rseed = 2
# 1. Randomly choose clusters
# 配列をランダムに入れ替えて最初の４つの項目を取得し、それを最初のデータポイントとする。
rng = np.random.RandomState(rseed) # Random
i = rng.permutation(X.shape[0])[:n_clusters]
centers = X[i]

while True:
    # 2a. Assign labels based on closest center
    # もっとも近いCentersにラベル付けする 0～3
    labels = pairwise_distances_argmin(X, centers)

    # 2b. Find new centers from means of points
    # 各クラスタについて平均をとり新しい中心とする
    new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
 
    # 2c. Check for convergence
    # 変化しなければおしまい　
    if np.all(centers == new_centers):
        break
    centers = new_centers

plt.scatter(X[:, 0], X[:, 1], c=labels,
                   s=50, cmap='viridis');


# In[33]:


from sklearn.datasets import make_moons
X, y = make_moons(200, noise=.05, random_state=0)


# In[34]:


labels = KMeans(2, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels,s=50, cmap='viridis');


# In[35]:


from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=2,
                                   affinity='nearest_neighbors',
                                   assign_labels='kmeans')
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels,
                    s=50, cmap='viridis');


# In[36]:


from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape


# In[37]:


kmeans = KMeans(n_clusters=10, random_state=0)
# 数字に関係なくクラスタに分類
clusters = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape
# (10, 64)
# クラスタ中心が64次元→数字イメージに変換できる


# In[39]:


fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)


# In[50]:


from scipy.stats import mode

# 値が0の10次元のリスト
labels = np.zeros_like(clusters)

for i in range(10):
    mask = (clusters == i) 
    # 0～9のクラスタについてTrue, Falseリスト
    labels[mask] = mode(digits.target[mask])[0]
    # digits.target[mask]で該当マスクのデータをすべて取得
    # modeで最頻値を取得
    # リスト0番目に最頻値が入っている
    # labelを正しい数値で付け替える


# In[56]:


from sklearn.metrics import accuracy_score
accuracy_score(digits.target, labels)


# In[57]:


from sklearn.metrics import confusion_matrix
mat = confusion_matrix(digits.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                    xticklabels=digits.target_names,
                    yticklabels=digits.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[61]:


from sklearn.datasets import load_sample_image
china = load_sample_image("china.jpg")
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(china);


# In[59]:


china.shape


# In[60]:


data = china / 255.0 # use 0...1 scale
# RGBだから0～255
data = data.reshape(427 * 640, 3)
data.shape
# この形なら3次元（2次元＋色）で見ることができる


# In[64]:


def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data

    # choose a random subset
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    fig.suptitle(title, size=20);


# In[65]:


plot_pixels(data, title='Input color space: 16 million possible colors')


# In[66]:


from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(16)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

plot_pixels(data, colors=new_colors,
                    title="Reduced color space: 16 colors")


# In[67]:


china_recolored = new_colors.reshape(china.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 6),
          subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(china)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(china_recolored)
ax[1].set_title('16-color Image', size=16);


# In[ ]:




