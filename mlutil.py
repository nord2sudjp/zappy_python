# coding: utf-8
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns; sns.set()

from sklearn.datasets import load_digits
from sklearn.datasets.samples_generator import make_blobs

# Name: show_digits
# Desc: digitsのイメージ表示
# Usage : 
#     show_digits(n=50)
def show_digits(n=100):
    digits = load_digits()
    # view first 100 images
    fig, axes = plt.subplots(10, int(n/10)+1, figsize=(8, 8),subplot_kw={'xticks':[], 'yticks':[]})
    for i, ax in enumerate(axes.flat):
        ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
        ax.text(0, 0, str(digits.target[i]))

# Name: make_shift_data
# Desc: 平均値をシフトしたデータの生成
# Usage : 
#     x = make_shift_data(1000)
#     hist = plt.hist(x, bins=30, normed=True)
def make_shift_data(N, f=0.3, avg=5, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N) # 1000個のデータを生成
    x[int(f * N):] += 5 # 7割のデータについては平均を5つずらす
    return x

# Name: plot_pixels
# Desc: 画像の色分析
# Usage:
#    china = load_sample_image("china.jpg")
#    data = china / 255.0 # use 0...1 scale
#    data = data.reshape(427 * 640, 3)
#    plot_pixels(data, title='Input color space: 16 million possible colors')
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

# Name: plot_blobs
# Desc: blobを表示
# Usage:
#    plot_blobs()
def plot_blobs(centers=3, dia=[0.01, 0.1, 0.5, 1, 10, 1000]): 
    dia = [0.01, 0.1, 0.5, 1, 10, 1000]
    fig, ax = plt.subplots(len(dia), figsize=(10,20))
    for i in range(len(dia)):
        X, y = make_blobs(n_samples=150, centers=centers,random_state=0, cluster_std=dia[i])
        ax[i].scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');
    
    plt.show()

# Name: make_randomxy
# desc: dim次元データを生成
# Usage
def make_randomxy(n=200, dim=2):
    rng = np.random.RandomState(1)
    X = np.dot(rng.rand(dim, dim), rng.randn(dim, n)).T
    #
    # rand : 0～1の範囲の一様分布 からランダム値を取得
    # randn : 標準正規分布（ガウス分布）からランダム値を取得
    # np.dotで内積を取得。この時点で2x200
    # 転置して200x2列のデータが出来上がる。
    # 
    plt.scatter(X[:, 0], X[:, 1])
    plt.axis('equal'); # x軸, y軸ラベルの範囲を一致させる
    return X
make_randomxy()

# Name: make_noise
# desc: データにノイズを入れる
# Usage:
#   x = 10 * rng.rand(200)
#   y = model(x)
def make_noise(x, sigma=0.3):
    fast_oscillation = np.sin(5 * x) # 長期的傾向 xの5倍
    slow_oscillation = np.sin(0.5 * x) # 短期的傾向
    noise = sigma * rng.randn(len(x)) # ノイズを加える
    return slow_oscillation + fast_oscillation + noise



# Name: 
# desc: 
# Usage:
