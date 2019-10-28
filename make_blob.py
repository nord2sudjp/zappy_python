import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets.samples_generator import make_blobs

centers_l = [0.01, 0.1, 0.5, 1, 10, 1000]

fig, ax = plt.subplots(6, figsize=(10,20))

for i in range(len(centers_l)):
    X, y = make_blobs(n_samples=150, centers=3,random_state=0, cluster_std=centers_l[i])
    ax[i].scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');
    
plt.show()