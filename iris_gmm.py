import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np

# データセット読み込み
iris = sns.load_dataset('iris')
iris.head()

# データ前処理
X_iris = iris.drop('species', axis=1)
X_iris.head()

from sklearn.mixture import GaussianMixture as gm 
model = gm(n_components=3,covariance_type='full')
model.fit(X_iris)
y_gmm = model.predict(X_iris)
print(y_gmm)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)     
pca.fit(X_iris)               
pca_transformed = pca.transform(X_iris)
# データマージ
iris["PCA1"] = pca_transformed[:, 0]
iris["PCA2"] = pca_transformed[:, 1]

sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False)
sns.lmplot("PCA1", "PCA2", hue='cluster', data=iris, fit_reg=False)

sns.lmplot("PCA1", "PCA2", hue='species', col="cluster", data=iris, fit_reg=False);