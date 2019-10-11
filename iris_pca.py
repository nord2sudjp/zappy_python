from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np

# データセット読み込み
iris = sns.load_dataset('iris')

# データ前処理
X_iris = iris.drop('species', axis=1)

# PCA
pca = PCA(n_components=2)     
pca.fit(X_iris)               
pca_transformed = pca.transform(X_iris)

print(pca_transformed)

# データマージ
iris["PCA1"] = pca_transformed[:, 0]
iris["PCA2"] = pca_transformed[:, 1]

# プロット
species = y_test.unique()
print(species) #['setosa' 'versicolor' 'virginica']
colors = ['navy', 'turquoise', 'darkorange']

for color,label in zip(colors, species):
        plt.scatter(iris[iris["species"]==label]["PCA1"],iris[iris["species"]=="setosa"]["PCA2"], color=color, label=label)
plt.title("PCA for IRIS")
plt.legend(loc="best")