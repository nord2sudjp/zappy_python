from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np

# �f�[�^�Z�b�g�ǂݍ���
iris = sns.load_dataset('iris')

# �f�[�^�O����
X_iris = iris.drop('species', axis=1)

# PCA
pca = PCA(n_components=2)     
pca.fit(X_iris)               
pca_transformed = pca.transform(X_iris)

print(pca_transformed)

# �f�[�^�}�[�W
iris["PCA1"] = pca_transformed[:, 0]
iris["PCA2"] = pca_transformed[:, 1]

# �v���b�g
species = y_test.unique()
print(species) #['setosa' 'versicolor' 'virginica']
colors = ['navy', 'turquoise', 'darkorange']

for color,label in zip(colors, species):
        plt.scatter(iris[iris["species"]==label]["PCA1"],iris[iris["species"]=="setosa"]["PCA2"], color=color, label=label)
plt.title("PCA for IRIS")
plt.legend(loc="best")