from sklearn.datasets import load_digits
digits = load_digits()

# PCA
from sklearn.manifold import Isomap
iso = Isomap(n_components=2) # 次元縮約 n_components=2を指定にする。
iso.fit(digits.data)
data_projected = iso.transform(digits.data) # 次元縮約
data_projected.shape # 縮約されたデータ
# (1797,2)


# Visualize PCA result
import matplotlib.pyplot as plt
import matplotlib.cm as cm
%matplotlib inline
plt.style.use('seaborn-darkgrid')
plt.scatter(data_projected[:, 0], data_projected[:, 1], c=digits.target,alpha=1, cmap = cm.Accent)
plt.colorbar(ticks=range(10))


# split into train and test
from sklearn.model_selection import train_test_split
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# Build NaiveBayes model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
y_model = model.predict(X_test)

# Review model
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_model)
# 0.8333333333333334