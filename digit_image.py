# イメージデータの読み込みおよび確認
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)
print(digits.data)

# view first image
import matplotlib.pyplot as plt 
%matplotlib inline
plt.gray()
plt.matshow(digits.images[0]) 
plt.show()

print(digits.target)

# view first 100 images
fig, axes = plt.subplots(10, 10, figsize=(8, 8),subplot_kw={'xticks':[], 'yticks':[]})
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0, 0, str(digits.target[i]))