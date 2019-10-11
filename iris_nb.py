# iris ÉfÅ[É^ÇÃì«Ç›çûÇ›
from sklearn.datasets import load_iris 
iris = load_iris()
print(iris.target_names)

X_iris = iris.data
y_iris = iris.target

# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris,random_state=1)

# Naive Bayes model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB() 
model.fit(X_train, y_train) 
y_model = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_model)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_model)
print(cm)

# visualize confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

plt.figure(figsize = (6,4))
sns.heatmap(cm, annot=True)
plt.show()