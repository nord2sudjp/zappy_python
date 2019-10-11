import seaborn as sns
iris = sns.load_dataset('iris')
iris.head()


X_iris = iris.drop('species', axis=1)
X_iris.head()

y_iris = iris['species']
y_iris.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris,random_state=1)

X_train.shape
# (112, 4)

X_test.shape
# (38, 4)

X_train.shape


from sklearn import svm
clf=svm.LinearSVC(max_iter=10000)
clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_model = clf.predict(X_test)
ac_score = accuracy_score(y_test, y_model)
print("accuracy score: ", ac_score)

