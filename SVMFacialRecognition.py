#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)


# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
ig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[],
    xlabel=faces.target_names[faces.target[i]])
# 1348人の写真があり、各写真は3000ピクセル(62x47)


# In[5]:


from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline

pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
# 150次元まで特徴量を削減
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)


# In[8]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target,random_state=42)


# In[11]:


from sklearn.model_selection import GridSearchCV
param_grid = {'svc__C': [1, 5, 10, 50],'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)

get_ipython().run_line_magic('time', 'grid.fit(Xtrain, ytrain)')
print(grid.best_params_)


# In[12]:


model = grid.best_estimator_
yfit = model.predict(Xtest)


# In[14]:


fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                           color='black' if yfit[i] == ytest[i] else 'red') # 名前が異なる場合には赤字にする
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14);


# In[15]:


from sklearn.metrics import classification_report
print(classification_report(ytest, yfit,target_names=faces.target_names))
# Ariel SharonとRumsfeldが小さい→なぜ?
# 誰と間違っているのか


# In[18]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                    xticklabels=faces.target_names,
                    yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');


# In[ ]:




