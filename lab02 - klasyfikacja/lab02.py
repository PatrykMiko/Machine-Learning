#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)


# In[2]:


import numpy as np
print((np.array(mnist.data.loc[42]).reshape(28,28)>0).astype(int))


# In[3]:


from matplotlib import pyplot as plt

pixels = np.array(mnist.data.loc[42]).reshape(28,28)
plt.imshow(pixels, cmap='gray')
plt.show()


# In[4]:


X, y = mnist["data"], mnist["target"].astype(np.uint8)
y_sorted = y.sort_values()
X_sorted = X.reindex(y_sorted.index)
X_train, X_test = X_sorted[:56000], X_sorted[56000:]
y_train, y_test = y_sorted[:56000], y_sorted[56000:]
print("Klasy w y_train:", np.sort(y_train.unique()))
print("Klasy w y_test:", np.sort(y_test.unique()))


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("Klasy w losowym y_train:", np.sort(y_train.unique()))
print("Klasy w losowym y_test:", np.sort(y_test.unique()))


# In[6]:


from sklearn.linear_model import SGDClassifier
y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_0)
y_train_pred = sgd_clf.predict(X_train)
y_test_pred = sgd_clf.predict(X_test)
acc_train = sum(y_train_pred == y_train_0)/len(y_train_0)
acc_test = sum(y_test_pred == y_test_0)/len(y_test_0)
accuracy = [float(acc_train), float(acc_test)]
print(accuracy)
import pickle
with open('sgd_acc.pkl', 'wb') as file:
    pickle.dump(accuracy, file)


# In[7]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(sgd_clf, X_train, y_train_0, cv=3, scoring="accuracy", n_jobs=-1)
score = np.array(score)
score = score.reshape(3,)
print(score)
with open('sgd_cva.pkl', 'wb') as file:
    pickle.dump(score, file)


# In[8]:


from sklearn.metrics import confusion_matrix
sgd_m_clf = SGDClassifier(random_state = 42, n_jobs=-1)
sgd_m_clf.fit(X_train, y_train)
y_test_pred = sgd_m_clf.predict(X_test)
sgd_cmx = confusion_matrix(y_test, y_test_pred)
print(sgd_cmx)
with open('sgd_cmx.pkl', 'wb') as file:
    pickle.dump(sgd_cmx, file)


# In[9]:


from sklearn.svm import SVC
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
y_test_pred = svm_clf.predict(X_test)
svc_cmx = confusion_matrix(y_test, y_test_pred)
print(svc_cmx)
with open('svc_cmx.pkl', 'wb') as file:
    pickle.dump(svc_cmx, file)


# In[ ]:




