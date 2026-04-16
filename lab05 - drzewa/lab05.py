#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)
print(data_breast_cancer['DESCR'])


# In[2]:


import numpy as np
import pandas as pd
size = 900
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})
df.plot.scatter(x='x',y='y')


# In[3]:


from sklearn.model_selection import train_test_split

X = data_breast_cancer.data[['mean texture', 'mean symmetry']]
y = data_breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[4]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

for i in range(2, 11):
    tree_clf = DecisionTreeClassifier(max_depth=i)
    tree_clf.fit(X_train, y_train)

    f1_train = f1_score(y_train, tree_clf.predict(X_train))
    print(f"F1-score dla zbioru uczącego (max_depth={i}) {f1_train}")
    f1_test = f1_score(y_test, tree_clf.predict(X_test))
    print(f"F1-score dla zbioru testującego (max_depth={i}) {f1_test} \n")



# In[5]:


from sklearn.tree import export_graphviz
import graphviz

tree_clf = DecisionTreeClassifier(max_depth=3)
tree_clf.fit(X_train, y_train)

str_dot = export_graphviz(
    tree_clf,
    out_file=None, 
    feature_names=data_breast_cancer.feature_names[[1, 8]],
    class_names=[str(num) + ", " + name for num, name in zip(set(data_breast_cancer.target), data_breast_cancer.target_names)],
    rounded=True,
    filled=True
)

graph = graphviz.Source(str_dot)
graph.render("bc", format="png", cleanup=True)


# In[6]:


from sklearn.metrics import accuracy_score
import pickle

f1_train = f1_score(y_train, tree_clf.predict(X_train))
f1_test = f1_score(y_test, tree_clf.predict(X_test))
acc_train = accuracy_score(y_train, tree_clf.predict(X_train))
acc_test = accuracy_score(y_test, tree_clf.predict(X_test))

f1acc_tree = [3, f1_train, f1_test, acc_train, acc_test]

with open('f1acc_tree.pkl', 'wb') as f:
    pickle.dump(f1acc_tree, f)


# In[7]:


size = 900
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)


# In[8]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

for i in range(2, 11):
    tree_reg = DecisionTreeRegressor(max_depth=i)
    tree_reg.fit(X_train, y_train)

    mse_train = mean_squared_error(y_train, tree_reg.predict(X_train))
    print(f"MSE dla zbioru uczącego (max_depth={i}) {mse_train}")
    mse_test = mean_squared_error(y_test, tree_reg.predict(X_test))
    print(f"MSE dla zbioru testującego (max_depth={i}) {mse_test} \n")



# In[9]:


import matplotlib.pyplot as plt

tree_reg = DecisionTreeRegressor(max_depth=4)
tree_reg.fit(X_train, y_train)

plt.clf()
plt.scatter(X, y, c="blue")
sort_index = np.argsort(X_test[:, 0])
plt.plot(X_test[sort_index], tree_reg.predict(X_test[sort_index]), c="red")


# In[10]:


str_dot = export_graphviz(
    tree_reg,
    out_file=None, 
    feature_names=['x'],
    rounded=True,
    filled=True
)

graph = graphviz.Source(str_dot)
graph.render("reg", format="png", cleanup=True)


# In[11]:


mse_train = mean_squared_error(y_train, tree_reg.predict(X_train))
mse_test = mean_squared_error(y_test, tree_reg.predict(X_test))

mse_tree = [4, mse_train, mse_test]

with open('mse_tree.pkl', 'wb') as f:
    pickle.dump(mse_tree, f)

