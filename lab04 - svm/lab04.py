#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets

data_breast_cancer = datasets.load_breast_cancer(as_frame=True)
print(data_breast_cancer['DESCR'])


# In[2]:


data_iris = datasets.load_iris(as_frame=True)
print(data_iris['DESCR'])


# In[3]:


from sklearn.model_selection import train_test_split

X = data_breast_cancer.data[['mean area', 'mean smoothness']]
y = data_breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[4]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pickle

svm_unscaled = LinearSVC(loss="hinge")
svm_unscaled.fit(X_train, y_train)

svm_scaled = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(loss="hinge"))]) 
svm_scaled.fit(X_train, y_train)

acc_train_unscaled = accuracy_score(y_train, svm_unscaled.predict(X_train))
print(f"Accuracy zbiory uczącego bez skalowania {acc_train_unscaled}")
acc_test_unscaled = accuracy_score(y_test, svm_unscaled.predict(X_test))
print(f"Accuracy zbiory testującego bez skalowania {acc_test_unscaled}")
acc_train_scaled = accuracy_score(y_train, svm_scaled.predict(X_train))
print(f"Accuracy zbiory uczącego ze skalowaniem {acc_train_scaled}")
acc_test_scaled = accuracy_score(y_test, svm_scaled.predict(X_test))
print(f"Accuracy zbiory testującego ze skalowaniem {acc_test_scaled}")

class_acc = [acc_train_unscaled, acc_test_unscaled, acc_train_scaled, acc_test_scaled]

with open('bc_acc.pkl', 'wb') as f:
    pickle.dump(class_acc, f)


# In[5]:


import numpy as np

X = data_iris.data[['petal length (cm)', 'petal width (cm)']]
y = (data_iris.target == 2).astype(np.int8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[6]:


svm_unscaled = LinearSVC(loss="hinge")
svm_unscaled.fit(X_train, y_train)

svm_scaled = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(loss="hinge"))]) 
svm_scaled.fit(X_train, y_train)

acc_train_unscaled = accuracy_score(y_train, svm_unscaled.predict(X_train))
print(f"Accuracy zbiory uczącego bez skalowania {acc_train_unscaled}")
acc_test_unscaled = accuracy_score(y_test, svm_unscaled.predict(X_test))
print(f"Accuracy zbiory testującego bez skalowania {acc_test_unscaled}")
acc_train_scaled = accuracy_score(y_train, svm_scaled.predict(X_train))
print(f"Accuracy zbiory uczącego ze skalowaniem {acc_train_scaled}")
acc_test_scaled = accuracy_score(y_test, svm_scaled.predict(X_test))
print(f"Accuracy zbiory testującego ze skalowaniem {acc_test_scaled}")

class_acc = [acc_train_unscaled, acc_test_unscaled, acc_train_scaled, acc_test_scaled]

with open('iris_acc.pkl', 'wb') as f:
    pickle.dump(class_acc, f)


# In[7]:


import pandas as pd
size = 900
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})
df.plot.scatter(x='x',y='y')


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)


# In[9]:


from sklearn.svm import LinearSVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

linear_svr_poly = Pipeline([("poly_features", PolynomialFeatures(degree=4, include_bias=False)), ("scaler", StandardScaler()), ("linear_svr", LinearSVR())])
linear_svr_poly.fit(X_train, y_train)

mse_train_linear = mean_squared_error(y_train, linear_svr_poly.predict(X_train))
mse_test_linear = mean_squared_error(y_test, linear_svr_poly.predict(X_test))

print(f"MSE zbiór uczący: {mse_train_linear}")
print(f"MSE zbiór testowy: {mse_test_linear}")


# In[10]:


from sklearn.svm import SVR

svr_poly = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="poly", degree=4))])
svr_poly.fit(X_train, y_train)

mse_train_svr = mean_squared_error(y_train, svr_poly.predict(X_train))
mse_test_svr = mean_squared_error(y_test, svr_poly.predict(X_test))

print(f"MSE zbiór uczący: {mse_train_svr}")
print(f"MSE zbiór testowy: {mse_test_svr}")


# In[11]:


from sklearn.model_selection import GridSearchCV

X = X.reshape(-1, 1)

param_grid = {
    "svr__C" : [0.1, 1, 10],
    "svr__coef0" : [0.1, 1, 10]
}
search = GridSearchCV(svr_poly, param_grid, scoring="neg_mean_squared_error", n_jobs=-1)
search.fit(X, y)
print(f"Najlepsze parametry {search.best_params_}")

best_C = search.best_params_["svr__C"]
best_coef0 = search.best_params_["svr__coef0"]


# In[12]:


from sklearn.svm import SVR

svr_poly_best = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="poly", degree=4, C=best_C, coef0=best_coef0))])
svr_poly_best.fit(X_train, y_train)

mse_train_svr_best = mean_squared_error(y_train, svr_poly_best.predict(X_train))
mse_test_svr_best = mean_squared_error(y_test, svr_poly_best.predict(X_test))

print(f"MSE zbiór uczący: {mse_train_svr_best}")
print(f"MSE zbiór testowy: {mse_test_svr_best}")


# In[13]:


reg_mse = [mse_train_linear, mse_test_linear, mse_train_svr_best, mse_test_svr_best]

with open('reg_mse.pkl', 'wb') as f:
    pickle.dump(reg_mse, f)

