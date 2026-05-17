#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
import numpy as np
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]


# In[2]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

kmeans_sil = []
for k in range(6, 13):
    kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(X)
    if k == 10:
        y_pred10 = labels
    sil = silhouette_score(X, labels)
    kmeans_sil.append(float(sil))
    print(f"Wskaźnik sylwetkowy dla k={k}: {sil}")


# In[3]:


import pickle

with open('kmeans_sil.pkl', 'wb') as f:
    pickle.dump(kmeans_sil, f)


# In[4]:


from sklearn.metrics import confusion_matrix

cmx = confusion_matrix(y, y_pred10)
print(cmx)


# In[5]:


argmax_indices = np.argmax(cmx, axis=1)
argmax_indices_list = sorted(list(set(int(x) for x in argmax_indices)))

with open('kmeans_argmax.pkl', 'wb') as f:
    pickle.dump(argmax_indices_list, f)


# In[6]:


from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=2, metric='euclidean', n_jobs=-1)
nn.fit(X)

distances, indices = nn.kneighbors(X[:300])

min_distances = []
for dists in distances:
    non_zero = dists[dists > 0]
    if len(non_zero) > 0:
        min_distances.append(float(non_zero[0]))

top_10 = sorted(min_distances)[:10]
print(f"10 najmniejszych odległości: {top_10}")
with open('dist.pkl', 'wb') as f:
    pickle.dump(top_10, f)


# In[7]:


from sklearn.cluster import DBSCAN

s = np.mean(min_distances)
eps_values = [s, s + 0.25 * s, s + 0.50 * s, s + 0.75 * s]

unique_labels_counts = []

for eps_val in eps_values:
    dbscan = DBSCAN(eps=eps_val, min_samples=5)
    dbscan.fit(X)
    num_unique_labels = len(set(dbscan.labels_))
    unique_labels_counts.append(int(num_unique_labels))
    print(f"Dla eps={eps_val:.4f} znaleziono {num_unique_labels} unikalnych etykiet.")

with open('dbscan_len.pkl', 'wb') as f:
    pickle.dump(unique_labels_counts, f)


# In[8]:


from collections import Counter

dbscan = DBSCAN(eps=1523.4446, min_samples=5, n_jobs=-1)
dbscan.fit(X)

cluster_distribution = Counter(dbscan.labels_)
for label, count in sorted(cluster_distribution.items()):
    if label == -1:
        print(f" -> Szum (etykieta -1): {count} instancji")
    else:
        print(f" -> Klaster {label}: {count} instancji")


# In[ ]:




