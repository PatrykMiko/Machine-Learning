#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import urllib.request
import tarfile
import gzip
import shutil

data_dir = "data"
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
tgz_path = os.path.join(data_dir, "housing.tgz")
csv_path = os.path.join(data_dir, "housing.csv")
gz_path = os.path.join(data_dir, "housing.csv.gz")

# (odpowiednik: mkdir data)
os.makedirs(data_dir, exist_ok=True)

# (odpowiednik: wget ...)
urllib.request.urlretrieve(url, tgz_path)

# (odpowiednik: tar xfz housing.tgz)
with tarfile.open(tgz_path, "r:gz") as tar:
    tar.extractall(path=data_dir)

# (odpowiednik: gzip housing.csv)
with open(csv_path, 'rb') as f_in:
    with gzip.open(gz_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Polecenie 'gzip' w systemie Linux domyślnie usuwa oryginalny plik po kompresji
os.remove(csv_path) 

#(odpowiednik: rm housing.tgz)
os.remove(tgz_path)

#(odpowiednik: zcat data/housing.csv.gz | head -4)
with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
    for i in range(4):
        print(f.readline().strip())


# In[2]:


import pandas as pd
df = pd.read_csv('data/housing.csv.gz')


# In[3]:


df.head()
df.info()


# In[4]:


df['ocean_proximity'].value_counts()
df['ocean_proximity'].describe()


# In[5]:


import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(20,15))
plt.savefig("obraz1.png")
plt.show()


# In[6]:


df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1, figsize=(7,4))
plt.savefig("obraz2.png")
plt.show()


# In[7]:


df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, figsize=(7,3), colorbar=True, 
        s=df["population"]/100, label="population", c="median_house_value", cmap=plt.get_cmap("jet"))
plt.savefig("obraz3.png")
plt.show()


# In[8]:


df.corr(numeric_only=True)["median_house_value"] \
  .sort_values(ascending=False) \
  .reset_index() \
  .rename(columns={"index": "atrybut", "median_house_value": "wspolczynnik_korelacji"}) \
  .to_csv("korelacja.csv", index=False)


# In[9]:


import seaborn as sns
sns.pairplot(df)


# In[10]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
len(train_set), len(test_set)


# In[12]:


train_set.to_pickle("train_set.pkl")
test_set.to_pickle("test_set.pkl")


# In[ ]:




