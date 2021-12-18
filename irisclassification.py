#!/usr/bin/env python
# coding: utf-8

# In[5]:


##loading the datasets


# In[23]:


from sklearn.datasets import load_iris
iris_dataset = load_iris()
import pandas as pd


# In[2]:


print("\nKeys of iris dataset: \n{}".format(iris_dataset.keys()))


# In[6]:


print(iris_dataset['DESCR'][:200]+"\n")


# In[8]:


print(iris_dataset['target_names'])
print(iris_dataset['feature_names'])


# In[12]:


print(type(iris_dataset['data']))
print(iris_dataset['data'].shape)


# In[13]:


print("first five columns of iris dataset\n")
print(iris_dataset['data'][:5])


# In[15]:


print("Type of target:\n{}".format(type(iris_dataset['target'])))


# In[17]:


print("Shape of target:\n{}".format(iris_dataset['target'].shape))


# In[18]:


print("target: \n{}".format(iris_dataset['target']))
#The meanings of the numbers are given by the iris['target_names'] array:
#0 means setosa, 1 means versicolor, and 2 means virginica.


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
 iris_dataset['data'], iris_dataset['target'], random_state=0)


# In[20]:


print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))


# In[21]:


print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))


# In[27]:


# creating dataframe from data in X_train
# labeling the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# creating a scatter matrix from the dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
 hist_kwds={'bins': 20}, s=60, alpha=.8)

