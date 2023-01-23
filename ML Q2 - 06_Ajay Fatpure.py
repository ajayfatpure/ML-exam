#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r'C:\Users\Hp\Desktop\ML Exam\Question2.csv')


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.shape
#105 rows and 3 columns are present in dataset


# In[6]:


df.nunique()
#number of unique values for all columns


# In[7]:


df.describe()


# In[8]:


sns.pairplot(df)
# pairplot of all columns with each other


# In[9]:


sns.heatmap(df.corr(),annot=True)
#heat map for better visualisation of correlatin betwwwn all columns


# In[11]:



#Spliting our data into dependant and independant features
X=df.drop('price',axis=1)
Y=df['price']
     


# In[12]:


X.head()


# In[13]:


Y.shape


# In[14]:



#splitting dataset into training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=29)


# In[15]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[16]:



model=LinearRegression()
model.fit(x_train,y_train)
model.score(x_test,y_test)


# In[17]:



poly_reg = PolynomialFeatures(degree=4)
x_poly_train = poly_reg.fit_transform(X)
     

     


# In[18]:



poly_reg = PolynomialFeatures(degree=4)
x_poly_train = poly_reg.fit_transform(x_train)
x_poly_test = poly_reg.fit_transform(x_test)


# In[19]:


x_poly_test


# In[ ]:




