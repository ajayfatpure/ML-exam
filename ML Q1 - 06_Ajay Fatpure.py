#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[1]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#importing libraries


# In[2]:


df=pd.read_csv(r'C:\Users\Hp\Desktop\ML Exam\data_1.csv')
#reading file(dataset)


# In[3]:


df.shape
#row and columns


# In[4]:


df.columns
#columns names


# In[5]:


df.info()
#discriptions of data


# In[6]:


df.head(10)
#top 10 rows of data set


# In[7]:


df.describe(include='all')
#discription of all columns 


# # Data cleaning

# In[8]:


df.isnull().sum()
# checking null values
#no null values found


# In[9]:


df.isnull().sum()/len(df.index)


# In[10]:


sns.pairplot(df)
# pairplot of all columns with each other


# In[11]:


#plt.figure(figsize=(8,8))
sns.heatmap(df.corr(),annot=True)
#heat map for better visualisation of correlatin betwwwn all columns


# In[12]:


plt.figure(figsize=(8,8))
sns.countplot('Year',hue='Fuel_Type',data=df)
#count of fuel_type car with year
# it is found that petrol car is more in number and has positive relation with year


# In[13]:


sns.countplot(x='Owner', data= df)


# In[14]:


df.Seller_Type.value_counts()
#dealer are more in number in Seller_Type


# In[15]:


sns.boxplot(df.Selling_Price)
#some 


# In[16]:


sns.boxplot(df.Kms_Driven)


# In[17]:


plt.figure(figsize=(15,10))
sns.countplot(df['Year'], hue = df.Seller_Type)
plt.xticks(rotation =90)
plt.show()


# In[18]:


df.Car_Name.value_counts()


# In[19]:


plt.figure(figsize=(10,5))
sns.countplot(df['Fuel_Type'], hue = df.Seller_Type)
plt.xticks(rotation =90)
plt.show()
#petrol car more in number and dealer are more in it..


# In[20]:


plt.figure(figsize=(10,5))
sns.countplot(df['Seller_Type'], hue = df.Fuel_Type)
plt.xticks(rotation =90)
plt.show()


# In[21]:


plt.figure(figsize=(10,5))
sns.countplot(df['Transmission'], hue = df.Seller_Type)
plt.xticks(rotation =90)
plt.show()
#manual car more demanding specally from dealer side


# In[22]:


df.Owner.value_counts()
#what kind of values are present in owner with thier count


# In[23]:


df['Owner'] = df['Owner'].replace(3,1)
#3 is comming only once so we can replace it with 1


# In[24]:


df.Owner.value_counts()


# In[25]:


plt.figure(figsize = (10,5))
sns.heatmap(df.corr(), annot = True)
plt.show()


# In[26]:


df['CurrentYear']=2022
df['Number_of_year']=df['CurrentYear']-df['Year']
#adding a current year and then calulation number of year has occurs since car is bought


# In[27]:


df = df.drop(['Present_Price'], axis =1)
#droping Present_Price column


# In[28]:


df = pd.get_dummies(df.drop('Car_Name', axis =1),drop_first=True)
df
#creating dummies for categorial columns and droping car_name column simuntanesouly


# In[29]:


df=df.drop(['Year','CurrentYear'], axis=1)
#dropping year and CurrentYear year column as Number of year column is more suitable in thier place


# In[30]:


df


# In[31]:


# Spliting the data

x = df.drop('Selling_Price', axis=1).values
y = df['Selling_Price'].values


# In[32]:


x


# In[33]:


y


# In[34]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
#splitting the data into x_train, x_test, y_train, y_test for model training


# In[35]:


#Scaling our independant variables using MinMax scaler to bring all the values in same range
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# In[36]:


# Model Building
from sklearn.ensemble import  RandomForestRegressor
model = RandomForestRegressor()
model1 = model.fit(x_train, y_train)
model1.score(x_test, y_test)


# ### we got 80.21% accuracy 

# In[37]:


y_pred=model1.predict(x_test)
y_pred
#feeding X_test into model and calculating its response in y_pred


# In[ ]:





# In[38]:


#Finding Errors
from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[39]:


print('MAE: ' ,mean_absolute_error(y_test,y_pred))
print('MSE: ',mean_squared_error(y_test,y_pred))
print('RMSE: ', np.sqrt(mean_squared_error(y_test,y_pred)))
#all  errors with thier values


# In[ ]:





# In[ ]:




