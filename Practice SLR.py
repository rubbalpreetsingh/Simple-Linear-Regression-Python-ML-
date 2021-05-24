#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[24]:


data = pd.read_csv('salary_data.csv')


# In[25]:


data.head()


# In[26]:


real_x = data.iloc[:,[0]].values
real_y = data.iloc[:,[1]].values


# In[27]:


train_x,test_x,train_y,test_y = train_test_split(real_x,real_y,test_size=0.3,random_state=0)


# In[28]:


slr = LinearRegression()


# In[29]:


slr.fit(train_x,train_y)


# In[30]:


y_pred = slr.predict(test_x)


# In[33]:


y_pred[3]


# In[34]:


test_y[3]


# In[37]:


slr.intercept_


# In[39]:


slr.coef_


# In[ ]:





# In[51]:


plt.scatter(train_x,train_y,color="green")
plt.plot(train_x,slr.predict(train_x))
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.title("For Training Dataset")


# In[55]:


plt.scatter(test_x,test_y)
plt.plot(train_x,slr.predict(train_x))
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.title("For Testing Dataset")


# In[ ]:




