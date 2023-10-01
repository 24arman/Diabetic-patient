#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv(r"C:\Users\hplap\Downloads\diabetes.csv")
df.head()


# In[4]:


df['Outcome'].value_counts()
df.isnull().sum()


# # 0 - Non diabetic
# # 1 -  Diabetic

# In[5]:


x = df.drop(columns = 'Outcome',axis =1)
y = df['Outcome']


# In[6]:


df


# In[7]:


s =StandardScaler()


# In[8]:


s.fit(x)


# In[9]:


vm = s.transform(x)
vm


# In[10]:


x = vm
x


# In[11]:


y


# In[12]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)  # Lets split into 80-20 ratio


# In[14]:


x_train.shape


# In[15]:


x_test.shape


# # train the model

# In[17]:


clf = svm.SVC(kernel = 'linear' )
clf.fit(x_train,y_train)

# Lets predict the accuracy
x_train_prediction = clf.predict(x_train)
accuracy_score(x_train_prediction,y_train )


# # Accuracy on test data

# In[20]:


x_test_prediction =clf.predict(x_test)
accuracy_score(x_test_prediction,y_test)



#Lets take sample from the data sets
input_sample = (5,166,72,19,175,22.7,0.6,51)
input_np_array = np.asarray(input_sample)

#reshape the array
input_np_array_reshaped = input_np_array.reshape(1,-1)
syd= s.transform(input_np_array_reshaped)
syd
prediction = clf.predict(syd)
prediction


# In[21]:


if (prediction[0]==0):
    print("person is not diabetic")
else:
    print('Person is diabetic')


# In[ ]:




