#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import accuracy_score
import pickle

# In[7]:


fish = pd.read_csv('Fish.csv')
fish.head()


# In[8]:


fish_x=pd.DataFrame(fish.iloc[:,[False,True,True,True,True,True,True]])
fish_x.head()


# In[9]:


fish_y=pd.DataFrame(fish.iloc[:,[True,False,False,False,False,False,False]])
fish_y.head()


# In[10]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(fish_x, fish_y, test_size=0.3)


# In[11]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, Y_train)


# In[12]:


y_pred=classifier.predict(X_test)

Y_pred_df=pd.DataFrame(y_pred, columns=["Predicted"])
Y_pred_df.head()


# In[13]:


Y_test.head()


# In[15]:


accuracy=accuracy_score(Y_test,y_pred)*100
print(accuracy)


# In[ ]:

pickle.dump(classifier, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[242.0, 23.2, 25.4,30.0,11.445,4.05]]))


