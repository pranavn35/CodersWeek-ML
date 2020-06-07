#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import pickle


# In[2]:


s=pd.read_csv("Social_Network_Ads.csv")
s.head()


# In[3]:


import seaborn as sb
corr = s.corr()
sb.heatmap(corr, vmax=1., square=False)


# In[10]:


s_x=pd.DataFrame(s.iloc[:,[False,False,True,True,False]])
s_x.head()


# In[7]:


s_y=pd.DataFrame(s.iloc[:,-1])
s_y.head()


# In[35]:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, Y_train, Y_test = train_test_split(s_x, s_y, test_size=0.3)
regression=LinearRegression()
regression.fit(X_train, Y_train)



Y_pred_lin=regression.predict(X_test)

Y_pred_lin
Y_pred_df=pd.DataFrame(Y_pred_lin, columns=["Predicted"])

from sklearn.tree import DecisionTreeRegressor 
tree_regressor=DecisionTreeRegressor()
tree_regressor.fit(X_train, Y_train)


# In[36]:


Y_pred_tree=tree_regressor.predict(X_test)
Y_tree_pred_df=pd.DataFrame(Y_pred_tree, columns=["Predicted"])
Y_tree_pred_df.head()


# In[38]:


sum_of_error_dt=0
Y_test_df=Y_test.reset_index().iloc[:,1:2]
Y_test_df.index

for i in Y_tree_pred_df.index:
    sum_of_error_dt=sum_of_error_dt+((Y_tree_pred_df.loc[i]["Predicted"]-Y_test_df.loc[i]["Purchased"])**2)
print(sum_of_error_dt)


# In[40]:


sum_of_error_lr=0
for i in Y_pred_df.index:
    sum_of_error_lr=sum_of_error_lr+((Y_pred_df.loc[i]["Predicted"]-Y_test_df.loc[i]["Purchased"])**2)
print(sum_of_error_lr)


# In[47]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, Y_train)


# In[48]:


y_pred=classifier.predict(X_test)

Y_pred_df=pd.DataFrame(y_pred, columns=["Predicted"])
Y_pred_df.head()


# In[49]:


Y_test.head()


# In[51]:


count=0
for i in Y_pred_df.index:
    if(Y_pred_df.loc[i]["Predicted"]==Y_test_df.loc[i]["Purchased"]):
        count+=1
len_test=len(Y_test_df)
print("Accuracy=",count/len_test*100)


# In[52]:


# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[22, 29000]]))

# In[ ]:




