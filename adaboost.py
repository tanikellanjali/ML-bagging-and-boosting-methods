#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[2]:


from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


# In[3]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import seaborn as sns


# In[4]:


from sklearn.metrics import confusion_matrix, f1_score


# In[5]:


dt_train = pd.read_csv('titanic_train.csv')


# In[6]:


dt_train.shape


# In[7]:


dt_train.info()


# In[8]:


dt_train.dtypes


# In[9]:


# we can see the suvival clearl has a dependency on Sex but on on Age hence we can drop the age column and make the sex column into a data type 
sns.set(rc = {'figure.figsize':(10,8)})  
g = sns.relplot(data=dt_train, x="Survived", y="Age"  ,hue="Sex",height = 7)
g.ax.axline(xy1=(10, 2), slope=.2, color="b", dashes=(5, 2))


# In[10]:


dt_train.drop(columns = 'Age',inplace = True)


# In[11]:


dt_train['Sex'].unique()


# In[12]:


for i in range (len(dt_train.index)):
          
        if dt_train['Sex'][i] == 'male' :    
            dt_train['Sex'][i] = 0
        if dt_train['Sex'][i] == 'female' :    
            dt_train['Sex'][i] = 1
          


# In[13]:


dt_train.Sex=dt_train.Sex.astype('int64')


# In[14]:


dt_train.dtypes


# In[15]:


# we can see the suvival clearl has no dependcies on Ticket  
sns.set(rc = {'figure.figsize':(10,8)})  
ax = sns.relplot(data=dt_train, x="Survived", y="Ticket" ,height = 7)
g.ax.axline(xy1=(10, 2), slope=.2, color="b", dashes=(5, 2))


# In[16]:


# clearly there is a dependencies 
g = sns.relplot(data=dt_train, x="PassengerId",y ="Embarked", hue= 'Survived',height = 7  )


# In[17]:


dt_train.drop(columns = ['Ticket','Name','Cabin'],inplace=True)


# In[18]:


dt_train.dtypes


# In[19]:


dt_train.isnull().sum()


# In[20]:


dt_train.fillna(4,inplace =True)


# In[21]:


for i in range (len(dt_train.index)):
          
        if dt_train['Embarked'][i] == 'S' :    
            dt_train['Embarked'][i] = 0
        if dt_train['Embarked'][i] == 'C' :    
            dt_train['Embarked'][i] = 1
        if dt_train['Embarked'][i] == 'Q' :    
            dt_train['Embarked'][i] = 2
            
          


# In[22]:


dt_train.Embarked=dt_train.Embarked.astype('int64')


# In[23]:


dt_train.dtypes


# In[24]:


corr = dt_train.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="Blues", annot=True)


# In[25]:


dtx = dt_train.drop(columns = 'Survived')
dty = dt_train['Survived']


# In[26]:


train_dtx,test_dtx,train_dty,test_dty = train_test_split(dtx,dty,test_size = 0.3,random_state=43)


# In[27]:


classifierdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=200)
classifierdt.fit(train_dtx, train_dty)


# In[28]:


preddt = classifierdt.predict(test_dtx)


# In[29]:


preddt


# In[30]:


l = np.arange(len(preddt))
plt.scatter(preddt , l)
plt.scatter(test_dty,l,c ='crimson')


# In[31]:


from sklearn import metrics


# In[32]:


metrics.accuracy_score(test_dty,preddt)


# In[33]:


confusion_matrix(test_dty, preddt)


# In[34]:


plt.figure(figsize=(10,10))
plt.scatter(test_dty, preddt, c='crimson')
plt.title('Adaboost Classifier ')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


# In[ ]:





# In[ ]:




