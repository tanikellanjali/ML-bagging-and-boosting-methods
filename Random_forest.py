#!/usr/bin/env python
# coding: utf-8

# https://github.com/prosperityai/random-forest/blob/master/random-forest.ipynb
# https://www.youtube.com/watch?v=zP1mBAJQNX0
# https://machinelearningmastery.com/implement-random-forest-scratch-python/
# https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
# https://learn-us-east-1-prod-fleet02-xythos.content.blackboardcdn.com/5e00ea752296c/12322840?X-Blackboard-Expiration=1669809600000&X-Blackboard-Signature=nlf%2FXIrY4E8Tc6ZYgoQFoUTY6odtmQ4nk3FWe5HnNmQ%3D&X-Blackboard-Client-Id=100310&response-cache-control=private%2C%20max-age%3D21600&response-content-disposition=inline%3B%20filename%2A%3DUTF-8%27%27random_forests.pdf&response-content-type=application%2Fpdf&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEFYaCXVzLWVhc3QtMSJGMEQCIAUtdeaUfUdbBi4jjh06%2B9g9lkVdfhs9MVTAKTuT0gylAiAJf8hXizIT7ZSmgyP5d1DeW67GLyljnZSJtWG52jKLPCrMBAhvEAIaDDYzNTU2NzkyNDE4MyIM%2Be132F8XR865WNSWKqkEgxAoKn8CizMjzZt%2Ff6%2Fb3CuNM5alt%2Fwc%2F9CU9QCSrjR%2FNecX7gN0kdmERy65qANvtXn667m5p8RomEftEcXP%2BNZEW448tqvKGJ84PsrSx6OhFoSTUxZLL4xLfslvUtlRtZ8hvsTK8YB3zW9Soj5rGisxuqT%2FxeeieloCMuVVdFEMJJZL3n%2B680yAOKOhuemjF4ukqsWBOI9FrohYj22Ob1FjndjLlCtPdrXw8fek7WRRbJKtu7w6Ig9moYPOON1YSxXGF07hzr5vwD2juofbcID3H93I5GhekKeUFALvYncQpca42BX2IkieB5kALGU%2BZe0kLgZDfU1BfqDsgg8ENcjr%2Fzo8plosHF7nbMuA2bdN73PHR8gwiDs%2FKSd8n9ajD6vpytbJLp1LL0lemXHMjOQJ4TflN3Ys%2FXf08rpCSdeR9u4iZhKAc47m6cRs2fw8clZpZGM%2B93H33m1SyePGAdK12jBhNDbn9CsmLgQreNebHR430%2FZEJxl%2BmAaamA11nNEQDVlnZAdz78PKcrEe3KhB7k0FJa%2FcYzuhF2tF3q9lKFc%2FKtXf6KkNPDbEbQtunSdQnZig2hxgTf9o4s%2FGJlpRBHob0GAOy4e4FSHhLc9me7VHXo30HWx%2BDZeAEwX56bLAqZ%2FPPpSHj5E81JL%2Fp750Jwx2PBvZi3zlqgHBnZlQn2w3OuP4gWaTxfsBPY37SOeGor9lTgyJsdiRJdgPoH%2FdvLDblfoo0TCv4JucBjqqAQNJmgP8wLu6KulqMSN2yzdGLrQE8Fw5k%2FkRqWcYftL6pp8WlpUiHQojUTXX6xv%2FgbYk%2FEamLIRyk20wYRcUmVztK6b1O%2FWcz80nKoBYKTy%2Fp585MGCjpMeD3na2Xd0cjL9vm4BqjTlXs6yZPgGkp0AoEuzFizbAUkSVxLYKoO%2BokDBPTVcliMpjetgaJPX0u5NX2V3y1bDN7EHSFhOC78E1AP6MQWHnMZZQ&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20221130T060000Z&X-Amz-SignedHeaders=host&X-Amz-Expires=21600&X-Amz-Credential=ASIAZH6WM4PLXR5HPR4Y%2F20221130%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=74aa80912eefe8f0b42accf739f861b6f70d07dda1791b5266a990489aec1384

# In[1]:


import pandas as pd
import pandas as pd 
import numpy as np 
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score


# In[2]:


df = pd.read_csv('https://pkgstore.datahub.io/machine-learning/sonar/sonar_csv/data/71dc2b4593995a5a1cff52824511fc61/sonar_csv.csv')


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.dtypes


# In[7]:


df['Class']


# In[8]:


df['Class'].unique()


# In[9]:


for i in range (len(df.index)):
          
        if df['Class'][i] == 'Rock' :    
            df['Class'][i] = 0
        if df['Class'][i] == 'Mine' :    
            df['Class'][i] = 1
          


# In[10]:


plt.figure(figsize=(16, 30))
g = sns.displot(data=df, x="Class", kind="kde", height = 7)


# In[11]:


df['Class'] =df['Class'].astype('int')


# In[12]:


corr = df.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="Blues", annot=True)


# In[13]:


x = df.drop(columns = 'Class',axis = 0 )
y = df['Class'].astype('int')


# In[14]:


train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = 0.3 , random_state = 43)


# In[15]:


from sklearn.ensemble import RandomForestClassifier


# In[16]:


clf=RandomForestClassifier(n_estimators=100)


# In[17]:


clf.fit(train_x,train_y)


# In[18]:


pred_y =clf.predict(test_x)


# In[19]:


from sklearn import metrics
metrics.accuracy_score(test_y,pred_y)


# In[20]:


confusion_matrix(test_y, pred_y)


# In[21]:


plt.figure(figsize=(10,10))
plt.scatter(test_y, pred_y, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(pred_y), max(test_y))
p2 = min(min(pred_y), min(test_y))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.title('Random Forest ')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


# In[ ]:


from sklearn import tree

features = x.columns.values # The name of each column
classes = ['0', '1'] # The name of each class
# You can also use low, medium and high risks in the same order instead
# classes = ['low risk', 'medium risk', 'high risk']

for estimator in clf.estimators_:
    print(estimator)
    plt.figure(figsize=(12,6))
    tree.plot_tree(estimator,
                   feature_names=features,
                   class_names=classes,
                   fontsize=8, 
                   filled=True, 
                   rounded=True)
    plt.show()


# In[ ]:


print(clf.estimators_[1])
plt.figure(figsize=(20,10))
tree.plot_tree(clf.estimators_[1],
                   feature_names=features,
                   class_names=classes,
                   fontsize=8, 
                   filled=True, 
                   rounded=True)
plt.show()


# In[ ]:


print(clf.estimators_[5])
plt.figure(figsize=(20,12))
tree.plot_tree(clf.estimators_[5],
                   feature_names=features,
                   class_names=classes,
                   fontsize=8, 
                   filled=True, 
                   rounded=True)
plt.show()


# In[ ]:


print(clf.estimators_[10])
plt.figure(figsize=(20,12))
tree.plot_tree(clf.estimators_[10],
                   feature_names=features,
                   class_names=classes,
                   fontsize=8, 
                   filled=True, 
                   rounded=True)
plt.show()


# In[52]:


i = 0
for estimators in clf.estimators_:
    i = i + 1
print(i)
    


# In[ ]:




