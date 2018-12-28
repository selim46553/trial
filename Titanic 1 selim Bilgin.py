
# coding: utf-8

# In[18]:


import pandas as pd


# In[29]:


cd /Users/SelimBilgin/Desktop/titanic_dataset_kaggle/


# In[30]:


train= pd.read_csv("train.csv")


# In[32]:


test= pd.read_csv("test.csv")


# In[33]:


train.head()


# In[36]:


train.isnull().sum()


# In[52]:


train = train.drop(["Name"], axis=1)


# In[44]:


train.head()


# In[45]:


train.isnull().sum()


# In[50]:


train.head()


# In[53]:


test= test.drop(["Name"], axis= 1)


# In[54]:


test.head()


# In[55]:


grand_data = [train, test]
sex_bool = { "male": 0, "female" : 1}
for x in grand_data:
    x["Sex"] = x["Sex"].map(sex_bool)


# In[57]:


train.head()


# In[58]:


train.isnull().sum()


# In[63]:


train.Cabin.value_counts()


# In[62]:


for x in grand_data:
    x["Cabin"]= x["Cabin"].str[:1]


# In[64]:


cabin_val = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for x in grand_data:
    x["Cabin"]= x["Cabin"].map(cabin_val)


# In[65]:


train.isnull().sum()


# In[68]:


train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace= True)


# In[69]:


train.isnull().sum()


# In[71]:


train = train.drop(["Embarked"], axis= 1)


# In[72]:


test = test.drop(["Embarked"], axis= 1)


# In[75]:


train.isnull().sum()


# In[76]:


train.head()


# In[77]:


train= train.drop(["Ticket"], axis=1)


# In[78]:


test= test.drop(["Ticket"], axis=1)


# In[79]:


train.head()


# In[80]:


train= train.drop(["Fare"], axis= 1)


# In[82]:


test= test.drop(["Fare"], axis= 1)


# In[83]:


train.head()


# In[84]:


train.isnull().sum()


# In[85]:


test_trial= test


# In[87]:


test_trial.isnull().sum()


# In[88]:


test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace= True)


# In[96]:


test.isnull().sum()


# In[108]:


test_trial3 = test


# In[111]:


test_trial3.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)


# In[116]:


train= train_trial3


# In[118]:


train_trial3.dropna(axis=0, how= "any", thresh= None, subset= None, inplace= True)


# In[120]:


train_trial3.isnull().sum()


# In[122]:


from sklearn.tree import DecisionTreeClassifier


# In[125]:


clf = DecisionTreeClassifier()


# In[142]:


test_trial3.shape


# In[144]:


train_trial3.shape


# In[149]:


train_trial3.drop(train_trial3.index[:295], inplace=True)


# In[150]:


train_trial3.shape


# In[151]:


train_trial3.drop(train_trial3.index[:1], inplace=True)


# In[152]:


train_trial3.shape


# In[163]:


import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
slptsy= KFold(n_splits=10, shuffle= True, random_state=0)
scoring = "accuracy"
score = cross_val_score(clf, train_trial3, test_trial3, cv= slptsy, n_jobs=1, scoring = scoring)
print(score)
round(np.mean(score)*100,2)

