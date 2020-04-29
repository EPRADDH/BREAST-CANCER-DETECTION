#!/usr/bin/env python
# coding: utf-8

# Let's start 
# 
# first we import the dataset and required library to read the data 
# we are importing dataset from the Sklearn dataset module.

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# import datda set
from sklearn.datasets import load_breast_cancer


# In[4]:


cancer = load_breast_cancer()


# In[5]:


cancer  # it has a some messed up information data with some feature and target value


# In[6]:


#first simplyfiy the data and read the information 
#information looks in a key :value pair
#print the information one by one for all the keys


# In[7]:


cancer.keys()


# In[8]:


#print("data","\n",cancer['data'],"\n",'-'*50)

#print("target","\n",cancer['target'],"\n",'-'*50)
#print("target_names",cancer['target_names'],"\n",'-'*50)
#print("DESCR",cancer['DESCR'],"\n",'-'*50)
#print("feature_names",cancer['feature_names'],"\n",'-'*50)


# In[9]:


print(cancer['DESCR'])


# In[10]:


cancer['data']


# In[11]:


cancer['target']


# In[12]:


cancer['target_names']


# In[13]:


cancer['feature_names']


# In[14]:


cancer['data'].shape


# In[15]:


#createing the data frame form the dataset

df_cancer = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])
df_cancer['target']= cancer['target']


# In[16]:


df_cancer.head()


# In[17]:


df_cancer.tail()


# In[18]:


df_cancer.describe()


# In[80]:


#df_cancer.isnull().sum()


# # Visualizing the Data

# In[71]:


#df_cancer = sns.load_dataset("df_cancer")
g = sns.PairGrid(df_cancer)
g.map(plt.scatter);


# In[ ]:


#df_cancer = sns.load_dataset("df_cancer")
g = sns.PairGrid(df_cancer)
g.map(plt.scatter);


# In[44]:


#heat map for correlation check
plt.figure(figsize = (25,20))
sns.heatmap(df_cancer.corr(),annot = True)


# In[ ]:





# ## Model training 

# In[24]:


X = df_cancer.drop(['target'],axis=1)


# In[ ]:





# In[25]:


y = df_cancer['target']


# In[26]:


from sklearn.model_selection import train_test_split
 
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size= 0.25, random_state = 355)


# In[35]:


from sklearn.svm import SVC


# In[36]:


from sklearn.metrics import classification_report,confusion_matrix


# In[86]:


svc_model = SVC()


# In[88]:


svc_model.fit(x_train , y_train)


# ## Evaluating the model

# In[89]:


#Let's see how well our model performs on the test data set

y_pred = svc_model.predict(x_test)


# In[90]:


y_pred


# In[91]:


# Confusion Matrix for validation
conf_mat = confusion_matrix(y_test,y_pred)
conf_mat


# it's shows somthing wrong in our model,becouse there is lost of variation in the feature values so first we need to normalize our datasets 

# In[95]:


## improving our model using few technique


# In[20]:


# let's see how data is distributed for every column
plt.figure(figsize=(20,100), facecolor='white')
plotnumber = 1

for column in df_cancer:
    if plotnumber<=30 :     # as there are 9 columns in the data
        ax = plt.subplot(10,3,plotnumber)
        sns.distplot(df_cancer[column])
        plt.xlabel(column,fontsize=20)
        #plt.ylabel('Salary',fontsize=20)
    plotnumber+=1
plt.show()


# We can see there is some skewness in the data, let's deal with data.
# 
# Now we have dealt with the 0 values and data looks better. But, there still are outliers present in some columns. Let's deal with them.

# In[22]:


df_cancer.isnull().sum()


# In[ ]:


#box plot is used to identify the outliers in the data sets

fig, ax = plt.subplots(figsize=(15,10))
sns.boxplot(data=df_cancer, width= 0.5,ax=ax, fliersize=3)


# Let's proceed by checking multicollinearity in the dependent variables. Before that, we should scale our data. Let's use the standard scaler for that.

# In[27]:


from sklearn.preprocessing import StandardScaler 

scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)


# This is how our data looks now after scaling. Great, now we will check for multicollinearity using VIF(Variance Inflation factor)

# In[28]:



X_scaled


# In[30]:


#variance_inflation_factor
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[31]:


#multicollinearity
vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(X_scaled,i) for i in range(X_scaled.shape[1])]
vif["Features"] = X.columns

#let's check the values
vif


# All the VIF values are more than 5 and are very low. That means  multicollinearity is prsence in the dependent variables.
# so we take all variables in our model without removing 
# 
# Now, we can go ahead with fitting our data to the model.
# 
# Before that, let's split our data in test and training set.

# In[33]:


x_train,x_test,y_train,y_test = train_test_split(X_scaled,y, test_size= 0.25, random_state = 355)


# In[37]:


svc_model = SVC()


# In[38]:


svc_model.fit(x_train , y_train)


# In[42]:


y_pred = svc_model.predict(x_test)


# Let's see how well our model performs on the test data set.

# In[43]:


y_pred


# In[52]:


# Confusion Matrix
conf_mat = confusion_matrix(y_test,y_pred)
conf_mat


# In[57]:


sns.heatmap(conf_mat, annot = True,fmt="d")


# In[59]:


#Summry report we have just did feature Normalization get the result
print(classification_report(y_test, y_pred))


# ### Let's use gridsearch to tune hyperparametre and see the improvement in model Result

# In[70]:


grid_param = {'C':[0.1,1,10,100],'gamma': [1,0.1,0.01,0.001],'random_state':[50,100,120] , 'kernel':['rbf'], 'max_iter':[-2,-1,1,2]}


# In[71]:


from sklearn.model_selection import GridSearchCV


# In[72]:


grid = GridSearchCV(SVC(),grid_param,refit = True,verbose = 5)


# In[73]:


grid.fit(x_train , y_train)


# In[74]:


grid.best_params_


# In[76]:


grid_pred = grid.predict((x_test))


# In[77]:


# Confusion Matrix
conf_mat = confusion_matrix(y_test,grid_pred)
conf_mat


# ### Finaly we get same result in both the cases it seem's our model is goodfit 

# In[ ]:




