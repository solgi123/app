#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


# In[46]:


df=pd.read_csv('clean heart.csv')
df.head()


# In[47]:

df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.nunique()


# In[8]:


df['target'].value_counts()


# In[9]:


df['target'].unique()
target_age=pd.crosstab(df['target'],df['sex'])
target_age


# In[10]:


#max heart rate
df['target'].unique()
target_thalach=pd.crosstab(df['target'],df['thalach'])
target_thalach


# In[11]:


# filter with chol,target is not 1 and trestbps(blood presure) more than 170
df[(df['chol']>200) &(df['target']!=0) & (df['trestbps']>170)]


# In[12]:


# from 30 people we have 2 people with thalach 180
def clip_thalach(thalach):
    if thalach>180:
        thalach=180
        return thalach
df['thalach'].apply(lambda x:clip_thalach(x))[:30] 


# In[13]:


# find the average of each features based on the chol and sort the thalach from down to up
df.groupby('chol').mean().sort_values('thalach',ascending=False)[:10]


# In[14]:


#categoroized with the age and sex with target(output)
pd.pivot_table(df,index=['sex','age'],values='target')[:20]


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(data=df,y='target',palette='hls',hue='sex')
plt.title('amount of the target')
plt.figure(figsize=(20,10))
plt.show


# In[16]:


# abundance of the with age
sns.swarmplot(df['age'])


# In[17]:


sns.pairplot(df[['target','fbs','age','sex','thalach']])


# In[18]:


# ca(major vessel and cp is the chain pain)
sns.countplot(data=df,x='fbs',hue='target')
plt.show


# In[19]:


plt.hist(df['target'])
plt.show()


# In[20]:


sns.relplot('chol','age',data=df,kind='line',ci=None)


# In[21]:


# comapre with men and women that who have more target zero and who have not
fig,ax=plt.subplots(figsize=(10,5))
sns.countplot(df['target'],hue=df['sex'],ax=ax)
plt.xlabel('target')
plt.ylabel('sex')
plt.xticks(rotation=50)
plt.show


# In[22]:


nums=['age','sex','trestbps','chol','trestbps','target']
for i in nums:
    plt.figure(figsize=(20,10))
    sns.jointplot(x=df[i],y=df['target'],kind='reg')
    plt.xlabel(i)
    plt.ylabel('resposne')
    plt.grid()
    plt.show()


# In[23]:


plt.bar(df['target'],df['age'],alpha=.5,width=0.8,label='chart')
plt.show()


# In[24]:


sns.catplot('sex','target',data=df,kind='box',hue='fbs')


# In[25]:


# abundance for each of the columns
import itertools
columns=df.columns[:8]
plt.subplots(figsize=(30,28))
length=len(columns)
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot((length/2),5,j+1)
    plt.subplots_adjust(wspace=0.3,hspace=0.8)
    df[i].hist(bins=30,edgecolor='black')
    plt.title(i)
plt.show()


# In[26]:


sns.jointplot('target','thalach',data=df,kind='kde',color='pink')


# # Finding the outliers

# # With box tucky

# In[27]:


df=pd.read_csv('clean heart.csv')
df.head()


# In[28]:

x=df.iloc[:,0:4].values
y=df.iloc[:4]
df[:4]


# In[30]:


df.boxplot(return_type='dict')
plt.plot()


# In[31]:


thalach=x[:,1]
iris_outliers=(thalach<40)
df[iris_outliers]


# # Applying tucky outlier labeling

# In[32]:


pd.options.display.float_format='{:.1f}'.format
x_df=pd.DataFrame(df)
x_df.describe()


# In[33]:


# we want to calculate this:
# iqr(for age) = 61.0 - 48.0 = 13.0
#iqr(1.5)= 19.5
#48.0 - 19.5 = 28.5
# 61.0 + 19.5 = 80.5 


# # Make a model

# In[34]:


import warnings
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans


# In[35]:


df.isnull().sum()


# In[36]:


X = df.drop(columns=['target','age','fbs','age','chol','trestbps','rectecg'],axis=1)
y = df['target']


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30, random_state=145)


# # Logistic Regression

# In[39]:


model = LogisticRegression()


# In[40]:


model.fit(X_train,y_train)


# In[ ]:


#We find the pickle and save it to our jupyter notebook(find the percentage)


# In[41]:


import pickle


# In[55]:


Pkl_Filename = "Pickle_RL_Model.pkl"
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)


# In[57]:


with open(Pkl_Filename, 'rb') as file:  
    Pickled_LR_model = pickle.load(file)

Pickled_LR_model


# In[60]:


# Calculate the Score 
score = Pickled_LR_model.score(X_test, y_test)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

# Predict the Labels using the reloaded Model
Ypredict = Pickled_LR_model.predict(X_test)  

Ypredict


# In[10]:


predictions = model.predict(X_test)


# In[11]:


predictions


# In[12]:


score=accuracy_score(y_test,predictions)


# In[13]:


score


# In[50]:


metrics.confusion_matrix(y_test,predictions)


# In[51]:


sns.heatmap(confusion_matrix(y_test,predictions), annot=True, cmap="mako")


# In[52]:


classification_report(y_test,predictions)


# # Decision Tree

# In[68]:


model=DecisionTreeClassifier()


# In[69]:


model.fit(X_train,y_train)


# In[70]:


import pickle
Pkl_Filename = "Pickle_DT_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)


# In[73]:


# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:  
    Pickled_DT_model = pickle.load(file)

Pickled_DT_model


# In[75]:


# Calculate the Score 
score = Pickled_DT_model.score(X_test, y_test)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

# Predict the Labels using the reloaded Model
Ypredict = Pickled_DT_model.predict(X_test)  

Ypredict


# In[16]:


predictions=model.predict(X_test)


# In[17]:


predictions


# In[18]:


score=accuracy_score(y_test,predictions)


# In[19]:


score


# In[21]:


import seaborn as sns
sns.heatmap(confusion_matrix(y_test,predictions), annot=True, cmap="mako")


# # Pickle the model

# In[33]:


Decision_tree_model_pkl=open('Random_Forest_regressor_model.pkl',"wb")


# In[34]:


import pickle
pickle.dump(model,Decision_tree_model_pkl)


# In[35]:


Decision_tree_model_pkl.close()


# # Random Forest

# In[76]:


model=RandomForestClassifier()


# In[77]:


model.fit(X_train,y_train)


# In[78]:


import pickle
Pkl_Filename = "Pickle_RF_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)
# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:  
    Pickled_RF_model = pickle.load(file)

Pickled_RF_model


# In[80]:


# Calculate the Score 
score = Pickled_RF_model.score(X_test, y_test)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

# Predict the Labels using the reloaded Model
Ypredict = Pickled_RF_model.predict(X_test)  

Ypredict


# In[24]:


predictions=model.predict(X_test)


# In[25]:


predictions


# In[26]:


score=accuracy_score(y_test,predictions)


# In[27]:


score


# In[28]:


sns.heatmap(confusion_matrix(y_test,predictions), annot=True, cmap="mako")


# # Neural Networks

# In[81]:


model=MLPClassifier()


# In[82]:


model.fit(X_train,y_train)


# In[83]:


import pickle
Pkl_Filename = "Pickle_NN_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)
# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:  
    Pickled_NN_model = pickle.load(file)

Pickled_NN_model


# In[85]:


# Calculate the Score 
score = Pickled_NN_model.score(X_test, y_test)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

# Predict the Labels using the reloaded Model
Ypredict = Pickled_NN_model.predict(X_test)  

Ypredict


# In[70]:


predictions=model.predict(X_test)


# In[71]:


predictions


# In[72]:


score=accuracy_score(y_test,predictions)


# In[73]:


score


# In[74]:


sns.heatmap(confusion_matrix(y_test,predictions), annot=True, cmap="mako")


# # Suport Vector Machine

# In[86]:


model = SVC()


# In[87]:


model.fit(X_train,y_train)


# In[88]:


import pickle
Pkl_Filename = "Pickle_SVM_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)
# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:  
    Pickled_SVM_model = pickle.load(file)

Pickled_SVM_model


# In[89]:


# Calculate the Score 
score = Pickled_SVM_model.score(X_test, y_test)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

# Predict the Labels using the reloaded Model
Ypredict = Pickled_SVM_model.predict(X_test)  

Ypredict


# In[77]:


predictions=model.predict(X_test)


# In[78]:


predictions


# In[79]:


score=accuracy_score(y_test,predictions)


# In[80]:


score


# # KMeans

# In[81]:


model=KMeans(n_clusters=3)


# In[82]:


model.fit(X_train,y_train)


# In[83]:


df['clusters']=df['target']


# In[84]:


predictions=model.predict(X_test)


# In[85]:


predictions


# In[86]:


score=accuracy_score(y_test,predictions)


# In[87]:


score


# In[88]:


classification_report(y_test,predictions)


# In[17]:


#You can see that the value of root mean squared error is 0.5524, which is almost same  as 10% of the mean value which is 0.05131.
#This means that our algorithm was almsot accurate
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('10% of Mean Price:', df['target'].mean() * 0.1)
# In[5]:

# # Time continuous for thalach
import time
import os
import numpy as np
import pandas as pd
from datetime import datetime
from IPython.display import clear_output

 

def doEvery_1_Seconds():
    datetime_now = datetime.now()
    print("Heart rate at time {}:{}:{}".format((datetime_now.hour%12),datetime_now.minute,datetime_now.second))
    df = np.random.randint(71,202,size=10)
    df = pd.DataFrame(df, columns=['thalach'])
    print(df)
    time.sleep(1)
#set timer limit in seconds
TIMER_LIMIT = 10

 

start_time = datetime.now()
while(1):
    present_time = datetime.now()
    init = start_time.minute*60 +start_time.second
    ending = present_time.minute*60 + present_time.second
    
    if(ending - init >= TIMER_LIMIT):  
        break
    else:
        clear_output(wait=True)
        doEvery_1_Seconds()
        setTimer = time.time()


# In[ ]:




