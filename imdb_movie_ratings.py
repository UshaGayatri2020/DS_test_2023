#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[3]:


df = pd.read_csv("movie_metadata.csv")


# In[4]:


df.sample(5)


# In[5]:


df.columns


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


df['gross'].fillna(df['gross'].median(), inplace=True)
df['budget'].fillna(df['budget'].median(), inplace=True)
df.dropna(inplace=True)


# In[9]:


df.shape


# In[ ]:


#To know which type of movie, creating a new column to store main genre of movie.


# In[10]:


df['main_genre'] = df['genres'].apply(lambda x: x.split('|')[0] if '|' in x else x)


# In[11]:


df.sample(2)


# In[13]:


plt.figure(figsize=(12,10))
sns.boxplot(x='imdb_score', y='main_genre',data=df)
plt.title("Movie gners with their imdb socres", fontsize=18)
plt.show()


# In[15]:


numeric_cols = df.select_dtypes(include=np.number).columns
z_scores = np.abs((df[numeric_cols]-df[numeric_cols].mean())/df[numeric_cols].std())
thershold=3
df=df[(z_scores < thershold).all(axis=1)]


# In[16]:


df.shape


# In[17]:


df.title_year.value_counts(dropna=True).sort_index().plot(kind='barh',figsize=(15,20))
plt.title("Number of movies released every year", fontsize=18)
plt.show()


# In[22]:


df.columns


# In[27]:


plt.figure(figsize=(15,10))
squarify.plot(Counter(df['main_genre']).values(),label=Counter(df['main_genre']).keys(),text_kwargs={'fontsize':10},bar_kwargs={'alpha':.7},pad=True)
plt.title("Geners", fontsize=18)
plt.axis("off")
plt.show()


# In[31]:


#movies with lowest imdb ratings
df[df['imdb_score']<=3.3]


# In[32]:


#movies with Highest imdb ratings
df[df['imdb_score']>=8.9]


# In[35]:


df.hist(bins=30,figsize=(15,15), color='r')
plt.show()


# In[36]:


df['num_genres']=df.genres.apply(lambda x: len(x.split('|')))
df.sample(2)


# In[37]:


df.num_genres.max()


# In[38]:


df[df.num_genres ==8]


# In[40]:


plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True,linewidths=5,cmap='coolwarm',square=True,cbar_kws={'label':'Correlation Coefficient'})
plt.title("Correlation Polit",fontsize=18)
plt.show()


# In[ ]:


#Selected Columns for model criteria 
#num_critic_for_reviews, duration, num_voted_users, num_user_for_reviews, movie_facebook_likes, director_facbook_likes


# In[41]:


df.columns


# In[44]:


X=df[['num_critic_for_reviews','duration','num_voted_users','num_user_for_reviews','movie_facebook_likes','director_facebook_likes']]
y=df['imdb_score']


# In[45]:


X.shape,y.shape


# In[46]:


X_train, X_test, y_train, y_test = tts(X,y,test_size=0.2,random_state=32)


# In[51]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# In[54]:


lm=LinearRegression()
lm.fit(X_train,y_train)
pred_lm=lm.predict(X_test)
print("Mean squared error using linear regression", mean_squared_error(y_test,pred_lm))
print("Mean absolute error using linear regression", mean_absolute_error(y_test,pred_lm))


# In[59]:


dtc = DecisionTreeRegressor()
dtc.fit(X_train,y_train)
pred_dt=dtc.predict(X_test)
print("Mean squared error using Decisiontree regression", mean_squared_error(y_test,pred_dt))
print("Mean absolute error using Decisiontree regression", mean_absolute_error(y_test,pred_dt))


# In[56]:


svr = SVR(kernel='rbf')
svr.fit(X_train,y_train)
pred_svr = svr.predict(X_test)
print("Mean squared error using Support Vector regression", mean_squared_error(y_test,pred_svr))
print("Mean absolute error using Support Vector regression", mean_absolute_error(y_test,pred_svr))


# In[58]:


knn = KNeighborsRegressor(n_neighbors=12)
knn.fit(X_train,y_train)
pred_knn = knn.predict(X_test)
print("Mean squared error using KNN regression", mean_squared_error(y_test,pred_knn))
print("Mean absolute error using KNN regression", mean_absolute_error(y_test,pred_knn))


# In[ ]:


#Linear regression model is the best compared to Decision Tree, SVM, and KNN, i.e.,
#Mean squared error using linear regression 0.6764002653438028
#Mean absolute error using linear regression 0.6577251825742132

