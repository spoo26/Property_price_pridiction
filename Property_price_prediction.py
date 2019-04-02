#!/usr/bin/env python
# coding: utf-8

# In[347]:


#Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[348]:


import seaborn as sns


# In[349]:


#Create dataframe of dataset stored in csv
df = pd.read_csv("price_paid_records.csv")
df


# In[350]:


#Check for null values in the dataset
df.isnull().any().any()


# In[351]:


df = df.drop(['Transaction unique identifier', 'PPDCategory Type', 'Record Status - monthly file only'], axis=1)


# In[352]:


#Filter the dataset based on year
df['Date of Transfer'] = pd.to_datetime(df['Date of Transfer'])
dfv1 = df.iloc[:,:3]
dfv1 = dfv1[dfv1['Date of Transfer'].dt.year == 1995]
dfv1.sort_values(by='Date of Transfer',inplace=True)


# In[353]:


#Plot month v/s mean price based on the year given
fig, ax = plt.subplots(figsize=(20,7))
dfv1.groupby(dfv1['Date of Transfer'].dt.month)['Price'].mean().plot(marker="o",ax=ax)
index = ['January', 'Febraury', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
plt.xticks(range(1, 13),index)
plt.xlabel("Month")
plt.ylabel("Price")
plt.title("The average price per motn in the year of 1995")
plt.show()


# In[354]:


#Updating the coloumn name and creating a coloumn 'year' which has only year values
dfv2 = df.rename(columns={'Date of Transfer':'Date_of_Transfer'})
dfv2['year'] = pd.DatetimeIndex(dfv2['Date_of_Transfer']).year


# In[355]:


dfv2


# In[356]:


#Extracting the Price and year coloumn
x = dfv2.iloc[:,[0,8]]


# In[357]:


#Grouping years and getting the mean price values for the respective year
y = x.groupby(['year']).mean()


# In[358]:


#ploting average price of land in a given year
plt.rcParams["figure.figsize"] = (20,9)
y.plot.bar(color=(0.1, 0.5, 0.5, 0.5),  edgecolor='blue',width=.4)
plt.title('Avg price of land in a given year',fontsize=16)
plt.ylabel('price', fontsize=16)
plt.xlabel('year', fontsize=16)
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
plt.show()


# In[359]:


# Multiple Linear Regression
county = input("What is the name of the county? ")
district = input("Which district? ")
town = input("Name the town please: ")


# In[360]:


df_l_data = df.loc[(df['County'] == 'THURROCK') & (df['District'] == 'THURROCK') & (df['Town/City'] == 'GRAYS')]


# In[361]:


df_l_data


# In[362]:


X = df_l_data.iloc[:,2:5].values
y = df_l_data.iloc[:,0].values
X


# In[363]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
X[:,1]=labelencoder_X.fit_transform(X[:,1])
X[:,2]=labelencoder_X.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()


# In[364]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[365]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test) 


# In[366]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[367]:


y_pred=regressor.predict(X_test)
y_pred = y_pred.astype(int)
print(y_pred)


# In[368]:


print(regressor.intercept_)
print(regressor.coef_)


# In[319]:


from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[370]:


#print('Variance score: {}'.format(regressor.score(X_test, y_test)))


# In[371]:


from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(y_test, y_pred))


# In[372]:


from matplotlib.colors import ListedColormap
X_set, y_set = X_test,y_test # can use X_train and y_train also
X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step=0.01), np.arange(start = X_set[:,1].min() -1 , stop=X_set[:,1].max() + 1, step=0.01))

#plt.contourf(X1,X2, regressor.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('red', 'green')))

#plt.xlim(X1.min(), X1.max())
#plt.xlim(X2.min(), X2.max())

#for i,j in enumerate(np.unique(y_set)):
#	plt.scatter(X_set[y_set == j,0], X_set[y_set == j, 1], c= ListedColormap(('red', 'green'))(i), label = j)
#plt.title('NB (Training set)')
#plt.xlabel('Age')
#plt.ylabel("Estimated Salary")
#plt.legend()
#plt.show()


# In[246]:


# Logistic Regression
df_log = df.loc[(df['County'] == 'GREATER MANCHESTER') & (df['District'] == 'OLDHAM') & (df['Town/City'] == 'OLDHAM')]
max_price = df_log.loc[df_log['Price'].idxmax()].values[0]


# In[247]:


lower_threshold = max_price//3
upper_threshold = int(2*lower_threshold)


# In[248]:


upper_threshold


# In[249]:


df_log


# In[250]:


df_log['Price'].values[df_log['Price'] < lower_threshold] = 0
df_log['Price'].values[(df_log['Price'] > lower_threshold) & (df_log['Price'] < upper_threshold)] = 1
df_log['Price'].values[df_log['Price'] > upper_threshold] = 2
df_log


# In[251]:


X_log = df_log.iloc[:,2:5].values
y_log = df_log.iloc[:,0].values


# In[252]:


import sklearn
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_log=LabelEncoder()
X_log[:,0]=labelencoder_X_log.fit_transform(X_log[:,0])
X_log[:,1]=labelencoder_X_log.fit_transform(X_log[:,1])
X_log[:,2]=labelencoder_X_log.fit_transform(X_log[:,2])


# In[253]:


from sklearn.model_selection import train_test_split
x_train_log,x_test_log,y_train_log,y_test_log=train_test_split(X_log,y_log,test_size=.20,random_state=5)


# In[255]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train_log,y_train_log)
y_pred_log=logreg.predict(x_test_log)


# In[256]:


sklearn.metrics.accuracy_score(y_test_log,y_pred_log)


# In[257]:


y_pred_log


# In[258]:


from sklearn.metrics import confusion_matrix
cm_log=confusion_matrix(y_test_log,y_pred_log)


# In[259]:


cm_log

