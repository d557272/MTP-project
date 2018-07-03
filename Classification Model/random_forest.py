
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix


# In[2]:

df = pd.read_csv('features1.csv', sep=',', header=None, low_memory=False)


# In[3]:

Y = df[856]


# In[4]:

X = df.iloc[:,2:856]


# In[5]:

X.shape


# In[6]:

Y.shape


# In[7]:

train_x, test_x, train_y, test_y = train_test_split(X, Y, train_size=0.85)


# In[8]:

train_x.shape


# In[9]:

test_x.shape


# In[10]:

train_y.shape


# In[11]:

test_y.shape


# In[12]:

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(train_x, train_y)


# In[14]:

predicted_train_y = clf.predict(train_x)


# In[15]:

mean_absolute_train_error = mean_absolute_error(train_y, predicted_train_y)


# In[16]:

print mean_absolute_train_error


# In[17]:

mean_squared_train_error = mean_squared_error(train_y, predicted_train_y)


# In[18]:

print mean_squared_train_error


# In[20]:

predicted_test_y = clf.predict(test_x)


# In[21]:

mean_absolute_test_error = mean_absolute_error(test_y, predicted_test_y)


# In[22]:

print mean_absolute_test_error


# In[23]:

mean_squared_test_error = mean_squared_error(test_y, predicted_test_y)


# In[24]:

print mean_squared_test_error


# In[25]:

confusion_matrix(test_y, predicted_test_y)


# In[26]:

print(predicted_test_y)


# In[27]:

bp = pd.read_csv('app_data/budget_planner.csv', sep=',', header=None, low_memory=False)


# In[28]:

bp = bp.iloc[1:2, 1:855]


# In[30]:

bp_y = clf.predict(bp)


# In[31]:

print bp_y


# In[32]:

bhim = pd.read_csv('app_data/bhim.csv', sep=',', header=None, low_memory=False)


# In[33]:

bhim = bhim.iloc[1:2,1:855]


# In[34]:

bhim_y = clf.predict(bhim)


# In[35]:

print bhim_y


# In[36]:

funnyys = pd.read_csv('app_data/funnys.csv', sep=',', header=None, low_memory=False)


# In[37]:

funnyys =  funnyys.iloc[1:2,1:855]


# In[38]:

fun_y = clf.predict(funnyys)


# In[39]:

print fun_y


# In[40]:

omingo = pd.read_csv('app_data/omingo.csv', sep=',', header=None, low_memory=False)


# In[41]:

omingo =  omingo.iloc[1:2,1:855]


# In[43]:

omingo_y = clf.predict(omingo)


# In[44]:

print omingo_y


# In[45]:

sys = pd.read_csv('app_data/system_administrator.csv', sep=',', header=None, low_memory=False)


# In[46]:

sys = sys.iloc[1:2,1:855]


# In[47]:

sys_y = clf.predict(sys)


# In[48]:

print sys_y


# In[49]:

tez = pd.read_csv('app_data/tez.csv', sep=',', header=None, low_memory=False)


# In[50]:

tez = tez.iloc[1:2,1:855]


# In[52]:

tez_y = clf.predict(tez)


# In[53]:

print tez_y


# In[54]:

mms = pd.read_csv('app_data/mms_beline.csv', sep=',', header=None, low_memory=False)


# In[55]:

mms = mms.iloc[1:2,1:855]


# In[57]:

mms_y = clf.predict(mms)


# In[58]:

print mms_y


# In[59]:

cal = pd.read_csv('app_data/calendar.csv', sep=',', header=None, low_memory=False)


# In[60]:

cal = cal.iloc[1:2,1:855]


# In[61]:

cal_y = clf.predict(cal)


# In[62]:

print cal_y


# In[63]:

laugh = pd.read_csv('app_data/laughtter.csv', sep=',', header=None, low_memory=False)


# In[64]:

laugh = laugh.iloc[1:2,1:855]


# In[65]:

laugh_y = clf.predict(laugh)


# In[66]:

print laugh_y


# In[67]:

andy = pd.read_csv('app_data/android_framework.csv', sep=',', header=None, low_memory=False)


# In[68]:

andy = andy.iloc[1:2,1:855]


# In[69]:

andy_y = clf.predict(andy)


# In[70]:

print andy_y

