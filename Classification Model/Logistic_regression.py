
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import preprocessing
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

linear_regression_model = LogisticRegression()
linear_regression_model.fit(train_x, train_y)


# In[13]:

train_accuracy = linear_regression_model.score(train_x, train_y)


# In[14]:

print train_accuracy


# In[15]:

test_accuracy = linear_regression_model.score(test_x, test_y)


# In[16]:

print test_accuracy


# In[17]:

predicted_y = linear_regression_model.predict(test_x)


# In[18]:

predicted_y


# In[19]:

confusion_matrix(test_y, predicted_y)


# In[45]:

bp = pd.read_csv('app_data/budget_planner.csv', sep=',', header=None, low_memory=False)


# In[47]:

bp = bp.iloc[1:2, 1:855]


# In[51]:

bp_y = linear_regression_model.predict(bp)


# In[ ]:

print bp_y


# In[54]:

bhim = pd.read_csv('app_data/bhim.csv', sep=',', header=None, low_memory=False)


# In[55]:

bhim = bhim.iloc[1:2,1:855]


# In[56]:

bhim_y = linear_regression_model.predict(bhim)


# In[57]:

print bhim_y


# In[58]:

funnyys = pd.read_csv('app_data/funnys.csv', sep=',', header=None, low_memory=False)


# In[59]:

funnyys =  funnyys.iloc[1:2,1:855]


# In[60]:

fun_y = linear_regression_model.predict(funnyys)


# In[61]:

print fun_y


# In[62]:

omingo = pd.read_csv('app_data/omingo.csv', sep=',', header=None, low_memory=False)


# In[63]:

omingo =  omingo.iloc[1:2,1:855]


# In[64]:

omingo_y = linear_regression_model.predict(omingo)


# In[65]:

print omingo_y


# In[66]:

sys = pd.read_csv('app_data/system_administrator.csv', sep=',', header=None, low_memory=False)


# In[68]:

sys = sys.iloc[1:2,1:855]


# In[69]:

sys_y = linear_regression_model.predict(sys)


# In[70]:

print sys_y


# In[72]:

tez = pd.read_csv('app_data/tez.csv', sep=',', header=None, low_memory=False)


# In[73]:

tez = tez.iloc[1:2,1:855]


# In[74]:

tez_y = linear_regression_model.predict(tez)


# In[75]:

print tez_y


# In[76]:

mms = pd.read_csv('app_data/mms_beline.csv', sep=',', header=None, low_memory=False)


# In[77]:

mms = mms.iloc[1:2,1:855]


# In[78]:

mms_y = linear_regression_model.predict(mms)


# In[79]:

print mms_y


# In[80]:

cal = pd.read_csv('app_data/calendar.csv', sep=',', header=None, low_memory=False)


# In[81]:

cal = cal.iloc[1:2,1:855]


# In[82]:

cal_y = linear_regression_model.predict(cal)


# In[83]:

print cal_y


# In[84]:

laugh = pd.read_csv('app_data/laughtter.csv', sep=',', header=None, low_memory=False)


# In[85]:

laugh = laugh.iloc[1:2,1:855]


# In[86]:

laugh_y = linear_regression_model.predict(laugh)


# In[87]:

print laugh_y


# In[88]:

andy = pd.read_csv('app_data/android_framework.csv', sep=',', header=None, low_memory=False)


# In[89]:

andy = andy.iloc[1:2,1:855]


# In[90]:

andy_y = linear_regression_model.predict(andy)


# In[91]:

print andy_y

