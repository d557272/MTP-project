
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix


# In[4]:

df = pd.read_csv('features1.csv', sep=',', header=None, low_memory=False)


# In[5]:

Y = df[856]


# In[6]:

X = df.iloc[:,2:856]


# In[7]:

X.shape


# In[8]:

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

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)


# In[13]:

predicted_train_y = clf.predict(train_x)


# In[14]:

mean_absolute_train_error = mean_absolute_error(train_y, predicted_train_y)


# In[15]:

print mean_absolute_train_error


# In[16]:

mean_squared_train_error = mean_squared_error(train_y, predicted_train_y)


# In[17]:

print mean_squared_train_error


# In[18]:

predicted_test_y = clf.predict(test_x)


# In[19]:

mean_absolute_test_error = mean_absolute_error(test_y, predicted_test_y)


# In[20]:

print mean_absolute_test_error


# In[21]:

mean_squared_test_error = mean_squared_error(test_y, predicted_test_y)


# In[22]:

print mean_squared_test_error


# In[23]:

confusion_matrix(test_y, predicted_test_y)


# In[24]:

print(predicted_test_y)


# In[25]:

bp = pd.read_csv('app_data/budget_planner.csv', sep=',', header=None, low_memory=False)


# In[26]:

bp = bp.iloc[1:2, 1:855]


# In[28]:

bp_y = clf.predict(bp)


# In[29]:

print bp_y


# In[30]:

bhim = pd.read_csv('app_data/bhim.csv', sep=',', header=None, low_memory=False)


# In[31]:

bhim = bhim.iloc[1:2,1:855]


# In[32]:

bhim_y = clf.predict(bhim)


# In[33]:

print bhim_y


# In[34]:

funnyys = pd.read_csv('app_data/funnys.csv', sep=',', header=None, low_memory=False)


# In[35]:

funnyys =  funnyys.iloc[1:2,1:855]


# In[36]:

fun_y = clf.predict(funnyys)


# In[37]:

print fun_y


# In[38]:

omingo = pd.read_csv('app_data/omingo.csv', sep=',', header=None, low_memory=False)


# In[39]:

omingo =  omingo.iloc[1:2,1:855]


# In[40]:

omingo_y = clf.predict(omingo)


# In[41]:

print omingo_y


# In[42]:

sys = pd.read_csv('app_data/system_administrator.csv', sep=',', header=None, low_memory=False)


# In[43]:

sys = sys.iloc[1:2,1:855]


# In[45]:

sys_y = clf.predict(sys)


# In[46]:

print sys_y


# In[47]:

tez = pd.read_csv('app_data/tez.csv', sep=',', header=None, low_memory=False)


# In[48]:

tez = tez.iloc[1:2,1:855]


# In[49]:

tez_y = clf.predict(tez)


# In[50]:

print tez_y


# In[51]:

mms = pd.read_csv('app_data/mms_beline.csv', sep=',', header=None, low_memory=False)


# In[52]:

mms = mms.iloc[1:2,1:855]


# In[53]:

mms_y = clf.predict(mms)


# In[54]:

print mms_y


# In[55]:

cal = pd.read_csv('app_data/calendar.csv', sep=',', header=None, low_memory=False)


# In[56]:

cal = cal.iloc[1:2,1:855]


# In[57]:

cal_y = clf.predict(cal)


# In[58]:

print cal_y


# In[59]:

laugh = pd.read_csv('app_data/laughtter.csv', sep=',', header=None, low_memory=False)


# In[60]:

laugh = laugh.iloc[1:2,1:855]


# In[61]:

laugh_y = clf.predict(laugh)


# In[62]:

print laugh_y


# In[63]:

andy = pd.read_csv('app_data/android_framework.csv', sep=',', header=None, low_memory=False)


# In[64]:

andy = andy.iloc[1:2,1:855]


# In[65]:

andy_y = clf.predict(andy)


# In[66]:

print andy_y

