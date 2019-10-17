#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
counts = pd.read_csv('dev_rsc/SeattleBike-master/FremontHourly.csv', index_col='Date', parse_dates=True)
weather = pd.read_csv('dev_rsc/SeattleBike-master/SeaTacWeather.csv', index_col='DATE', parse_dates=True)


# In[3]:


daily = counts.resample('d', how='sum')
daily['Total'] = daily.sum(axis=1)
daily = daily[['Total']] # remove other columns
daily.head()


# In[4]:


days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i in range(7):
    daily[days[i]] = (daily.index.dayofweek == i).astype(float)
daily.head()


# In[5]:


from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays('2012', '2016')
print(holidays)


# In[6]:


print(pd.Series(1, index=holidays, name='holiday'))


# In[7]:


daily = daily.join(pd.Series(1, index=holidays, name='holiday'))
daily['holiday'].fillna(0, inplace=True)
daily.head()


# In[26]:


def hours_of_daylight(date, axis=23.44, latitude=47.61):
            """Compute the hours of daylight for the given date"""
            days = (date - pd.datetime(2000, 12, 21)).days
            m = (1. - np.tan(np.radians(latitude))
                 * np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.25)))
            return 24. * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.


# In[10]:


# temperatures are in 1/10 deg C; convert to C
weather['TMIN'] /= 10
weather['TMAX'] /= 10
weather['Temp (C)'] = 0.5 * (weather['TMIN'] + weather['TMAX'])

# precip is in 1/10 mm; convert to inches
weather['PRCP'] /= 254
weather['dry day'] = (weather['PRCP'] == 0).astype(int)

daily = daily.join(weather[['PRCP', 'Temp (C)', 'dry day']])
daily.head()


# In[11]:


daily['annual'] = (daily.index - daily.index[0]).days / 365.
daily.head()


# In[12]:


column_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'holiday',
                'daylight_hrs', 'PRCP', 'dry day', 'Temp (C)', 'annual']
X = daily[column_names]
y = daily['Total']

from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=False)
model.fit(X, y)
daily['predicted'] = model.predict(X)


# In[28]:


daily['daylight_hrs'] = list(map(hours_of_daylight, daily.index))
plt.plot(daily[['daylight_hrs']])
plt.xticks(rotation=90)


# In[13]:


daily[['Total', 'predicted']].plot(alpha=0.5);


# In[14]:


params = pd.Series(model.coef_, index=X.columns)
params


# In[15]:


from sklearn.utils import resample
np.random.seed(1)
err = np.std([model.fit(*resample(X, y)).coef_
for i in range(1000)], 0)


# In[16]:


print(pd.DataFrame({'effect': params.round(0),'error': err.round(0)}))


# In[25]:


daily[['daylight_hrs']]


# In[ ]:




