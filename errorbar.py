#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

x = np.linspace(0,10,10);
y = np.sin(x/2);
yerr = 0.3 * np.ones(len(y));
xerr = 0.5 * np.ones(len(y));
e = plt.errorbar(x,y,yerr=err, xerr=err)


# In[22]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

x = np.linspace(0,10,10);
y = np.sin(x/2);
yerr = 0.3 * np.ones(len(y));
xerr = 0.5 * np.ones(len(y));
e = plt.errorbar(x,y,yerr=err, xerr=err, ecolor="red", elinewidth=0.5)

