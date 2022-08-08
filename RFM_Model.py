#!/usr/bin/env python
# coding: utf-8

# # Customer segmentation
# 
# #### Best Customers	111	
# Customers who bought most recently, most often and spend the most. Strategy: No price incentives, New products and loyalty programs
# 
# #### Loyal Customers	X1X	
# Customers who bought most recently. Strategy: Use R and M to further segment.
# 
# #### Big Spenders XX1 
# Customers who spend the most. Strategy: Market your most expensive products.
# 
# #### Almost Lost	311	
# Haven't purchased for some time, but purchased frequently and spend the most. Strategy: Agressive price incentives
# 
# #### Lost Customers 411	
# Haven't purchased for some time, but purchased frequently and spend the most. Strategy: Agressive price incentives.
# 
# #### Lost Cheap Customers 444	
# Last purchase long ago, purchased few and spend little. Strategy: Don't spend too much trying to re-acquire.

# In[154]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[155]:


loans = pd.read_csv('withdrawals-2021_06_01-to-2023_08_01.csv',sep=',')


# In[156]:


loans.head()


# # Create the RFM Table
# 
# Since recency is calculated for a point in time and the loan dataset last order date is July 31st 2022, that is the date we will use to calculate recency.
# 
# Set this date to the current day and extract all orders until yesterday.

# In[157]:


import datetime as dt
NOW = dt.datetime(2022,7,31)


# In[158]:


# Make the date_placed column datetime
loans['completed'] = pd.to_datetime(loans['completed'])


# Create the RFM table

# In[159]:


rfmTable = loans.groupby('user_id').agg({'completed': lambda x: ((NOW - x.max()).days)+1, # Recency
                                        'loan_id': lambda x: len(x),      # Frequency
                                        'amount': lambda x: x.sum()}) # Monetary Value

rfmTable['completed'] = rfmTable['completed'].astype(int)
rfmTable.rename(columns={'completed': 'recency', 
                         'loan_id': 'frequency', 
                         'amount': 'monetary_value'}, inplace=True)


# # Validating the RFM Table

# In[160]:


rfmTable.head()


# # Determining RFM quantiles

# In[161]:


quantiles = rfmTable.quantile(q=[0.25,0.5,0.75])
quantiles


# Here the three quantiles for our rfm table have been determined. They will be sent to dictionary for easier use below

# In[162]:


quantiles = quantiles.to_dict()
quantiles


# # Creating RFM segmentation table

# In[163]:


rfmSegmentation = rfmTable


# We create two classes for the RFM segmentation since, having a high recency is bad, while high frequency and monetary value is good.

# In[164]:


# Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
def RClass(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
# Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
def FMClass(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1


# In[165]:


rfmSegmentation['R_Quartile'] = rfmSegmentation['recency'].apply(RClass, args=('recency',quantiles,))
rfmSegmentation['F_Quartile'] = rfmSegmentation['frequency'].apply(FMClass, args=('frequency',quantiles,))
rfmSegmentation['M_Quartile'] = rfmSegmentation['monetary_value'].apply(FMClass, args=('monetary_value',quantiles,))


# In[166]:


rfmSegmentation['RFMClass'] = rfmSegmentation.R_Quartile.map(str)                             + rfmSegmentation.F_Quartile.map(str)                             + rfmSegmentation.M_Quartile.map(str)


# In[167]:


rfmSegmentation.head()


# Who are the top 5 best customers? by RFM Class (111), high spenders who take loans recently and frequently?

# In[171]:


rfmSegmentation[rfmSegmentation['RFMClass']=='111'].sort_values('monetary_value', ascending=False).head()


# In[150]:


rfmSegmentation.to_csv('data.csv')

