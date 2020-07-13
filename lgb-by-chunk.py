# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import gc
import time
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

import lightgbm as lgb
from lightgbm import plot_importance
import matplotlib.pyplot as plt
import sys

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

def ip_analysis(df):
 #Adding time features
 df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
 df['minute'] = pd.to_datetime(df.click_time).dt.minute.astype('uint8')
 df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
 
 #General features about IP, good channels and apps
 df['ip_device_os_channel_count'] = df.groupby(['ip','device','os','channel'])['app'].transform('count').astype('uint16')
 df['ip_device_os_app_count'] = df.groupby(['ip','device','os','app'])['channel'].transform('count').astype('uint16')

 #Disregarding IP, time-related behaviors
 df['device_os_time_channel_count'] = df.groupby(['device','os','day','hour','minute','channel'])['app'].transform('count').astype('uint16')
 df['device_os_time_channel_nunique'] = df.groupby(['device','os','day','hour','minute','channel'])['app'].transform('nunique').astype('uint16')
 df['device_os_time_channel_avg'] = df['device_os_time_channel_count']/df['device_os_time_channel_nunique']

 df['device_os_time_app_count'] = df.groupby(['device','os','day','hour','minute','app'])['channel'].transform('count').astype('uint16')
 df['device_os_time_app_nunique'] = df.groupby(['device','os','day','hour','minute','app'])['channel'].transform('nunique').astype('uint16')
 df['device_os_time_app_avg'] = df['device_os_time_app_count']/df['device_os_time_app_nunique']
 
 df['ip_day_hour_channel'] = df.groupby(['ip','day','hour'])['channel'].transform('count').astype('uint16')
 df['ip_app_channel'] = df.groupby(['ip','app'])['channel'].transform('count').astype('uint16')
 df['ip_app_os_channel'] = df.groupby(['ip','app','os'])['channel'].transform('count').astype('uint16')
 df['ip_day_app_hour'] =df.groupby(['ip','app','channel'])['hour'].transform('mean')
 
 #Drop useless features
 df.drop('day',axis=1,inplace=True)   
 df.drop('minute',axis=1,inplace=True)
 df.drop(['click_time'], axis=1, inplace=True)
 df.drop(['ip'], axis=1, inplace=True)
 
 return df


dtypes = {'ip': 'uint32','app': 'uint16','device': 'uint16','os': 'uint16','channel': 'uint16','is_attributed' : 'uint8'}



############ The LGB parameters
lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':'auc',
        'learning_rate': 0.1,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0,
        'scale_pos_weight':99
    }


chunk_size = 20000000
predictions = pd.DataFrame() #predictions from different chunks
sub = pd.DataFrame()

print('loading test data...')
test_df = pd.read_csv("../input/test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time','click_id'])
test_df = ip_analysis(test_df)

sub['click_id'] = test_df['click_id'].astype('int')
test_df.drop(['click_id'], axis=1, inplace=True)



############ The function processing each chunk
def process(test_df,chunk_num): 
  
  print('Loading training data for chunk {}...'.format(chunk_num))

  df = pd.read_csv('../input/train.csv', dtype=dtypes, skiprows=range(1,184903891-chunk_size*chunk_num), nrows=chunk_size, 
  usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
 
  
  print('Using samples from {} to {}.'.format(chunk_size*(chunk_num-1),chunk_size*chunk_num))

  y = df['is_attributed']
  df = ip_analysis(df)
  df.drop(['is_attributed'], axis=1, inplace=True)
  predictors = list(df.columns)
  categorical = ['app','device','os','channel']
  
  x1, x2, y1, y2 = train_test_split(df, y, test_size=0.15, random_state=99)
  del df,y
  gc.collect()

  print('Starting the training for chunk {}...'.format(chunk_num))
  lgb_train = lgb.Dataset(x1.values, label=y1.values, feature_name=predictors, categorical_feature=categorical)
  lgb_valid = lgb.Dataset(x2.values, label=y2.values, feature_name=predictors, categorical_feature=categorical)


  del x1, y1, x2, y2 
  gc.collect()

  model_lgb = lgb.train(lgb_params, lgb_train, valid_sets=lgb_valid, num_boost_round=1500, early_stopping_rounds=30, verbose_eval=10)
  del lgb_train, lgb_valid
  gc.collect()

  print('Training for chunk {} finished. Making predictions...'.format(chunk_num))
  #print('The LGB model takes {:.0f} MB'.format(sys.getsizeof(model_lgb)/2**20))

  #Plot the feature importance from lightGBM
  plot_importance(model_lgb)
  fig_name = 'feature_importance_lgb_'+str(chunk_num)+'.png'
  plt.gcf().savefig(fig_name)
  
  return model_lgb.predict(test_df, num_iteration=model_lgb.best_iteration) 


############ The final prediction
print('Final prediction...')

N_chunk = 9
sub['is_attributed'] = 1.

for j in range(N_chunk):
  sub['is_attributed'] = sub['is_attributed'] * process(test_df, j+1)


sub['is_attributed'] = sub['is_attributed']**(1./N_chunk)
sub.to_csv('sub1.csv', float_format='%.8f', index=False)

