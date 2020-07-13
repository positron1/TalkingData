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

# The processing method is from "Callum Gundlach LGBM Single Model LB: .9791"

most_freq_hours_in_test_data = [4,5,9,10,13,14]
least_freq_hours_in_test_data = [6, 11, 15]

def add_counts(df, cols):
    arr_slice = df[cols].values
    unq, unqtags, counts = np.unique(np.ravel_multi_index(arr_slice.T, arr_slice.max(axis=0)+1), return_inverse=True, return_counts=True)
    df["_".join(cols)+"_count"] = counts[unqtags]

def add_next_click(df):
    D = 2**26
    df['category'] = (df['ip'].astype(str) + "_" + df['app'].astype(str) + "_" + df['device'].astype(str) + "_" + df['os'].astype(str)).apply(hash) % D
    click_buffer = np.full(D, 3000000000, dtype=np.uint32)
    df['epochtime'] = df['click_time'].astype(np.int64) // 10 ** 9
    next_clicks = []
    for category, time in zip(reversed(df['category'].values), reversed(df['epochtime'].values)):
        next_clicks.append(click_buffer[category] - time)
        click_buffer[category] = time
    del click_buffer
    df['next_click'] = list(reversed(next_clicks))
    df.drop(['category', 'epochtime'], axis=1, inplace=True)

def ip_analysis(df):
    
    #Extrace date info
    df['click_time']= pd.to_datetime(df['click_time'])
    df['hour'] = df['click_time'].dt.hour.astype('uint8')
    df['day'] = df['click_time'].dt.day.astype('uint8')
    df['wday'] = df['click_time'].dt.dayofweek.astype('uint8')
    gc.collect()

    #Groups
    df['in_test_hh'] = ( 3 - 2 * df['hour'].isin( most_freq_hours_in_test_data ) - 1 * df['hour'].isin( least_freq_hours_in_test_data )).astype('uint8')

    print('Adding next_click...')
    add_next_click(df)

    print('Grouping...')
    
    add_counts(df, ['ip'])
    add_counts(df, ['os', 'device'])
    add_counts(df, ['os', 'app', 'channel'])
    add_counts(df, ['ip', 'device'])
    add_counts(df, ['app', 'channel'])
    add_counts(df, ['ip', 'wday', 'in_test_hh'])
    add_counts(df, ['ip', 'wday', 'hour'])
    add_counts(df, ['ip', 'os', 'wday', 'hour'])
    add_counts(df, ['ip', 'app', 'wday', 'hour'])
    add_counts(df, ['ip', 'device', 'wday', 'hour'])
    add_counts(df, ['ip', 'app', 'os'])
    add_counts(df, ['wday', 'hour', 'app'])
    
    #df.drop(['ip', 'day', 'click_time'], axis=1, inplace=True )
    df.drop(['day'], axis=1, inplace=True )
    gc.collect()

    #print( df.info() )

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


chunk_size = 35000000
predictions = pd.DataFrame() #predictions from different chunks
sub = pd.DataFrame()

#######Process the test set
extra_df = pd.read_csv('../input/test_supplement.csv', dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time'])
df = extra_df[(pd.to_datetime(extra_df['click_time']).dt.hour>=4)&(pd.to_datetime(extra_df['click_time']).dt.hour<=15)]
print('Size of extra set is {}'.format(extra_df.shape))
del extra_df
gc.collect()

df = ip_analysis(df)
#Here need to deal with the degeneracy in extra_df
df.drop_duplicates(keep='first',inplace=True)

test_df = pd.read_csv("../input/test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel'])
test_df['click_time'] = pd.to_datetime(test_df['click_time'])
test_df = pd.merge(test_df, df, on = list(test_df.columns), how='left')


print('Merged successfully!')
test_df.drop(['ip','click_time'], axis=1, inplace=True )


print('Size of test set is {}'.format(test_df.shape))
print('Test_df successfully processed!')

sub['click_id'] = list(range(test_df.shape[0]))


############ The function processing each chunk
def process(test_df,chunk_num): 
  
  print('Loading training data for chunk {}...'.format(chunk_num))

  df = pd.read_csv('../input/train.csv', dtype=dtypes, skiprows=range(1,184903891-chunk_size*chunk_num), nrows=chunk_size, 
  usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
 
  
  print('Using samples from {} to {}.'.format(chunk_size*(chunk_num-1),chunk_size*chunk_num))

  y = df['is_attributed']
  df = ip_analysis(df)
  df.drop(['is_attributed','ip','click_time'], axis=1, inplace=True)
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
print('Processing training data ...')

N_chunk = 2
sub['is_attributed'] = 1.

for j in range(N_chunk):
  sub['is_attributed'] = sub['is_attributed'] * process(test_df, j+1)


sub['is_attributed'] = sub['is_attributed']**(1./N_chunk)
sub.to_csv('sub1.csv', float_format='%.8f', index=False)
    