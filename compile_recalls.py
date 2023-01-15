#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 10:40:33 2023

@author: WBR
"""

# Import and compile manually scored recalls

import glob.glob
import pandas as pd

#%%
global exp_flag
exp_flag = 21
# project path
if exp_flag ==2:
    home_dir = '/Volumes/GoogleDrive/My Drive/grad_school/DML_WBR/dissertation_drive/cna_recall/rifa_exp2_mturk/'
elif exp_flag ==21:
    home_dir = '/Volumes/GoogleDrive/My Drive/grad_school/DML_WBR/dissertation_drive/cna_recall/rifa_exp2_endo/'
#%% import and compile individual recalls into one df

def rater_loop(data_dirs):
    dfs=[]
    for idir in data_dirs:
        files = glob.glob(idir + '*.csv')
        sub_dfs = []
        for file in files:
            # read in scored file
            try:
                df = pd.read_csv(file,header=0)
                df = df.set_index(['sub_id','response','para','sent']).stack().str.split(',',expand=True).stack().unstack(-2).reset_index(-1,drop=True).reset_index()
                sub_dfs.append(df)
            except:
                print('error in'+ file+'\n')
        # create 1 df from n subject df's
        dfs.append(pd.concat(sub_dfs, ignore_index = True))
    return dfs # dfs[0],dfs[1]
        
#%%

data_dirs =[home_dir + 'wbr_scored_2_14_22/']
dfs =rater_loop(data_dirs)
df = dfs[0] # just using wbr's scores

# some cleanup
filt = df['idea_units'] == '---' 
df = df[~filt]
filt = df['idea_units'] == 'None'
df = df[~filt]
filt = df['idea_units'] == '0'
df = df[~filt]

#%%
df['idea_units'] = pd.to_numeric(df['idea_units'])

# import recall key
if exp_flag ==2:
    recall_key = pd.read_csv('/Volumes/GoogleDrive/My Drive/grad_school/DML_WBR/dissertation_drive/cna_recall/viruses_recall_key.csv',header=0)
elif exp_flag ==21:
    recall_key = pd.read_csv('/Volumes/GoogleDrive/My Drive/grad_school/DML_WBR/dissertation_drive/cna_recall/endocrine_recall_key.csv',header=0)

# new column for the SOURCE paragrph
df  = df.merge(recall_key, left_on='idea_units', right_on='idea_unit',suffixes=['','_y']) 
df['para_idea_source'] = df['para_y']   
df.drop(['idea_unit','para_y'],inplace=True,axis=1)
    
# new column for source/recall paragraph match    
filt = df.para == df.para_idea_source
df.loc[filt,'para_match'] = 1
filt = df.para != df.para_idea_source
df.loc[filt,'para_match'] = 0

sumstats = df.groupby('sub_id',as_index=False)['para_match'].agg({'sum':'sum','count':'count','mean':'mean'})
sumstats['count'].mean()

df.to_csv(home_dir + 'recall_wbr_scored_3_8_22.csv',index = False) # excludes 0's
# df.to_csv(home_dir + 'recall_wbr_scored_11_5_21.csv',index = False)
# df.to_csv(home_dir + 'recall_wbr_scored_11_5_21_including_zero_idea_units.csv',index = False)
#%%