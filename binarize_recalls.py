#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 11:10:02 2023

@author: WBR
"""

#%%

# 10_18_22 Convert idea units to binary variable then output to R
# Necessary for mixed effects logistic regression models
import pandas as pd
import numpy as np

#%%
exp_flag =21
# home_dir = '/Volumes/GoogleDrive/My Drive/grad_school/DML_WBR/dissertation_drive/cna_recall/rifa_exp2_mturk/'
# rdf = pd.read_csv(home_dir + 'rdf_n=170_11_5_21.csv')
home_dir = '/Volumes/GoogleDrive/My Drive/grad_school/DML_WBR/dissertation_drive/cna_recall/rifa_exp2_endo/'
rdf = pd.read_csv(home_dir + 'rdf_n=190_8_7_22.csv')

# gather idea units into single cell
rdf['str_idea_units'] = rdf['idea_units'].astype(str)
# rdf['all_units'] = rdf.groupby('mturk_id')['str_idea_units'].transform(lambda x: [x.tolist()]*len(x)) # A LIST OF STRINGS, BUT CAN'T DROP DUPLICATES ON LIST
rdf['all_units'] = rdf.groupby('mturk_id')['str_idea_units'].transform(lambda x: ','.join(x)) # NOT A LIST OF STRINGS
rdf = rdf.drop_duplicates(subset=['mturk_id','all_units'])
rdf['all_units'] = rdf['all_units'].apply(lambda x: x.split(',')) # A LIST OF STRINGS

# make df with idea units and merge with ids
ids = pd.DataFrame({'mturk_id':rdf.mturk_id.unique()})
ius = pd.DataFrame(ids.values.repeat(42, axis = 0), columns = ids.columns)
ius['idea_units_bin'] = np.arange(1,43).tolist() * 190
ius['idea_units_bin'] = ius['idea_units_bin'].astype(str)
# merge with rdf
rdf = rdf.merge(ius,on='mturk_id')

# populate correct column
rdf['correct'] = rdf.apply(lambda x: x.idea_units_bin in x.all_units, axis =1)
rdf['correct'] = rdf['correct'].astype(int)

# drop columns that will be joined from iu_df
to_drop = ['idea_units',
 'content',
 'source_sentence',
 'RPi_idea_bin',
 'RPp_idea_bin',
 'NRP_idea_bin',
 'RP_any',
 'RP_imp',
 'RP_per']
rdf = rdf.drop(to_drop,axis=1)
# need to rename for func to work. 
# Everywhere else,'idea units' means recalled, 
# here, 'correct' indicates recalled or not for each idea unit
rdf['idea_units'] = rdf['idea_units_bin'].astype(int)
rdf = rdf_conditionalizing(rdf)

df = merge_GM(rdf)
out = m_center(df,cvars=cvars)
out['RP_count_c'] = out['RP_count_c'].replace('',np.nan)
out.to_csv(home_dir + 'binary_correct_n=190_10_21_22.csv',index=False)

#%%