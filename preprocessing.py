#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 20:11:43 2023

@author: Walter Reilly
"""

# This script is run interactively. It combines the three experimental phases, engineers features, identifies bad data, and, finally, outputs clean dataframes for plotting in publication_ready_plots.py and computing inferential statistics in R

#%%
import pandas as pd
import glob
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
import re

#%% prepare raw recall csvs for manual recall scoring
def prep_recall_for_scoring(exp_flag):
    if exp_flag ==2:
        recall_dir = '/Volumes/GoogleDrive/My Drive/grad_school/DML_WBR/dissertation_drive/cna_recall/rifa_exp2_mturk/part2/results3/'
        save_dir = '/Volumes/GoogleDrive/My Drive/grad_school/DML_WBR/dissertation_drive/cna_recall/rifa_exp2_mturk/for_scoring/'
    elif exp_flag ==21:
        recall_dir = '/Volumes/GoogleDrive/My Drive/grad_school/DML_WBR/dissertation_drive/cna_recall/rifa_exp2_endo/part2/results/'
        save_dir = '/Volumes/GoogleDrive/My Drive/grad_school/DML_WBR/dissertation_drive/cna_recall/rifa_exp2_endo/for_scoring/'
    files = glob.glob(recall_dir + '*.csv')
    names = [Path(file).stem for file in files]
    
    for counter,file in enumerate(files):
        df = pd.read_csv(file,header=2)
        
        # weed out test files
        if len(df) < 5:
            continue
        try:
            # just want recall responses
            filt = df['trialText'].str.contains('Please try to recall')
            # filt = filt.na(False) supposed to be filt.fillna(false)?
            df = df[filt]
        except:
            print(file)
            continue
        
        out = df[['rowNo','response']].copy()
        out['sub_id'] = names[counter]
        
        # separte paragraph and sentence
        try:
            out[['para','sent']] = out['rowNo'].str.split('_',n=1,expand=True)
        except: #participants who skipped through task
            out[['para','sent']] = None
        
        out['sent'] = out['sent'].fillna(1)
        del out['rowNo']
                    
        # sick way to recode values. Data has row/trial value. This replaces with paragaph number
        vals =out['para'].unique()
        mapper ={k:i+1 for i,k in enumerate(vals)}
        out['para'] = out['para'].map(mapper)
        
        # make it nice for RAs
        out[['idea_units']] = '---'
        out = out[['sub_id','para','sent','idea_units','response']]
        
        # split up multi-sentence responses
        sent_split = lambda x: x.split('.')
        out = out.set_index(['sub_id','para'])
        out.loc[:,'response'] = out.response.apply(sent_split)
        out = out.explode('response') # explode is clutch!
        out = out[out['response'] != ''] # get rid of blank rows created by response terminated with '.'
        out = out.reset_index()
        out.loc[:,'sent'] = out.groupby((out['para'] != out['para'].shift(1)).cumsum()).cumcount()+1
        
        # drop everything except worker id from filename. Removes chronological file sorting
        name = re.sub('\d*_','',names[counter])
        
        if name in note_takers:
            out.to_csv(save_dir + 'note_takers/'+ name + '.csv',index=False)
        else:
            out.to_csv(save_dir + name + '.csv',index=False)
            
prep_recall_for_scoring(exp_flag=21)

#%% import data for analysis

global exp_flag
exp_flag = 21
# project path
if exp_flag ==2:
    home_dir = '/Volumes/GoogleDrive/My Drive/grad_school/DML_WBR/dissertation_drive/cna_recall/rifa_exp2_mturk/'
elif exp_flag ==21:
    home_dir = '/Volumes/GoogleDrive/My Drive/grad_school/DML_WBR/dissertation_drive/cna_recall/rifa_exp2_endo/'
# read in data 
def import_data_to_dfs(part_path,part_flag=0):
    files = glob.glob(part_path + '*.csv')
    names = [Path(file).stem for file in files]
    dfs=[]
    for counter,file in enumerate(files):
        try:
            # get some metadata
            id_df = pd.read_csv(file,header=0,nrows=1)
            link = id_df['link'][0]
            mturk_id = id_df['id'][0]
            comp_code = id_df['completionCode'] if part_flag == 1 else 0 # exp2 part 2 has no comp code, links to Gates-MacGintie instead
            timestamp = id_df['GMT_timestamp'].tolist()[0]
            
            # import df and attach metadata
            df = pd.read_csv(file,header=2)
            df[['mturk_id','comp_code','link','timestamp','file_id']] = [mturk_id,comp_code,link,timestamp,names[counter]]
            # df['mturk_id'] = df['mturk_id'].astype(int)
            
            # fix rowNo mixed dtype issue
            if df.dtypes['rowNo'].kind == 'O':
                df['rowNo'] =  df['rowNo'].str.replace('_','0')
                df['rowNo'] = pd.to_numeric(df['rowNo'])
            
            # subjectGroup 3 doesn't have these columns in part2
            if part_flag ==1 & ('responseType' in df.columns):
                df.drop('responseType',axis=1,inplace=True)
                df.drop('responseOptions',axis=1,inplace=True)
                
            dfs.append(df)
        except:
            print(file + 'error \n')    
    metadf = pd.concat(dfs, ignore_index = True)
    return metadf

def import_part1_raw():
    return import_data_to_dfs(home_dir +  'part1/results/',part_flag=1)

def import_part2_raw():
    part2 = import_data_to_dfs(home_dir +  'part2/results/') 
    # in RIFA experiment 1, part2 subjectGroup is 1 or NaN.  
    part2.drop('subjectGroup',inplace=True,axis=1)
    return part2

def import_part3_raw():
    # just get the goods from part3 and remove single row of test data (no mturk_id)
    if exp_flag ==2:
        part3_dirty = pd.read_csv(home_dir + 'part3/Gates-MacGintie_November+4,+2021_16.39.csv',header=0)
    elif exp_flag ==21:
        part3_dirty = pd.read_csv(home_dir + 'part3/Gates-MacGintie_March+8,+2022_20.08.csv',header=0)
    
    part3 = part3_dirty.loc[2:,['Q139','SC0','SC1','EndDate','IPAddress']].copy()
    part3['mturk_id'] = part3['Q139']
    part3.drop('Q139',inplace=True,axis=1)
    filt = part3['mturk_id'].notnull()
    part3 = part3[filt]
    
    part3[['SC0','SC1']] = part3[['SC0','SC1']].apply(pd.to_numeric)
    part3.columns = ['vocabulary','comprehension','EndDate','IPAddress','mturk_id']
    
    # drop second of any repeated submissions
    part3 = part3.sort_values('EndDate').drop_duplicates(subset='mturk_id',keep='first')

    return part3
    
def import_raw_3parts():
    part1 = import_part1_raw()
    part2 = import_part2_raw()        
    part3 = import_part3_raw()    
    return part1,part2,part3

def import_good_subs():
    if exp_flag == 2:
        df = pd.read_csv(home_dir + 'subs_included_n=170_11_8_21.csv')['mturk_id'].tolist()
    elif exp_flag ==21:
        df = pd.read_csv(home_dir + 'subs_included_n=190_3_8_22.csv')['mturk_id'].tolist()
    return df

# part3 = import_part3_raw()
# p3dir = '/Volumes/GoogleDrive/My Drive/grad_school/DML_WBR/dissertation_drive/cna_recall/paid_rifa/part3/'
# part3.to_csv(p3dir + 'paid_RIFA_gates-macgintie_6_16_21.csv',index=False)

#%% map part1 meta onto part2/df

def map_part1_meta_onto_df(df):
    part1 = import_part1_raw()
    # Grab Reading Time Data. rowNo = 10:36 are reading rows, RT is the outcome
    if exp_flag ==2:
        readrts = part1[part1['rowNo'].isin(np.arange(8,35))].copy()
    elif exp_flag ==21:
        readrts = part1[part1['rowNo'].isin(np.arange(8,41))].copy()
    readrts['mean_reading_ms'] = readrts.groupby('mturk_id')['RT'].transform('sum')
    readrts = readrts[['mturk_id','mean_reading_ms']].drop_duplicates()
    
    # grab others, merge with reading time
    sub_dict = part1[['mturk_id','subjectGroup','timestamp']].copy()
    sub_dict = sub_dict.merge(readrts,on='mturk_id')
    sub_dict = sub_dict[sub_dict['subjectGroup'].notnull()]
    sub_dict = sub_dict.drop_duplicates()
    sub_dict.rename(columns={'timestamp':'p1_timestamp'},inplace=True)
    df.rename(columns={'timestamp':'p2_timestamp'},inplace=True)
    df = df.merge(sub_dict,on='mturk_id') # 185 here
    
    # convenience processing
    df['p1_timestamp'] =  pd.to_datetime(df['p1_timestamp'].str.replace('_',' '))
    df['p2_timestamp'] =  pd.to_datetime(df['p2_timestamp'].str.replace('_',' '))
    if exp_flag ==2:
        df['p1_start'] = pd.to_datetime(df['p1_timestamp']) - pd.Timestamp('2021-08-31')
    elif exp_flag ==21:
        df['p1_start'] = pd.to_datetime(df['p1_timestamp']) - pd.Timestamp('2021-12-13')
        
    df['p1_start'] = df['p1_start'].astype('timedelta64[D]')
    df['time_diff'] = (df['p2_timestamp'] - df['p1_timestamp']).astype('timedelta64[h]')
    df['read_time_minutes'] = df['mean_reading_ms'].apply(lambda x: x/1000/60)
    # 185 here
    
    # instructions cols
    filt = part1[(~part1['trialText'].isnull()) & (part1['trialText'].str.contains('Instructions check!'))].copy() 
    filt['p1_instructions_acc'] = filt.groupby('mturk_id')['correct'].transform('mean')
    filt = filt[['mturk_id','p1_instructions_acc']].drop_duplicates()
    df= df.merge(filt,on='mturk_id')
    # 185 here
    
    # RP attempts. Doesn't distinguish between RP paragraph
    filt = part1[(~part1['trialText'].isnull()) & (part1['trialText'].str.contains('Search your memory'))].copy() 
    # only 168 here..search string isn't working. at least 185 finished part 1
    filt['RP_count'] = filt.groupby('mturk_id')['trialText'].transform('count')
    # RP RT
    filt['RP_RT'] = filt.groupby('mturk_id')['RT'].transform('sum')
    filt['RP_RT'] = filt['RP_RT'].apply(lambda x: x/1000/60)
    filt = filt[['mturk_id','RP_count','RP_RT']].drop_duplicates()
    df= df.merge(filt,on='mturk_id',how='left') # left join because group 3 subjects did no RP
    return df

#%% part2 instructions check

def p2_instructions_acc(part2):
    filt = part2[(~part2['trialText'].isnull()) & (part2['trialText'].str.contains('Instructions check!'))].copy() 
    filt['p2_instructions_acc'] = filt.groupby('mturk_id')['correct'].transform('mean')
    filt = filt[['mturk_id','p2_instructions_acc']].drop_duplicates()
    part2= part2.merge(filt,on='mturk_id')
    return part2
    
#%% continuous bio measure 
# part2 = import_part2_raw()
# filt = part2[(~part2['trialText'].isnull()) & (part2['trialText'].str.contains('How many biological'))].copy() 
# filt = filt[['mturk_id','response']]
# filt.to_csv(home_dir + 'bio_sciences_raw_3_8_22.csv',index=False) # opened in excel and cleaned responses manually due high variability in input

def continuous_bio(part2):
    if exp_flag ==2:
        df = pd.read_csv(home_dir + 'bio_sciences_clean_11_5_21.csv')
    elif exp_flag ==21:
        df = pd.read_csv(home_dir + 'bio_sciences_clean_3_8_22.csv')
    df = df[['mturk_id','response_clean']]
    df.rename(columns={'response_clean':'n_bio_courses'},inplace=True)
    df = df.drop_duplicates()
    part2 = part2.merge(df,on='mturk_id')
    return part2

#%% merge questionaire responses into part2 df 

def map_questionaire_responses(part2):
    rowNos = np.arange(26,35)
    filt = part2['rowNo'].isin(rowNos)
    q_aire = part2[filt].copy()
    q_aire = q_aire[['mturk_id','response','rowNo','trialText']].drop_duplicates() # this works, not sure why there are duplicates in mturk exp2 though
    # clean up response    
    q_aire['response'] = q_aire['response'].apply(lambda x: x if x.isnumeric() else x.split(';')[0])
    
    # look = q_aire.groupby(['mturk_id','rowNo']).count() # duplicate index error. us to id any remaining subjects with duplicate files
    
    piv = q_aire.pivot(index='mturk_id',columns ='rowNo',values='response')
    piv = piv.reset_index()
    piv.columns = ['mturk_id','bio_maj','bio_current','n_bio','familiarity','effort','acc_answers','use_data','p1_notes','p2_notes']
    
    # n_bio was cleaned up in continuous_bio function added later
    piv = piv.apply(lambda x: pd.to_numeric(x) if x.name not in ['mturk_id','n_bio'] else x)
    # recode response of 'no' (coded as 2) to 0
    recode = {2:0,1:1}
    piv = piv.apply(lambda x:  x.map(recode) if x.name in ['bio_maj','bio_current','acc_answers','use_data'] else x)
    # peek at correlations 
    # piv_cor  = piv.corr()
    
    return part2.merge(piv,on='mturk_id') # default on='inner'

#%% MC question meta data

def import_mc_meta():
    if exp_flag ==2:
        qdir='/Volumes/GoogleDrive/My Drive/grad_school/DML_WBR/dissertation_drive/cna_recall/'
        metaq = pd.read_csv(qdir + 'viruses_question_df.csv')
        metaq['q_id'] = metaq['q_num'].apply(str) + "_" + metaq['q_type'] + "_" + metaq['source_para'].apply(str) 
    elif exp_flag ==21:
        qdir='/Volumes/GoogleDrive/My Drive/grad_school/DML_WBR/dissertation_drive/cna_recall/'
        metaq = pd.read_csv(qdir + 'endocrine_question_df.csv')
        metaq['q_id'] = metaq['q_num'].apply(str) + "_" + metaq['q_type'] + "_" + metaq['source_para'].apply(str) 
    return metaq

def map_mc_meta_and_drop_fluff(part2):
    
    metaq = import_mc_meta()
    
    # prep for merge
    part2['question'] = part2['trialText']
    part2.drop('trialText', axis=1, inplace=True)  
    
    #merge 
    # inner join drops all non MC rows. Cool.
    return part2.merge(metaq, on='question',how='inner')

#%% filter out subjects

# if flag_logic = True, return df with participants who affirmed accurate answers
def acc_answers(df,flag_logic):
    if flag_logic:
        filt = df['acc_answers'] == 1
        return df[filt]
    else:
        filt = df['acc_answers'] == 0
        warnings.warn('You wanted bad data, right??')
        return df[filt]
    
# bad subs id'd for bad recall 
def rem_bad_subs(df,bad_subs):
    return df[~df['mturk_id'].isin(bad_subs)]

# good_subs are simply those included in final recall analysis, passing all checks
def good_recall_subs_only(df,good_subs): 
    return df[df['mturk_id'].isin(good_subs)]

# fraud_subs defined by repeat IP addresses and suspicious usernames
def rem_fraud_subs(df,fraud_subs):
    return df[~df['mturk_id'].isin(fraud_subs)]

#%% # convert  factor to category and order

def category_factor(df,var,ordered=False,categories=None):
    # df[var] = df[var].astype('category')
    if categories:
        cat_type = pd.CategoricalDtype(categories=categories, ordered=ordered)
    else:
        cat_type = pd.CategoricalDtype(categories=df[var].unique(), ordered=ordered)   
    df[var] = df[var].astype(cat_type)
    return df

#%%

def add_gates_to_df(df):
    clean_gates = import_part3_raw()
    return df.merge(clean_gates, on='mturk_id')

#%%
good_subs = import_good_subs()

# df merges probably messed up. Fix it 
part2 = (import_part2_raw()
          .pipe(map_part1_meta_onto_df) #subject group, reading rt, timestamp,p1 instructions_acc
          .pipe(p2_instructions_acc)
          .pipe(continuous_bio)
          .pipe(map_questionaire_responses)
          .pipe(map_mc_meta_and_drop_fluff)
          .pipe(acc_answers,flag_logic = True)
          .pipe(add_gates_to_df)
          .pipe(category_factor,var='subjectGroup',categories=['nsg:1','nsg:2','nsg:3'])
          .pipe(good_recall_subs_only,good_subs))        


part2.to_csv(home_dir + 'part2_df_uptoMC_3_8_22_good_subs.csv',index=False)
# part2 = pd.read_csv(home_dir + 'part2_df_uptoMC_11_5_21_good_subs.csv') 

#%% basic plotting function

def add_title(g,title):
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(title, fontsize=14)

def plot_funk(cat=False,hist=False,meta=False,scatter_meta=False,lm_meta=False,title=None,save_dir=None,args=None,kwargs={}):
    # best attempt at a plotting helper. 
    if cat:
        g = sns.catplot(**kwargs)
        add_title(g,title) if title else print('No title given\n')
        g.savefig(save_dir + 'bar_' + title + '.png') if save_dir else print('Did not save', title)
    if hist:
        g = sns.displot(**kwargs)
        add_title(g,title) if title else print('No title given\n')
        g.savefig(save_dir + 'hist_' + title + '.png') if save_dir else print('Did not save', title)
    if meta:
        g = sns.catplot(**kwargs)
        add_title(g,title) if title else print('No title given\n')
        g.savefig(save_dir + 'meta_bar_' + title + '.png') if save_dir else print('Did not save', title)
    if scatter_meta:
        g = sns.FacetGrid(**kwargs)
        g.map(sns.lmplot,*args)
        add_title(g,title) if title else print('No title given\n')
        g.savefig(save_dir + 'meta_scatter_'  + title + '.png') if save_dir else print('Did not save', title) 
    if lm_meta:
        g = sns.lmplot(**kwargs)
        add_title(g,title) if title else print('No title given\n')
        g.savefig(save_dir + 'meta_scatter_'  + title + '.png') if save_dir else print('Did not save', title) 
    if not cat|hist|meta|scatter_meta|lm_meta:
         print('\nSpecify plot type!!\n')
    # else:
        # return g
        
#%% META DATA PLOTS

# n good subs per group
nsubs = part2[['mturk_id','subjectGroup']].drop_duplicates().groupby(['subjectGroup']).count().reset_index()

# meta data by group
meta_df = pd.melt(part2,id_vars=['mturk_id','subjectGroup'],value_vars=['comprehension','vocabulary','familiarity', 'effort','bio_maj','bio_current', 
                         'acc_answers','use_data'], value_name='meta_val',var_name='meta_var').drop_duplicates()

if exp_flag ==2:
    sdir='/Volumes/GoogleDrive/My Drive/grad_school/DML_WBR/dissertation_drive/cna_recall/rifa_exp2_mturk/figures/n=170/'
elif exp_flag ==21:
    sdir='/Volumes/GoogleDrive/My Drive/grad_school/DML_WBR/dissertation_drive/cna_recall/rifa_exp2_endo/figures/'
title='Meta Data'
cat_args = {'data':meta_df,'col':'meta_var','y':'meta_val','x':'subjectGroup',
            'kind':'bar','ci':68,'sharey':False,'col_wrap':4}
g = plot_funk(kwargs=cat_args,cat=True,save_dir=sdir,title='Meta Data')

#%% MC plot functions
         
def mc_grand_group_mean(part2,save_dir=None,title=None):
    data = part2.copy()
    data['sub_mean']= part2.groupby(['mturk_id'])['correct'].transform('mean')
    data = data.drop_duplicates(subset=['mturk_id','sub_mean'])
    cat_args = {'data':data,'x':'subjectGroup','y':'sub_mean','kind':'bar','ci':68}
    plot_funk(kwargs=cat_args,cat=True,save_dir=save_dir,title=title)
    return data
    
def mc_2factor_mean(part2,factor1,factor2,save_dir=None,title=None):
    data = part2.copy()
    data['sub_mean']= part2.groupby(['mturk_id',factor1,factor2])['correct'].transform('mean')
    data = data.drop_duplicates(subset=['mturk_id','sub_mean',factor1,factor2])
    cat_args = {'data':data,'x':factor1,'col':factor2,'y':'sub_mean','hue':'subjectGroup','kind':'bar','ci':68}
    plot_funk(kwargs=cat_args,cat=True,save_dir=save_dir,title=title)
    
def mc_factor_mean(part2,factor,save_dir=None,title=None,meta=False,lm_meta=False):

    df = part2.copy()
    df['sub_mean']= df.groupby(['mturk_id',factor])['correct'].transform('mean')
    df = df.drop_duplicates(subset=['mturk_id','sub_mean',factor])
        
    if not meta or lm_meta:
        cat_args = {'data':df,'x':factor,'y':'sub_mean','hue':'subjectGroup','kind':'bar','ci':68}
        plot_funk(kwargs=cat_args,cat=True,save_dir=save_dir,title=title)

    if meta:
        #convert two continuous to qtile so can be plotted as categorical data
        #format as long data so meta_var can be passed as plotting factor
        
        def qcut_func(x):
            return pd.qcut(x,q=2,labels=range(1,3))
        
        df['p1_start_qtile'] = df['p1_start'].transform(lambda x: qcut_func(x))        
        df['fam_qtile'] = df['familiarity'].transform(lambda x: qcut_func(x))
        # df['eff_qtile'] = df['effort'].transform(lambda x: qcut_func(x))
        df['vocab_qtile'] = df['vocabulary'].transform(lambda x: qcut_func(x))
        df['comp_qtile'] = df['comprehension'].transform(lambda x: qcut_func(x))
        df['read_qtile'] = df['read_time_minutes'].transform(lambda x: qcut_func(x))
        df['RP_qtile'] = df['RP_count'].transform(lambda x: qcut_func(x))
        df['RP_RT_qtile'] = df['RP_RT'].transform(lambda x: qcut_func(x))
                   
        meta_df = pd.melt(df,id_vars=['mturk_id'],value_vars=['comp_qtile','vocab_qtile','fam_qtile', 
                        'bio_maj','bio_current', 'read_qtile','RP_qtile','RP_RT_qtile','acc_answers','use_data','p1_start_qtile'], value_name='meta_val',var_name='meta_var') # 'eff_qtile',
        df = df.merge(meta_df, on='mturk_id')

        #plot
        kwargs = {'data':df,'x':'meta_val','hue':'subjectGroup','y':'sub_mean','kind':'bar','ci':68,
                'col':'meta_var','row':factor,'sharex':False}
        plot_funk(meta=True,kwargs=kwargs,save_dir=save_dir,title=title)
        
    if lm_meta:
        meta_df = pd.melt(df,id_vars=['mturk_id'],value_vars=['comprehension','vocabulary','familiarity',
                        'effort','read_time_minutes','RP_count','RP_RT','n_bio_courses','p1_start'], value_name='meta_val',var_name='meta_var')
        df = df.merge(meta_df, on='mturk_id')
        kwargs = {'data':df,'x':'meta_val','hue':'subjectGroup','y':'sub_mean',
                'row':factor,'col':'meta_var','sharex':False}
        plot_funk(lm_meta=True,kwargs=kwargs,save_dir=save_dir,title=title)

#%%

look = mc_grand_group_mean(part2,title='MC Group Mean Barplot',save_dir=sdir)
look.groupby('subjectGroup').agg({'sub_mean':['count','std','mean']})
look.agg({'sub_mean':['count','std','mean']})
# mc_factor_mean(part2,factor='source_para',title='MC x Source Para',save_dir=sdir)
mc_factor_mean(part2,factor='q_type',title='MC x Q type',save_dir=sdir)
mc_factor_mean(part2,factor='q_num',title='MC x q_num',save_dir=sdir)
# mc_2factor_mean(part2, factor1='q_type', factor2='source_para',title='MC x q_type x source_para',save_dir=sdir)

mc_factor_mean(part2,factor='q_type',title='MC x Q type',meta=True,save_dir=sdir)
mc_factor_mean(part2,factor='q_type',title='MC x Q type',lm_meta=True,save_dir=sdir)

# cross with bio maj
# mc_factor_mean(part2,factor='bio_maj',title='MC x bio_maj',save_dir=sdir)

#%% RECALL ANALYSES 

#%% Prepare sentence importance as covariate

def import_and_standardize_importance():
    idir = '/Volumes/GoogleDrive/My Drive/grad_school/DML_WBR/dissertation_drive/cna_recall/'
    if exp_flag ==2:
        imp = pd.read_csv(idir + 'viruses_node_df.csv')
    elif exp_flag==21:
        imp = pd.read_csv(idir + 'endocrine_extra_peripheral_node_df.csv')
    filt = imp['depth'] ==2
    imp =imp[filt]
    
    # standardize importance 
    imp['s_importance'] = imp['importance'].transform(lambda x:(x - x.mean())/x.std())
    return imp

def import_edge_df():
    idir = '/Volumes/GoogleDrive/My Drive/grad_school/DML_WBR/dissertation_drive/cna_recall/'
    if exp_flag ==2:
        edge_df = pd.read_csv(idir + 'viruses_edge_df.csv')
    elif exp_flag ==21:
        edge_df = pd.read_csv(idir + 'endocrine_extra_peripheral_edge_df.csv')
    return edge_df

#%%
# RIFA recall subjectGroup merge on file_id to link recall with mturk_id

# import recall df
def import_recall_df(incl_zeros=False):
    if exp_flag ==2:
        if incl_zeros:
            rdf = pd.read_csv(home_dir + 'recall_wbr_scored_11_5_21_including_zero_idea_units.csv')
        else:
            rdf = pd.read_csv(home_dir + 'recall_wbr_scored_11_5_21.csv')
    elif exp_flag ==21:
        if incl_zeros:
            NotImplementedError
        else:
            rdf = pd.read_csv(home_dir + 'recall_wbr_scored_3_8_22.csv')

    # fix column name incosisency
    rdf['file_id'] = rdf['sub_id']
    rdf.drop('sub_id',inplace=True,axis=1)
    return rdf

# import part2/mc df
def import_part2_mc_df():
    if exp_flag ==2:
        mc_df = pd.read_csv(home_dir + 'part2_df_uptoMC_11_5_21_all.csv') 
    elif exp_flag ==21:
        mc_df = pd.read_csv(home_dir + 'part2_df_uptoMC_3_8_22_all_subs.csv') 
    return mc_df

def extract_metadata_cols(p2df):
    meta_cols = p2df[['mturk_id','file_id','subjectGroup','bio_maj','bio_current', 'n_bio_courses',
                      'familiarity', 'effort','p1_notes', 'p2_notes', 'acc_answers','use_data','p1_timestamp',
                      'mean_reading_ms', 'time_diff', 'read_time_minutes','p1_start',
                      'p1_instructions_acc','p2_instructions_acc','RP_count','RP_RT']].copy()
    return  meta_cols.drop_duplicates()

def add_meta_to_rdf(rdf):
    p2df = import_part2_mc_df()
    meta_cols = extract_metadata_cols(p2df)
    rdf = rdf.merge(meta_cols, on='file_id')
    return rdf

#%% RIFA recall conditionalizing funcs

def import_idea_units_df():
    if exp_flag ==2:
        iu_df = pd.read_csv(home_dir + 'viruses_idea_units_dataframe_exp2.csv')
    elif exp_flag ==21:
        iu_df = pd.read_csv(home_dir + 'endocrine_idea_units_df.csv')
    return iu_df
    
def RP_func(x):
    if x['subjectGroup']  == 'nsg:1':
        return x['RPi_idea_bin']
    elif x['subjectGroup']  == 'nsg:2':
        return x['RPp_idea_bin']
    elif x['subjectGroup']  == 'nsg:3':
        return x['NRP_idea_bin']

def rdf_conditionalizing(rdf):
    iu_df = import_idea_units_df()
    rdf = rdf.merge(iu_df,on='idea_units')
    rdf['RP_bins'] = rdf.apply(RP_func,axis=1)
    return rdf
    
# add an arbitrary column of 1s so I can count and transform..
def arbitrary_col(rdf):
    rdf['arbitrary'] = 1
    return rdf

def condition_n(df):
    nstats = df.groupby('subjectGroup')['mturk_id'].nunique().reset_index()
    nstats = nstats.set_index('subjectGroup')
    nstats.rename(columns={"mturk_id": "condition_n"},inplace=True)
    return df.merge(nstats,on='subjectGroup')

#%% 

def standardize_meta(rdf,meta):
    aggs = rdf[['mturk_id',meta]].drop_duplicates().agg({meta : ['mean','std']}) # compute summary stats excluding duplicates within subject
    rdf['standardized_' + meta] = rdf[meta].apply(lambda x: (x - aggs['mean'])/aggs['std']) # standardize meta
    return rdf

def m_center(df,cvars):    
    for var in cvars:
        df[var+'_c'] = df[var].transform(lambda x: x-x.mean())
    return df

#%%    
# remove idea units used as cues
def remove_cues(df):
    if exp_flag ==2:
        # there was something called experiment 2 but exp2_mturk isn't it and those data aren't being used
        NotImplementedError #df = df[~df['idea_units'].isin([12,23,24,32,33,34,41,42])]
    elif exp_flag ==21:
        NotImplementedError
    return df

#%% QC functions

def outlier_cols(df):
    df['out_comprehension'] = df['comprehension'].transform(lambda x: 1 if x < 10 else 0)
    df['out_vocabulary'] = df['vocabulary'].transform(lambda x: 1 if x < 10 else 0)
    df['out_read'] = df['read_time_minutes'].transform(lambda x: 1 if x > 10 else 0)
    df['out_effort'] = df['effort'].transform(lambda x: 1 if x < 60 else 0)
    df['out_p1_instructions'] = df['p1_instructions_acc'].transform(lambda x: 1 if x < .75 else 0)
    df['out_read'] = df['read_time_minutes'].transform(lambda x: 1 if x < 1 else 0)
    
    return df 

def rem_low_comprehension(df):
    df = df[~(df['comprehension'] < 10)]  
    return df 

def rem_low_vocabulary(df):
    df = df[~(df['vocabulary'] < 20)] 
    return df 

def rem_slow_read(df):
    df = df[~(df['read_time_minutes'] >  10)] 
    return df 

def rem_low_effort(df):
    df = df[~(df['effort'] <  60)]
    return df 

def rem_instructions_acc(df):
    df = df[~(df['p1_instructions_acc'] <  .75)] 
    return df 

def rem_read(df):
    df = df[~(df['read_time_minutes'] <  1)] 
    return df 

#%% bad data

if exp_flag ==2:
    bad_subs = ['A1EG4DWTRISH7X','A1WRJZC8E2ZCR0'] # single note taker who slipped through and 35 unit recall
elif exp_flag==21:
    bad_subs=['A2J43GKTP92XM2','A2DQPK76L6YR7A']  # outlier for 36 idea units recalled. Probable note taker. A2DQ...reported taking notes..slipped through into scored recalls

#%%

rdf = (import_recall_df()
       .pipe(add_meta_to_rdf)
       .pipe(add_gates_to_df)
       .pipe(rdf_conditionalizing)
       .pipe(acc_answers,flag_logic=True)
       .pipe(arbitrary_col)
       .pipe(rem_bad_subs,bad_subs)
       .pipe(category_factor,var='subjectGroup',categories=['nsg:1','nsg:2','nsg:3'])
       .pipe(outlier_cols)
       .pipe(rem_instructions_acc)
       .pipe(rem_low_comprehension)
       .pipe(rem_low_vocabulary)
       .pipe(rem_slow_read)
       .pipe(rem_low_effort)
       .pipe(rem_read)
       .pipe(condition_n))

#%%
rdf.mturk_id.nunique() 
rdf.drop_duplicates(subset=['mturk_id','subjectGroup']).groupby('subjectGroup').count()

# save good subs for use in MC
rdf[['mturk_id']].drop_duplicates()['mturk_id'].to_csv(home_dir + 'subs_included_n=190_3_8_22.csv',index=False)

good_subs = import_good_subs()
rdf = rdf[rdf.mturk_id.isin(good_subs)]
rdf.to_csv(home_dir + 'rdf_n=190_8_7_22.csv',index=False)

# rdf = pd.read_csv(home_dir + 'rdf_n=170_11_5_21.csv') 
rdf = pd.read_csv(home_dir + 'rdf_n=190_8_7_22.csv')
rdf = category_factor(rdf, var='subjectGroup',categories=['nsg:1','nsg:2','nsg:3'])
#%% visual inspection of outliers

look = rdf[['effort','comprehension','vocabulary','read_time_minutes','p1_instructions_acc','p2_instructions_acc','RP_count','RP_RT','mturk_id']].drop_duplicates()

look['s_comp'] = look['comprehension'].transform(lambda x: (x-x.mean())/x.std())
look['s_voc'] = look['vocabulary'].transform(lambda x: (x-x.mean())/x.std())
look['s_read'] = look['read_time_minutes'].transform(lambda x: (x-x.mean())/x.std())
look['s_RP'] = look['RP_count'].transform(lambda x: (x-x.mean())/x.std())
look['s_RP_RT'] = look['RP_RT'].transform(lambda x: (x-x.mean())/x.std())
look['s_eff'] = look['effort'].transform(lambda x: (x-x.mean())/x.std())

kwargs = {'data':look,'x':'effort','binwidth':1,'height':4}
plot_funk(hist=True,title=None,kwargs=kwargs,save_dir=None) # effort 60% or z < -2

kwargs = {'data':look,'x':'comprehension','binwidth':1,'height':4}
plot_funk(hist=True,title=None,kwargs=kwargs,save_dir=None) # comprehension 15 or z < -2

kwargs = {'data':look,'x':'vocabulary','binwidth':1,'height':4}
plot_funk(hist=True,title=None,kwargs=kwargs,save_dir=None) # vocabulary 28 or z < -2

kwargs = {'data':look,'x':'read_time_minutes','binwidth':.25,'height':4}
plot_funk(hist=True,title=None,kwargs=kwargs,save_dir=None) # read time greater than 6 or z < 2

kwargs = {'data':look,'x':'RP_count','binwidth':1,'height':4}
plot_funk(hist=True,title=None,kwargs=kwargs,save_dir=None) 

kwargs = {'data':look,'x':'RP_RT','binwidth':1,'height':4}
plot_funk(hist=True,title=None,kwargs=kwargs,save_dir=None) 
# in actaul data anything over 22 looked like shit. call it over 25 or include as covariate?
# can't use SD because average subject was so shitty..

# combinations of exclusions 
look = rdf[['mturk_id','out_comprehension', 'out_vocabulary', 'out_read','out_effort', 'out_p1_instructions']].drop_duplicates()
look = look.set_index('mturk_id')
look['out_sum'] = look.sum(axis=1) # 29 subs excluded
sum(look.out_sum == 0)