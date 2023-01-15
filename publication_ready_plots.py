#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 17:04:44 2022

@author: Walter Reilly

"""

# This script produces the figures included in Reilly et al.'s investigation of selective retrieval effects on recall of science text information.
# This project was designed to investigate the consequences of practicing some information on a final test on all information. I was therefore interested in not only how much information was recalled, but which specific ideas were recalled, in which order, and which condition each participant had been assigned to. Furthermore, I collected several individual difference measures and explored their interactions with recall performance. These plotting functions were crucial to the iterative experimental designed process, which involved two pilot experiments prior to the final two experiments included in Reilly et al. These functions and this script were pared down to include only what was used in the published analayses.

#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib

#%% dir setup

global exp_flag
exp_flag = 21

if exp_flag ==2:
    home_dir = '/Volumes/GoogleDrive/My Drive/grad_school/DML_WBR/dissertation_drive/cna_recall/rifa_exp2_mturk/'
    sdir='/Volumes/GoogleDrive/My Drive/grad_school/DML_WBR/dissertation_drive/cna_recall/rifa_exp2_mturk/figures/pub_ready/'
elif exp_flag ==21:
    home_dir = '/Volumes/GoogleDrive/My Drive/grad_school/DML_WBR/dissertation_drive/cna_recall/rifa_exp2_endo/'
    sdir='/Volumes/GoogleDrive/My Drive/grad_school/DML_WBR/dissertation_drive/cna_recall/rifa_exp2_endo/figures/pub_ready/'

#%% helper functions

def add_title(g,title):
    g.fig.subplots_adjust(top=.85)
    g.fig.suptitle(title, fontsize=22)
    
# adds subjects who were dropped froma plot for not having anY idea units recalled in a bin
# after adding subjects back, zeros can be added to the colum of interest
def add_0recall_subs(df,rdf):
    all_subs = import_good_subs()
    missing_subs = list(set(all_subs) - set(df.mturk_id.unique().tolist()))
    rdf_missing = rdf[rdf.mturk_id.isin(missing_subs)]
    return pd.concat([df,rdf_missing])

def category_factor(df,var,ordered=False,categories=None):
    if categories:
        cat_type = pd.CategoricalDtype(categories=categories, ordered=ordered)
    else:
        cat_type = pd.CategoricalDtype(categories=df[var].unique(), ordered=ordered)   
    df[var] = df[var].astype(cat_type)
    return df

def import_good_subs():
    if exp_flag == 2:
        df = pd.read_csv(home_dir + 'subs_included_n=170_11_8_21.csv')['mturk_id'].tolist()
    elif exp_flag ==21:
        df = pd.read_csv(home_dir + 'subs_included_n=190_3_8_22.csv')['mturk_id'].tolist()
    return df

def change_condition_labs(df):
    di = {"nsg:1":'RPm',"nsg:2":'RPp',"nsg:3":'NRP'}
    df['subjectGroup'] = df['subjectGroup'].map(di)
    df = category_factor(df, var='subjectGroup',categories=['RPm','RPp','NRP'])
    return df

def change_mc_labs(df):
    di = {"memory":'Detail',"inference":'Inference'}
    df['q_type'] = df['q_type'].map(di)
    return df

def merge_GM(df):
    df['GMRT'] = df['comprehension']  + df['vocabulary']
    return df

def m_center(df,cvars):    
    for var in cvars:
        df[var+'_c'] = df[var].transform(lambda x: x-x.mean())
    return df

def scale_vars(df,cvars):
    for var in cvars:
        df[var+'_s'] = df[var].transform(lambda x: x/x.std())
    return df

def pub_plot(g,base=None,ymin=None,ymax=None):
    g.set_titles("")
    g.set_titles(col_template = "{col_name}") 

    loc = plticker.MultipleLocator(base=base) # this locator puts ticks at regular intervals
    for ax in g.axes.flat:
        ax.set(ylim=(ymin, ymax))
        ax.yaxis.set_major_locator(loc)
        ax.yaxis.set_major_formatter(plticker.ScalarFormatter())
        ax.set_xlabel("")
        ax.tick_params(bottom=False)
    # plt.tick_params(axis='both', which='major', labelsize=14)


#%% import rdf

if exp_flag == 2:
    rdf = pd.read_csv(home_dir + 'rdf_n=170_11_5_21.csv') 
elif exp_flag ==21:
    rdf = pd.read_csv(home_dir + 'rdf_n=190_8_7_22.csv')

rdf = change_condition_labs(rdf)

#%%

def plot_funk(cat=False,base=None,ymin=None,ymax=None,bar=False,hist=False,meta=False,scatter_meta=False,lm_meta=False,title=None,save_dir=None,args=None,kwargs={}):
    sns.set_style("ticks")
    params = {'axes.facecolor':'white', 'figure.facecolor':'white','figure.dpi': 300,'axes.labelsize': 18,'axes.titlesize':20, 'legend.fontsize': 20, 'xtick.labelsize': 20, 'ytick.labelsize': 20}
    matplotlib.rcParams.update(params)
    if cat:
        g = sns.catplot(**kwargs)
        pub_plot(g, base, ymin, ymax)    
        add_title(g,title) if title else print('No title given\n')
        g.savefig(save_dir + 'bar_' + title + '.png') if save_dir else print('Did not save', title)
    if meta:
        g = sns.catplot(**kwargs)
        pub_plot(g, base, ymin, ymax)  
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
    return g
        
#%%

def generalized_recall_plotter(rdf,agg_flag,threeway=False,base=None,ymax=None,ymin=None,rem_cues=True,bar=False,plot_bar=False,plot_hist=False,meta=False,lm_meta=False,title=None,save_dir=None):
    df= rdf.copy()
        
    if agg_flag == 1: # no RPi or RPp idea units
        y_var_name = 'Idea Units Recalled'
        df = df[df['RP_any'] == 0]        
        df.drop_duplicates(inplace=True,subset=['mturk_id','idea_units']) # drop repeats!
        df[y_var_name] = df.groupby(['mturk_id'])['arbitrary'].transform('count')
        
        df = add_0recall_subs(df,rdf) 
        df[y_var_name] = df[y_var_name].fillna(0)
        
    elif agg_flag == 2: # RPm only
        y_var_name = 'Idea Units Recalled'
        df = df[df['RP_imp'] == 1]
        df.drop_duplicates(inplace=True,subset=['mturk_id','idea_units']) # drop repeats!
        df[y_var_name] = df.groupby(['mturk_id'])['arbitrary'].transform('count')

        df = add_0recall_subs(df,rdf) 
        df[y_var_name] = df[y_var_name].fillna(0)
        
    elif agg_flag == 3: # only RPp 
        y_var_name = 'Idea Units Recalled'
        df = df[df['RP_per'] == 1]
        df.drop_duplicates(inplace=True,subset=['mturk_id','idea_units']) # drop repeats!
        df[y_var_name] = df.groupby(['mturk_id'])['arbitrary'].transform('count')

        df = add_0recall_subs(df,rdf) 
        df[y_var_name] = df[y_var_name].fillna(0)
    
    if plot_hist:
        kwargs = {'data':df,'hue':'subjectGroup','x':'idea_units','binwidth':1,'height':4}
        plot_funk(hist=True,title=title,kwargs=kwargs,save_dir=save_dir)
        
    # drop duplicates
    if not plot_hist or plot_prop:
        df.drop_duplicates(inplace=True,subset=['mturk_id',y_var_name]) 
        
    if plot_bar:
        kwargs = {'data':df,'x':'subjectGroup','y':y_var_name,'kind':'bar','ci':68}
        plot_funk(cat=True,title=title,kwargs=kwargs,save_dir=save_dir,base=base,ymax=ymax,ymin=ymin)
        
    if meta:
        #convert continuous to qtile
        #format as long data so meta_var can be passed as plotting factor
        
        def qcut_func(x):
            return pd.qcut(x,q=2,labels=range(1,3))
        
        df['p1_start_qtile'] = df['p1_start'].transform(lambda x: qcut_func(x))
        df['fam_qtile'] = df['familiarity'].transform(lambda x: qcut_func(x))
        df['vocab_qtile'] = df['vocabulary'].transform(lambda x: qcut_func(x))
        df['comp_qtile'] = df['comprehension'].transform(lambda x: qcut_func(x))
        df['read_qtile'] = df['read_time_minutes'].transform(lambda x: qcut_func(x))
        df['RP_qtile'] = df['RP_count'].transform(lambda x: qcut_func(x))
        df['RP_RT_qtile'] = df['RP_RT'].transform(lambda x: qcut_func(x))
                   
        meta_df = pd.melt(df,id_vars=['mturk_id'],value_vars=['comp_qtile','vocab_qtile','fam_qtile',
                        'bio_maj','bio_current', 'read_qtile','RP_qtile','RP_RT_qtile','acc_answers','use_data','p1_start_qtile'], value_name='meta_val',var_name='meta_var') # 'eff_qtile',
        df = df.merge(meta_df, on='mturk_id')

        kwargs = {'data':df,'x':'meta_val','hue':'subjectGroup','y':y_var_name,'kind':'bar','ci':68,
                'col_wrap':4,'col':'meta_var','sharex':False,'sharey':False}
        plot_funk(meta=True,kwargs=kwargs,save_dir=save_dir,title=title)
        
    if threeway:
        def qcut_func(x):
            return pd.qcut(x,q=2,labels=range(1,3))
        
        df['GMRT_ntile'] = df['GMRT_c_s'].transform(lambda x: qcut_func(x))
        df['familiarity_ntile'] = df['familiarity_c_s'].transform(lambda x: qcut_func(x))
                   
        meta_df = pd.melt(df,id_vars=['mturk_id'],value_vars=['GMRT_ntile','familiarity_ntile'], value_name='meta_val',var_name='meta_var') # 'eff_qtile',
        df = df.merge(meta_df, on='mturk_id')
        #plot
        kwargs = {'data':df,'x':'meta_val','hue':'subjectGroup','y':y_var_name,'kind':'bar','ci':68
                  ,'col':'meta_var','sharex':False,'sharey':False}
        plot_funk(meta=True,kwargs=kwargs,save_dir=save_dir,title=title)
        
    if lm_meta:
        meta_df = pd.melt(df,id_vars=['mturk_id'],value_vars=['comprehension','vocabulary','familiarity','RP_RT',
                        'effort','read_time_minutes','RP_count','n_bio_courses','p1_start'], value_name='meta_val',var_name='meta_var')
        df = df.merge(meta_df, on='mturk_id')
        df['meta_val'] = df['meta_val'].fillna(0)
        df['meta_val'] = pd.to_numeric(df['meta_val'])
        kwargs = {'data':df,'x':'meta_val','hue':'subjectGroup','y':y_var_name,
                'col_wrap':3,'col':'meta_var','sharex':False}
        plot_funk(lm_meta=True,kwargs=kwargs,save_dir=save_dir,title=title)
        
    return df

#%% MAIN RECALL PLOTS

# RP bin plots
df = generalized_recall_plotter(rdf,agg_flag=1,base=1,ymin=0,ymax=7,plot_bar=True,rem_cues=False,title='Untested Idea Units',save_dir=sdir) 
df = generalized_recall_plotter(rdf,agg_flag=2,base=1,ymin=0,ymax=3,plot_bar=True,rem_cues=False,title='Main Idea Units',save_dir=sdir) 
df = generalized_recall_plotter(rdf,agg_flag=3,base=1,ymin=0,ymax=2,plot_bar=True,rem_cues=False,title='Peripheral Idea Units',save_dir=sdir) 

#%% plot 3way interaction as median split

if exp_flag == 2:
    rdf = pd.read_csv(home_dir + 'rdf_n=170_11_5_21.csv') 
elif exp_flag ==21:
    rdf = pd.read_csv(home_dir + 'rdf_n=190_8_7_22.csv')

rdf = change_condition_labs(rdf)

cvars = ['GMRT','effort','familiarity','n_bio_courses','RP_count']
rdf = merge_GM(rdf)
rdf = m_center(rdf,cvars=cvars)
rdf = scale_vars(rdf,cvars=['GMRT_c','familiarity_c'])

df = generic_recall_plotter(rdf,threeway=True,agg_flag=6,base=1,ymin=0,ymax=7,plot_bar=False,meta=False,rem_cues=False,title='Untested Idea Units',save_dir=sdir) 

df['Prior Knowledge'] = df['familiarity_ntile'].map({1:'Low Knowledge',2:'High Knowledge'})
df['Reading Ability'] = df['GMRT_ntile'].map({1:'Less-skilled Readers',2:'Skilled Readers'})

def title_helper(g):
    g.fig.subplots_adjust(top=.8)
    g.fig.suptitle(title, fontsize=24)

# split by knowledge
dflowk = df[df['familiarity_ntile'] ==1]
dfhighk = df[df['familiarity_ntile'] ==2]

kwargs = {'data':dflowk,'col':'Reading Ability','x':'subjectGroup','y':'Idea Units Recalled','kind':'bar','ci':68,'sharex':False,'sharey':False}
title='Low Knowledge'
g = plot_funk(cat=True,kwargs=kwargs,save_dir=sdir,title=title,base=1,ymin=0,ymax=9)
add_title(g,title)
title_helper(g)

kwargs = {'data':dfhighk,'col':'Reading Ability','x':'subjectGroup','y':'Idea Units Recalled','kind':'bar','ci':68,'sharex':False,'sharey':False}
title='High Knowledge'
g =plot_funk(cat=True,kwargs=kwargs,save_dir=sdir,title=title,base=1,ymin=0,ymax=9)
add_title(g,title)
title_helper(g)
#%% IMPORT MULTIPLE CHOICE DATA

if exp_flag == 2:
    part2 = pd.read_csv(home_dir + 'part2_df_uptoMC_11_5_21_good_subs.csv') 
elif exp_flag == 21:
    part2 = pd.read_csv(home_dir + 'part2_df_uptoMC_3_8_22_good_subs.csv')
    
part2 = change_condition_labs(part2)
part2 = change_mc_labs(part2)

cvars = ['GMRT','effort','familiarity','n_bio_courses','RP_count']
part2 = merge_GM(part2)
part2 = m_center(part2, cvars=cvars)

#%% Multiple Choice Function

def mc_factor_mean(part2,factor,save_dir=None,title=None,meta=False,lm_meta=False,base=None,ymax=None,ymin=None):

    df = part2.copy()
    yvar = 'Proportion Correct'
    df[yvar]= df.groupby(['mturk_id',factor])['correct'].transform('mean')
    df = df.drop_duplicates(subset=['mturk_id',yvar,factor])
        
    if not meta or lm_meta:
        kwargs = {'data':df,'col':factor,'y':yvar,'x':'subjectGroup','kind':'bar','ci':68}
        plot_funk(kwargs=kwargs,cat=True,save_dir=save_dir,title=title,base=base,ymax=ymax,ymin=ymin)

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
        kwargs = {'data':df,'x':'meta_val','hue':'subjectGroup','y':yvar,'kind':'bar','ci':68,
                'col':'meta_var','row':factor,'sharex':False}
        plot_funk(meta=True,kwargs=kwargs,save_dir=save_dir,title=title)
        
    if lm_meta:
        meta_df = pd.melt(df,id_vars=['mturk_id'],value_vars=['comprehension','vocabulary','familiarity',
                        'effort','read_time_minutes','RP_count','RP_RT','n_bio_courses','p1_start'], value_name='meta_val',var_name='meta_var')
        df = df.merge(meta_df, on='mturk_id')
        kwargs = {'data':df,'x':'meta_val','hue':'subjectGroup','y':yvar,
                'row':factor,'col':'meta_var','sharex':False}
        plot_funk(lm_meta=True,kwargs=kwargs,save_dir=save_dir,title=title)

#%% Run MC plot

mc_factor_mean(part2,base=.1,ymin=.5,ymax=1, factor='q_type',title='Multiple Choice',save_dir=sdir)

#%% Not used

def mc_grand_group_mean_meta(part2,covariate=None,save_dir=None,title=None):
    data = part2.copy()
    yvar = 'Multiple Choice Proportion Correct'
    data[yvar]= part2.groupby(['mturk_id'])['correct'].transform('mean')
    data = data.drop_duplicates(subset=['mturk_id',yvar])
    kwargs = {'data':data,'x':covariate,'hue':'subjectGroup','y':yvar,'ci':68,'sharex':False}
    plot_funk(kwargs=kwargs,lm_meta=True,save_dir=save_dir,title=title)
    return data
    
mc_grand_group_mean_meta(part2,covariate='GMRT',save_dir=sdir,title='MC x GMRT')
mc_grand_group_mean_meta(part2,covariate='familiarity',save_dir=sdir,title='MC x Familiarity')
mc_grand_group_mean_meta(part2,covariate='n_bio_courses',save_dir=sdir,title='MC x N Bio')
#%%
