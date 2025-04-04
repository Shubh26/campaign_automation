import sys
sys.path.extend(['D:\work\project\cac\sales_measurment_service'])
import pandas as pd
import numpy as np
import kaleido
# if used in jupyter lab need to have the extension - https://www.npmjs.com/package/jupyterlab-plotly
import plotly
import plotly.offline as pyo
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import os,sys
import copy
import glob,shutil
from datetime import datetime,timedelta
import re
import time
import seaborn as sns

# from absl import logging
import logging

import importlib
from utils import date_utils, file_utils,zipcode_utils
from utils import measurement_utils as meas
#, processor, campaign_helper

from pathlib import Path
import statistics

import math
import geopy.distance
import sklearn
import collections,itertools

from utils import constants as c
from utils import dataframe_utils
from warnings import warn
import pylab as pl
from IPython import display
from pandas.util.testing import assert_frame_equal
from utils import ab_group_utils as ab

sklearn.__version__



def get_store_avg_sales_v2(df_sales, df_group):#, historical = False):
    df1 = df_sales[df_sales['store_id'].isin(df_group['store_id'])].groupby(['store_id','week'])['sales_dollars'].agg('sum').groupby(['store_id']).agg('mean').reset_index()
    df2 = df_sales[df_sales['store_id'].isin(df_group['store_id'])].groupby(['store_id'])['sales_dollars'].agg('sum').reset_index()
    # if historical:
    #     df1.rename(columns = {'sales_dollars': f'Historical Avg Sales'}, inplace = True)
    #     df2.rename(columns = {'sales_dollars': f'Historical Total Sales'}, inplace = True)
    #else:
    df1.rename(columns = {'sales_dollars': f'Avg Sales'}, inplace = True)
    df2.rename(columns = {'sales_dollars': f'Total Sales'}, inplace = True)
    t = pd.merge(df1, df2, on = 'store_id', how = 'inner')
    return t


def get_incremental_lift(df_sales, df_hist, df_test, df_control):
    groups = ['Test', 'Control']
    df_group = [df_test, df_control]
    # df_sales = meas.get_group_col(df_sales, df_test, df_control)
    # df_sales = meas.get_non_reported_sales_store(df_sales, df_test, df_control , False)
    for idx, group in enumerate(groups):
        #tmp = group.lower()
        #df_temp = df_sales[df_sales['group'] == tmp]
        df1 = get_store_avg_sales_v2(df_sales, df_group[idx])
        df1.rename(columns = {'Avg Sales':f'{group} Avg Sales', 'Total Sales':f'{group} Total Sales'}, inplace = True)
        df2 = get_store_avg_sales_v2(df_hist, df_group[idx])
        df2.rename(columns = {'Avg Sales': f'{group} Historical Avg Sales', 'Total Sales': f'{group} Historical Total Sales'}, inplace = True)
        t = pd.merge(df1, df2, on = 'store_id', how = 'inner')
        if group == 'Test':
            t['Lift'] = t[f'{group} Avg Sales'] - t[f'{group} Historical Avg Sales']
            temp = t.copy()
        else:
            t = t[f'{group} Avg Sales'].mean() - t[f'{group} Historical Avg Sales'].mean()
    temp['Lift'] = (temp['Lift'] - t)/t
    return temp

def get_running_rate_lift(df_sales, df_test, df_control):
    groups = ['Test', 'Control']
    df_group = [test, control]
    for idx, group in enumerate(groups):
        t = get_store_avg_sales_v2(df_sales, df_group[idx])
        t.rename(columns = {'Avg Sales':f'{group} Avg Sales', 'Total Sales':f'{group} Total Sales'}, inplace = True)
        #print(t['Test Avg Sales'].mean())
        if group == 'Control':
            t = t[f'{group} Avg Sales'].mean()
        else:
            temp = t.copy()
    temp['Lift'] = ((temp['Test Avg Sales'] - t)/t)*100
    return temp

def get_historical_lift(df_sales, df_hist, df_test):
    df1 = get_store_avg_sales_v2(df_sales, df_test)
    df1.rename(columns = {'Avg Sales':'Test Avg Sales', 'Total Sales': 'Test Total Sales'}, inplace = True)
    df2 = get_store_avg_sales_v2(df_hist, df_test)
    df2.rename(columns = {'Avg Sales': 'Test Historical Avg Sales', 'Total Sales': 'Test Historical Total Sales'}, inplace = True)
    t = pd.merge(df1, df2, on = 'store_id', how = 'inner')
    t['Lift'] = ((t['Test Avg Sales'] - t['Test Historical Avg Sales'])/t['Test Historical Avg Sales'])*100
    return t

def get_lift_plot(df, ul, ll):
    #warn('This method will be deprecated. Please use get_dist_plot() for generating lift plots', DeprecationWarning, stacklevel=2)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.kdeplot(df['Lift'], ax=ax, legend=False)
    x = ax.lines[-1].get_xdata()
    y = ax.lines[-1].get_ydata()
    mean = df['Lift'].mean()
    ax.vlines(mean, 0, np.interp(mean, x, y), linestyle='--', alpha=0.5)
    plt.text(x = mean, y = np.interp(mean, x, y), s = f'{mean:.2f}', color = 'black')
    x_special1 = ll
    x_special2 = ul
    ax.vlines(x_special1, 0, np.interp(x_special1, x, y), linestyle='-', color='crimson', alpha=0.5, label = f'{ll}')
    plt.text(x = x_special1, y = np.interp(x_special1, x, y), s = f'{x_special1:.2f}', color = 'black', horizontalalignment = 'right')
    ax.fill_between(x, 0, y, where=x < x_special1, color='gold', alpha=0.2)
    ax.vlines(x_special2, 0, np.interp(x_special2, x, y), linestyle='-', color='crimson', alpha=0.5, label = f'{ul}')
    plt.text(x = x_special2, y = np.interp(x_special2, x, y), s = f'{x_special2:.2f}', color = 'black', horizontalalignment = 'left')
    ax.fill_between(x, 0, y, where=x > x_special2, color='gold', alpha=0.2)
    plt.show()
    return fig
    
def get_lift_plot_v1(df, ul, ll):
    #warn('This method will be deprecated. Please use get_dist_plot() for generating lift plots', DeprecationWarning, stacklevel=2)
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.kdeplot(df['Lift'], ax=ax, legend=False)
    x = ax.lines[-1].get_xdata()
    y = ax.lines[-1].get_ydata()
    mean = df['Lift'].mean()
    ax.vlines(mean, 0, np.interp(mean, x, y), linestyle='--', alpha=0.5)
    plt.text(x = mean, y = np.interp(mean, x, y), s = f'{mean:.2f}', color = 'black')
    x_special1 = ll
    x_special2 = ul
    ax.vlines(x_special1, 0, np.interp(x_special1, x, y), linestyle='-', color='crimson', alpha=0.5, label = f'{ll}')
    plt.text(x = x_special1, y = np.interp(x_special1, x, y), s = f'{x_special1:.2f}', color = 'black', horizontalalignment = 'right')
    ax.fill_between(x, 0, y, where=x < x_special1, color='gold', alpha=0.2)
    ax.vlines(x_special2, 0, np.interp(x_special2, x, y), linestyle='-', color='crimson', alpha=0.5, label = f'{ul}')
    plt.text(x = x_special2, y = np.interp(x_special2, x, y), s = f'{x_special2:.2f}', color = 'black', horizontalalignment = 'left')
    ax.fill_between(x, 0, y, where=x > x_special2, color='gold', alpha=0.2)
    display.display(pl.gcf())
    display.clear_output(wait=True)
    time.sleep(0.01)
    #plt.show()
    
def get_dist_plot(df, ul, ll, col_name):
    cols = df.columns
    for col in cols:
        if col == col_name:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.kdeplot(df[col], ax=ax, legend=False)
            x = ax.lines[-1].get_xdata()
            y = ax.lines[-1].get_ydata()
            mean = df[col].mean()
            ax.vlines(mean, 0, np.interp(mean, x, y), linestyle='--', alpha=0.5)
            plt.text(x = mean, y = np.interp(mean, x, y), s = f'{mean:.2f}', color = 'black')
            x_special1 = ll
            x_special2 = ul
            ax.vlines(x_special1, 0, np.interp(x_special1, x, y), linestyle='-', color='crimson', alpha=0.5, label = f'{ll}')
            plt.text(x = x_special1, y = np.interp(x_special1, x, y), s = f'{x_special1:.2f}', color = 'black', horizontalalignment = 'right')
            ax.fill_between(x, 0, y, where=x < x_special1, color='gold', alpha=0.2)
            ax.vlines(x_special2, 0, np.interp(x_special2, x, y), linestyle='-', color='crimson', alpha=0.5, label = f'{ul}')
            plt.text(x = x_special2, y = np.interp(x_special2, x, y), s = f'{x_special2:.2f}', color = 'black', horizontalalignment = 'left')
            ax.fill_between(x, 0, y, where=x > x_special2, color='gold', alpha=0.2)
            plt.show()
        
def get_dual_plot(df, ul, ll):
    #warn('This method will be deprecated. Please use get_dist_plot() for generating lift plots', DeprecationWarning, stacklevel=2)
    temp = df.copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.kdeplot(temp['Lift'], ax=ax, legend=False)
    #sns.kdeplot(data = pd.melt(temp[['store_id', 'Test Avg Sales', 'Lift']], ['store_id']), x='value', hue='variable', fill=True, ax= ax, legend=True)
    #sns.kdeplot(temp['Test Avg Sales'], shade=True, color="b", ax=ax)
    x1 = ax.lines[-1].get_xdata()
    y1 = ax.lines[-1].get_ydata()
    mean = temp['Lift'].mean()
    ax.vlines(mean, 0, np.interp(mean, x1, y1), linestyle='--', alpha=0.5)
    plt.text(x = mean, y = np.interp(mean, x1, y1), s = f'{mean:.2f}', color = 'black')
    # ll = 0.25
    # ul = 0.4
    x_special1 = ll['Lift']
    x_special2 = ul['Lift']
    ax.vlines(x_special1, 0, np.interp(x_special1, x1, y1), linestyle='-', color='crimson', alpha=0.5, label = f'{ll}')
    plt.text(x = x_special1, y = np.interp(x_special1, x1, y1), s = f'{x_special1:.2f}', color = 'black', horizontalalignment = 'right')
    ax.fill_between(x1, 0, y1, where=x1 < x_special1, color='gold', alpha=0.2)
    ax.vlines(x_special2, 0, np.interp(x_special2, x1, y1), linestyle='-', color='crimson', alpha=0.5, label = f'{ul}')
    plt.text(x = x_special2, y = np.interp(x_special2, x1, y1), s = f'{x_special2:.2f}', color = 'black', horizontalalignment = 'left')
    ax.fill_between(x1, 0, y1, where=x1 > x_special2, color='gold', alpha=0.2)

    sns.kdeplot(temp['Test Avg Sales'], ax=ax, legend=False)
    x2 = ax.lines[-1].get_xdata()
    y2 = ax.lines[-1].get_ydata()
    mean = temp['Test Avg Sales'].mean()
    ax.vlines(mean, 0, np.interp(mean, x2, y2), linestyle='--', alpha=0.5)
    plt.text(x = mean, y = np.interp(mean, x2, y2), s = f'{mean:.2f}', color = 'black')
    # ll = 0.1
    # ul = 0.5
    x_special1 = ll['Test Avg Sales']
    x_special2 = ul['Test Avg Sales']
    ax.vlines(x_special1, 0, np.interp(x_special1, x2, y2), linestyle='-', color='crimson', alpha=0.5, label = f'{ll}')
    plt.text(x = x_special1, y = np.interp(x_special1, x2, y2), s = f'{x_special1:.2f}', color = 'black', horizontalalignment = 'right')
    ax.fill_between(x2, 0, y2, where=x2 < x_special1, color='lime', alpha=0.2)
    ax.vlines(x_special2, 0, np.interp(x_special2, x2, y2), linestyle='-', color='crimson', alpha=0.5, label = f'{ul}')
    plt.text(x = x_special2, y = np.interp(x_special2, x2, y2), s = f'{x_special2:.2f}', color = 'black', horizontalalignment = 'left')
    ax.fill_between(x2, 0, y2, where=x2 > x_special2, color='lime', alpha=0.2)
    plt.show()
    
def get_dual_plot_v1(df, ul, ll):
    #warn('This method will be deprecated. Please use get_dist_plot() for generating lift plots', DeprecationWarning, stacklevel=2)
    temp = df.copy()
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.kdeplot(temp['Lift'], ax=ax, legend=False)
    #sns.kdeplot(data = pd.melt(temp[['store_id', 'Test Avg Sales', 'Lift']], ['store_id']), x='value', hue='variable', fill=True, ax= ax, legend=True)
    #sns.kdeplot(temp['Test Avg Sales'], shade=True, color="b", ax=ax)
    x1 = ax.lines[-1].get_xdata()
    y1 = ax.lines[-1].get_ydata()
    mean = temp['Lift'].mean()
    ax.vlines(mean, 0, np.interp(mean, x1, y1), linestyle='--', alpha=0.5)
    plt.text(x = mean, y = np.interp(mean, x1, y1), s = f'{mean:.2f}', color = 'black')
    # ll = 0.25
    # ul = 0.4
    x_special1 = ll['Lift']
    x_special2 = ul['Lift']
    ax.vlines(x_special1, 0, np.interp(x_special1, x1, y1), linestyle='-', color='crimson', alpha=0.5, label = f'{ll}')
    plt.text(x = x_special1, y = np.interp(x_special1, x1, y1), s = f'{x_special1:.2f}', color = 'black', horizontalalignment = 'right')
    ax.fill_between(x1, 0, y1, where=x1 < x_special1, color='gold', alpha=0.2)
    ax.vlines(x_special2, 0, np.interp(x_special2, x1, y1), linestyle='-', color='crimson', alpha=0.5, label = f'{ul}')
    plt.text(x = x_special2, y = np.interp(x_special2, x1, y1), s = f'{x_special2:.2f}', color = 'black', horizontalalignment = 'left')
    ax.fill_between(x1, 0, y1, where=x1 > x_special2, color='gold', alpha=0.2)

    sns.kdeplot(temp['Test Avg Sales'], ax=ax, legend=False)
    x2 = ax.lines[-1].get_xdata()
    y2 = ax.lines[-1].get_ydata()
    mean = temp['Test Avg Sales'].mean()
    ax.vlines(mean, 0, np.interp(mean, x2, y2), linestyle='--', alpha=0.5)
    plt.text(x = mean, y = np.interp(mean, x2, y2), s = f'{mean:.2f}', color = 'black')
    # ll = 0.1
    # ul = 0.5
    x_special1 = ll['Test Avg Sales']
    x_special2 = ul['Test Avg Sales']
    ax.vlines(x_special1, 0, np.interp(x_special1, x2, y2), linestyle='-', color='crimson', alpha=0.5, label = f'{ll}')
    plt.text(x = x_special1, y = np.interp(x_special1, x2, y2), s = f'{x_special1:.2f}', color = 'black', horizontalalignment = 'right')
    ax.fill_between(x2, 0, y2, where=x2 < x_special1, color='lime', alpha=0.2)
    ax.vlines(x_special2, 0, np.interp(x_special2, x2, y2), linestyle='-', color='crimson', alpha=0.5, label = f'{ul}')
    plt.text(x = x_special2, y = np.interp(x_special2, x2, y2), s = f'{x_special2:.2f}', color = 'black', horizontalalignment = 'left')
    ax.fill_between(x2, 0, y2, where=x2 > x_special2, color='lime', alpha=0.2)
    display.display(pl.gcf())
    display.clear_output(wait=True)
    time.sleep(0.01)
    #plt.show()
    
def norm_col(df, col_name):
    cols = df.columns
    for col in cols:
        if col == col_name:
            df[col] = (df[col] - df[col].min())/(df[col].max() - df[col].min())
    return df

def get_optimized_stores_v2(df, threshold = 0.5, lift_plot = True):
    temp = df.copy()
    #norm_col(temp, 'Lift')
    std = np.std(temp['Lift'])
    mean = np.mean(temp['Lift'])
    counter = 0.1
    actual = 0
    while actual < threshold:
        # print("actual", actual)
        # print("threshold", threshold)
        # print("actual type", type(actual))
        # print("threshold type",type(threshold))
        #print("counter1:", counter)
        ul = mean + counter*std
        ll = mean - counter*std
        t = temp[(temp['Lift'] > ll) & (temp['Lift'] < ul)]['store_id'].unique()
        temp['Performance'] = 'Low'
        temp.loc[temp['store_id'].isin(t), 'Performance'] = 'High'
        actual = len(t)/temp['store_id'].nunique()
        counter+=0.01
    #     print("actual:", actual)
    #     print("counter2:",counter)
    # print("ul:", ul)
    # print("ll:", ll)
    t = temp.groupby(['Performance'])['store_id'].nunique().reset_index()
    t.rename(columns={'store_id':'store_count'}, inplace = True)
    t['split'] = t['store_count'].apply(lambda x: round((x/t['store_count'].sum())*100,2))
    t['split'] = t['split'].astype(str) + '%'
    print("optimization summary:\nTotal stores %d" %(t['store_count'].sum()))
    print(t.to_string(index=False))
    if lift_plot:
        get_lift_plot(temp, ul, ll)
    return temp

def get_optimized_stores_v21(df, threshold = 0.5, lift_plot = True):
    temp = df.copy()
    std = np.std(temp['Lift'])
    mean = np.mean(temp['Lift'])
    counter = 0.1
    actual = 0
    while actual < threshold:
        # print("actual", actual)
        # print("threshold", threshold)
        # print("actual type", type(actual))
        # print("threshold type",type(threshold))
        #print("counter1:", counter)
        ul = mean + counter*std
        ll = mean - counter*std
        t = temp[(temp['Lift'] > ll) & (temp['Lift'] < ul)]['store_id'].unique()
        temp['Performance'] = 'Low'
        temp.loc[temp['store_id'].isin(t), 'Performance'] = 'High'
        actual = len(t)/temp['store_id'].nunique()
        counter+=0.01
        get_lift_plot_v1(temp, ul, ll)
        # display.display(pl.gcf())
        #display.clear_output(wait=True)
        # time.sleep(0.01)
    #display.clear_output(wait=True)
    #plt.show()
    #     print("actual:", actual)
    #     print("counter2:",counter)
    # print("ul:", ul)
    # print("ll:", ll)
    t = temp.groupby(['Performance'])['store_id'].nunique().reset_index()
    t.rename(columns={'store_id':'store_count'}, inplace = True)
    t['split'] = t['store_count'].apply(lambda x: round((x/t['store_count'].sum())*100,2))
    t['split'] = t['split'].astype(str) + '%'
    print("optimization summary:\nTotal stores %d" %(t['store_count'].sum()))
    print(t.to_string(index=False))
    # if lift_plot:
    #     get_lift_plot(temp, ul, ll)
    return temp

def get_optimized_stores_v22(df, threshold = 0.5, lift_plot = True):
    temp = df.copy()
    norm_col(temp, 'Lift')
    norm_col(temp, 'Test Avg Sales')
    # std = np.std(temp['Lift'])
    # mean = np.mean(temp['Lift'])
    std = {'Lift': np.std(temp['Lift']), 'Test Avg Sales': np.std(temp['Test Avg Sales'])}
    mean = {'Lift': np.mean(temp['Lift']), 'Test Avg Sales': np.mean(temp['Test Avg Sales'])}
    counter = 0.1
    actual = 0
    while actual < threshold:
        # print("actual", actual)
        # print("threshold", threshold)
        # print("actual type", type(actual))
        # print("threshold type",type(threshold))
        #print("counter1:", counter)
        # ul = mean + counter*std
        # ll = mean - counter*std
        ul = {'Lift': mean['Lift'] + counter*std['Lift'], 'Test Avg Sales': mean['Test Avg Sales'] + counter*std['Test Avg Sales']}
        ll = {'Lift': mean['Lift'] - counter*std['Lift'], 'Test Avg Sales': mean['Test Avg Sales'] - counter*std['Test Avg Sales']}
        t = temp[((temp['Lift'] > ll['Lift']) & (temp['Test Avg Sales'] > ll['Test Avg Sales'])) & ((temp['Lift'] < ul['Lift']) & (temp['Test Avg Sales'] < ul['Test Avg Sales']))]['store_id'].unique()
        temp['Performance'] = 'Low'
        temp.loc[temp['store_id'].isin(t), 'Performance'] = 'High'
        actual = len(t)/temp['store_id'].nunique()
        counter+=0.01
    #     print("actual:", actual)
    #     print("counter2:",counter)
    # print("ul:", ul)
    # print("ll:", ll)
    t = temp.groupby(['Performance'])['store_id'].nunique().reset_index()
    t.rename(columns={'store_id':'store_count'}, inplace = True)
    t['split'] = t['store_count'].apply(lambda x: round((x/t['store_count'].sum())*100,2))
    t['split'] = t['split'].astype(str) + '%'
    print("optimization summary:\nTotal stores %d" %(t['store_count'].sum()))
    print(t.to_string(index=False))
    if lift_plot:
        #get_lift_plot(temp, ul, ll)
        get_dual_plot(temp, ul, ll)
    return temp

def get_optimized_stores_v23(df, threshold = 0.5, lift_plot = True):
    temp = df.copy()
    norm_col(temp, 'Lift')
    norm_col(temp, 'Test Avg Sales')
    # std = np.std(temp['Lift'])
    # mean = np.mean(temp['Lift'])
    std = {'Lift': np.std(temp['Lift']), 'Test Avg Sales': np.std(temp['Test Avg Sales'])}
    mean = {'Lift': np.mean(temp['Lift']), 'Test Avg Sales': np.mean(temp['Test Avg Sales'])}
    counter = 0.1
    actual = 0
    while actual < threshold:
        # print("actual", actual)
        # print("threshold", threshold)
        # print("actual type", type(actual))
        # print("threshold type",type(threshold))
        #print("counter1:", counter)
        # ul = mean + counter*std
        # ll = mean - counter*std
        ul = {'Lift': mean['Lift'] + counter*std['Lift'], 'Test Avg Sales': mean['Test Avg Sales'] + counter*std['Test Avg Sales']}
        ll = {'Lift': mean['Lift'] - counter*std['Lift'], 'Test Avg Sales': mean['Test Avg Sales'] - counter*std['Test Avg Sales']}
        t = temp[((temp['Lift'] > ll['Lift']) & (temp['Test Avg Sales'] > ll['Test Avg Sales'])) & ((temp['Lift'] < ul['Lift']) & (temp['Test Avg Sales'] < ul['Test Avg Sales']))]['store_id'].unique()
        temp['Performance'] = 'Low'
        temp.loc[temp['store_id'].isin(t), 'Performance'] = 'High'
        actual = len(t)/temp['store_id'].nunique()
        counter+=0.01
        get_dual_plot_v1(temp, ul, ll)
    #     print("actual:", actual)
    #     print("counter2:",counter)
    # print("ul:", ul)
    # print("ll:", ll)
    t = temp.groupby(['Performance'])['store_id'].nunique().reset_index()
    t.rename(columns={'store_id':'store_count'}, inplace = True)
    t['split'] = t['store_count'].apply(lambda x: round((x/t['store_count'].sum())*100,2))
    t['split'] = t['split'].astype(str) + '%'
    print("optimization summary:\nTotal stores %d" %(t['store_count'].sum()))
    print(t.to_string(index=False))
    # if lift_plot:
    #     #get_lift_plot(temp, ul, ll)
    #     get_dual_plot(temp, ul, ll)
    return temp

def get_optimized_stores_v3(df, lower_limit = 0.5, upper_limit = 0.5, lift_plot = True):
    temp = df.copy()
    std = np.std(temp['Lift'])
    mean = np.mean(temp['Lift'])
    limits = {'lower_limit': lower_limit, 'upper_limit': upper_limit}
    # actual_ul = 0
    # actual_ll = 0
    # actual = 0
    for k, limit in limits.items():
        if k == 'lower_limit':
            actual = 0
            counter = 0.1
            t1 = temp[temp['Lift'] < mean]
            while actual < limit:
                #print("counter1:", counter)
                ll = mean - counter*std
                t = t1[t1['Lift'] > ll]['store_id'].unique()
                #t = temp[(temp['Lift'] > ll) & (temp['Lift'] < ul)]['store_id'].unique()
                t1['Performance'] = 'Low'
                t1.loc[t1['store_id'].isin(t), 'Performance'] = 'High'
                actual = len(t)/t1['store_id'].nunique()
                counter+=0.01
        else:
            actual = 0
            counter = 0.1
            t2 = temp[temp['Lift'] >= mean]
            while actual < limit:
                #print("counter1:", counter)
                ul = mean + counter*std
                t = t2[t2['Lift'] < ul]['store_id'].unique()
                #t = temp[(temp['Lift'] > ll) & (temp['Lift'] < ul)]['store_id'].unique()
                t2['Performance'] = 'Low'
                t2.loc[t2['store_id'].isin(t), 'Performance'] = 'High'
                actual = len(t)/t2['store_id'].nunique()
                counter+=0.01
                
    temp = pd.concat([t1,t2])
            
    #     print("actual:", actual)
    #     print("counter2:",counter)
    # print("ul:", ul)
    # print("ll:", ll)
    t = temp.groupby(['Performance'])['store_id'].nunique().reset_index()
    t.rename(columns={'store_id':'store_count'}, inplace = True)
    t['split'] = t['store_count'].apply(lambda x: round((x/t['store_count'].sum())*100,2))
    t['split'] = t['split'].astype(str) + '%'
    print("optimization summary:\nTotal stores %d" %(t['store_count'].sum()))
    print(t.to_string(index=False))
    if lift_plot:
        fig = get_lift_plot(temp, ul, ll)
    return temp, fig

def get_optimized_stores_v31(df, lower_limit = 0.5, upper_limit = 0.5, lift_plot = True):
    temp = df.copy()
    norm_col(temp, 'Lift')
    norm_col(temp, 'Test Avg Sales')
    std = {'Lift': np.std(temp['Lift']), 'Test Avg Sales': np.std(temp['Test Avg Sales'])}
    mean = {'Lift': np.mean(temp['Lift']), 'Test Avg Sales': np.mean(temp['Test Avg Sales'])}
    for k, limit in limits.items():
        if k == 'lower_limit':
            actual = 0
            counter = 0.1
            t1 = temp[temp['Lift'] < mean]
            while actual < limit:
                #print("counter1:", counter)
                #ll = mean - counter*std
                ll = {'Lift': mean['Lift'] - counter*std['Lift'], 'Test Avg Sales': mean['Test Avg Sales'] - counter*std['Test Avg Sales']}
                t = t1[((t1['Lift'] > ll['Lift']) & (t1['Test Avg Sales'] > ll['Test Avg Sales']))]['store_id'].unique()
                #t = temp[(temp['Lift'] > ll) & (temp['Lift'] < ul)]['store_id'].unique()
                t1['Performance'] = 'Low'
                t1.loc[t1['store_id'].isin(t), 'Performance'] = 'High'
                actual = len(t)/t1['store_id'].nunique()
                counter+=0.01
        else:
            actual = 0
            counter = 0.1
            t2 = temp[temp['Lift'] >= mean]
            while actual < limit:
                #print("counter1:", counter)
                #ul = mean + counter*std
                ul = {'Lift': mean['Lift'] + counter*std['Lift'], 'Test Avg Sales': mean['Test Avg Sales'] + counter*std['Test Avg Sales']}
                t = t2[t2['Lift'] < ul]['store_id'].unique()
                #t = temp[(temp['Lift'] > ll) & (temp['Lift'] < ul)]['store_id'].unique()
                t2['Performance'] = 'Low'
                t2.loc[t2['store_id'].isin(t), 'Performance'] = 'High'
                actual = len(t)/t2['store_id'].nunique()
                counter+=0.01
                
    temp = pd.concat([t1,t2])
            
    #     print("actual:", actual)
    #     print("counter2:",counter)
    # print("ul:", ul)
    # print("ll:", ll)
    t = temp.groupby(['Performance'])['store_id'].nunique().reset_index()
    t.rename(columns={'store_id':'store_count'}, inplace = True)
    t['split'] = t['store_count'].apply(lambda x: round((x/t['store_count'].sum())*100,2))
    t['split'] = t['split'].astype(str) + '%'
    print("optimization summary:\nTotal stores %d" %(t['store_count'].sum()))
    print(t.to_string(index=False))
    if lift_plot:
        get_lift_plot(temp, ul, ll)
    return temp

def generate_contour_plot(df_lift):
    x = df_lift['Lift']
    y = df_lift['Test Avg Sales']

    fig = go.Figure()
    #for each in color:
    fig.add_trace(go.Histogram2dContour(
            x = x,
            y = y,
            colorscale = 'earth',
            reversescale = True,
            xaxis = 'x',
            yaxis = 'y'
        ))
    fig.update_xaxes(title_text='Lift')
    fig.update_yaxes(title_text='Test Avg Sales')
    fig.add_trace(go.Scatter(
            x = x,
            y = y,
            xaxis = 'x',
            yaxis = 'y',
            mode = 'markers',
            marker = dict(
                color = 'rgba(0,0,0,0.3)',
                size = 3
            )
        ))
    fig.add_trace(go.Histogram(
            y = y,
            xaxis = 'x2',
            marker = dict(
                color = 'rgba(0,0,0,1)'
            )
        ))
    fig.add_trace(go.Histogram(
            x = x,
            yaxis = 'y2',
            marker = dict(
                color = 'rgba(0,0,0,1)'
            )
        ))

    fig.update_layout(
        autosize = False,
        xaxis = dict(
            zeroline = False,
            domain = [0,0.85],
            showgrid = False
        ),
        yaxis = dict(
            zeroline = False,
            domain = [0,0.85],
            showgrid = False
        ),
        xaxis2 = dict(
            zeroline = False,
            domain = [0.85,1],
            showgrid = False
        ),
        yaxis2 = dict(
            zeroline = False,
            domain = [0.85,1],
            showgrid = False
        ),
        height = 600,
        width = 600,
        bargap = 0,
        hovermode = 'closest',
        showlegend = False
    )

    fig.add_vline(x=df_lift['Lift'].mean(), line_width=0.8, line_dash="dash", line_color="white", annotation_text = f"lift mean:{round(df_lift['Lift'].mean(),2)}", annotation_position = "top right")
    fig.add_hline(y=df_lift['Test Avg Sales'].mean(), line_width=0.8, line_dash="dash", line_color="white", annotation_text = f"avg sales mean:{round(df_lift['Test Avg Sales'].mean(),2)}", annotation_position = "top right")
    fig.show()
    return fig

#def calculate_lift(df_sales, df_hist, df_test, df_control, lift_type, optimization = True, threshold = 0.5, lift_plot = True):
def calculate_lift_bckup(df_sales, df_hist, df_test, df_control, lift_type, optimization = True, lower_limit = 0.5, upper_limit = 0.5, lift_plot = True):
    """
    Calculate lift in test group
    
    lift_type: {'incremental lift', 'running rate lift', 'historical lift'}
              * incremental lift : a) Historical_lift_test_per_store = (test_stores_current_week_sales - test_stores_historical_sales_avg)
                                   b) Historical_lift_control_across_all_stores = ∑(control_stores_historical_sales_avg - control_stores_current_week_sales)/Number of control stores
                                   c) incremental_lift_per_test_store =  (Historical_lift_test_per_store -  Historical_lift_control_across_all_stores)
              * running rate lift : a) avg_sales_control_across_all_stores =  (Total Sales in control for that week)/ Number of  Control Stores
                                    b) incremental_lift_per_test_store =  current week test store sales - avg_sales_control_across_all_stores
              * historical lift : a) incremental_lift_per_test_store compared to historical  =  (test_stores_historical_avg - test_stores_current_week_sales)
    optimization: boolean
                  to flag high and low performing stores
    threshold: can take value from 0.0 to 1.0
               to exlcude outliers, what portion of stores to be categories in high performing stores
    lift_plot: boolean
               to generate lift plot
    
    """
    
    if lift_type == 'incremental lift':
        t = get_incremental_lift(df_sales, df_hist, df_test, df_control)
    elif lift_type == 'running rate lift':
        t = get_running_rate_lift(df_sales, df_test, df_control)
    else:
        t = get_historical_lift(df_sales, df_hist, df_test)
    if optimization:
        #t = get_optimized_stores_v2(t, threshold, lift_plot)
        #t = get_optimized_stores_v21(t, threshold, lift_plot)
        #t = get_optimized_stores_v22(t, threshold, lift_plot)
        #t = get_optimized_stores_v23(t, threshold, lift_plot)
        t = get_optimized_stores_v3(t, lower_limit, upper_limit, lift_plot)
    return t


def __create_optimization_folder(input_folder):
    #default_folder = 'ab_group'
    optimization_path = os.path.join(input_folder, ab.default_folder, 'optimization')
    if not os.path.exists(optimization_path):
        print("Creating optimization folder, campaign optimization files will be generated at: %s" % (optimization_path))
        os.mkdir(optimization_path)
        print("Optimization folder created")
    return optimization_path


def __create_optimization_folder_v1(campaign_details):
    optimization_path = os.path.join(os.path.realpath(campaign_details['output_path']), campaign_details['campaign'],f"{campaign_details['client']}_{campaign_details['brand']}" ,'optimization', campaign_details['store'])
    if not os.path.exists(optimization_path):
        print("Creating optimization folder for %s, campaign optimization files will be generated at: %s" % (campaign_details['store'],optimization_path))
        os.makedirs(optimization_path)
        print("Optimization folder created")
    return optimization_path


def get_opt_test_df(input_folder, df_opt):
    trg_folder_path = os.path.join(input_folder, ab.default_folder, f'targetingPackage')
    isExist = os.path.exists(trg_folder_path)
    if not isExist:
        raise Exception(print("Given path does not have target package folder, further processed is aborted"))
    test = os.path.join(trg_folder_path, 'test_df.csv')
    test_zip = os.path.join(trg_folder_path, 'test_zipcodes.csv')
    df_test = pd.read_csv(test)
    df_test_zip = pd.read_csv(test_zip)
    
    df_test['IO'] = 'Low'
    df_test.loc[df_test['store_id'].isin(df_opt[df_opt['Performance'] == 'High']['store_id']), 'IO'] = 'High'
    t = df_test.groupby(['zipcode_expanded', 'IO'])['store_id'].agg('count').reset_index().groupby(['zipcode_expanded'])['IO'].agg('count').reset_index()
    
    #checking if same zipcode is marked both High and Low IO's, and marking it to only High
    df_test.loc[df_test['zipcode_expanded'].isin(t[t['IO'] > 1]['zipcode_expanded']), 'IO'] = 'High'
    
    # t = df_test.groupby(['zipcode_expanded', 'IO'])['store_id'].agg('count').reset_index().groupby(['zipcode_expanded'])['IO'].agg('count').reset_index()
    # t[t['IO'] > 1]['zipcode_expanded']
    # try:
    # if not t[t['IO'] > 1]['zipcode_expanded'].empty:
    #     t[0]
    # except:
    #     print("test df expanded zipcodes have duplicate IO's assignemnt")
    
    df_test_zip['IO'] = 'Low'
    df_test_zip.loc[df_test_zip['zipcode_expanded'].isin(df_test[df_test['IO'] == 'High']['zipcode_expanded']), 'IO'] = 'High'
    df_test_zip.groupby(['IO'])['zipcode_expanded'].nunique().reset_index()
    return df_test, df_test_zip


def get_opt_test_df_v1(campaign_details, df_opt):
    trg_folder_path = os.path.join(os.path.realpath(campaign_details['output_path']), campaign_details['campaign'], f"{campaign_details['client']}_{campaign_details['brand']}", 'ab_group', 'targetingPackage', campaign_details['store'])
    #trg_folder_path = os.path.join(input_folder, ab.default_folder, f'targetingPackage')
    isExist = os.path.exists(trg_folder_path)
    if not isExist:
        raise Exception(print("Given path does not have target package folder for %s, further processed is aborted" %campaign_details['store']))
    test = os.path.join(trg_folder_path, 'test_df.csv')
    test_zip = os.path.join(trg_folder_path, 'test_zipcodes.csv')
    df_test = pd.read_csv(test)
    df_test_zip = pd.read_csv(test_zip)
    
    df_test['IO'] = 'Low'
    df_test.loc[df_test['store_id'].isin(df_opt[df_opt['Performance'] == 'High']['store_id']), 'IO'] = 'High'
    t = df_test.groupby(['zipcode_expanded', 'IO'])['store_id'].agg('count').reset_index().groupby(['zipcode_expanded'])['IO'].agg('count').reset_index()
    
    #checking if same zipcode is marked both High and Low IO's, and marking it to only High
    df_test.loc[df_test['zipcode_expanded'].isin(t[t['IO'] > 1]['zipcode_expanded']), 'IO'] = 'High'
    
    # t = df_test.groupby(['zipcode_expanded', 'IO'])['store_id'].agg('count').reset_index().groupby(['zipcode_expanded'])['IO'].agg('count').reset_index()
    # t[t['IO'] > 1]['zipcode_expanded']
    # try:
    # if not t[t['IO'] > 1]['zipcode_expanded'].empty:
    #     t[0]
    # except:
    #     print("test df expanded zipcodes have duplicate IO's assignemnt")
    
    df_test_zip['IO'] = 'Low'
    df_test_zip.loc[df_test_zip['zipcode_expanded'].isin(df_test[df_test['IO'] == 'High']['zipcode_expanded']), 'IO'] = 'High'
    df_test_zip.groupby(['IO'])['zipcode_expanded'].nunique().reset_index()
    return df_test, df_test_zip




def generate_optimization_file(input_folder, df_lift, lower_limit, upper_limit, lift_plot, lift_type):
    df_opt, fig = get_optimized_stores_v3(df_lift, lower_limit, upper_limit, lift_plot)
    df_opt_test, df_opt_test_zip = get_opt_test_df(input_folder, df_opt)
    optimization_path = __create_optimization_folder(input_folder)
    x, t = ab.get_sales_df(input_folder, data_categories=['historical'])
    fig.savefig(os.path.join(optimization_path, f"{t['campaign']}_{t['client']}_{t['brand']}_{t['store']}_{lift_type}_plot.jpeg"))
    df_opt_test.to_csv(os.path.join(optimization_path, 'test_opt_df.csv'), index= False)
    df_opt_test_zip.to_csv(os.path.join(optimization_path, 'test_opt_zipcodes.csv'), index = False)
    print("Optimized zipcode file has been generated at: %s" %optimization_path)

    
def generate_optimization_file_v1(campaign_details, df_lift, lower_limit, upper_limit, lift_plot, lift_type):
    df_opt, fig = get_optimized_stores_v3(df_lift, lower_limit, upper_limit, lift_plot)
    df_opt_test, df_opt_test_zip = get_opt_test_df_v1(campaign_details, df_opt)
    optimization_path = __create_optimization_folder_v1(campaign_details)
    
    #x, t = ab.get_sales_df(input_folder, data_categories=['historical'])
    fig.savefig(os.path.join(optimization_path, f"{campaign_details['client']}_{campaign_details['brand']}_{campaign_details['store']}_{lift_type}_plot.jpeg"))
    df_opt_test.to_csv(os.path.join(optimization_path, 'test_opt_df.csv'), index= False)
    df_opt_test_zip.to_csv(os.path.join(optimization_path, 'test_opt_zipcodes.csv'), index = False)
    print("Optimized zipcode file for %s has been generated at: %s" %(campaign_details['store'], optimization_path))


def get_df_lift(input_folder, lift_type):
    df_hist, camp_details = ab.get_sales_df(input_folder)
    df_sales, x = ab.get_sales_df(input_folder, data_categories=['incampaign'])
    df_test, df_control = meas.get_ab_group(input_folder)
    
    if lift_type == 'incremental lift':
        t = get_incremental_lift(df_sales, df_hist, df_test, df_control)
    elif lift_type == 'running rate lift':
        t = get_running_rate_lift(df_sales, df_test, df_control)
    else:
        t = get_historical_lift(df_sales, df_hist, df_test)
    return t, camp_details


def get_lift_df_v1(campaign_details, historical_files, campaign_files, lift_type):
    df_hist_sales = []
    file_count = 0
    for file in historical_files:
        t = file.split('\\')[-1].split('_')[-1].split('.',1)[-1]
        if t == 'csv.bz2' or t == 'csv':
            df_hist_sales.append(pd.read_csv(file))
            file_count+=1
    print("Total historical sales files used for %s optimization: %d" %(campaign_details['store'], file_count))
    df_hist_sales = pd.concat(df_hist_sales)
    
    df_sales = []
    file_count = 0
    for file in campaign_files:
        t = file.split('\\')[-1].split('_')[-1].split('.',1)[-1]
        if t == 'csv.bz2' or t == 'csv':
            df_sales.append(pd.read_csv(file))
            file_count+=1
    print("Total campaign sales files used for %s optimization: %d" %(campaign_details['store'], file_count))
    df_sales = pd.concat(df_sales)
    
    # df_hist, camp_details = ab.get_sales_df(input_folder)
    # df_sales, x = ab.get_sales_df(input_folder, data_categories=['incampaign'])
    df_test, df_control = meas.get_ab_group_df(campaign_details)
    
    if lift_type == 'incremental lift':
        df_lift = get_incremental_lift(df_sales, df_hist_sales, df_test, df_control)
    elif lift_type == 'running rate lift':
        df_lift = get_running_rate_lift(df_sales, df_test, df_control)
    else:
        df_lift = get_historical_lift(df_sales, df_hist_sales, df_test)
    return df_lift


def calculate_lift_preprocessing(input_folder, lift_type, optimization, lower_limit, upper_limit, lift_plot):
    np.random.seed(123)
    df_lift, t = get_df_lift(input_folder, lift_type)
    fig = generate_contour_plot(df_lift)
    if optimization:
        #t = get_optimized_stores_v2(t, threshold, lift_plot)
        #t = get_optimized_stores_v21(t, threshold, lift_plot)
        #t = get_optimized_stores_v22(t, threshold, lift_plot)
        #t = get_optimized_stores_v23(t, threshold, lift_plot)
        generate_optimization_file(input_folder, df_lift, lower_limit, upper_limit, lift_plot, lift_type)
    optimization_path = __create_optimization_folder(input_folder)
    fig_path = os.path.join(optimization_path,f"{t['campaign']}_{t['client']}_{t['brand']}_{t['store']}_lift_vs_avg_sales_2D.jpeg")
    fig.write_image(fig_path, engine='kaleido')
    print("Optimization process has been completed !!")


def calculate_lift_preprocessing_v1(campaign_details, historical_files, campaign_files, lift_type, optimization, lower_limit, upper_limit, lift_plot):
    #np.random.seed(123)
    #df_lift, t = get_df_lift(input_folder, lift_type)
    df_lift = get_lift_df_v1(campaign_details, historical_files, campaign_files, lift_type)
    fig = generate_contour_plot(df_lift)
    if optimization:
        #t = get_optimized_stores_v2(t, threshold, lift_plot)
        #t = get_optimized_stores_v21(t, threshold, lift_plot)
        #t = get_optimized_stores_v22(t, threshold, lift_plot)
        #t = get_optimized_stores_v23(t, threshold, lift_plot)
        generate_optimization_file_v1(campaign_details, df_lift, lower_limit, upper_limit, lift_plot, lift_type)
    optimization_path = __create_optimization_folder_v1(campaign_details)
    fig_path = os.path.join(optimization_path,f"{campaign_details['client']}_{campaign_details['brand']}_{campaign_details['store']}_lift_vs_avg_sales_2D.jpeg")
    fig.write_image(fig_path, engine='kaleido')
    print("Optimization process has been completed !!")
    
    
def get_kde_plot(temp):
    sns.kdeplot(x=temp['Lift'], y=temp['Test Avg Sales'],cmap="Reds", shade=True, bw_adjust=.5)
    sns.kdeplot(data = pd.melt(temp[['store_id', 'Test Avg Sales', 'Lift']], ['store_id']), x='value', hue='variable', fill=True)
    plt.show()
    
def get_scatter_plot(temp):
    plt.rcParams['figure.figsize'] = [8, 8]
    sns.scatterplot(data=temp, x="Lift", y="Test Avg Sales", size="Test Total Sales", legend=False, sizes=(20, 500), alpha = 0.2)
    plt.show()
    
def get_line_plot(temp):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax =sns.lineplot(x = "Lift", y = "Test Avg Sales", data=temp)
    plt.show()
    
# color = ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
#              'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
#              'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
#              'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
#              'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
#              'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
#              'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
#              'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
#              'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
#              'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
#              'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
#              'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
#              'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
#              'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
#              'ylorrd']
    

#def calculate_lift(df_sales, df_hist, df_test, df_control, lift_type, optimization = True, threshold = 0.5, lift_plot = True):
def calculate_lift(input_folder, lift_type, optimization=True, lower_limit=0.5, upper_limit=0.5, lift_plot=True ):
    """
    Calculate lift in test group
    
    lift_type: {'incremental lift', 'running rate lift', 'historical lift'}
              * incremental lift : a) Historical_lift_test_per_store = (test_stores_current_week_sales - test_stores_historical_sales_avg)
                                   b) Historical_lift_control_across_all_stores = ∑(control_stores_historical_sales_avg - control_stores_current_week_sales)/Number of control stores
                                   c) incremental_lift_per_test_store =  (Historical_lift_test_per_store -  Historical_lift_control_across_all_stores)
              * running rate lift : a) avg_sales_control_across_all_stores =  (Total Sales in control for that week)/ Number of  Control Stores
                                    b) incremental_lift_per_test_store =  current week test store sales - avg_sales_control_across_all_stores
              * historical lift : a) incremental_lift_per_test_store compared to historical  =  (test_stores_historical_avg - test_stores_current_week_sales)
    optimization: boolean
                  to flag high and low performing stores
    threshold: can take value from 0.0 to 1.0
               to exlcude outliers, what portion of stores to be categories in high performing stores
    lift_plot: boolean
               to generate lift plot
    
    """
    calculate_lift_preprocessing(input_folder, lift_type, optimization, lower_limit, upper_limit, lift_plot)


    #def calculate_lift(df_sales, df_hist, df_test, df_control, lift_type, optimization = True, threshold = 0.5, lift_plot = True):
def calculate_lift_v1(campaign_details, historical_files, campaign_files, lift_type, optimization=True, lower_limit=0.5, upper_limit=0.5, lift_plot=True):
    """
    Calculate lift in test group
    
    lift_type: {'incremental lift', 'running rate lift', 'historical lift'}
              * incremental lift : a) Historical_lift_test_per_store = (test_stores_current_week_sales - test_stores_historical_sales_avg)
                                   b) Historical_lift_control_across_all_stores = ∑(control_stores_historical_sales_avg - control_stores_current_week_sales)/Number of control stores
                                   c) incremental_lift_per_test_store =  (Historical_lift_test_per_store -  Historical_lift_control_across_all_stores)
              * running rate lift : a) avg_sales_control_across_all_stores =  (Total Sales in control for that week)/ Number of  Control Stores
                                    b) incremental_lift_per_test_store =  current week test store sales - avg_sales_control_across_all_stores
              * historical lift : a) incremental_lift_per_test_store compared to historical  =  (test_stores_historical_avg - test_stores_current_week_sales)
    optimization: boolean
                  to flag high and low performing stores
    threshold: can take value from 0.0 to 1.0
               to exlcude outliers, what portion of stores to be categories in high performing stores
    lift_plot: boolean
               to generate lift plot
    
    """
    calculate_lift_preprocessing_v1(campaign_details, historical_files, campaign_files, lift_type, optimization, lower_limit, upper_limit, lift_plot)