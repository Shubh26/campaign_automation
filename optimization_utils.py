import sys
import pandas as pd
import numpy as np
import kaleido
# if used in jupyter lab need to have the extension - https://www.npmjs.com/package/jupyterlab-plotly
import plotly
import plotly.offline as pyo
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import os, copy, glob, shutil
from datetime import datetime,timedelta
import re, time
import seaborn as sns
import logging
from utils import date_utils
from utils import measurement_utils as meas
from pathlib import Path
from warnings import warn
import pylab as pl
from IPython import display
from utils import ab_group_utils as ab

def get_store_avg_sales_v3(df_sales, df_group=None):
    df_all = []
    if df_group is not None:
        df_sales = df_sales[df_sales['store_id'].isin(df_group['store_id'])]
    for time_frame in df_sales['time_frame'].unique():
        df_temp = df_sales[df_sales['time_frame'] == time_frame]
        df_temp = df_temp.groupby(['store_id', 'week'])['sales_dollars'].agg('sum').reset_index()
        if time_frame == 'weekly':
            df_all.append(df_temp)
        elif time_frame == 'yearly':
            df_temp['sales_dollars'] = df_temp['sales_dollars']/52
            df_all.append(df_temp)
        elif time_frame == 'halfyearly':
            df_temp['sales_dollars'] = df_temp['sales_dollars']/26
            df_all.append(df_temp)
        elif time_frame == 'quarterly':
            df_temp['sales_dollars'] = df_temp['sales_dollars']/13
            df_all.append(df_temp)
        elif time_frame == 'monthly':
            df_temp['sales_dollars'] = df_temp['sales_dollars']/4
            df_all.append(df_temp)
    t = pd.concat(df_all)
    df_store_avg = t.groupby(['store_id', 'week'])['sales_dollars'].agg('sum').groupby(['store_id']).agg('mean').reset_index()
    return df_store_avg

def get_store_avg_and_total_sales(df_sales, df_group):
    df1 = get_store_avg_sales_v3(df_sales, df_group)
    df2 = df_sales[df_sales['store_id'].isin(df_group['store_id'])].groupby(['store_id'])['sales_dollars'].agg('sum').reset_index()
    df1.rename(columns = {'sales_dollars': f'Avg Sales'}, inplace = True)
    df2.rename(columns = {'sales_dollars': f'Total Sales'}, inplace = True)
    t = pd.merge(df1, df2, on = 'store_id', how = 'inner')
    return t

def get_incremental_lift(df_sales, df_hist, df_test, df_control):
    groups = ['Test', 'Control']
    df_group = [df_test, df_control]
    for idx, group in enumerate(groups):
        #df1 = get_store_avg_sales_v2(df_sales, df_group[idx])
        df1 = get_store_avg_and_total_sales(df_sales, df_group[idx])
        df1.rename(columns = {'Avg Sales':f'{group} Avg Sales', 'Total Sales':f'{group} Total Sales'}, inplace = True)
        #df2 = get_store_avg_sales_v2(df_hist, df_group[idx])
        df2 = get_store_avg_and_total_sales(df_hist, df_group[idx])
        df2.rename(columns = {'Avg Sales': f'{group} Historical Avg Sales', 'Total Sales': f'{group} Historical Total Sales'}, inplace = True)
        t = pd.merge(df1, df2, on = 'store_id', how = 'inner')
        if group == 'Test':
            t['Lift'] = t[f'{group} Avg Sales'] - t[f'{group} Historical Avg Sales']
            temp = t.copy()
        else:
            t = t[f'{group} Avg Sales'].mean() - t[f'{group} Historical Avg Sales'].mean()
    temp['Lift'] = ((temp['Lift'] - t)/t)*100
    return temp

def get_running_rate_lift(df_sales, df_test, df_control):
    groups = ['Test', 'Control']
    df_group = [test, control]
    for idx, group in enumerate(groups):
        #t = get_store_avg_sales_v2(df_sales, df_group[idx])
        t = get_store_avg_and_total_sales(df_sales, df_group[idx])
        t.rename(columns = {'Avg Sales':f'{group} Avg Sales', 'Total Sales':f'{group} Total Sales'}, inplace = True)
        if group == 'Control':
            t = t[f'{group} Avg Sales'].mean()
        else:
            temp = t.copy()
    temp['Lift'] = ((temp['Test Avg Sales'] - t)/t)*100
    return temp

def get_historical_lift(df_sales, df_hist, df_test):
    #df1 = get_store_avg_sales_v2(df_sales, df_test)
    df1 = get_store_avg_and_total_sales(df_sales, df_test)
    df1.rename(columns = {'Avg Sales':'Test Avg Sales', 'Total Sales': 'Test Total Sales'}, inplace = True)
    #df2 = get_store_avg_sales_v2(df_hist, df_test)
    df2 = get_store_avg_and_total_sales(df_hist, df_test)
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

def get_optimized_stores_v3(df, lower_limit = 0.5, upper_limit = 0.5, lift_plot = True):
    temp = df.copy()
    std = np.std(temp['Lift'])
    mean = np.mean(temp['Lift'])
    limits = {'lower_limit': lower_limit, 'upper_limit': upper_limit}
    for k, limit in limits.items():
        if k == 'lower_limit':
            actual = 0
            counter = 0.1
            t1 = temp[temp['Lift'] < mean]
            while actual < limit:
                ll = mean - counter*std
                t = t1[t1['Lift'] > ll]['store_id'].unique()
                t1['Performance'] = 'Low'
                t1.loc[t1['store_id'].isin(t), 'Performance'] = 'High'
                actual = len(t)/t1['store_id'].nunique()
                counter+=0.01
        else:
            actual = 0
            counter = 0.1
            t2 = temp[temp['Lift'] >= mean]
            while actual < limit:
                ul = mean + counter*std
                t = t2[t2['Lift'] < ul]['store_id'].unique()
                t2['Performance'] = 'Low'
                t2.loc[t2['store_id'].isin(t), 'Performance'] = 'High'
                actual = len(t)/t2['store_id'].nunique()
                counter+=0.01
                
    temp = pd.concat([t1,t2])
    t = temp.groupby(['Performance'])['store_id'].nunique().reset_index()
    t.rename(columns={'store_id':'store_count'}, inplace = True)
    t['split'] = t['store_count'].apply(lambda x: round((x/t['store_count'].sum())*100,2))
    t['split'] = t['split'].astype(str) + '%'
    print("optimization summary:\nTotal stores %d" %(t['store_count'].sum()))
    print(t.to_string(index=False))
    if lift_plot:
        fig = get_lift_plot(temp, ul, ll)
    return temp, fig


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

def __create_optimization_folder_v1(campaign_details):
    optimization_path = os.path.join(os.path.realpath(campaign_details['output_path']), campaign_details['campaign'],f"{campaign_details['client']}_{campaign_details['brand']}" ,'optimization', campaign_details['store'])
    if not os.path.exists(optimization_path):
        print("Creating optimization folder for %s, campaign optimization files will be generated at: %s" % (campaign_details['store'],optimization_path))
        os.makedirs(optimization_path)
        print("Optimization folder created")
    return optimization_path

def get_opt_test_df_v1(campaign_details, df_opt):
    trg_folder_path = os.path.join(os.path.realpath(campaign_details['output_path']), campaign_details['campaign'], f"{campaign_details['client']}_{campaign_details['brand']}", 'ab_group', 'targetingPackage', campaign_details['store'])
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
    df_test_zip['IO'] = 'Low'
    df_test_zip.loc[df_test_zip['zipcode_expanded'].isin(df_test[df_test['IO'] == 'High']['zipcode_expanded']), 'IO'] = 'High'
    df_test_zip.groupby(['IO'])['zipcode_expanded'].nunique().reset_index()
    return df_test, df_test_zip
    
    
def generate_optimization_file_v1(campaign_details, df_lift, lower_limit, upper_limit, lift_plot, lift_type):
    df_opt, fig = get_optimized_stores_v3(df_lift, lower_limit, upper_limit, lift_plot)
    df_opt_test, df_opt_test_zip = get_opt_test_df_v1(campaign_details, df_opt)
    optimization_path = __create_optimization_folder_v1(campaign_details)
    fig.savefig(os.path.join(optimization_path, f"{campaign_details['client']}_{campaign_details['brand']}_{campaign_details['store']}_{lift_type}_plot.jpeg"))
    df_opt_test.to_csv(os.path.join(optimization_path, 'test_opt_df.csv'), index= False)
    df_opt_test_zip.to_csv(os.path.join(optimization_path, 'test_opt_zipcodes.csv'), index = False)
    print("Optimized zipcode file for %s has been generated at: %s" %(campaign_details['store'], optimization_path))


def get_lift_df_v2(campaign_details, historical_files, campaign_files, lift_type):
    df_hist_sales = get_historical_sales(campaign_details, historical_files)
    df_sales = meas.get_campaign_sales(campaign_details, campaign_files)
    if 'inclusion' in campaign_details:
        if campaign_details['store'] in campaign_details['inclusion']:
            df_hist_sales = meas.impose_inclusion(campaign_details, df_hist_sales, 'historical')
            df_sales = meas.impose_inclusion(campaign_details, df_sales, 'campaign')
    if 'exclusion' in campaign_details:
        if campaign_details['store'] in campaign_details['exclusion']:
            df_hist_sales = meas.impose_exclusion(campaign_details, df_hist_sales, 'historical')
            df_sales = meas.impose_exclusion(campaign_details, df_sales, 'campaign')
    df_test, df_control = meas.get_ab_group_df(campaign_details)
    if lift_type == 'incremental lift':
        df_lift = get_incremental_lift(df_sales, df_hist_sales, df_test, df_control)
    elif lift_type == 'running rate lift':
        df_lift = get_running_rate_lift(df_sales, df_test, df_control)
    else:
        df_lift = get_historical_lift(df_sales, df_hist_sales, df_test)
    return df_lift
    
def calculate_lift_preprocessing_v2(campaign_details, historical_files, campaign_files, lift_type, optimization, lower_limit, upper_limit, lift_plot):
    df_lift = get_lift_df_v2(campaign_details, historical_files, campaign_files, lift_type)
    fig = generate_contour_plot(df_lift)
    if optimization:
        generate_optimization_file_v1(campaign_details, df_lift, lower_limit, upper_limit, lift_plot, lift_type)
    optimization_path = __create_optimization_folder_v1(campaign_details)
    fig_path = os.path.join(optimization_path,f"{campaign_details['client']}_{campaign_details['brand']}_{campaign_details['store']}_lift_vs_avg_sales_2D.jpeg")
    fig.write_image(fig_path, engine='kaleido')
    print("Optimization process for %s has been completed !!" %campaign_details['store'])
    
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
    
def is_sales_recent(df_sales, n_weeks):
    """filter df for stores that have sales in given recent weeks"""
    end_date = date_utils.get_date(df_sales['week'].max(), date_utils.DATE_FORMAT_ISO)
    start_date = date_utils.add_days(end_date, -(n_weeks * 7))
    temp = df_sales[df_sales['week'] > start_date.strftime("%Y-%m-%d")]
    temp = temp.groupby(['store_id'])['week'].nunique().reset_index()
    recent_sales_stores = temp[temp['week'] == n_weeks]['store_id'].unique()
    df_sales = df_sales[df_sales['store_id'].isin(recent_sales_stores)]
    return df_sales

def get_historical_sales(campaign_details, historical_files):
    df_hist_sales = []
    file_count = 0
    for file in historical_files:
        #t = file.split('\\')[-1].split('_')[-1].split('.',1)[-1]
        t = Path(file).name.split('.', 1)[-1]
        if t == 'csv.bz2' or t == 'csv':
            df_hist_sales.append(pd.read_csv(file))
            file_count+=1
    print("Total historical sales files used for %s %s: %d" %(campaign_details['store'],campaign_details['task'],file_count))
    df_hist_sales = pd.concat(df_hist_sales)
    
    if 'recent_sales_week' in campaign_details:
        if campaign_details['store'] in campaign_details['recent_sales_week']:
            df_hist_sales = is_sales_recent(df_hist_sales, campaign_details['recent_sales_week'][campaign_details['store']])
            
    df_hist_sales = df_hist_sales[df_hist_sales['sales_dollars'] != 0]
    return df_hist_sales    
    
def calculate_lift_v2(campaign_details, historical_files, campaign_files, lift_type, optimization=True, lower_limit=0.5, upper_limit=0.5, lift_plot=True):
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
    calculate_lift_preprocessing_v2(campaign_details, historical_files, campaign_files, lift_type, optimization, lower_limit, upper_limit, lift_plot)