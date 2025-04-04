import sys
sys.path.extend(['D:\work\project\cac\sales_measurment_service'])
import pandas as pd
import numpy as np
# if used in jupyter lab need to have the extension - https://www.npmjs.com/package/jupyterlab-plotly
import plotly
import plotly.offline as pyo
import plotly.express as px
import plotly.graph_objects as go
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
#, processor, campaign_helper

from pathlib import Path

import math
import geopy.distance
import sklearn
import collections,itertools
from collections import defaultdict

from utils import constants as c
from utils import dataframe_utils
from utils import ab_group_utils as ab


sklearn.__version__

product = 'product'
product_category = 'product_cat'
product_sub_category = 'product_sub_cat'
store_division = 'store_division'

def get_avg_sales_bckup(df, level=None):
#     total sales
#     df.groupby(['week'])['sales_dollars'].agg('sum').reset_index()
#     df.groupby(['week','product'])['sales_dollars'].agg('sum').reset_index()
    
#     store count    
#     df.groupby(['week'])['store_id'].nunique().reset_index()
#     df.groupby(['week', 'product'])['store_id'].nunique().reset_index()
    
#     avg sales    
#     df = df.groupby(['week', 'store_id'])['sales_dollars'].agg('sum').reset_index().groupby(['week'])['sales_dollars'].agg('mean').reset_index()
#     df = df.groupby(['week','product', 'store_id'])['sales_dollars'].agg('sum').reset_index().groupby(['week', 'product'])['sales_dollars'].agg('mean').reset_index()
    
    avg = True
    df = df.groupby(get_groupby(level, avg))['sales_dollars'].agg('sum').reset_index().groupby(get_groupby(level))['sales_dollars'].agg('mean').reset_index()
    return df


def get_group_col(df_sales, df_test, df_control):
    GROUP_COL = 'group'
    t = 'not reported'
    df_sales[GROUP_COL] = t
    df_all = [df_test, df_control]
    group = ['test', 'control']
    for idx, df in enumerate(df_all):
        df_sales.loc[df_sales['store_id'].isin(df['store_id']), GROUP_COL] = group[idx]
        #df_sales.loc[~df_sales['store_id'].isin(df['store_id']), GROUP_COL] = f'{group[idx]} {t}'
    return df_sales

def get_non_reported_sales_store(df_sales, df_test, df_control, not_reported_stores = True):
    if not_reported_stores:
        df_sales = df_sales[df_sales['group'] == 'not reported']
    else:
        df_sales = df_sales[~(df_sales['group'] == 'not reported')]
    return df_sales

def get_total_sales(df, level=None):
    # df = df.groupby(['week'])['sales_dollars'].agg('sum').reset_index()
    # df = df.groupby(['week','product'])['sales_dollars'].agg('sum').reset_index()
    df = df.groupby(get_groupby(level))['sales_dollars'].agg('sum').reset_index()
    return df

def get_store_count(df, level=None):
    # df = df.groupby(['week'])['store_id'].nunique().reset_index()
    # df = df.groupby(['week', 'product'])['store_id'].nunique().reset_index()
    df = df.groupby(get_groupby(level))['store_id'].nunique().reset_index()
    return df

def get_groupby(level=None, avg=False):
    if not avg:
        if level == 'product':
            t = ['week', 'product']
        elif level == 'product_cat':
            t = ['week','product_cat']
        elif level == 'product_sub_cat':
            t = ['week','product_sub_cat']
        elif level == 'store_division':
            t = ['week','store_division']
        else:
            t = ['week']
    else:
        if level == 'product':
            t = ['week','product', 'store_id']
        elif level == 'product_cat':
            t = ['week','product_cat', 'store_id']
        elif level == 'product_sub_cat':
            t = ['week','product_sub_cat', 'store_id']
        elif level == 'store_division':
            t = ['week','store_division', 'store_id']
        else:
            t = ['week', 'store_id']
    return t

def get_avg_sales(df, level=None):
    avg = True
    df = df.groupby(get_groupby(level, avg))['sales_dollars'].agg('sum').reset_index().groupby(get_groupby(level))['sales_dollars'].agg('mean').reset_index()
    return df

def __get_overall_sales(df):
    overall = df.mean().reset_index().transpose()
    overall.rename(columns=overall.iloc[0], inplace = True)
    overall.drop(overall.index[0], inplace = True)
    overall.insert(loc=0, column='week' , value='overall')
    overall.iloc[:,1:] = overall.iloc[:,1:].apply(pd.to_numeric)
    overall = overall.round()
    overall.iloc[:,1:] = overall.iloc[:,1:].astype(int)
    temp = pd.concat([df, overall])
    overall_test_avg = temp.loc[temp['week'] == 'overall']['Test Sales']/temp.loc[temp['week'] == 'overall']['Test Stores']
    overall_control_avg = temp.loc[temp['week'] == 'overall']['Control Sales']/temp.loc[temp['week'] == 'overall']['Control Stores']
    overall_sales_lift = ((overall_test_avg[0] - overall_control_avg[0])/overall_control_avg[0])*100
    temp.loc[temp['week'] == 'overall', ['Test Avg','Control Avg', 'Sales Lift']] = [round(overall_test_avg[0],2), round(overall_control_avg[0],2), round(overall_sales_lift,2)]
    return temp

def get_measurement_metric(df_sales, level=None):
    #t = df_sales.copy()
    groups = ['Test', 'Control']
    temp = []
    for group in groups:
        tmp = group.lower()
        #print(group)
        df_temp = df_sales[df_sales['group'] == tmp]
        total_sales = get_total_sales(df_temp, level)
        total_sales.rename(columns={'sales_dollars': f'{group} Sales'}, inplace = True)
        total_sales = round(total_sales)
        total_sales[f'{group} Sales'] = total_sales[f'{group} Sales'].astype(int)
        store_count = get_store_count(df_temp, level)
        store_count.rename(columns={'store_id': f'{group} Stores'}, inplace = True)
        avg_sales = get_avg_sales(df_temp, level)
        avg_sales.rename(columns={'sales_dollars': f'{group} Avg'}, inplace = True)
        avg_sales = round(avg_sales, 2)
        t = pd.concat([total_sales, store_count, avg_sales], axis=1)
        t = t.loc[:,~t.columns.duplicated()]
        temp.append(t)
    if level is None:
        temp = pd.merge(temp[0], temp[1], on = ['week'], how = 'inner')
    else:
        temp = pd.merge(temp[0], temp[1], on = ['week', level], how = 'inner')
    # temp = pd.concat(temp, axis=1)
    # temp = temp.loc[:,~temp.columns.duplicated()]
    temp['Sales Lift'] = round(((temp['Test Avg'] - temp['Control Avg'])/ temp['Control Avg'])*100, 2)
    temp = __get_overall_sales(temp)
    temp.rename(columns={'week':'Week'}, inplace=True)
    temp = temp[['Week','Test Sales','Control Sales','Test Stores','Control Stores','Test Avg','Control Avg','Sales Lift']]
    return temp

def get_group_store_count(df_test, df_control):
    test = df_test['store_id'].nunique()
    control = df_control['store_id'].nunique()
    temp = [test, control]
    return temp

def get_nonreported_store(df_sales, df_test, df_control):
    groups = ['Test', 'Control']
    t = get_group_store_count(df_test, df_control)
    temp = []
    for idx, group in enumerate(groups):
        tmp = group.lower()
        df_temp = df_sales[df_sales['group'] == tmp]
        store_count = get_store_count(df_temp)
        store_count.rename(columns={'store_id': f'{group} Stores'}, inplace = True)
        store_count[f'Non Reported {group} Stores'] = t[idx] - store_count[f'{group} Stores']
        store_count[f'Total {group} Stores'] = t[idx]
        temp.append(store_count)
    temp = pd.merge(temp[0], temp[1], on = ['week'], how = 'inner')
    # if level is None:
    #     temp = pd.merge(temp[0], temp[1], on = ['week'], how = 'inner')
    # else:
    #     temp = pd.merge(temp[0], temp[1], on = ['week', level], how = 'inner')
    return temp

def generate_measurement_file_bckup(df_sales, df_test, df_control, file_path, result_path, list_of_metrics=[None]):
    sales = get_group_col(df_sales, df_test, df_control)
    sales = get_non_reported_sales_store(sales, df_test, df_control , False)
    file_path.split('\\')[-1].split('_')
    client = file_path.split('\\')[-1].split('_')[3]
    brand = file_path.split('\\')[-1].split('_')[4]
    retail_chain = file_path.split('\\')[-1].split('_')[5]
    results_file = os.path.join(result_path,f'{client}_{brand}_{retail_chain}_measurement.xlsx')
    xlsxwriter_obj = pd.ExcelWriter(results_file, engine='xlsxwriter')
    for lvl in list_of_metrics:
        temp = get_measurement_metric(df_sales, lvl)
        if lvl is None:
            temp.to_excel(xlsxwriter_obj, sheet_name='weekly_sales_report', index = False)
        else:
            temp.to_excel(xlsxwriter_obj, sheet_name=f'{lvl}_sales_report', index = False)
    temp = get_nonreported_store(df_sales, df_test, df_control)
    temp.to_excel(xlsxwriter_obj, sheet_name=f'nonreported_stores', index = False)
    xlsxwriter_obj.save()
    
def __create_measurement_folder(input_folder):
    #default_folder = 'ab_group'
    measurement_path = os.path.join(input_folder, ab.default_folder, 'measurement')
    if not os.path.exists(measurement_path):
        print("Creating measurement folder, campaign measurement files will be generated at: %s" % (measurement_path))
        os.mkdir(measurement_path)
        print("Measurement folder created")
    return measurement_path

def __create_measurement_folder_v1(campaign_details):
    measurement_path = os.path.join(os.path.realpath(campaign_details['output_path']), campaign_details['campaign'],f"{campaign_details['client']}_{campaign_details['brand']}" ,'measurement')
    if not os.path.exists(measurement_path):
        print("Creating measurement folder, campaign measurement files will be generated at: %s" % (measurement_path))
        os.mkdir(measurement_path)
        print("Measurement folder created")
    return measurement_path

def get_ab_group(input_folder):
    ab_group_data = {}
    #default_folder = 'ab_group'
    ab_group_folder = 'targetingPackage'
    ab_group_path = os.path.join(input_folder, ab.default_folder, ab_group_folder)
    isExist = os.path.exists(ab_group_path)
    if not isExist:
        raise Exception(print("Given path does not have target package folder, further processed is aborted"))
    test = os.path.join(ab_group_path, 'test_df.csv')
    control = os.path.join(ab_group_path, 'control_df.csv')
    df_test = pd.read_csv(test)
    df_control = pd.read_csv(control)
    return df_test, df_control

def get_ab_group_df(campaign_details):
    #ab_group_path
    trg_folder_path = os.path.join(os.path.realpath(campaign_details['output_path']), campaign_details['campaign'], f"{campaign_details['client']}_{campaign_details['brand']}", 'ab_group', 'targetingPackage', campaign_details['store'])
    isExist = os.path.exists(trg_folder_path)
    if not isExist:
        raise Exception(print("Given campaign details are not correct or target package folder for %s is not available, further processed is aborted" %campaign_details['store']))
    test = os.path.join(trg_folder_path, 'test_df.csv')
    control = os.path.join(trg_folder_path, 'control_df.csv')
    df_test = pd.read_csv(test)
    df_control = pd.read_csv(control)
    return df_test, df_control

def measurement_file_preprocessing(input_folder, list_of_metrics):
    df_sales, t = ab.get_sales_df(input_folder, data_categories=['incampaign'])
    df_test, df_control = get_ab_group(input_folder)
    df_sales = get_group_col(df_sales, df_test, df_control)
    df_sales = get_non_reported_sales_store(df_sales, df_test, df_control , False)
    measurement_path = __create_measurement_folder(input_folder)
    measurement_file = os.path.join(measurement_path,f"{t['campaign']}_{t['client']}_{t['brand']}_{t['store']}_measurement.xlsx")
    with pd.ExcelWriter(measurement_file, engine='xlsxwriter') as xlsxwriter: # openpyxl
        for lvl in list_of_metrics:
            temp = get_measurement_metric(df_sales, lvl)
            if lvl is None:
                temp.to_excel(xlsxwriter, sheet_name='weekly_sales_report', index = False)
            else:
                temp.to_excel(xlsxwriter, sheet_name=f'{lvl}_sales_report', index = False)
        temp = get_nonreported_store(df_sales, df_test, df_control)
        temp.to_excel(xlsxwriter, sheet_name=f'nonreported_stores', index = False)
        xlsxwriter.save()
        # xlsxwriter.close()
        # xlsxwriter.handles = None
    print("Measurement file has been generated at: %s" %measurement_path)
    print("Measurement file generation process has been completed !!")
    
def measurement_file_preprocessing_v1(campaign_details, df_sales, list_of_metrics):
    df_test, df_control = get_ab_group_df(campaign_details)
    df_sales = get_group_col(df_sales, df_test, df_control)
    df_sales = get_non_reported_sales_store(df_sales, df_test, df_control , False)
    measurement_path = __create_measurement_folder_v1(campaign_details)
    measurement_file = os.path.join(measurement_path,f"{campaign_details['client']}_{campaign_details['brand']}_{campaign_details['store']}_measurement.xlsx")
    with pd.ExcelWriter(measurement_file, engine='xlsxwriter') as xlsxwriter:
        for lvl in list_of_metrics:
            temp = get_measurement_metric(df_sales, lvl)
            if lvl is None:
                temp.to_excel(xlsxwriter, sheet_name='weekly_sales_report', index = False)
            else:
                temp.to_excel(xlsxwriter, sheet_name=f'{lvl}_sales_report', index = False)
        temp = get_nonreported_store(df_sales, df_test, df_control)
        temp.to_excel(xlsxwriter, sheet_name=f'nonreported_stores', index = False)
        xlsxwriter.save()
    print("Measurement file has been generated at: %s" %measurement_path)
    print("Measurement file generation process has been completed !!")

def measurement_file_preprocessing_v2(campaign_details, campaign_files, list_of_metrics):
    df_sales = []
    file_count = 0
    for file in campaign_files:
        t = file.split('\\')[-1].split('_')[-1].split('.',1)[-1]
        if t == 'csv.bz2' or t == 'csv':
            df_sales.append(pd.read_csv(file))
            file_count+=1
    print("Total campaign sales files used for %s measurement: %d" %(campaign_details['store'], file_count))
    df_sales = pd.concat(df_sales)
    df_test, df_control = get_ab_group_df(campaign_details)
    df_sales = get_group_col(df_sales, df_test, df_control)
    df_sales = get_non_reported_sales_store(df_sales, df_test, df_control , False)
    measurement_path = __create_measurement_folder_v1(campaign_details)
    measurement_file = os.path.join(measurement_path,f"{campaign_details['client']}_{campaign_details['brand']}_{campaign_details['store']}_measurement.xlsx")
    with pd.ExcelWriter(measurement_file, engine='xlsxwriter') as xlsxwriter:
        for lvl in list_of_metrics:
            temp = get_measurement_metric(df_sales, lvl)
            if lvl is None:
                temp.to_excel(xlsxwriter, sheet_name='weekly_sales_report', index = False)
            else:
                temp.to_excel(xlsxwriter, sheet_name=f'{lvl}_sales_report', index = False)
        temp = get_nonreported_store(df_sales, df_test, df_control)
        temp.to_excel(xlsxwriter, sheet_name=f'nonreported_stores', index = False)
        xlsxwriter.save()
    print("Measurement file has been generated at: %s" %measurement_path)
    print("Measurement file generation process has been completed !!")
    
def generate_measurement_file(input_folder, list_of_metrics=[None]):
    measurement_file_preprocessing(input_folder, list_of_metrics)
    
def generate_measurement_file_v1(campaign_details, df_sales, list_of_metrics=[None]):
    measurement_file_preprocessing_v1(campaign_details, df_sales, list_of_metrics)
    
def generate_measurement_file_v2(campaign_details, campaign_files, list_of_metrics=[None]):
    measurement_file_preprocessing_v2(campaign_details, campaign_files, list_of_metrics)    
    
def init_logging(logs_file):
    # https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout

    file_handler = logging.handlers.RotatingFileHandler(logs_file, maxBytes=(1048576*5), backupCount=7)
    # file_handler = logging.FileHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # logging.get_absl_handler().use_absl_log_file('absl_logging', os.path.join(main_data_folder,'Smithfield/AlwaysOn/logs'))
    file_handler.setFormatter(formatter)

    rootLogger = logging.getLogger()
    # customLogger = logging.getLogger('TestControl')
    rootLogger.setLevel(logging.INFO)
    # removing existing loggers
    while(len(rootLogger.handlers)>0):
        rootLogger.removeHandler(rootLogger.handlers[0])

    rootLogger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    rootLogger.addHandler(stream_handler)
    print('logging info',rootLogger.handlers)