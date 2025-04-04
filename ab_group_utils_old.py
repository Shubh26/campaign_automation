import sys
sys.path.extend(['D:\work\project\cac\z3_explorations', 'D:\work\project\cac\sales_measurment_service'])
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
from collections import defaultdict

from utils import constants as c
from utils import dataframe_utils
from warnings import warn
import pylab as pl
from IPython import display
#import mysql.connector

import MySQLdb
import mysql.connector
import pymysql

import random

import single_retailer_splitter as srs

sklearn.__version__

default_folder = 'ab_group'

def setup_db_connection():
    hostname = 'localhost'
    username = 'root'
    password = 'DSGtarget@247'
    database = 'target'
    targetConnect = MySQLdb.connect(host=hostname, user=username, passwd=password, db=database)
    return targetConnect

def get_store_expansion(store='7eleven', dbConnect=setup_db_connection()):
    start_time = datetime.now()
    con = setup_db_connection()
    print("Target db is connected, trying to fetch store expansion data, please wait it will take a while")
    t = pd.read_sql(f'SELECT * FROM target.zipcode_expansion_{store} LIMIT 90000;', con=dbConnect)
    finish_time = datetime.now()
    duration_sec = (finish_time - start_time).total_seconds()
    print("Data has been retrieved! (at=%s, total=%f sec)" % (str(finish_time), duration_sec))
    con.close()
    t = __format_expansion_df(t)
    return t

def get_campaign_details(file_path):
    t = file_path.split('\\')[-1].split('_')
    temp = {'campaign':t[3], 'client':t[4], 'brand':t[5], 'store':t[6]}
    return temp

def __format_hist_df(df_hist):
    np.random.seed(123)
    random_store = np.random.choice(df_hist['store_id'].unique(), size=400, replace=False)
    df_hist = df_hist[df_hist['store_id'].isin(random_store)]
    df_hist = dataframe_utils.StandardizeData.standardize_dataframe(df_hist)
    return df_hist

def get_sales_files(input_folder, data_categories=['historical']):
    input_path = os.path.join(input_folder,'*')
    files = glob.glob(input_path, recursive = False)
    #file_list = []
    cat_files = defaultdict(list)
    cat_count = defaultdict(int)
    for file in files:
        t = file.split('\\')[-1].split('_')[-1].split('.',1)[-1]
        if t == 'csv.bz2' or t == 'csv':
            t = file.split('\\')[-1].split('_')
            for each in t:
                for cat in data_categories:
                    if each == cat:
                        #file_list.append(file)
                        cat_files[cat].append(file)
                        cat_count[cat] += 1
    if not cat_count:
        raise Exception(print("Either of historical/campaign files are missing or there are no files at the given location"))

    for cat, count in cat_count.items():
        if cat == 'historical' and count == 1 and len(cat_count) == 1:
            print(f'{cat} sales file has been retreived')
            print(cat_files[cat])
            return cat_files
        elif cat == 'historical' and count > 1:
            raise Exception(print(f'There seems to be multiple {cat} file at the location, aborting the further process'))
        elif cat == 'historical' and count == 1 and len(cat_count) > 1:
            print(f'All data category sales files has been retrived: {sum(cat_count.values())}\nHistorical sales file: {count}')
        else:
            print(f'Total {cat} sales files found at the location: {count}')
            print(cat_files[cat])
            return cat_files
        
def get_sales_df(input_folder, data_categories=['historical']):
    cat_files = get_sales_files(input_folder, data_categories)
    if not cat_files:
        raise Exception(print("Further processing has been stopped"))
    df_sales = []
    for cat, files in cat_files.items():
        # if clause will be deprecated for historical cat this is only for testing purpsoe
        if cat == 'historical' and len(cat_files) == 1:
            for file in files:
                df = pd.read_csv(file)
                df = __format_hist_df(df)
                df_sales.append(df)
                t = get_campaign_details(file)
        else:
            for file in files:
                df = pd.read_csv(file)
                df_sales.append(df)
    df_sales = pd.concat(df_sales)
    df_sales = dataframe_utils.StandardizeData.standardize_dataframe(df_sales)
    t = get_campaign_details(cat_files[data_categories[0]][0])
    return df_sales, t

def get_hist_df(input_folder):
    """
    given input folder, it retreive historical file and convert it to dataframe
    param: input_folder - 'rD:\work\project\cac\ab_automation\lipton'
    """
    file = get_sales_files(input_folder)
    if file is None:
        raise Exception(print("Further processing has been stopped"))
    df_hist = pd.read_csv(file[0])
    df_hist = __format_hist_df(df_hist)
    t = get_campaign_details(file[0])
    return df_hist, t
        
def get_store_avg(df):
    t = df.groupby(['store_id', 'week'])['sales_dollars'].agg('sum').groupby(['store_id']).agg('mean').reset_index()
    #t.rename(columns={'sales_dollars': 'store_avg_sales'}, inplace = True)
    return t

def get_merge_df(df_exp, avg_sales):
    t = pd.merge(df_exp, avg_sales, how = 'inner', on = 'store_id')
    return t

def __create_ab_folder(input_folder, default_folder):
    ab_dir = os.path.join(input_folder, default_folder)
    if not os.path.exists(ab_dir):
        print("Creating ab_group folder, all computed files will be generated in this folder at: %s" % (ab_dir))
        os.mkdir(ab_dir)
        print("AB group folder created")
    return ab_dir

def __run_z3(ab_dir, temp, t, trg_folder, df_exp_db, df_hist, splits, avg_tol, size_tol):
    z3_input_file = os.path.join(ab_dir, f"{temp['campaign']}_{temp['client']}_{temp['brand']}_{temp['store']}_z3_input.json")
    z3_input = file_utils.save_json_format_for_z3(t, z3_input_file)
    z3_output_file = os.path.join(ab_dir, f"{temp['campaign']}_{temp['client']}_{temp['brand']}_{temp['store']}_z3_output.xlsx")
    srs.single_retailer_splitter(z3_input_file, splits, avg_tol, size_tol, z3_output_file)
    file_utils.convert_save_z3_results_to_platform_format(z3_output_file, df_exp_db, df_hist, trg_folder)
    print("Targeting groups has been generated at the location: %s" %trg_folder)
    
def __format_expansion_df(df_exp_db):
    df_exp_db['radius'] = df_exp_db['radius'].astype(int)
    df_exp_db['zipcode'] = df_exp_db['zipcode'].astype(str)
    df_exp_db['zipcode_expanded'] = df_exp_db['zipcode_expanded'].astype(str)
    return df_exp_db

def ab_group_preprocessing(input_folder, store, splits, avg_tol, size_tol):
    #default_folder = 'ab_group'
    ab_dir = __create_ab_folder(input_folder, default_folder)
    trg_folder = os.path.join(input_folder, default_folder, f'targetingPackage')
    
    df_exp_db = get_store_expansion(store)
    #df_hist, temp = get_hist_df(input_folder)
    df_hist, temp = get_sales_df(input_folder)
    t = get_store_avg(df_hist)
    t = get_merge_df(df_exp_db, t)
    __run_z3(ab_dir, temp, t, trg_folder, df_exp_db, df_hist, splits, avg_tol, size_tol)
    print("AB group creation process has been completed !!")
    
def generate_ab_group(input_folder, store, splits, avg_tol, size_tol):
    ab_group_preprocessing(input_folder, store, splits, avg_tol, size_tol)
    
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