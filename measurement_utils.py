import sys
import pandas as pd
import numpy as np
# if used in jupyter lab need to have the extension - https://www.npmjs.com/package/jupyterlab-plotly
import os, copy, glob, shutil
from datetime import datetime,timedelta
import re, time, logging
from pathlib import Path
from utils import ab_group_utils as ab

def get_group_col(df_sales, df_test, df_control):
    GROUP_COL = 'group'
    t = 'not reported'
    df_sales[GROUP_COL] = t
    df_all = [df_test, df_control]
    group = ['test', 'control']
    for idx, df in enumerate(df_all):
        df_sales.loc[df_sales['store_id'].isin(df['store_id']), GROUP_COL] = group[idx]
    return df_sales

def get_non_reported_sales_store(df_sales, df_test, df_control, not_reported_stores = True):
    if not_reported_stores:
        df_sales = df_sales[df_sales['group'] == 'not reported']
    else:
        df_sales = df_sales[~(df_sales['group'] == 'not reported')]
    return df_sales

def get_total_sales(df, level=None):
    df = df.groupby(get_groupby(level))['sales_dollars'].agg('sum').reset_index()
    return df

def get_store_count(df, level=None):
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
    overall.insert(loc=0, column='week' , value='Overall')
    overall.iloc[:,1:] = overall.iloc[:,1:].apply(pd.to_numeric)
    overall = overall.round()
    overall.iloc[:,1:] = overall.iloc[:,1:].astype(int)
    temp = pd.concat([df, overall])
    overall_test_avg = temp.loc[temp['week'] == 'Overall']['Test Sales']/temp.loc[temp['week'] == 'Overall']['Test Stores']
    overall_control_avg = temp.loc[temp['week'] == 'Overall']['Control Sales']/temp.loc[temp['week'] == 'Overall']['Control Stores']
    overall_sales_lift = ((overall_test_avg[0] - overall_control_avg[0])/overall_control_avg[0])*100
    temp.loc[temp['week'] == 'Overall', ['Test Avg','Control Avg', 'Sales Lift']] = [round(overall_test_avg[0],2), round(overall_control_avg[0],2), round(overall_sales_lift,2)]
    return temp

def get_measurement_metric(df_sales, level=None):
    groups = ['Test', 'Control']
    temp = []
    for group in groups:
        tmp = group.lower()
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
    temp['Sales Lift'] = round(((temp['Test Avg'] - temp['Control Avg'])/ temp['Control Avg'])*100, 2)
    
    temp['week'] = temp['week'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%d-%b-%y'))
    temp = __get_overall_sales(temp)
    temp.rename(columns={'week':'Week','Test Avg':'Test Avg.','Control Avg':'Control Avg.'}, inplace=True)
    if level is None:
        temp = temp[['Week','Test Sales','Control Sales','Test Stores','Control Stores','Test Avg.','Control Avg.','Sales Lift']]
    else:
        temp = temp[['Week',level,'Test Sales','Control Sales','Test Stores','Control Stores','Test Avg.','Control Avg.','Sales Lift']]
    temp = format_measurment_metric(temp)
    return temp


def format_measurment_metric(output_df):
    output_df.loc[:, "Test Sales"] = '$'+ output_df["Test Sales"].map('{:,.0f}'.format)
    output_df.loc[:, "Control Sales"] = '$'+ output_df["Control Sales"].map('{:,.0f}'.format)
    output_df.loc[:, "Test Avg."] = '$'+ output_df["Test Avg."].map('{:.0f}'.format)
    output_df.loc[:, "Control Avg."] = '$'+ output_df["Control Avg."].map('{:.0f}'.format)
    output_df.loc[:, "Sales Lift"] = output_df["Sales Lift"].map('{:.2f}'.format) + '%'
    return output_df

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
    return temp

def __create_measurement_folder_v1(campaign_details):
    measurement_path = os.path.join(os.path.realpath(campaign_details['output_path']), campaign_details['campaign'],f"{campaign_details['client']}_{campaign_details['brand']}" ,'measurement')
    if not os.path.exists(measurement_path):
        print("Creating measurement folder, campaign measurement files will be generated at: %s" % (measurement_path))
        os.mkdir(measurement_path)
        print("Measurement folder created")
    return measurement_path

def get_ab_group_df(campaign_details):
    trg_folder_path = os.path.join(os.path.realpath(campaign_details['output_path']), campaign_details['campaign'], f"{campaign_details['client']}_{campaign_details['brand']}", 'ab_group', 'targetingPackage', campaign_details['store'])
    isExist = os.path.exists(trg_folder_path)
    if not isExist:
        raise Exception(print("Given campaign details are not correct or target package folder for %s is not available, further processed is aborted" %campaign_details['store']))
    test = os.path.join(trg_folder_path, 'test_df.csv')
    control = os.path.join(trg_folder_path, 'control_df.csv')
    df_test = pd.read_csv(test)
    df_control = pd.read_csv(control)
    return df_test, df_control

def get_campaign_sales(campaign_details, campaign_files):
    df_sales = []
    file_count = 0
    for file in campaign_files:
        #t = file.split('\\')[-1].split('_')[-1].split('.',1)[-1]
        t = Path(file).name.split('.', 1)[-1]
        if t == 'csv.bz2' or t == 'csv':
            df_sales.append(pd.read_csv(file))
            file_count+=1
    print("Total campaign sales files used for %s %s: %d" %(campaign_details['store'], campaign_details['task'], file_count))
    df_sales = pd.concat(df_sales)
    df_sales = df_sales[df_sales['sales_dollars'] != 0]
    return df_sales

def impose_inclusion(campaign_details, df_sales, file_type):
    store = campaign_details['store']
    counter = 0
    for col in df_sales.columns:
        for k, v in campaign_details['inclusion'][store].items():
            if k == col:
                counter+=1
                df_sales = df_sales[df_sales[col].isin(v)]
    if counter == 0:
        raise Exception(print("One or more given inclusion columns are not available in %s sales files" %file_type))
    return df_sales

def impose_exclusion(campaign_details, df_sales, file_type):
    store = campaign_details['store']
    counter = 0
    for col in df_sales.columns:
        for k, v in campaign_details['exclusion'][store].items():
            if k == col:
                counter+=1
                df_sales = df_sales[~df_sales[col].isin(v)]
    if counter == 0:
        raise Exception(print("One or more given exclusion columns are not available in %s sales files" %file_type))
    return df_sales
    
def format_measurement_file(measurment_df, xlsxwriter, lvl):
    if lvl is None:
        lvl = 'weekly'
    measurment_df.reset_index(drop=True).style.set_properties(**{'text-align': 'center'}).to_excel(xlsxwriter, sheet_name=f'{lvl}_sales_report', index = False)
    worksheet = xlsxwriter.sheets[f'{lvl}_sales_report']
    workbook = xlsxwriter.book
    header_cell_format = workbook.add_format()
    header_cell_format.set_align('center')
    col_names = [{'header': col_name} for col_name in measurment_df.columns]
    worksheet.add_table(0, 0, measurment_df.shape[0], measurment_df.shape[1]-1, {'columns': col_names, 'style': 'Table Style Medium 7'})
    for i, col in enumerate(col_names):
        try:
            worksheet.write(0, i, col['header'], header_cell_format)
        except:
            pass
    last_cell_format = workbook.add_format({'bold': True, 'font_color': 'white', 'bg_color': '#ff9933'})
    last_cell_format.set_align('center')
    for idx, col in enumerate(measurment_df.iloc[-1]):
        try:
            worksheet.write(measurment_df.shape[0], idx, col, last_cell_format)
        except:
            worksheet.write(measurment_df.shape[0], idx, None, last_cell_format)
    return xlsxwriter
    
def measurement_file_preprocessing_v3(campaign_details, campaign_files, list_of_metrics):
    df_sales = get_campaign_sales(campaign_details, campaign_files)
    if 'inclusion' in campaign_details:
        if campaign_details['store'] in campaign_details['inclusion']:
            df_sales = impose_inclusion(campaign_details, df_sales, 'campaign')
    if 'exclusion' in campaign_details:
        if campaign_details['store'] in campaign_details['exclusion']:
            df_sales = impose_exclusion(campaign_details, df_sales, 'campaign')
    df_test, df_control = get_ab_group_df(campaign_details)
    df_sales = get_group_col(df_sales, df_test, df_control)
    df_sales = get_non_reported_sales_store(df_sales, df_test, df_control , False)
    measurement_path = __create_measurement_folder_v1(campaign_details)
    measurement_file = os.path.join(measurement_path,f"{campaign_details['client']}_{campaign_details['brand']}_{campaign_details['store']}_measurement.xlsx")
    with pd.ExcelWriter(measurement_file, engine='xlsxwriter') as xlsxwriter:
        for lvl in list_of_metrics:
            temp = get_measurement_metric(df_sales, lvl)
            if lvl is None:
                xlsxwriter = format_measurement_file(temp, xlsxwriter, lvl)
            else:
                xlsxwriter = format_measurement_file(temp, xlsxwriter, lvl)
        temp = get_nonreported_store(df_sales, df_test, df_control)
        temp['week'] = temp['week'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%d-%b-%y'))
        temp.to_excel(xlsxwriter, sheet_name=f'nonreported_stores', index = False)
        xlsxwriter.save()
    print("Measurement file for %s has been generated at: %s" %(campaign_details['store'], measurement_path))
    print("Measurement file generation process for %s has been completed !!" %campaign_details['store'])
    
def generate_measurement_file_v3(campaign_details, campaign_files, list_of_metrics=[None]):
    measurement_file_preprocessing_v3(campaign_details, campaign_files, list_of_metrics)  
    
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