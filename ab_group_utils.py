import sys
sys.path.extend(['D:\work\project\cac\z3_explorations', 'D:\work\project\cac\sales_measurment_service'])
import pandas as pd
import numpy as np
import os, copy, glob, shutil, json, subprocess
from datetime import datetime,timedelta
import re, time, logging
from pathlib import Path
from utils import file_utils
from utils import optimization_utils as opt
from pathlib import Path
from utils import dataframe_utils
from warnings import warn
import MySQLdb, mysql.connector, pymysql, random
import run_splitter as rs


def setup_db_connection(con='server'):
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', "db_credentials.csv")
    cred = pd.read_csv(file_path)
    temp = cred[cred['con'] == con]
    hostname = temp['hostname'].values[0]
    username = temp['username'].values[0]
    password = temp['password'].values[0]
    database = temp['database'].values[0]
    targetConnect = MySQLdb.connect(host=hostname, user=username, passwd=password, db=database)
    return targetConnect

def load_zipcode_data_to_db(zipcode_master_file_path, zipcode_not_validated_file_path):
    upload_zipcode_master_to_db(zipcode_master_file_path)
    upload_zipcode_validated_to_db(zipcode_not_validated_file_path)
    
def format_store_file(df_store):
    check_na(df_store)
    store_dict = {'store_id': str, 'store_address': str, 'store_banner': str, 'latitude': float, 'longitude': float, 'url': str, 'zipcode': int}
    for col in df_store.columns:
        if col not in store_dict:
            raise Exception(print("provided store sheet does not have standard columns, file should have only these columns 'store_id', 'store_address', 'store_banner', 'latitude', 'longitude', 'url', 'zipcode'. Further process is aborted"))
    for store, dtype in store_dict.items():
        df_store[store] = df_store[store].astype(dtype)
    return df_store


def check_na(df):
    for col in df.columns:
        if not df[df[col].isna()].empty:
            raise Exception(print("%s column has null or na values, please remove it and try again" %col))
            
def remove_zipcode(df_store):
    import re
    df_store['store_address'] = df_store['store_address'].apply(lambda x: re.sub(r' \d+$', '',  x))
    return df_store

def format_zip_not_validated_file(df_zip_not_val):
    check_na(df_zip_not_val)
    zip_not_val_dict = {'zipcode': int, 'validated': str}
    for col in df_zip_not_val.columns:
        if col not in zip_not_val_dict:
            raise Exception(print("provided zipcode not validated sheet does not have standard columns, file should have only these columns 'zipcode', 'validated'. Further process is aborted"))
    for col, dtype in zip_not_val_dict.items():
        df_zip_not_val[col] = df_zip_not_val[col].astype(dtype)
    return df_zip_not_val

def format_zip_master_file(df_zip):
    check_na(df_zip)
    zip_master_dict = {'zipcode': int, 'country': str, 'latitude': float, 'longitude': float}
    for col in df_zip.columns:
        if col not in zip_master_dict:
            raise Exception(print("provided zipcode master sheet does not have standard columns, file should have only these columns 'zipcode', 'country', 'latitude', 'longitude'. Further process is aborted"))
    for col, dtype in zip_master_dict.items():
        df_zip[col] = df_zip[col].astype(dtype)
    return df_zip
    
def upload_store_to_db(store_file_path):
    dbConnect = setup_db_connection()
    cursor = dbConnect.cursor()
    store_data = pd.read_csv(store_file_path)
    store_data = format_store_file(store_data)
    n_rows = store_data.shape[0]
    store = store_data['store_banner'].unique()[0]
    for i,row in store_data.iterrows():
            sql = "INSERT INTO target.all_stores VALUES (%s,%s,%s,%s,%s,%s,%s)"
            cursor.execute(sql, tuple(row))
            # the connection is not auto committed by default, so we must commit to save our changes
    # close the connection even if error occurs, else db will be involved and it will go in dead lock and DML statement won't be executed
    dbConnect.commit()
    cursor.close()
    print ("%d rows has been loaded for %s in target db in 'all_stores' table" %(n_rows, store))

def upload_zipcode_master_to_db(zipcode_master_file_path):
    dbConnect = setup_db_connection()
    cursor = dbConnect.cursor()
    zipcode_master = pd.read_csv(zipcode_master_file_path)
    zipcode_master = format_zip_master_file(zipcode_master)
    n_rows = zipcode_master.shape[0]
    for i,row in zipcode_master.iterrows():
            sql = "INSERT INTO target.zipcode_enterprise_master VALUES (%s,%s,%s,%s)"
            cursor.execute(sql, tuple(row))
    dbConnect.commit()
    cursor.close()
    print ("%d rows of zipcode master data has been loaded in target db in 'zipcode_enterprise_master' table" %n_rows)

def upload_zipcode_validated_to_db(zipcode_validated_file_path):
    dbConnect = setup_db_connection()
    cursor = dbConnect.cursor()
    zipcode_validated = pd.read_csv(zipcode_validated_file_path)
    zipcode_validated = format_zip_not_validated_file(zipcode_validated)
    n_rows = zipcode_validated.shape[0]
    for i,row in zipcode_validated.iterrows():
            sql = "INSERT INTO target.zipcode_not_validated VALUES (%s,%s)"
            cursor.execute(sql, tuple(row))
    dbConnect.commit()
    cursor.close()
    print ("%d rows of zipcode validated data has been loaded in target db in 'zipcode_not_validated' table" %n_rows)

def store_banner_check(df_store):
    store_banner = df_store['store_banner'].str.lower().unique()[0]
    df_store['store_address'] = df_store['store_address'].str.lower()
    temp = df_store[df_store['store_address'].str.contains(store_banner, regex=False)]
    if not temp.empty:
        raise Exception(print("store file has store banner as prefix in store address, please remove it and try again"))
    return df_store


def create_exp_view(store):
    dbConnect = setup_db_connection()
    cursor = dbConnect.cursor()
    store_banner =  f'"{store}"'
    sql = f"CREATE OR REPLACE VIEW target.zipcode_expansion_{store} AS\
    SELECT store_id, url, temp3.zipcode AS zipcode, zipcode_expanded, is_original_zipcode, distance, radius, CASE WHEN temp3.zipcode_expanded = not_z.zipcode then FALSE ELSE TRUE END AS validated\
    FROM\
    (SELECT store_id, url, zipcode, zipcode_expanded, is_original_zipcode, distance, radius\
    FROM\
    (SELECT store_id, url, zipcode, zipcode_expanded, is_original_zipcode,\
    (69.0*DEGREES(ACOS(LEAST(1.0, COS(RADIANS(temp1.store_latitude))*COS(RADIANS(temp1.zipcode_latitude))*COS(RADIANS(temp1.store_longitude-temp1.zipcode_longitude))+SIN(RADIANS(temp1.store_latitude))*SIN(RADIANS(temp1.zipcode_latitude)))))) AS distance,\
    CEILING(69.0*DEGREES(ACOS(LEAST(1.0, COS(RADIANS(temp1.store_latitude))*COS(RADIANS(temp1.zipcode_latitude))*COS(RADIANS(temp1.store_longitude-temp1.zipcode_longitude))+SIN(RADIANS(temp1.store_latitude))*SIN(RADIANS(temp1.zipcode_latitude)))))) AS radius\
    FROM\
    (SELECT s.store_id,\
    s.url,\
    s.zipcode AS zipcode,\
    s.latitude AS store_latitude,\
    s.longitude AS store_longitude,\
    z.zipcode AS zipcode_expanded,\
    z.latitude AS zipcode_latitude,\
    z.longitude As zipcode_longitude,\
    CASE WHEN s.zipcode = z.zipcode then TRUE ELSE FALSE END AS is_original_zipcode\
    FROM (SELECT store_id, latitude, longitude, url, zipcode from target.all_stores WHERE store_banner = {store_banner}) AS s CROSS JOIN target.zipcode_enterprise_master AS z) AS temp1) AS temp2\
    WHERE (radius <=5 OR is_original_zipcode = TRUE)) AS temp3 LEFT JOIN target.zipcode_not_validated AS not_z ON temp3.zipcode_expanded = not_z.zipcode;"
    cursor.execute(sql)
    cursor.close()
    print("view has been created for %s" %store)
    

def create_store_view(store):
    dbConnect = setup_db_connection()
    cursor = dbConnect.cursor()
    store_banner =  f'"{store}"'
    sql = f"CREATE OR REPLACE VIEW target.store_exp_zipcode AS\
    SELECT s.store_id,\
    s.url,\
    s.zipcode AS zipcode,\
    s.latitude AS store_latitude,\
    s.longitude AS store_longitude,\
    z.zipcode AS zipcode_expanded,\
    z.latitude AS zipcode_latitude,\
    z.longitude As zipcode_longitude,\
    CASE WHEN s.zipcode = z.zipcode then TRUE ELSE FALSE END AS is_original_zipcode\
    FROM  target.all_stores AS s\
    CROSS JOIN target.zipcode_enterprise_master AS z\
    WHERE s.store_banner = {store_banner};"
    cursor.execute(sql)
    cursor.close()
    print("view has been created for %s" %store)
    
# def get_store_expansion(store):
#     start_time = datetime.now()
#     dbConnect = setup_db_connection()
#     print("Target db is connected, trying to fetch %s store expansion data, please wait it will take a while" %store)
#     df_exp = pd.read_sql(f'SELECT * FROM target.zipcode_expansion_{store} LIMIT 90000;', con=dbConnect)
#     finish_time = datetime.now()
#     duration_sec = (finish_time - start_time).total_seconds()
#     print("Data has been retrieved! (at=%s, total=%f sec)" % (str(finish_time), duration_sec))
#     dbConnect.close()
#     df_exp = __format_expansion_df(df_exp)
#     return df_exp

def get_store_expansion(store):
    start_time = datetime.now()
    dbConnect = setup_db_connection()
    print("Target db is connected, trying to fetch %s store expansion data, please wait it will take a while" %store)
    create_store_view(store)
    df_exp = pd.read_sql(f'SELECT * FROM target.store_zipcode_expansion LIMIT 90000;', con=dbConnect)
    finish_time = datetime.now()
    duration_sec = (finish_time - start_time).total_seconds()
    print("Data has been retrieved! (at=%s, total=%f sec)" % (str(finish_time), duration_sec))
    dbConnect.close()
    df_exp = __format_expansion_df(df_exp)
    return df_exp

def users_connected():
    dbConnect = setup_db_connection()
    t = pd.read_sql("SELECT SUBSTRING_INDEX(host, ':', 1) AS host_short, GROUP_CONCAT(DISTINCT user) AS users, COUNT(*) AS threads FROM information_schema.processlist GROUP BY host_short ORDER BY COUNT(*), host_short;", con=dbConnect)
    return t

def get_merge_df(df_exp, avg_sales):
    t = pd.merge(df_exp, avg_sales, how = 'inner', on = 'store_id')
    return t

def __create_ab_folder_v1(campaign_details):
    ab_group_path = os.path.join(os.path.realpath(campaign_details['output_path']), campaign_details['campaign'],f"{campaign_details['client']}_{campaign_details['brand']}" ,'ab_group')
    if not os.path.exists(ab_group_path):
        print("Creating ab group folder, campaign test/control splits will be generated at: %s" % (ab_group_path))
        os.makedirs(ab_group_path)
        print("ab group folder created")
    return ab_group_path
    
def __format_expansion_df(df_exp_db):
    df_exp_db['radius'] = df_exp_db['radius'].astype(int)
    df_exp_db['zipcode'] = df_exp_db['zipcode'].astype(str)
    df_exp_db['zipcode_expanded'] = df_exp_db['zipcode_expanded'].astype(str)
    df_exp_db['is_original_zipcode'] = df_exp_db['is_original_zipcode'].astype(bool)
    df_exp_db['validated'] = df_exp_db['validated'].astype(bool)
    df_exp_db = dataframe_utils.StandardizeData.standardize_dataframe(df_exp_db)
    return df_exp_db

def __create_trg_folder(ab_group_path, store):
    trg_path = os.path.join(ab_group_path ,'targetingPackage', store)
    #if not os.path.exists(trg_path):
    print("Creating targeting package folder, %s test/control splits will be generated at: %s" % (store, trg_path))
    if os.path.exists(trg_path):
        shutil.rmtree(trg_path)
    os.makedirs(trg_path)
    print("targeting package folder is created for %s" %store)
    return trg_path

def get_random_stores(df, n_stores):
    np.random.seed(123)
    random_store = np.random.choice(df['store_id'].unique(), size=n_stores, replace=False)
    df = df[df['store_id'].isin(random_store)]
    return df

def format_list(temp):
    temp = [f'"{c}" ' for c in temp]
    #temp = [f"{c} " for c in temp]
    return temp

def run_parallel(ab_group_path, campaign_details_json, z3_input_files, mode, cluster, split, groups, avg_tol, size_tol):
    if mode == 'multi':
        store = ' , '.join([rs.get_retailer(file) for file in z3_input_files])
    elif mode == 'single':
        store = campaign_details['store']
    print("executing parallel jobs for %s" %store)
    #ab_group_path = r'D:\work\project\cac\campaign\campaign_automation\new_format\mtwdew_p3\pepsico_mtwdew\ab_group'
    file_path = os.path.join(os.path.realpath(ab_group_path), "z3_parallel.sh")
    file = open(file_path, 'w')
    file.write("#!/bin/bash\n")
    par = "parallel -u " 
    run_splitter = "python /data/users/shubhamg/z3_explorations/run_splitter.py"
    dem = " ::: "
    camp = campaign_details_json
    #z3_input_files = [["D:\\work\\project\\cac\\campaign\\campaign_automation\\new_format\\mtwdew_p3\\pepsico_mtwdew\\ab_group\\7eleven_z3_input.json"]]
    #z3_input_files = [["D:\\work\\project\\cac\\campaign\\campaign_automation\\new_format\\mtwdew_p3\\pepsico_mtwdew\\ab_group\\speedway_z3_input.json", "D:\\work\\project\\cac\\campaign\\campaign_automation\\new_format\\mtwdew_p3\\pepsico_mtwdew\\ab_group\\7eleven_z3_input.json"]]
    #mode = "single"
    #cluster = "True"
    #split = [[{"control": 0.2, "test": 0.8}, {"control": 0.3, "test": 0.7}]]
    #groups = [10, 20]
    #avg_tol = [[5, 6], [10, 11]]
    #size_tol = [[100, 150], [200, 250]]
    #t = [[100, 150], [200, 250], [300, 350]]
    file.writelines([par, run_splitter, dem, camp, dem])
    #file.writelines(format_list(z3_input_files))
    file.writelines(format_list([z3_input_files]))
    file.writelines([dem, mode, dem, cluster, dem])
    #file.writelines(format_list(split))
    file.writelines(format_list([split]))
    file.writelines(dem)
    file.writelines(format_list(groups))
    #file1.writelines(format_list(groups))
    file.writelines(dem)
    file.writelines(format_list(avg_tol))
    file.writelines(dem)
    file.writelines(format_list(size_tol))
    file.close()
    os.system("attrib +h " + file_path)
    os.chmod(Path(file_path), 0o775)
    subprocess.call(['sh', f'./{file_path}'])
    os.remove(file_path)
    print("%s parallel jobs has been completed!!" %store)

# def ab_group_preprocessing_v2(campaign_details):
#     ab_group_path = __create_ab_folder_v1(campaign_details)
#     z3_input_files = []
#     for idx, (store, args) in enumerate(campaign_details['z3_param']['retailers'].items()):
#         campaign_details['store'] = store
#         df_exp_db = get_store_expansion(store)
#         historical_files = args['input_files']['historical_files']
#         df_hist_sales = opt.get_historical_sales(campaign_details, historical_files)
#         df_hist_sales = get_random_stores(df_hist_sales, 100)
#         store_sales_avg = opt.get_store_avg_sales_v3(df_hist_sales)
#         merged_exp_store_avg = get_merge_df(df_exp_db, store_sales_avg)
        
#         if (campaign_details['z3_param']['mode'] == 'single') and not campaign_details['z3_param']['parallelization']:
#             if args['split_folder'] is None:
#                 z3_input_files = []
#                 z3_input_file = os.path.join(ab_group_path, f"{store}_z3_input.json")
#                 file_utils.save_json_format_for_z3(merged_exp_store_avg, z3_input_file)
#                 z3_input_files.append(z3_input_file)
                
#                 mode = campaign_details['z3_param']['mode']
#                 cluster = campaign_details['z3_param']['cluster']
#                 split = campaign_details['z3_param']['split'][idx]
#                 groups = campaign_details['z3_param']['groups'][idx][0]
#                 avg_tol = campaign_details['z3_param']['avg_tol'][idx][0]
#                 size_tol = campaign_details['z3_param']['size_tol'][idx][0]
#                 rs.run_splitter(campaign_details, z3_input_files, mode, cluster, split, groups, avg_tol, size_tol)
            
#             else:
#                 output_folder = __create_trg_folder(ab_group_path, store)
                
#                 if campaign_details['z3_param']['cluster']:
#                     temp_path = os.path.join(ab_group_path, 'z3_output', 'single_retailer_clustered', f"{args['split_folder']}")
#                     files = glob.glob(os.path.join(temp_path,'*'), recursive = False)
#                     for file in files:
#                         #t = file.split('\\')[-1].split('.')[-1]
#                         t = Path(file).name.split('.')[-1]
#                         if t == 'json':        
#                             #if store == file.split('\\')[-1].split('_')[0]:
#                             if store == Path(file).name.split('_')[0]:
#                                 z3_output_path = file
#                 else:
#                     z3_output_path = os.path.join(ab_group_path, 'z3_output', 'single_retailer_non_clustered', f"{args['split_folder']}", 'ab_split.xlsx')
                
                            
#                 file_utils.convert_save_z3_results_to_platform_format(z3_output_path, df_exp_db, store_sales_avg, output_folder, is_clustered_result = campaign_details['z3_param']['cluster'])
            
#         if (campaign_details['z3_param']['mode'] == 'multi') and not campaign_details['z3_param']['parallelization']:
#             if args['split_folder'] is None:
#                 z3_input_file = os.path.join(ab_group_path, f"{store}_z3_input.json")
#                 file_utils.save_json_format_for_z3(merged_exp_store_avg, z3_input_file)
#                 z3_input_files.append(z3_input_file)
                
#             else:
#                 output_folder = __create_trg_folder(ab_group_path, store)
                
#                 if campaign_details['z3_param']['cluster']:
#                     temp_path = os.path.join(ab_group_path, 'z3_output', 'multi_retailer_clustered', f"{args['split_folder']}")
#                     files = glob.glob(os.path.join(temp_path,'*'), recursive = False)
#                     for file in files:
#                         #t = file.split('\\')[-1].split('.')[-1]
#                         t = Path(file).name.split('.')[-1]
#                         if t == 'json':        
#                             #if store == file.split('\\')[-1].split('_')[0]:
#                             if store == Path(file).name.split('_')[0]:
#                                 z3_output_path = file
#                 else:
#                     z3_output_path = os.path.join(ab_group_path, 'z3_output', 'multi_retailer_non_clustered', f"{args['split_folder']}", 'ab_split.xlsx')
                
#                 file_utils.convert_save_z3_results_to_platform_format(z3_output_path, df_exp_db, store_sales_avg, output_folder, is_clustered_result = campaign_details['z3_param']['cluster'])
    
#     if (campaign_details['z3_param']['mode'] == 'multi') and not campaign_details['z3_param']['parallelization']:
#         if args['split_folder'] is None:
#             mode = campaign_details['z3_param']['mode']
#             cluster = campaign_details['z3_param']['cluster']
#             split = campaign_details['z3_param']['split']
#             groups = campaign_details['z3_param']['groups'][0]
#             avg_tol = campaign_details['z3_param']['avg_tol'][0]
#             size_tol = campaign_details['z3_param']['size_tol'][0]
#             rs.run_splitter(campaign_details, z3_input_files, mode, cluster, split, groups, avg_tol, size_tol)
    
#     print("ab group creation process has been completed !")
def generate_temp_json(ab_group_path, campaign_details):
    file_path = os.path.join(os.path.realpath(ab_group_path), "temp.json")
    os.system("attrib -h " + file_path)
    with open(file_path, "w") as outfile:
        json.dump(campaign_details, outfile)
    os.system("attrib +h " + file_path)
    return file_path


def ab_group_preprocessing_v2(campaign_details):
    ab_group_path = __create_ab_folder_v1(campaign_details)
    z3_input_files = []
    for idx, (store, args) in enumerate(campaign_details['z3_param']['retailers'].items()):
        campaign_details['store'] = store
        df_exp_db = get_store_expansion(store)
        historical_files = args['input_files']['historical_files']
        df_hist_sales = opt.get_historical_sales(campaign_details, historical_files)
        df_hist_sales = get_random_stores(df_hist_sales, 100)
        store_sales_avg = opt.get_store_avg_sales_v3(df_hist_sales)
        merged_exp_store_avg = get_merge_df(df_exp_db, store_sales_avg)
        
        if (campaign_details['z3_param']['mode'] == 'single') and not campaign_details['z3_param']['parallelization']:
            if args['split_folder'] is None:
                z3_input_files = []
                z3_input_file = os.path.join(ab_group_path, f"{store}_z3_input.json")
                file_utils.save_json_format_for_z3(merged_exp_store_avg, z3_input_file)
                z3_input_files.append(z3_input_file)
                
                mode = campaign_details['z3_param']['mode']
                cluster = campaign_details['z3_param']['cluster']
                split = campaign_details['z3_param']['split'][idx]
                groups = campaign_details['z3_param']['groups'][idx][0]
                avg_tol = campaign_details['z3_param']['avg_tol'][idx][0]
                size_tol = campaign_details['z3_param']['size_tol'][idx][0]
                campaign_details_json = generate_temp_json(ab_group_path, campaign_details)
                print("campaign_details", campaign_details)
                print("mode", mode)
                print("cluster", cluster)
                print("split", split)
                print("groups", groups)
                print("avg_tol", avg_tol)
                print("size_tol", size_tol)
                rs.run_splitter(campaign_details_json, z3_input_files, mode, cluster, split, groups, avg_tol, size_tol)
            
            else:
                output_folder = __create_trg_folder(ab_group_path, store)
                
                if campaign_details['z3_param']['cluster']:
                    temp_path = os.path.join(ab_group_path, 'z3_output', 'single_retailer_clustered', f"{args['split_folder']}")
                    files = glob.glob(os.path.join(temp_path,'*'), recursive = False)
                    for file in files:
                        #t = file.split('\\')[-1].split('.')[-1]
                        t = Path(file).name.split('.')[-1]
                        if t == 'json':        
                            #if store == file.split('\\')[-1].split('_')[0]:
                            if store == Path(file).name.split('_')[0]:
                                z3_output_path = file
                else:
                    z3_output_path = os.path.join(ab_group_path, 'z3_output', 'single_retailer_non_clustered', f"{args['split_folder']}", 'ab_split.xlsx')
                
                            
                file_utils.convert_save_z3_results_to_platform_format(z3_output_path, df_exp_db, store_sales_avg, output_folder, is_clustered_result = campaign_details['z3_param']['cluster'])
            
        if (campaign_details['z3_param']['mode'] == 'multi') and not campaign_details['z3_param']['parallelization']:
            if args['split_folder'] is None:
                z3_input_file = os.path.join(ab_group_path, f"{store}_z3_input.json")
                file_utils.save_json_format_for_z3(merged_exp_store_avg, z3_input_file)
                z3_input_files.append(z3_input_file)
                
            else:
                output_folder = __create_trg_folder(ab_group_path, store)
                
                if campaign_details['z3_param']['cluster']:
                    temp_path = os.path.join(ab_group_path, 'z3_output', 'multi_retailer_clustered', f"{args['split_folder']}")
                    files = glob.glob(os.path.join(temp_path,'*'), recursive = False)
                    for file in files:
                        #t = file.split('\\')[-1].split('.')[-1]
                        t = Path(file).name.split('.')[-1]
                        if t == 'json':        
                            #if store == file.split('\\')[-1].split('_')[0]:
                            if store == Path(file).name.split('_')[0]:
                                z3_output_path = file
                else:
                    z3_output_path = os.path.join(ab_group_path, 'z3_output', 'multi_retailer_non_clustered', f"{args['split_folder']}", 'ab_split.xlsx')
                
                file_utils.convert_save_z3_results_to_platform_format(z3_output_path, df_exp_db, store_sales_avg, output_folder, is_clustered_result = campaign_details['z3_param']['cluster'])
    
    if (campaign_details['z3_param']['mode'] == 'multi') and not campaign_details['z3_param']['parallelization']:
        if args['split_folder'] is None:
            mode = campaign_details['z3_param']['mode']
            cluster = campaign_details['z3_param']['cluster']
            split = campaign_details['z3_param']['split']
            groups = campaign_details['z3_param']['groups'][0]
            avg_tol = campaign_details['z3_param']['avg_tol'][0]
            size_tol = campaign_details['z3_param']['size_tol'][0]
            campaign_details_json = generate_temp_json(ab_group_path, campaign_details)
            rs.run_splitter(campaign_details_json, z3_input_files, mode, cluster, split, groups, avg_tol, size_tol)
    os.remove(campaign_details_json)
    print("ab group creation process has been completed !")



def ab_group_preprocessing_parallel(campaign_details):
    ab_group_path = __create_ab_folder_v1(campaign_details)
    z3_input_files = []
    for idx, (store, args) in enumerate(campaign_details['z3_param']['retailers'].items()):
        campaign_details['store'] = store
        df_exp_db = get_store_expansion(store)
        historical_files = args['input_files']['historical_files']
        df_hist_sales = opt.get_historical_sales(campaign_details, historical_files)
        df_hist_sales = get_random_stores(df_hist_sales, 100)
        store_sales_avg = opt.get_store_avg_sales_v3(df_hist_sales)
        merged_exp_store_avg = get_merge_df(df_exp_db, store_sales_avg)
        
        if (campaign_details['z3_param']['mode'] == 'single'):
            if args['split_folder'] is None:
                z3_input_files = []
                z3_input_file = os.path.join(ab_group_path, f"{store}_z3_input.json")
                file_utils.save_json_format_for_z3(merged_exp_store_avg, z3_input_file)
                z3_input_files.append(z3_input_file)
                mode = campaign_details['z3_param']['mode']
                cluster = campaign_details['z3_param']['cluster']
                split = campaign_details['z3_param']['split'][idx]
                                                                                                   
                if not campaign_details['z3_param']['parallelization']:
                    groups = campaign_details['z3_param']['groups'][idx][0]
                    avg_tol = campaign_details['z3_param']['avg_tol'][idx][0]
                    size_tol = campaign_details['z3_param']['size_tol'][idx][0]
                    rs.run_splitter(campaign_details, z3_input_files, mode, cluster, split, groups, avg_tol, size_tol)
                elif campaign_details['z3_param']['parallelization']:
                    groups = campaign_details['z3_param']['groups'][idx]
                    avg_tol = campaign_details['z3_param']['avg_tol'][idx]
                    size_tol = campaign_details['z3_param']['size_tol'][idx]
                    campaign_details_json = generate_temp_json(ab_group_path, campaign_details)
                    run_parallel(ab_group_path, campaign_details_json, z3_input_files, mode, cluster, split, groups, avg_tol, size_tol)
                    
            else:
                output_folder = __create_trg_folder(ab_group_path, store)
                
                if campaign_details['z3_param']['cluster']:
                    temp_path = os.path.join(ab_group_path, 'z3_output', 'single_retailer_clustered', f"{args['split_folder']}")
                    files = glob.glob(os.path.join(temp_path,'*'), recursive = False)
                    for file in files:
                        #t = file.split('\\')[-1].split('.')[-1]
                        t = Path(file).name.split('.')[-1]
                        if t == 'json':        
                            #if store == file.split('\\')[-1].split('_')[0]:
                            if store == Path(file).name.split('_')[0]:
                                z3_output_path = file
                else:
                    z3_output_path = os.path.join(ab_group_path, 'z3_output', 'single_retailer_non_clustered', f"{args['split_folder']}", 'ab_split.xlsx')
                
                            
                file_utils.convert_save_z3_results_to_platform_format(z3_output_path, df_exp_db, store_sales_avg, output_folder, is_clustered_result = campaign_details['z3_param']['cluster'])
            
        if (campaign_details['z3_param']['mode'] == 'multi'):
            if args['split_folder'] is None:
                z3_input_file = os.path.join(ab_group_path, f"{store}_z3_input.json")
                file_utils.save_json_format_for_z3(merged_exp_store_avg, z3_input_file)
                z3_input_files.append(z3_input_file)
                
            else:
                output_folder = __create_trg_folder(ab_group_path, store)
                
                if campaign_details['z3_param']['cluster']:
                    temp_path = os.path.join(ab_group_path, 'z3_output', 'multi_retailer_clustered', f"{args['split_folder']}")
                    files = glob.glob(os.path.join(temp_path,'*'), recursive = False)
                    for file in files:
                        #t = file.split('\\')[-1].split('.')[-1]
                        t = Path(file).name.split('.')[-1]
                        if t == 'json':        
                            #if store == file.split('\\')[-1].split('_')[0]:
                            if store == Path(file).name.split('_')[0]:
                                z3_output_path = file
                else:
                    z3_output_path = os.path.join(ab_group_path, 'z3_output', 'multi_retailer_non_clustered', f"{args['split_folder']}", 'ab_split.xlsx')
                
                file_utils.convert_save_z3_results_to_platform_format(z3_output_path, df_exp_db, store_sales_avg, output_folder, is_clustered_result = campaign_details['z3_param']['cluster'])
    
    if (campaign_details['z3_param']['mode'] == 'multi'):
        if args['split_folder'] is None:
            mode = campaign_details['z3_param']['mode']
            cluster = campaign_details['z3_param']['cluster']
            split = campaign_details['z3_param']['split']
            if not campaign_details['z3_param']['parallelization']:
                groups = campaign_details['z3_param']['groups'][0]
                avg_tol = campaign_details['z3_param']['avg_tol'][0]
                size_tol = campaign_details['z3_param']['size_tol'][0]
                rs.run_splitter(campaign_details, z3_input_files, mode, cluster, split, groups, avg_tol, size_tol)
            elif campaign_details['z3_param']['parallelization']:
                groups = campaign_details['z3_param']['groups']
                avg_tol = campaign_details['z3_param']['avg_tol']
                size_tol = campaign_details['z3_param']['size_tol']
                campaign_details_json = generate_temp_json(ab_group_path, campaign_details)
                run_parallel(ab_group_path, campaign_details_json, z3_input_files, mode, cluster, split, groups, avg_tol, size_tol)
    os.remove(campaign_details_json)
    print("ab group creation process has been completed !")
    
def generate_ab_group(campaign_details):
    ab_group_preprocessing_v2(campaign_details)
    
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