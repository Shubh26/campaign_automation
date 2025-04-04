#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os, sys, glob
from pathlib import Path
import filecmp

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import plotly.express as px
import plotly.graph_objects as go

import shutil
import logging
import requests
import re
import json

import tempfile
import zipfile
# import magic

# TODO keep only one version of constants
import utils.stats_helper
from utils import date_utils, zipcode_utils
from utils.constants import *
from utils import dataframe_utils

# resources folder is in the same level as utils folder
# https://stackoverflow.com/questions/1270951/how-to-refer-to-relative-paths-of-resources-when-working-with-a-code-repository
main_package_folder = Path(__file__).parent.parent
main_resources_folder = os.path.join(main_package_folder, 'resources', 'main')
custom_file_configs_folder = os.path.join(main_resources_folder,"custom_file_configs")
folder_config_path = os.path.join(main_resources_folder,"app_config.json")
app_config = {}
with open(folder_config_path, "r") as f:
    app_config = json.load(f)

def get_app_config():
    return app_config

main_data_folder = get_app_config().get("main_data_folder", "/data/cac/sales_data")

def get_main_resources_folder():
    return main_resources_folder


def get_temp_folder():
    return os.path.join(main_data_folder,'tmp')

def get_raw_data_folder(main_data_folder=main_data_folder):
    return os.path.join(main_data_folder,'raw')

def get_processed_data_folder(main_data_folder=main_data_folder):
    return os.path.join(main_data_folder,'processed')

def get_received_data_folder(main_data_folder=main_data_folder):
    return os.path.join(main_data_folder,'received')

def get_plots_folder(main_data_folder=main_data_folder):
    return os.path.join(main_data_folder,'plots')

def get_stores_data_folder(main_data_folder=main_data_folder, retail_chain=None):
    stores_folder = os.path.join(main_data_folder,'stores')
    if retail_chain:
        stores_folder = os.path.join(stores_folder, retail_chain)
    return stores_folder



raw_data_folder = get_raw_data_folder()
processed_data_folder = get_processed_data_folder()
received_data_folder = get_received_data_folder()
plots_folder = get_plots_folder()
stores_folder = get_stores_data_folder()


STORE_ADDRESS_CLEANED_COL = STORE_ADDRESS_COL +'_cleaned'

def get_campaign_folder(main_data_folder=main_data_folder, client_name='pepsico',
                        brand_name='pureleaf', retail_chain='circlek', campaign_start_date='2021-05-25'):
    """
    This function returns the campaign folder path & creates the directory if it does not exist

    Arguments:
        main_data_folder:string
            Main data folder for TARGET team in the server
            default:'/data/cac/sales_data',  specified in file_utils.main_data_folder
            Eg:- '/data/cac/sales_data' in dsgpool machines
        client_name:string
            client's name - eg:- pepsico, smithfield
        brand_name:string
            brand name Eg:- bubly, pureleaf, eckrich, nathans, pure_farmland, kretschmar
        retail_chain:string
            retain_chain(s) involved in this campaign. If multiple entries separate by comma (,)
            Eg:- circlek
                walmart
                circklek,walmart
        campaign_start_date:string
            date specified in 'YYYY-MM-DD' format. Eg:- 2021-05-25

    """
    campaign_start_date_str = campaign_start_date
    campaign_start_date = date_utils.get_date(campaign_start_date,date_utils.DATE_FORMAT_ISO)

    year_date = campaign_start_date.strftime("%Y/%m")
    retail_chain_str = '_'.join(sorted(retail_chain.split(',')))
    campaign_name = f'{brand_name}_{retail_chain_str}_{campaign_start_date_str}'
    campaign_folder = Path(os.path.join(main_data_folder,'campaigns',client_name, brand_name,year_date,
                                  campaign_name))
    print(f"campaign_folder {campaign_folder}")
    # creating campaign folder
    # for mode value refer - https://stackoverflow.com/questions/1627198/python-mkdir-giving-me-wrong-permissions
    campaign_folder.mkdir(mode=0o777, parents=True,exist_ok=True)
    return campaign_folder


def __get_ordered_values_for_file_path(client, brand, retail_chain, start_date=None, end_date=None, sep="_"):
    start_date_str = date_utils.get_date_string(start_date, date_utils.DATE_FORMAT_ISO) if date_utils.is_date(
        start_date) else start_date
    end_date_str = date_utils.get_date_string(end_date, date_utils.DATE_FORMAT_ISO) if date_utils.is_date(
        end_date) else end_date
    order_of_values = [client, brand, retail_chain, start_date_str, end_date_str]
    # filtering out only cases which are not None & converting to lower case
    order_of_values = __filter_standardize_values_for_file_path(order_of_values)
    # expanding cases which have comma separated values
    order_of_values = [sep.join(sorted(v.split(","))) for v in order_of_values]
    return order_of_values

def __filter_standardize_values_for_file_path(order_of_values):
    return [v.lower() for v in order_of_values if (v is not None and len(v) > 0)]

def get_filename(filename_prefix, client, brand, retail_chain,
                 start_date, end_date, filename_suffix=None, filename_extension="csv"):
    """
    Get a filename based on client, brand_name, retail_chain & start, end dates

    Arguments:
        filename_prefix:str
            prefix string for the filename, Eg:-sales or "store_level_sales"
        client:str
            client_name. Eg:smithfield, pepsico etc
        brand:str
            brand name.Eg:eckrich, nathans etc
        retail_chain:str
            retail chain. Eg: kroger, jewel etc
        start_date:datatime.datetime
            start_date for this file, should be datetime.datetime object
        end_date:datetime.datetime
            end_date for this file, should be a datetime.datetime object
        filename_suffix:str
            a suffix to keep for the file.
            Eg:- "raw"
        filename_extension:str
            extension for the file. Eg:- "csv" or "csv.bz2" or "xlsx" etc
    """
    sep = "_"
    if filename_extension is not None and len(filename_extension.strip())!=0:
        filename_extension = f".{filename_extension}"
    order_of_values = [filename_prefix]
    order_of_values.extend(__get_ordered_values_for_file_path(client, brand, retail_chain, start_date, end_date, sep))
    order_of_values.append(filename_suffix)
    order_of_values = __filter_standardize_values_for_file_path(order_of_values)
    filename = "_".join(order_of_values)
    return f"{filename}{filename_extension}"

def get_filepath_to_store(base_path, filename_prefix, file_info, filename_suffix, filename_extension):
    """
    Get file path given information like client, brand, retail_chain, start_date, end_date
    """
    required_fields = [CLIENT, BRAND, RETAIL_CHAIN, START_DATE, END_DATE]
    fetched_values = {k:file_info.get(k) for k in required_fields}
    start_date=file_info.get(START_DATE)
    file_path_fields = [CLIENT, BRAND, RETAIL_CHAIN]
    file_path_params_fetched = {k:v for (k,v) in fetched_values.items() if k in file_path_fields}
    file_path_params = __get_ordered_values_for_file_path(**file_path_params_fetched)
    filename_to_save = get_filename(filename_prefix=filename_prefix,
                                    filename_suffix=filename_suffix,
                                    filename_extension=filename_extension,
                                    **fetched_values)
    return os.path.join(base_path,*file_path_params, str(start_date.year), f'{start_date.month:02d}',filename_to_save)


def __flat_map(f, xs):
    ys = []
    for x in xs:
        ys.extend(f(x))
    return ys

def get_aggregate_filename(client,period='weekly',header=True):
    header_tag = '' if header else '_noheader'
    return f'{client}_{period}_sales_data_aggregate{header_tag}.csv'

def get_aggregate_path(client,period='weekly',header=True):
    return os.path.join(get_client_data_folder(client,processed_data_folder),get_aggregate_filename(client,period,header))



def download_file(url,downloaded_filename=None,download_dir=None):
    """
    This function is used to download file from a url
    Arguments:
        url:str
            url to download file from
        downloaded_filename:str
            filename to keep for the downloaded file
        download_dir:str
            directory to download the file to
    """
    if not downloaded_filename:
        downloaded_filename=url.split('/')[-1]
    if not download_dir:
        download_dir=tempfile.mkdtemp()
    filepath=os.path.join(download_dir,downloaded_filename)
    # https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
    with requests.get(url, allow_redirects=True,stream=True) as r:
        r.raise_for_status()
        with open(filepath,'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return filepath



def extract_all(zip_filepath):
    # sometimes the downloaded file may be a zip & doesn't have an extension, so adding _extract at the end
    extracted_folder=os.path.splitext(zip_filepath)[0]+'_extract'
    with zipfile.ZipFile(zip_filepath,'r') as zf:
        zf.extractall(extracted_folder)
    return extracted_folder


def get_extracted_files(extracted_folder):
    supported_exts=set(['csv'])
    extracted_files = [f for f in os.listdir(extracted_folder)]
    exts = set(f.split('.')[-1] for f in  extracted_files)
    s = exts-supported_exts
    if len(s)>0:
        raise TypeError('following file formats are not supported',','.join(s))
    extracted_filepaths = [os.path.join(extracted_folder,filename) for filename in extracted_files]
    return extracted_filepaths

def get_list_of_filepaths(filepath, filepath_prefix=None,
                      regex_pattern_file=None, regex_pattern_filepath=None, flags=re.IGNORECASE):
    """
    Given a glob expression get list of files/folders matching that format
    For glob expressions refer - https://en.wikipedia.org/wiki/Glob_%28programming%29
    https://man7.org/linux/man-pages/man7/glob.7.html
    https://docs.python.org/3/library/glob.html
    Arguments:
        filepath:str
            This can be either the entire filepath string/glob expression
            or it can be only the last part of the file_path
        filepath_prefix:str
            This if not None is prepended to the filepath to form the glob pattern to search for

        regex_pattern_file:str or re.pattern
            a regex pattern for further filtering based on only the final filename/foldername
        regex_pattern_filepath:str or re.pattern
            a regex pattern for further filtering based on the entire filepath
        flags:str
            flags to use for the regular expression
    """
    if filepath_prefix is not None:
        filepath = os.path.join(filepath_prefix, filepath)
    files_list = [f for f in glob.glob(filepath)]

    # applying further filtering based on filename
    if regex_pattern_file is not None:
        re_f = re.compile(regex_pattern_file, flags=flags)
        files_list = [f for f in files_list if re_f.search(Path(f).name)]

    # applying further filtering based on filepath
    if regex_pattern_filepath is not None:
        re_f = re.compile(regex_pattern_filepath, flags=flags)
        files_list = [f for f in files_list if re_f.search(f)]

    return files_list


def get_list_of_files(filepath, filepath_prefix=None,
                      regex_pattern_file=None, regex_pattern_filepath=None, flags=re.IGNORECASE):
    """
    Given a glob expression get list of files/folders matching that format. This would return only the filenames
    For glob expressions refer - https://en.wikipedia.org/wiki/Glob_%28programming%29
    https://man7.org/linux/man-pages/man7/glob.7.html
    https://docs.python.org/3/library/glob.html
    Arguments:
        filepath:str
            This can be either the entire filepath string/glob expression
            or it can be only the last part of the file_path
        filepath_prefix:str
            This if not None is prepended to the filepath to form the glob pattern to search for

        regex_pattern_file:str or re.pattern
            a regex pattern for further filtering based on only the final filename/foldername
        regex_pattern_filepath:str or re.pattern
            a regex pattern for further filtering based on the entire filepath
        flags:str
            flags to use for the regular expression
    """
    files_list = get_list_of_filepaths(filepath, filepath_prefix,
                                  regex_pattern_file, regex_pattern_filepath, flags=re.IGNORECASE)
    return [Path(f).name for f in files_list]


def get_brand_data_folder(brand, brand_info, main_folder=raw_data_folder):
    client = brand_info.get_client_name(brand)
    return os.path.join(main_folder,client,brand)

def get_client_data_folder(client,main_folder):
    return os.path.join(main_folder,client)

def copy_file(source_filepath, destination_path, replace=True):
    """
    copy file from one location to another
    Arguments:
    source_filepath:String
        file to copy
    destination_path:String
        filepath to copy to
    replace:boolean
        If replace is true file is replaced even if it already exists
        If False replacement is not done, if file already exists even if there is a change in file

    returns True if file copied

    """
    # https://stackoverflow.com/questions/36821178/how-to-shutil-copyfile-only-if-file-differ/36821211
    create_parent_dirs(destination_path)
    destination_path = Path(destination_path)
    if __is_copy(source_filepath, destination_path, replace):
        logging.info(f'file copied from {source_filepath} to {destination_path}')
        shutil.copyfile(source_filepath,destination_path)
        return True
    else:
        logging.info(f'file {destination_path} already exists & not copied')

def __is_copy(source_filepath, destination_path, replace):
    """
    copy if file does not exists or
    if file exists & replace=False don't copy
    if file exists & replace=True copy only if files differ, this will keep the modification time as the intial time
    """
    return not destination_path.exists() or (replace and not filecmp.cmp(source_filepath, destination_path))

def create_parent_dirs(file_path, mode=0o777, exist_ok=True):
    file_path = Path(file_path)
    ## creating parent directories
    # for mode value refer - https://stackoverflow.com/questions/1627198/python-mkdir-giving-me-wrong-permissions
    file_path.parent.mkdir(mode=mode, parents=True,exist_ok=exist_ok)


def create_dir(file_path, mode=0o777, exist_ok=True):
    file_path = Path(file_path)
    ## creating parent directories
    # for mode value refer - https://stackoverflow.com/questions/1627198/python-mkdir-giving-me-wrong-permissions
    file_path.mkdir(mode=mode, parents=True,exist_ok=exist_ok)

def delete_file(file_path):
    """
    Given a file path or a directory path this will delete the file/directory
    """
    # https://stackoverflow.com/questions/6996603/how-to-delete-a-file-or-folder-in-python
    file_path = Path(file_path)
    if file_path.is_dir():
        shutil.rmtree(file_path)
    else:
        file_path.unlink(missing_ok=True)

def is_exists(file_path):
    """
    Check if a file or folder exists at the given location
    """
    return (file_path is not None) and Path(file_path).exists()

def is_file(file_path):
    """
    Check if a file exists at the given location
    """

    return (file_path is not None) and Path(file_path).is_file()

def get_file_type(filepath):
    """
    Get filetype given filename or filepath
    """
    filepath = Path(filepath)
    filetype = filepath.suffix
    return filetype[1:]

def get_filename_from_path(filepath):
    """
    return filename given filepath
    """
    return Path(filepath).name

def get_filename_without_extension(filename):
    """
    return filename without extension
    """
    return Path(filename).stem

def get_non_duplicate_filename(filename, list_of_names):
    """
    Given a filename return a new name which is not present in the provided list
    If the filename is not present in the list returns the original name
    """
    original_filename = get_filename_without_extension(filename)
    file_type = get_file_type(filename)
    i = 1
    while filename in list_of_names:
        filename = f"{original_filename}_v{i}"
        # adding the extension
        if file_type is not None and len(file_type.strip())!=0:
            filename = f"{filename}.{file_type}"
        i+=1
    return filename

def get_historical_path(brand_name):
    return os.path.join(get_brand_data_folder(brand_name,processed_data_folder),f'{brand_name}.csv')

def get_historical_data(brand_name):
    # https://docs.python.org/3/reference/lexical_analysis.html#f-strings
    historical_path = get_historical_path(brand_name)
    df_historical = pd.read_csv(historical_path)
    df_historical[WEEK_START_COl] = pd.to_datetime(df_historical[WEEK_START_COl], format=date_utils.DATE_FORMAT_ISO)
    return df_historical



address_abbreviations = {}
def get_address_abbreviations():
    global address_abbreviations
    if address_abbreviations:
        logging.info('returning already available address abbreviations')
        return address_abbreviations
    else:
        # list taken from - http://maf.directory/zp4/abbrev
        address_abbreviations_file = os.path.join(main_resources_folder,'address_abbreviations.json')
        logging.info('loading address abbreviations from file')
        address_abbreviations = {}
        with open(address_abbreviations_file,'r') as f:
            address_abbreviations = json.load(f)
        address_abbreviations = {k.lower():v.lower() for k,v in address_abbreviations.items()}

        
def replace_nan(df):
    group = ['test_influence', 'control_influence']
    for col in group:
        df[col][df[col].isnull()] = df[col][df[col].isnull()].apply(lambda x: [])
    return df


def save_json_format_for_z3(df, output_path, store_id_col=STORE_ID_COL,sales_col=SALES_DOLLAR_COL, zipcode_col=ZIPCODE_COL,
                           zipcode_expanded_col=ZIPCODE_EXPANDED, radius_col=RADIUS_COL,
                           validated_col=VALIDATED_COL, is_original_zipcode_col=IS_ORIGINAL_ZIPCODE_COL):
    """
    saves a dataframe in json format for z3 code

    Output example
    {
        "circle k 1":{
            "score":162.3573584906,
            "control_influence":[
                "55012",
                "55045"
            ],
            "test_influence":[
                "55045"
            ]
        },
        "circle k 10":{
            "score":77.6528301887,
            "control_influence":[
                "59404"
            ],
            "test_influence":[
                "59404"
            ]
        }
    }
    """
    test_radius = 2
    control_radius = 5
    
    # for boston pizza as we might not find any zipcode within 2 miles radius so we increase radius to include zipcodes
    #test_radius = 5
    #control_radius = 10
    ## TODO can control max radius using some config,
    # as of now max of 5 radius is the max radius to which file is expanded
    t = dataframe_utils.get_store_list_expanded(df, radius=test_radius, max_radius=5, radius_col=radius_col,
                                zipcode_col=zipcode_col, zipcode_expanded_col=zipcode_expanded_col,
                                is_original_zipcode_col=is_original_zipcode_col, validated_col=validated_col
                                )
    test = t.groupby([store_id_col,sales_col])[zipcode_expanded_col].apply(list).reset_index(name='test_influence')
    t = dataframe_utils.get_store_list_expanded(df, radius=control_radius, max_radius=10, radius_col=radius_col,
                                zipcode_col=zipcode_col, zipcode_expanded_col=zipcode_expanded_col,
                                is_original_zipcode_col=is_original_zipcode_col, validated_col=validated_col
                                )
    control = t.groupby([store_id_col,sales_col])[zipcode_expanded_col].apply(list).reset_index(name='control_influence')
    
    # some stores may not have test influence zipcodes hence left join is used to inlcude such store
    final = pd.merge(control, test[[store_id_col,'test_influence']], on = store_id_col, how = 'left').set_index(store_id_col)
    rename_dict = {SALES_DOLLAR_COL:'score'}
    final.columns = [rename_dict.get(col,col) for col in final.columns]
    # saving json file
    #final.to_json(output_path, orient = 'index', indent = 4)
    #final.to_csv(r'D:\work\project\cac\campaign\campaign_automation\new_format\test_json.csv')
    final = replace_nan(final)
    final.to_json(output_path, orient = 'index')
    # returning a dataframe -  store_id, sales_dollar, control_influence, test_influence
    return final

def plot_save_weekly_group_avg_graph(df, output_folder, week_col=WEEK_END_COL, sales_col=SALES_DOLLAR_COL, group_col=GROUP_COL):
    #check this later https://github.home.247-inc.net/cac/sales/blob/master/measurement/pf_transition_matrix.ipynb
    fig = plt.figure(figsize=(45,22))
    # https://het.as.utexas.edu/HET/Software/Matplotlib/api/dates_api.html
    # https://kite.com/python/answers/how-to-plot-dates-on-the-x-axis-of-a-matplotlib-plot-in-python
    ax = plt.gca()
    formatter = mdates.DateFormatter("%Y-%b-%d")
    matplotlib.rcParams.update({'font.size': 35})
    ax.xaxis.set_major_formatter(formatter)

    group_sales_lineplot = sns.lineplot(x=week_col, y = sales_col, hue=group_col,style=group_col,markers=True, data=df,sort=True,lw=4)\
    .set_title("test vs control avg. sales")

    ticks = plt.xticks(df[week_col], rotation = 'vertical', fontsize = 25)
    # ticks = plt.xticks(np.arange(len(uniq_dates)),uniq_dates['date_formatted'], rotation = 'vertical', fontsize =14 )
    plt.xlabel("weeks")
    plt.ylabel("avg. sales")

    plt.show(group_sales_lineplot)



    fig_plotly = go.Figure()
    fig_plotly = px.line(df, x=week_col, y=sales_col, title='test vs control avg. sales',color=group_col)
    fig_plotly.show()
    # saving plots
    # saving seaborn plot
    seaborn_plot_path = os.path.join(output_folder,"sales_test-control.jpg")
    group_sales_lineplot.get_figure().savefig(seaborn_plot_path,orientation='landscape',bbox_inches='tight' ,pad_inches=.1)
    logging.info(f"seaborn plot saved to {seaborn_plot_path}")

    # saving plotly plot
    plotly_plot_path = os.path.join(output_folder,"sales_test-control.html")
    # include_plotlyjs='cdn', to reduce file size, if you need offline version keep as True
    # https://plotly.github.io/plotly.py-docs/generated/plotly.io.write_html.html
    fig_plotly.write_html(plotly_plot_path, include_plotlyjs='cdn')
    logging.info(f"plotly plot saved to {plotly_plot_path}")

def convert_save_z3_results_to_platform_format(z3_results_path, df_expanded, df_sales, output_folder, \
                                               store_id_col=STORE_ID_COL, sales_col = SALES_DOLLAR_COL, \
                                               week_col=WEEK_COL, \
                                               is_include_missing_weeks_for_average=False, \
                                               is_clustered_result=False, avg_sales_plot=False):
    """
    Saves files to be provided to the TARGET platform. This also saves graphs showing trend of sales
    for the during present in sales file

    Arguments:
        z3_results_path:string
            filepath to the z3 output
            if output is from z3 code using clustering then file should be a json file with the following content
                a dictionary with the following structure
                {'test': {'0': 'circle k 1765',
                        '1': 'circle k 2700025'
                        '2': 'circle k 2701700',...},
     '          control': {'0': 'circle k 2723388',
                        '1': 'circle k 2723421',...}
            if output is from z3 code without clustering then it should be a excel file with the following columns,
                subset_name,stores column being mandatory 
                S.No.	subset_name	subset_size	subset_size_pct	subset_score_sum	subset_score_avg	stores
                0	control	27	0.238938053	50497.16846	1870.265499	"store_id_1074\nstore_id_76.."
                1	test	86	0.761061947	161623.2506	1879.340123	"store_id_1189\nstore_id_1214..."

        df_expanded:pandas.Dataframe
            5 miles expanded store list
        df_sales:pandas.Dataframe
            sales dataframe with weekly data, should have store_id, sales
        store_id_col:string
            store id column name
        sales_col:string
            sales dollar column name
        week_col:string
            week column name
        is_include_missing_weeks_for_average:boolean
            true if we need to consider missing weeks as zero sales for average
            let's say data is present for 52weeks for store1, but for store2 it's only 40 weeks. 
            Keeping this as true will divide store2_sales/52
        is_clustered_result:boolean
            true, if output is from the clustering version of z3 code

    """
     # for mode value refer - https://stackoverflow.com/questions/1627198/python-mkdir-giving-me-wrong-permissions
    Path(output_folder).mkdir(mode=0o777, parents=True,exist_ok=True)
    df_sales_temp = df_sales.copy()
    if is_include_missing_weeks_for_average:
        df_sales_temp = utils.stats_helper.StatsHelper.fill_missing(df_sales_temp, store_id_col, week_col=week_col)
    GROUP_COL = 'group'
    aggregation_col = GROUP_COL
    df_sales_temp[GROUP_COL] = 'skipped'
    # Eg: z3_result = {'control':['store_id_1','store_id_2'...],
    # 'test':['store_id_3','store_id_4'..]}
    z3_results = {}
    if is_clustered_result:
        with open(z3_results_path, 'r') as f:
            z3_results_json = json.load(f)
        for group, z3_group_stores in z3_results_json.items():
            # the z3_results have None values present in them, removing it
            group_stores = list(filter(None, z3_group_stores.values()))
            z3_results[group] = group_stores
    else:
        df_z3_results = pd.read_excel(z3_results_path)
        # TODO spaces after each store_id is not handled
        z3_results = {group:values for (group,values) in zip(df_z3_results['subset_name'], df_z3_results['stores'].str.split("\n"))}
    for group, group_stores in z3_results.items():
        df = pd.DataFrame()
        df[STORE_ID_COL] = group_stores
        radius = 2
        if group=="control":
            radius = 5

        t = dataframe_utils.get_store_list_expanded(df_expanded, radius)
        #df_out = pd.merge(df, t, how='inner', on=store_id_col )
        df_out = pd.merge(df, t, how='left', on=store_id_col )
        #df_out['group'] = group

        df_zipcode = zipcode_utils.get_df_with_zipcode_for_dv360(df_out)
        #df_zipcode['group'] = group

        group_stores_filepath = os.path.join(output_folder,f"{group}_df.csv")
        zipcodes_filepath = os.path.join(output_folder, f"{group}_zipcodes.csv")

        logging.info(f"store output path {group_stores_filepath}, zipcodes filepath {zipcodes_filepath}")
        # saving store & zipcode info for this group
        df_out.to_csv(group_stores_filepath, index=False)
        df_zipcode.to_csv(zipcodes_filepath, index=False)

        df_sales_temp.loc[df_sales_temp[store_id_col].isin(group_stores),GROUP_COL] = group
    
    df_sales_temp = df_sales_temp[df_sales_temp[GROUP_COL] != 'skipped'] 
    t = df_sales_temp.copy()
    # t = t[t[GROUP_COL]!='skipped']
    # sales average per group
    if (week_col is not None) and (week_col in t.columns):
        df_sales_group_avg = t.groupby([aggregation_col,store_id_col, week_col])[sales_col].agg('sum') \
        .groupby([aggregation_col]).agg('mean').reset_index(name='sales_avg')
        
        sales_avg_path = os.path.join(output_folder,'sales_groupwise_avg.csv')
        df_sales_group_avg.to_csv(sales_avg_path, index=False)


        df_sales_group_avg_weekly = t.groupby([aggregation_col, store_id_col, week_col])[sales_col].agg('sum') \
        .groupby([aggregation_col, week_col]).agg('mean').reset_index()
        sales_weekly_avg_path = os.path.join(output_folder,'sales_groupwise_weekly_avg.csv')
        df_sales_group_avg_weekly.to_csv(sales_weekly_avg_path, index=False)

        #plotting graph
        if avg_sales_plot:
            plot_save_weekly_group_avg_graph(df_sales_group_avg_weekly, output_folder, week_col=week_col, \
                                        sales_col=sales_col)
    else:
        # week column is not present for this client, we would have used 52 or 13 week aggregate sales data
        #df_sales_group_avg = t.groupby([aggregation_col,store_id_col])[sales_col].agg('sum') \
        #.groupby([aggregation_col]).agg('mean').reset_index(name='sales_avg')
        logging.info(f"week_col {week_col} not present in sales data, so not calculating average")

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
    print('logging info',rootLogger.handlers )