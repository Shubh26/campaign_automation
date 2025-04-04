#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os,sys
from pathlib import Path

import logging
from utils import dataframe_utils

import math
import re
import random, string

from utils.constants import *

def identify_zipcode_columns(column_names):
    # TODO pull this regex compile out
    zip_col_regex = re.compile('^zip(_|\s)?(code)?$',flags=re.IGNORECASE)
    zip_cols = [col for col in column_names if zip_col_regex.search(col)]
    return zip_cols

def identify_zipcode_column(column_names):
    zip_cols = identify_zipcode_columns(column_names)
    if len(zip_cols)==0:
        raise ValueError('Unable to find zip code column')
    if len(zip_cols)>1:
        raise ValueError(f'Multiple zip code columns found value ambiguous -  {",".join(zip_cols)}. Specify using - {ZIPCODE_COL}')
    return zip_cols[0]

def validate_zipcode_type_pandas_df(df,zipcode_columns=None):
    if zipcode_columns==None:
        zipcode_columns = identify_zipcode_columns(df)
    dtypes = df.dtypes.to_dict()
    # zipcodes should be loaded as string (object)
    incorrect_type_columns = {col:dtypes.get(col) for col in zipcode_columns if str(dtypes.get(col))!="object"}
    if len(incorrect_type_columns)>0:
        logging.info(f'following columns have incorrect data type {incorrect_type_columns}')
        # TODO raise an exception
        # raise Exception(f'following columns have incorrect data type {incorrect_type_columns}')

def get_dataframe_with_standard_length_zipcode(df, zipcode_actual=ZIPCODE_COL, zipcode_standard=ZIPCODE_COL, drop=False, size=5):
    """
    Given a pandas dataframe with zipcode, returns another one with an additional column
    where zipcode is standardized to a size (5) digit length
    Args:
    zipcode_actual : the zipcode column with the non normalized zipcode
    zipcode_standard : the standardized 5 digit zipcode output column
    drop:boolean
        Keep true if you want to drop original zipcode column
    size : int the standard size (=5 for us zipcodes)
    """
    t = df.copy()
    columns = list(t.columns)
    if zipcode_standard==zipcode_actual:
        # generating a temp unique column name for zipcode actual & will drop this at the end
        temp_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(16))
        rename_dict = {zipcode_actual:temp_name}
        zipcode_actual = temp_name
        columns = [rename_dict.get(col,col) for col in columns]
        t.columns = columns
        drop = True

    # the 5 digit conversion step
    t[zipcode_standard] = _get_series_with_constant_length_zipcode(t[zipcode_actual], size)

    # if drop=True dropping the original column
    if drop:
        # if dropping keeping the added column at same location of removed column
        rename_dict = {zipcode_actual:zipcode_standard}
        columns = [rename_dict.get(col,col) for col in columns]
        t = t[columns]
        # t.drop(zipcode_actual,axis=1,inplace=True)

    return t

def get_dataframe_with_5digit_zipcode(df,zipcode_actual=ZIPCODE_COL,zipcode_5digit=ZIPCODE_COL,drop=False):
    """
    Given a pandas dataframe with zipcode, returns another one with an additional column
    where zipcode is standardized to a 5 digit length
    Args:
    zipcode_actual : the zipcode column with the non normalized zipcode
    zipcode_5digit : the standardized 5 digit zipcode output column
    drop:boolean
        Keep true if you want to drop original zipcode column
    """
    return get_dataframe_with_standard_length_zipcode(df, zipcode_actual=zipcode_actual, zipcode_standard=zipcode_5digit, drop=drop, size=5)

def _get_series_with_constant_length_zipcode(zipcode_series, size=5):
    """
    Given a zipcode series return a series with 5 digit zipcodes
    """
    t = zipcode_series.copy()
    # zipcodes should be loaded as string (object)
    if t.dtype!='object':
        t = change_zipcode_series_type(t)
    # slower method
    # t = t.apply(convert_to_5digit_zipcode)
    # faster method

    t = t.str.split('-').str[0].str.pad(width=size, side='left', fillchar='0').str.slice(stop=size)
    return t


class ZipcodeHelper(object):
    ZIPCODE_INVALID_DV360_PATH_KEY = 'zipcode_invalid_dv360_path'
    ZIPCODE_DATABASE_PATH_KEY = 'zipcode_database_path'
    def __init__(self,**kwargs):
        """
        This class helps with operations requiring the zipcode database & the invalid zipcodes list from dv360
        Arguments:
        zipcode_invalid_dv360_path:String
            path to the invalid zipcodes file for dv360
        zipcode_database_path:String
            path to the zipcode database file with population, approximate_latitude etc

        """
        super(ZipcodeHelper).__init__()
        ## TODO keep a default value specific to machine os & maybe even give the option of specifying the complete path to zipcode not validated file
        codes_folder='D:\projects\Personalization\code'
        zipcode_invalid_dv360_default_path = os.path.join(codes_folder,"sales","resources","zip_not_validated_dv_360.csv")
        self.zipcode_invalid_dv360_path = Path(kwargs.get(ZipcodeHelper.ZIPCODE_INVALID_DV360_PATH_KEY,zipcode_invalid_dv360_default_path))
        zipcode_database_default_path = os.path.join(codes_folder,"sales","resources","zip_code_database_enterprise.csv")
        self.zipcode_database_path = Path(kwargs.get(ZipcodeHelper.ZIPCODE_DATABASE_PATH_KEY,zipcode_database_default_path))

        # TODO can have different exceptions based on which one is missing
        # https://stackoverflow.com/questions/82831/how-do-i-check-whether-a-file-exists-without-exceptions
        # is file checks for exists as well
        if (not self.zipcode_invalid_dv360_path.is_file()) or (not self.zipcode_database_path.is_file()):
            raise Exception(f"specify a valid path for {ZipcodeHelper.ZIPCODE_INVALID_DV360_PATH_KEY} and {ZipcodeHelper.ZIPCODE_DATABASE_PATH_KEY}, current values {self.zipcode_invalid_dv360_path} {self.zipcode_database_path} respectively")

        self._load_zipcode_invalid_dv360()
        self._load_zipcode_database()

    def _load_zipcode_invalid_dv360(self):
        df_non_validated_dv360 = pd.read_csv(self.zipcode_invalid_dv360_path,dtype={VALIDATED_COL:bool,ZIPCODE_COL:str})
        t = df_non_validated_dv360
        # the should be already have 5 digit zipcodes but running this just to be on the safer side
        t = get_dataframe_with_5digit_zipcode(t,zipcode_actual=ZIPCODE_COL,zipcode_5digit=ZIPCODE_COL)
        self._df_invalid_dv360 = t[[ZIPCODE_COL, VALIDATED_COL]]

    def _load_zipcode_database(self):
        df_zipcode = pd.read_csv(self.zipcode_database_path,dtype={ZIPCODE_COL:str})

        t = df_zipcode
        # the should be already have 5 digit zipcodes but running this just to be on the safer side
        t = get_dataframe_with_5digit_zipcode(t,zipcode_actual=ZIPCODE_COL,zipcode_5digit=ZIPCODE_COL)

        self._df_zipcode = t

    def __is_validation_column_present(self,df):
        return VALIDATED_COL in df.columns


    def get_dataframe_with_validation_column(self,df,zipcode_col=ZIPCODE_COL,df_name='', suppress_exceptions=False):
        t = df.copy()
        if self.__is_validation_column_present(df):
            t = t.astype({VALIDATED_COL: bool})
            logging.info(f'{VALIDATED_COL} already present for {df_name} using the same, ignoring new addition')
            return t

        ### Adding non validated column
        t = pd.merge(t,self._df_invalid_dv360[[VALIDATED_COL,ZIPCODE_COL]],how='left',left_on=zipcode_col,right_on=ZIPCODE_COL)
        na_zipcode = t[t[zipcode_col].isna()]
        if not na_zipcode.empty:
            message = f'merging with dv360 for {df_name} rows with NA values found for required column {ZIPCODE_5DIGIT_COL}, no of cases {len(na_zipcode)} top few cases {na_zipcode.head(5)}'
            if suppress_exceptions:
                logging.error(message)
            else:
                raise Exception(message)

        t.loc[t[VALIDATED_COL].isna(),VALIDATED_COL]=True
#         https://stackoverflow.com/questions/15891038/change-data-type-of-columns-in-pandas
        t = t.astype({VALIDATED_COL: bool})
        logging.debug(f'validated vs non validated zips stats {t[VALIDATED_COL].value_counts()}')
#         logging.info(f'*****provided data dtypes {df.dtypes} ****data type validation before {self._df_invalid_dv360.dtypes} data type after {t.dtypes}')
        return t

    def retain_only_validated_rows(self,df,df_name=''):
        """
        This retains only rows with zipcode validated using dv360
        """
        if not self.__is_validation_column_present(df):
            df = self.get_dataframe_with_validation_column(df)
        t = df.copy()
        invalid = t[~t[VALIDATED_COL]]
        if len(invalid)>0:
            logging.info(f'removing invalid rows, {df_name} shape {invalid.shape} \n top invalid zipcode values {invalid.head(5)}')
        t = t[t[VALIDATED_COL]]
        return t

    def get_zipcode_database(self, columns = [ZIPCODE_COL,"approximate_latitude","approximate_longitude",POPULATION_COUNT_COL, COUNTY_COL, STATE_COL]):
        """
        This will return a pandas dataframe of the zipcode database

        Arguments:
        columns: list of string
            List of column names to pick from the zipcode database
            By default it returns only the most commonly useful ones
            If you want all the columns to be picked specify columns = None

        """
        df = self._df_zipcode.copy()
        # taking all columns if columns is None
        columns = columns if columns else df.columns
        df = df[columns]
        return df

    def __is_population_column_present(self,df):
        return POPULATION_COUNT_COL in df.columns

    def get_dataframe_with_population_column(self,df,zipcode_col=ZIPCODE_COL,df_name='', suppress_exceptions=False):
        """
        Add population count column to the given dataframe.

        Arguments:
        df: pandas dataframe
            A pandas dataframe with store list
            Mandatory column - zipcode_col
        zipcode_col: String
            header for zipcode column, default_value: zipcode
        df_name: String
            Pandas dataframe name. This is only used for logging information, so it's not mandatory but it will be helpful
            when debugging the code
        suppress_exceptions:boolean
            If the zipcode used for the store is not present in zipcode database, by default it will throw an exception.
            If True instead of exception it will just log an error message

        """
        t = df.copy()
        if self.__is_population_column_present(df):
            t = t.astype({POPULATION_COUNT_COL: np.int32})
            logging.info(f'{POPULATION_COUNT_COL} already present for {df_name} using the same, ignoring new addition')
            return t

        ### Adding population count column
        t = pd.merge(t,self.get_zipcode_database(columns=[ZIPCODE_COL,POPULATION_COUNT_COL]),how='left',left_on=zipcode_col,right_on=ZIPCODE_COL)
        na_population_count = t[t[POPULATION_COUNT_COL].isna()]
        if not na_population_count.empty:
            message = f'merging with zipcode database for for {df_name} rows with NA values found for required column {POPULATION_COUNT_COL}, no of cases {len(na_population_count)} top few cases {na_population_count.head(5)}'
            if suppress_exceptions:
                logging.error(message)
            else:
                raise Exception(message)

        # ideally there won't be any zipcodes which would be missing from the zipcode database
        # assigning a value of zero is population count is na
        t.loc[t[POPULATION_COUNT_COL].isna(),POPULATION_COUNT_COL]=0
#         https://stackoverflow.com/questions/15891038/change-data-type-of-columns-in-pandas
        t = t.astype({POPULATION_COUNT_COL: np.int32})
        return t

    def get_dataframe_with_additional_columns(self, df, zipcode_col=ZIPCODE_COL, additional_columns=[CITY_COL, STATE_COL]):
        t = df.copy()
        # removing items from list if already present in the dataframe
        additional_columns_temp = list(set(additional_columns)-set(df.columns))
        if len(additional_columns_temp)==0:
            logging.info(f'all the required columns already present in the dataframe {additional_columns}')
            return df
        if len(additional_columns_temp) < len(additional_columns):
            logging.info(f'skipped the following columns as they were already present in the df {set(additional_columns).intersection(set(df.columns))}')

        additional_columns = additional_columns_temp

        # we will use county instead of city for now, but keep column header as city
        rename_dict = {}
        if CITY_COL in additional_columns:
            rename_dict = {COUNTY_COL : CITY_COL}
            # reversing the dictionary
            temp_dict = {v:k for k,v in rename_dict.items()}
            additional_columns = [temp_dict.get(col,col) for col in additional_columns]

        additional_columns.append(ZIPCODE_COL)
        t = pd.merge(t,self.get_zipcode_database(additional_columns),how='left',left_on=zipcode_col,right_on=ZIPCODE_COL)
        t = dataframe_utils.rename_headers(t, rename_dict)
        return t

zipcode_helper = None
def get_zipcode_helper():
    # using the variable available in global scope
    global zipcode_helper
    if not zipcode_helper:
        # TODO create this based on the provided paths
        zipcode_helper = ZipcodeHelper()
    return zipcode_helper

def haversine_distance(lat1,lng1,lat2,lng2):
    earthRadius = 3958.75 # miles (or 6371.0 kilometers)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    lng1 = math.radians(lng1)
    lng2 = math.radians(lng2)

    dlat = lat2-lat1
    dlng = lng2-lng1
    a = math.sin(dlat/2)**2 + math.sin(dlng/2)**2 * math.cos(lat1) * math.cos(lat1)
    c = 2 * math.atan2(math.sqrt(a),math.sqrt(1-a))
    distance = earthRadius * c
    return distance


def _trim_longer_zipcode(zipcode):
    zipcode=str(zipcode)
    if '-' in zipcode or len(zipcode)>5:
        zipcode = zipcode.split('-')[0]
        zipcode = zipcode[:5]
    return zipcode


def _zero_pad_zipcode(zipcode):
    zipcode=str(zipcode)
    max_len=5
    return f"{'0'*(max_len-len(zipcode))}{zipcode}"


def convert_to_5digit_zipcode(zipcode):
    """
    zipcode: either as string or number
    """
    # sometimes zip comes in as number
    zipcode = _trim_longer_zipcode(zipcode)
    zipcode = _zero_pad_zipcode(zipcode)
    return zipcode

def _update_row(row,zipcode_nearby_zipcodes_dict,zipcode_col='zipcode'):
    zipcode = row[zipcode_col]
    zipcode5=convert_to_5digit_zipcode(zipcode)
    row.loc['Zipcode_5digit']= zipcode5
    # keeping set for removing duplicate entries
    nearby_zipcodes = set(zipcode_nearby_zipcodes_dict.get(zipcode5,[zipcode5]))
    # since we are going to be creating only as many rows as there are entries & we also indeed the original entry
    # adding that to the list as well
    nearby_zipcodes.add(zipcode5)
    # sorting is not necessary, TODO it might be better to keep the actual zip as the 1st entry
    row.loc['Zipcode_all']  = sorted(list(nearby_zipcodes))
    return row


def update_df_add_rows(df,zipcode_nearby_zipcodes_dict,zipcode_col='zipcode'):
#     https://stackoverflow.com/questions/53860398/pandas-dataframe-how-do-i-split-one-row-into-multiple-rows-by-multi-value-colum
    # this step will create a new column called Zipcode_all
    level_regex = re.compile('^level_\d+$')
    ZIPCODE_ALL = 'Zipcode_all'
    t = df.apply(lambda row: _update_row(row,zipcode_nearby_zipcodes_dict,zipcode_col),axis=1)
    index_cols = list(t.columns)
    # want to keep every column except ZIPCODE_ALL into index
    index_cols.remove(ZIPCODE_ALL)
    t = t.set_index(index_cols)[ZIPCODE_ALL].apply(pd.Series).stack().reset_index()
#     there would ideally be only one column with level_number eg:- ['level_2']
    col_to_drop=[col for col in t.columns if level_regex.search(str(col))]
    if len(col_to_drop)!=1:
        raise ValueError(f'multiple cols found with level_ {col_to_drop}')
    t = t.drop(labels=col_to_drop,axis=1)
    t = t.rename(columns={zipcode_col:'Zipcode_actual',0:zipcode_col})
    return t




def update_table(df,zipcode_col='zipcode'):
    t = update_df_add_rows(df,zipcode_nearby_zipcodes_dict,zipcode_col)

    t['is_original_zipcode'] = t['Zipcode_5digit']==t[zipcode_col]
    # in this case we are only looking at store in 2 mile radius, when zipcode is exactly the same keeping it as 0
    t['Radius'] = 2
    t.loc[t['is_original_zipcode'],'Radius']=0
    # t['Radius'].loc[t['is_original_zipcode']]=0
    return t


def change_zipcode_type(df,col_name):
    if df.dtypes.loc[col_name]=='float64':
        # if type if float then decimal places would be there, to avoid it casting to integer first
        df.loc[:,col_name] = pd.to_numeric(df[col_name],downcast='integer')
    # the converted integer for zip code col is cast to string,
    # in cases where the zipcode col have hyphens or spaces in between, col type will be string itself, but no harm in doing it
    df.loc[:,col_name] = df[col_name].astype(str)
    return df

def change_zipcode_series_type(zipcode_series):
    t = zipcode_series.copy()
    if t.dtype =='float64':
        print('zipcode float type')
        # if type if float then decimal places would be there, to avoid it casting to integer first
        t = pd.to_numeric(t,downcast='integer')
    # the converted integer for zip code col is cast to string,
    # in cases where the zipcode col have hyphens or spaces in between, col type will be string itself, but no harm in doing it
    t = t.astype(str)
    return t


def identify_zip_col_modify_and_write(filepath):
    filename=os.path.basename(filepath)
    df = pd.read_csv(filepath)
    zipcode_col = identify_zipcode_column(df.columns)
    df = change_zipcode_type(df,zipcode_col)

    df_updated = update_table(df,zipcode_col)
    output_path=os.path.join(os.path.splitext(filepath)[0]+'_2mile_zipcode.csv')
    df_updated.to_csv(output_path)
    print(f'No of zipcodes before taking 2 mile radius {len(df[zipcode_col].unique())} filename {filename}')
    print(f'No of zipcodes after taking 2 mile radius {len(df_updated[zipcode_col].unique())} filename {filename}')
#     print(f'Done writing {output_path}')
    return (df_updated,zipcode_col)


def _check_is_overlap(df_test,df_control,col_name):
    # if there is intersection then there is overlap
    return len(set(df_g_c['ZIP'].unique()).intersection(df_g_t['ZIP'].unique()))>0


def get_df_with_zipcode_for_dv360(df,zipcode_col=ZIPCODE_EXPANDED):
    t = df.copy()
    t = t[[zipcode_col]]
    t.columns=[zipcode_col]
    # running 5 digit standardization, the zipcode might already be in 5 digit form just ensuring that
    t = get_dataframe_with_5digit_zipcode(t,zipcode_actual=zipcode_col,zipcode_5digit=zipcode_col)
    # zipcode_formatted is currently used only for the dv360 file
    # t['zipcode_formatted'] = t[zipcode_col].apply(lambda x: f"{x}, United States (Postal Code)")
    t['zipcode_formatted'] = t[zipcode_col].apply(lambda x: f"US/{x}")
    # TODO document it dropping duplicates here might be unexpected
    t = t.drop_duplicates()
    return t
