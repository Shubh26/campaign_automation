import os
import re

import pandas as pd
import numpy as np
import pytest

import utils.sql_utils
from utils.constants import STORE_ID_COL, STORE_ADDRESS_COL, BANNER_COL, ZIPCODE_COL, SALES_DOLLAR_COL, ZIPCODE_EXPANDED
from utils import dataframe_utils #import split_to_columns, transpose, StandardizeData
from utils.test_utils import test_resources_folder, sales_data_variants_folder, assert_dfs_equal

standardize_file = os.path.join(test_resources_folder,'standardize_test_file.csv')
standardize_file2 = os.path.join(test_resources_folder,'standardize_test_file2.csv')

def test_standardize_column_names1():
    df = pd.read_csv(standardize_file)
    out_expected = ["store_id",BANNER_COL,"address",ZIPCODE_COL,ZIPCODE_EXPANDED,"week",SALES_DOLLAR_COL]

    df = dataframe_utils.StandardizeData.standardize_column_names(df)
    out = df.columns
    np.testing.assert_array_equal(out,out_expected,err_msg="columns not renamed properly")

def test_standardize_store_id1():
    df = pd.read_csv(standardize_file)
    df = dataframe_utils.StandardizeData.standardize_column_names(df)

    out_expected = ['walmart_1', 'walmart_2', 'kroger_3', 'kroger_4', 'walmart_5']
    out = dataframe_utils.StandardizeData.standardize_store_id_column(df,store_id_col=STORE_ID_COL,store_banner_col=BANNER_COL)[STORE_ID_COL]
    np.testing.assert_array_equal(out,out_expected,err_msg="standardized store ids does not match")


def test_standardize_store_id2():
    df = pd.read_csv(standardize_file2)
    df = dataframe_utils.StandardizeData.standardize_column_names(df)

    # out_expected = ['walmart_1', 'walmart_2', 'kroger_3', 'kroger_4', 'walmart_5']
    try:
        out = dataframe_utils.StandardizeData.standardize_store_id_column(df,store_id_col=STORE_ID_COL,store_banner_col=BANNER_COL)[STORE_ID_COL]
        pytest.fail("there are mis matching store id & banner, it should have thrown an error for those cases")
    except ValueError as e:
        # print(f"correctly threw error:- {e}")
        expected_message = "current store id kroger_5 doesn't correspond to the current store banner walmart"
        actual_message = ' '.join(e.args)
        np.testing.assert_array_equal(actual_message, expected_message, err_msg="error message not matching for invalid standardize store id entry")


def test_standardize_store_address_column1():
    df = pd.read_csv(standardize_file)
    df = dataframe_utils.StandardizeData.standardize_column_names(df)
    out_expected = 'street no 10, random street, random city'
    out = dataframe_utils.StandardizeData.standardize_store_address_column(df,store_address_col=STORE_ADDRESS_COL)[STORE_ADDRESS_COL][0]
    np.testing.assert_array_equal(out,out_expected,err_msg="test store address is not standardized properly")


def test_split_columns():
    filepath = os.path.join(test_resources_folder, "standardize_test_file_split_column.csv")
    expected_output_filepath = os.path.join(test_resources_folder, "standardize_test_file_split_column_processed.csv")
    df = pd.read_csv(filepath)
    df_out = dataframe_utils.split_to_columns(df, column_to_split="Geography", column_names=[STORE_ID_COL, STORE_ADDRESS_COL], separator=":")

    df_expected = pd.read_csv(expected_output_filepath)
    assert df_expected.equals(df_out), f"expected output is not present after splitting a column into multiple columns"


def test_transpose_data():
    filepath = os.path.join(sales_data_variants_folder, "header_to_transpose.csv")
    df = pd.read_csv(filepath, skiprows=3, header=[0,1])
    df_out = dataframe_utils.transpose(df, fixed_column_indices=[0], headers_to_transpose=[0],transposed_header_names=["store"])
    expected_output_filepath = os.path.join(sales_data_variants_folder, "header_to_transpose_processed.csv")
    df_expected = pd.read_csv(expected_output_filepath)
    df_expected.columns = ['Product', 'store', 'Dollar Sales', 'Unit Sales', 'Volume Sales']
    assert df_expected.equals(df_out), f"transposing headers into row values didn't work properly for the file {filepath}"


def test_transpose_data2():
    filepath = os.path.join(sales_data_variants_folder, "header_to_transpose2.csv")
    df = pd.read_csv(filepath, skiprows=3, header=[0,1,2])
    df_out = dataframe_utils.transpose(df, fixed_column_indices=[0,1], headers_to_transpose=[0,1],transposed_header_names=["division", "store"])
    expected_output_filepath = os.path.join(sales_data_variants_folder, "header_to_transpose2_processed.csv")
    df_expected = pd.read_csv(expected_output_filepath)
    df_expected.columns = ['Product', 'Product_dup', 'division', 'store', 'Dollar Sales',
                           'Unit Sales', 'Volume Sales']
    assert df_expected.equals(df_out), f"transposing headers into row values didn't work properly for the file {filepath}"

def __convert_dtype_to_string(dtype_actual):
    dtype_actual = dtype_actual.to_dict()
    return {k: v.str[1:] for k, v in dtype_actual.items()}

def test_change_datatype1():
    d = {'col1': [1, 2], 'col2': [3, 4], 'col3': [5.0, 6.0]}
    df = pd.DataFrame(data=d)
    dtype_actual = __convert_dtype_to_string(df.dtypes)
    dtype_expected = {'col1': 'i8', 'col2': 'i8', 'col3': 'f8'}
    assert dtype_actual == dtype_expected
    # test functionality of changing all columns datatype to same type
    df_actual = dataframe_utils.change_datatype(df,"str")
    dtype_actual = __convert_dtype_to_string(df_actual.dtypes)
    dtype_expected = {'col1': 'O', 'col2': 'O', 'col3': 'O'}
    assert dtype_actual == dtype_expected


def test_change_datatype2():
    d = {'col1': [1, 2], 'col2': [3, 4], 'col3': [5.0, 6.0]}
    df = pd.DataFrame(data=d)
    dtype_actual = __convert_dtype_to_string(df.dtypes)
    dtype_expected = {'col1': 'i8', 'col2': 'i8', 'col3': 'f8'}
    assert dtype_actual == dtype_expected
    #selectively change the datatype of columns
    df_actual = dataframe_utils.change_datatype(df, {'col1': "float",'col2':"str"})
    dtype_actual = __convert_dtype_to_string(df_actual.dtypes)
    dtype_expected = {'col1': 'f8', 'col2': 'O', 'col3': 'f8'}
    assert dtype_actual == dtype_expected

def test_regex_replace_df1():
    d = {'col1': [1, 2], 'col2': ['39774', '23265']}
    df = pd.DataFrame(data=d)
    df_actual = dataframe_utils.regex_replace_df(df,'col2',"^",'7eleven_')
    d = {'col1': [1, 2], 'col2': ['7eleven_39774', '7eleven_23265']}
    df_expected = pd.DataFrame(data=d)
    assert_dfs_equal(df_expected,df_actual,"")

def test_regex_replace_df2():
    d = {'col1': [1, 2], 'col2': ['39774', '23265']}
    df = pd.DataFrame(data=d)
    df_actual = dataframe_utils.regex_replace_df(df,'col2',"^(\d)",r'7eleven_\1')
    d = {'col1': [1, 2], 'col2': ['7eleven_39774', '7eleven_23265']}
    df_expected = pd.DataFrame(data=d)
    assert_dfs_equal(df_expected,df_actual,"")

def test_regex_replace_df3():
    d = {'col1': [1, 2], 'col2': ['39774', '23265']}
    df = pd.DataFrame(data=d)
    compiled_regex=re.compile("^(\d)")
    df_actual = dataframe_utils.regex_replace_df(df,'col2',compiled_regex,r'7eleven_\1')
    d = {'col1': [1, 2], 'col2': ['7eleven_39774', '7eleven_23265']}
    df_expected = pd.DataFrame(data=d)
    assert_dfs_equal(df_expected,df_actual,"")

if __name__=="__main__":
    test_regex_replace_df3()
