import pytest
import pandas as pd
import os
import json
import copy
import time
import logging

from utils import file_loader
from utils.test_utils import assert_dfs_equal

sales_data_variants_folder = "resources/test/sales_data_variants"

def test_update_metacell_values():
    multi_table_filepath = os.path.join(sales_data_variants_folder, "multi_table_file.csv")
    tables_config_path = os.path.join(sales_data_variants_folder, "multi_table_file_all_tables_config.json")
    with open(tables_config_path, 'r') as f:
        tables_config = json.load(f)
    table1_config = tables_config[0]
    metadata_configs = table1_config.get('metadata', [])
    metadata_configs_expected = copy.deepcopy(metadata_configs)
    # deleting the value so that we can load it again
    del metadata_configs[0]['value']
    metadata_configs = file_loader.update_metacell_values(multi_table_filepath, metadata_configs)
    assert metadata_configs_expected==metadata_configs

def test_load_dataframe_given_config_csv():
    multi_table_filepath = os.path.join(sales_data_variants_folder, "multi_table_file.csv")
    df_expected = pd.read_csv(multi_table_filepath, skiprows=2, nrows=3, header=[0,1])
    table_map = {'metadata': [],
                 "table_loading_config": {
                        "start_row": 2,
                        "end_row": 6
                    }
                 }
    common_config = {
                   "table_loading_config": {
                        "header": [
                            0,
                            1
                        ]
                        }}
    df_out = file_loader.load_dataframe(multi_table_filepath, common_config, table_map)
    pd.testing.assert_frame_equal(df_expected,
                                  df_out)  # pd.testing.assert_frame_equal will show the difference between dataframes as well
    assert df_expected.equals(df_out)

def test_load_dataframe_given_config_xlsx():
    multi_table_xlsx_filepath = os.path.join(sales_data_variants_folder, "multi_sheet_file_with_transpose.xlsx")
    df_expected = pd.read_excel(multi_table_xlsx_filepath, skiprows=7, header=[0,1], sheet_name='Bubly 16oz (Arizona)')
    table_map = {'metadata': [],
                 'table_loading_config': {'start_row': 7,
                  'end_row': 43,
                  'sheet_name': 'Bubly 16oz (Arizona)',
                  'header': [0, 1]}}
    df_out = file_loader.load_dataframe(multi_table_xlsx_filepath, {}, table_map)
    pd.testing.assert_frame_equal(df_expected,
                                  df_out)  # pd.testing.assert_frame_equal will show the difference between dataframes as well
    assert df_expected.equals(df_out)

def test_load_dataframe_given_config_xlsx_jewel():
    filepath = os.path.join(sales_data_variants_folder, "kretschmar_jewel_week_30_store_level_data.xlsx")
    table_map = {'metadata': [],
     'table_loading_config': {'start_row': 8,
                              'sheet_name': 0,
                              'header': [0],
                              "usecols": "B:M"}}
    df_out = file_loader.load_dataframe(filepath,{}, table_map)
    df_expected = pd.read_excel(filepath, skiprows=8, header=[0],usecols='B:M')
    pd.testing.assert_frame_equal(df_expected, df_out)

    # ensuring that it works irrespective of whether it's common config or tables config
    df_out = file_loader.load_dataframe(filepath, table_map, {})
    pd.testing.assert_frame_equal(df_expected, df_out)

    # ensuring that end row parameter also works
    df_expected = pd.read_excel(filepath, skiprows=8, header=[0], usecols='B:M', nrows=2)
    table_map['table_loading_config']['end_row']=10
    df_out = file_loader.load_dataframe(filepath, {}, table_map)
    assert_dfs_equal(df_expected, df_out, f"original path - {filepath}")

def test_add_loading_config_common1():
    file_loader_config_creator = file_loader.FileLoaderConfigCreator({},[])
    file_loader_config_creator.add_common_loading_config(header=[0], start_row=10)
    common_config = file_loader_config_creator.get_common_config()
    common_config_expected = {"table_loading_config":{"header":[0], "start_row":10}}
    assert common_config_expected == common_config

def test_add_loading_config_table_specific1():
    file_loader_config_creator = file_loader.FileLoaderConfigCreator({},[])
    file_loader_config_creator.add_loader_config_table_specific(table_number=0, header=[0], start_row=10)
    tables_config = file_loader_config_creator.get_table_configs()
    tables_config_expected = [{"table_loading_config":{"header":[0], "start_row":10}}]
    assert tables_config_expected == tables_config


def test_get_index_row_iterator_time_taken():
    filepath = os.path.join(sales_data_variants_folder, "kretschmar_jewel_week_30_store_level_data.xlsx")
    start_time = time.time()
    index_row_iterator = file_loader.get_index_row_iterator(filepath, "xlsx", 0, read_only=True)
    count = 0
    for line_no, row in index_row_iterator:
        count+=1
    assert count==6705
    end_time = time.time()
    time_taken1 = end_time - start_time
    assert time_taken1<5, "it should be faster to read & iterate through data in read_only mode"
    start_time = time.time()
    index_row_iterator = file_loader.get_index_row_iterator(filepath, "xlsx", 0)
    count = 0
    for line_no, row in index_row_iterator:
        count += 1
    assert count==6705
    end_time = time.time()
    time_taken2 = end_time - start_time
    # it takes about ~40s to load data in read_only=False mode
    assert time_taken2 > 10, "it is slower to read & iterate through data with read_only mode = False, which is the default"

def test_get_index_row_iterator_merged_cells():
    # test to check values of merged cells
    filepath = os.path.join(sales_data_variants_folder, "merged_cells.xlsx")
    index_row_iterator = file_loader.get_index_row_iterator(filepath, "xlsx", 0, read_only=False)
    count = 0
    data_expected = [
        ['', '', '', '', ''],
        ['header1', 'header2', 'header2', 'header2', 'header3'],
        ['header1', '', '', '', 'header3_1']
    ]
    data = []
    for line_no, row in index_row_iterator:
        count += 1
        data.append(row)
    assert data_expected==data
    count = 0
    data_expected2 = [
        ['', '', '', '', ''],
        ['header1', 'header2', '', '', 'header3'],
        ['', '', '', '', 'header3_1']
    ]
    data = []
    # merged cell values are not available in read_only option
    index_row_iterator = file_loader.get_index_row_iterator(filepath, "xlsx", 0, read_only=True)
    for line_no, row in index_row_iterator:
        count += 1
        data.append(row)
    assert data_expected2 == data

if __name__=="__main__":
    # test_add_loading_config_table_specific1()
    test_get_index_row_iterator_merged_cells()