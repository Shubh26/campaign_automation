import pytest
import pandas as pd
import os,json,copy
from pathlib import Path
import time
import logging
import cProfile

from utils import file_utils, file_processor, constants, date_utils, config_objects
from utils.file_utils import custom_file_configs_folder

from utils.test_utils import assert_dfs_equal, sales_data_variants_folder


def test_process_file():
    multi_table_filepath = os.path.join(sales_data_variants_folder, "multi_table_file.csv")
    multi_table1_processed_filepath = os.path.join(sales_data_variants_folder, "multi_table_file_table1_processed.csv")
    common_config_path = os.path.join(sales_data_variants_folder, "multi_table_file_common_config.json")
    tables_config_path = os.path.join(sales_data_variants_folder, "multi_table_file_all_tables_config.json")
    with open(common_config_path, 'r') as f:
        common_config = json.load(f)
    with open(tables_config_path, 'r') as f:
        tables_config = json.load(f)
    # only taking the 1st table configuration & processing it
    table1_config = tables_config[0]
    df_out = file_processor.process_file(multi_table_filepath, common_config, [table1_config])
    df_expected = pd.read_csv(multi_table1_processed_filepath, parse_dates=[constants.WEEK_COL])
    assert_dfs_equal(df_expected, df_out, multi_table1_processed_filepath)

def test_multi_sheet_file1():

    multi_sheet_filepath = os.path.join(sales_data_variants_folder, "multi_sheet_file_with_transpose.xlsx")
    multi_joined_table_processed_filepath = os.path.join(sales_data_variants_folder,
                                                   "multi_sheet_file_with_transpose_processed.csv")
    common_config_path = os.path.join(sales_data_variants_folder, "multi_sheet_file_common_config.json")
    table_specific_config_path = os.path.join(sales_data_variants_folder, "multi_sheet_file_config.json")
    with open(common_config_path, 'r') as f:
        common_config = json.load(f)
    with open(table_specific_config_path, 'r') as f:
        table_specific_configs = json.load(f)
    df_out = file_processor.process_file(multi_sheet_filepath, common_config, table_specific_configs)
    df_expected = pd.read_csv(multi_joined_table_processed_filepath, parse_dates=[constants.WEEK_COL])
    assert_dfs_equal(df_expected, df_out, multi_joined_table_processed_filepath)

def test_multi_sheet_file_path_to_config1():
    # similar to test_multi_sheet_file1
    multi_sheet_filepath = os.path.join(sales_data_variants_folder, "multi_sheet_file_with_transpose.xlsx")
    multi_joined_table_processed_filepath = os.path.join(sales_data_variants_folder,
                                                   "multi_sheet_file_with_transpose_processed.csv")
    common_config_path = os.path.join(sales_data_variants_folder, "multi_sheet_file_common_config.json")
    table_specific_config_path = os.path.join(sales_data_variants_folder, "multi_sheet_file_config.json")

    df_out = file_processor.process_file(multi_sheet_filepath, common_config_path, table_specific_config_path)
    df_expected = pd.read_csv(multi_joined_table_processed_filepath, parse_dates=[constants.WEEK_COL])
    assert_dfs_equal(df_expected, df_out, multi_joined_table_processed_filepath)


def test_process_file_eckrich1():
    eckrich_filepath = os.path.join(sales_data_variants_folder, "eckrich_kroger_store_level_sales_sample.csv")
    eckrich_processed_filepath = os.path.join(sales_data_variants_folder, "eckrich_kroger_store_level_sales_sample_processed.csv")
    common_config_path = os.path.join(custom_file_configs_folder, "smithfield_kroger_common_config.json")
    tables_config_path = os.path.join(sales_data_variants_folder, "smithfield_kroger_eckrich_sample_config.json")
    with open(common_config_path, 'r') as f:
        common_config = json.load(f)
    with open(tables_config_path, 'r') as f:
        tables_config = json.load(f)
    # only taking the 1st table configuration & processing it
    table1_config = tables_config[0]
    #TODO complete test case
    df_out = file_processor.process_file(eckrich_filepath, common_config, [table1_config])
    # df_out.to_csv(eckrich_processed_filepath, index=False)
    df_expected = pd.read_csv(eckrich_processed_filepath, parse_dates=[constants.DATE, constants.WEEK_COL],
                              dtype={constants.ZIPCODE_COL:str})
    assert_dfs_equal(df_expected, df_out, eckrich_processed_filepath)

def test_squashed_header_file1():
    # similar to test_multi_sheet_file1
    header_squashed_filepath = os.path.join(sales_data_variants_folder, "header_squashed.csv")
    header_squashed_processed_filepath = os.path.join(sales_data_variants_folder,
                                                   "header_squashed_processed.csv")
    common_config_path = os.path.join(sales_data_variants_folder, "header_squashed_common_config.json")
    table_specific_config_path = os.path.join(sales_data_variants_folder, "header_squashed_tables_config.json")

    df_out = file_processor.process_file(header_squashed_filepath, common_config_path, table_specific_config_path)
    df_out = df_out.reset_index(drop=True)
    df_expected = pd.read_csv(header_squashed_processed_filepath, parse_dates=[constants.WEEK_COL])
    assert_dfs_equal(df_expected, df_out, header_squashed_processed_filepath)

def test_transform_jewel_data1():
    start_time = time.time()
    jewel_data_filepath = os.path.join(sales_data_variants_folder, "kretschmar_jewel_week_30_store_level_data.xlsx")
    jewel_processed_filepath = os.path.join(sales_data_variants_folder, "kretschmar_jewel_week_30_store_level_data_processed.csv")
    common_config_path = os.path.join(custom_file_configs_folder, "kretschmar_jewel_common_config.json")
    tables_config_path = os.path.join(custom_file_configs_folder, "kretschmar_jewel_file_config.json")
    # https://stackoverflow.com/questions/31391275/using-like-inside-pandas-query
    df_out = file_processor.process_file(jewel_data_filepath, common_config_path, tables_config_path)
    # df_out.to_csv(jewel_processed_filepath, index=False)
    df_expected = pd.read_csv(jewel_processed_filepath, parse_dates=['week'])
    assert_dfs_equal(df_expected, df_out, jewel_processed_filepath)
    end_time = time.time()
    time_taken = end_time - start_time
    assert time_taken<5

def test_transform_jewel_data2():
    start_time = time.time()
    jewel_data_filepath = os.path.join(sales_data_variants_folder, "kretschmar_jewel_week_30_store_level_data.xlsx")
    jewel_processed_filepath = os.path.join(sales_data_variants_folder, "kretschmar_jewel_week_30_store_level_data_processed.csv")
    common_config_path = os.path.join(custom_file_configs_folder, "kretschmar_jewel_common_config_v1.json")
    tables_config_path = os.path.join(custom_file_configs_folder, "kretschmar_jewel_file_config.json")
    # https://stackoverflow.com/questions/31391275/using-like-inside-pandas-query
    df_out = file_processor.process_file(jewel_data_filepath, common_config_path, tables_config_path)
    # df_out.to_csv(jewel_processed_filepath, index=False)
    df_expected = pd.read_csv(jewel_processed_filepath, parse_dates=['week'])
    assert_dfs_equal(df_expected, df_out, jewel_processed_filepath)
    end_time = time.time()
    time_taken = end_time - start_time
    assert time_taken < 5

def test_get_raw_filepath1():
    eckrich_filepath = os.path.join(sales_data_variants_folder, "eckrich_kroger_store_level_sales_sample.csv")
    common_config_path = os.path.join(custom_file_configs_folder, "smithfield_kroger_common_config.json")
    tables_config_path = os.path.join(sales_data_variants_folder, "smithfield_kroger_eckrich_sample_config.json")
    # converting to Path so that we can the test on windows & linux machines. In windows path separation is backslash "\"
    raw_filepath_expected = Path(r"/data/cac/sales_data/raw/smithfield/eckrich/kroger/2020/08/store_level_sales_smithfield_eckrich_kroger_2020-08-02_2020-08-08_raw.csv")
    raw_filepath = file_processor.get_raw_filepath(eckrich_filepath, common_config_path, tables_config_path)
    raw_filepath = Path(raw_filepath)
    assert raw_filepath_expected==raw_filepath

def test_get_processed_filepath1():
    eckrich_filepath = os.path.join(sales_data_variants_folder, "eckrich_kroger_store_level_sales_sample.csv")
    common_config_path = os.path.join(custom_file_configs_folder, "smithfield_kroger_common_config.json")
    tables_config_path = os.path.join(sales_data_variants_folder, "smithfield_kroger_eckrich_sample_config.json")
    # converting to Path so that we can the test on windows & linux machines. In windows path separation is backslash "\"
    processed_filepath_expected = Path(r"/data/cac/sales_data/processed/smithfield/eckrich/kroger/2020/08/store_level_sales_smithfield_eckrich_kroger_2020-08-02_2020-08-08_processed.csv.bz2")
    processed_filepath = file_processor.get_processed_filepath(eckrich_filepath, common_config_path, tables_config_path)
    processed_filepath = Path(processed_filepath)
    assert processed_filepath_expected==processed_filepath
    # checking with a different extension
    processed_filepath_expected = Path(
        r"/data/cac/sales_data/processed/smithfield/eckrich/kroger/2020/08/store_level_sales_smithfield_eckrich_kroger_2020-08-02_2020-08-08_processed.csv")
    processed_filepath = file_processor.get_processed_filepath(eckrich_filepath, common_config_path, tables_config_path, filename_extension="csv")
    processed_filepath = Path(processed_filepath)
    assert processed_filepath_expected == processed_filepath

def test_parse_metadata_and_raw_processed_filepath_with_config_content1():
    filepath = "dummy.xlsx"
    common_config = {
        "metadata": [{
            "id": "retail_chain",
            "value": "kroger"
        },
        {
            "id": "brand",
            "value": "nathans"
        }
        ]
    }
    df = pd.DataFrame(data={"c1": [1, 2, 3], "date": ["2021-08-16", "2021-08-22", "2021-08-17"]})
    df['date'] = pd.to_datetime(df['date'])
    metadata_expected_dict = {'filename': 'dummy.xlsx', 'file_extension': 'xlsx',
                              'retail_chain': 'kroger',
                              'brand': 'nathans',
                              'client': 'smithfield',
                              'start_date': date_utils.get_date("2021-08-16"),
        'end_date': date_utils.get_date("2021-08-22")}
    metadata_expected = config_objects.FileInfo(**metadata_expected_dict)
    metadata = file_processor.parse_metadata(filepath, common_config, {}, df)

    assert metadata_expected == metadata

    def __get_expected_filepath(file_category="raw", extension="csv"):
        return Path(
            rf"/data/cac/sales_data/{file_category}/smithfield/nathans/kroger/2021/08/store_level_sales_smithfield_nathans_kroger_2021-08-16_2021-08-22_{file_category}.{extension}")
    raw_filepath_expected = __get_expected_filepath("raw", "xlsx")
    raw_filepath = file_processor.get_raw_filepath(filepath, common_config, {}, df)
    raw_filepath = Path(raw_filepath)
    assert raw_filepath_expected==raw_filepath

    processed_filepath_expected = __get_expected_filepath("processed", "csv.bz2")
    processed_filepath = file_processor.get_processed_filepath(filepath, common_config, {}, df)
    processed_filepath = Path(processed_filepath)
    assert processed_filepath_expected==processed_filepath

def __get_file_content(brand):
    return f"{brand} Store Level Sales: 8/12/2020 3:35:17 PM Eastern Standard Time,\n" +\
            "Division(s)        :  'All Divisions`,\n" +\
            "Days               :  'From: 8/2/2020 to 8/8/2020`,\n" +\
            "Level:   'Consumer UPC`,\n" +\
            "GTINs              :  '25 Items`,\n" +\
            ",,,,,,,\n" +\
            "date,dummy_header\n" +\
            "2021-08-16,d1\n" +\
            "2021-08-22,d2\n" +\
            "2021-08-17,d2\n"

def test_parse_metadata_and_raw_processed_filepath_with_config_content2():
    # metadata extracted from file content, if file content have info then filename is not used
    filepath = os.path.join(sales_data_variants_folder, "test_parse_metadata_bubly.csv")
    brand_client_mapping = {
        "eckrich":config_objects.SMITHFIELD,
        "nathans":config_objects.SMITHFIELD,
        "pure_farmland":config_objects.SMITHFIELD,
        "bubly":config_objects.PEPSICO,
        "lipton":config_objects.PEPSICO
    }

    brands_text = ["Eckrich", "Nathans", "Pure Farmland"]
    brands = ["eckrich", "nathans", "pure_farmland"]
    for brand, brand_text in zip(brands, brands_text):
        client = brand_client_mapping.get(brand)
        file_content = __get_file_content(brand_text)
        with open(filepath, 'w') as f:
            f.write(file_content)
        common_config = {
            "metadata": [{
                "id": "retail_chain",
                "value": "kroger"
            }
            ]
        }

        df = pd.read_csv(filepath, skiprows=6)
        df['date'] = pd.to_datetime(df['date'])
        metadata_expected_dict = {'filename': filepath, 'file_extension': 'csv',
                                  'retail_chain': 'kroger',
                                  'brand': brand,
                                  'client': client,
                                  'start_date': date_utils.get_date("2021-08-16"),
            'end_date': date_utils.get_date("2021-08-22")}
        metadata_expected = config_objects.FileInfo(**metadata_expected_dict)
        metadata = file_processor.parse_metadata(filepath, common_config, {}, df)
        print(f"metadata {metadata.__dict__}")
        print(f"metadata_expected {metadata_expected.__dict__}")

        assert metadata_expected == metadata

        def __get_expected_filepath(file_category="raw", extension="csv"):
            return Path(
            rf"/data/cac/sales_data/{file_category}/{client}/{brand}/kroger/2021/08/store_level_sales_{client}_{brand}_kroger_2021-08-16_2021-08-22_{file_category}.{extension}")
        raw_filepath_expected = __get_expected_filepath("raw")
        raw_filepath = file_processor.get_raw_filepath(filepath, common_config, {}, df)
        raw_filepath = Path(raw_filepath)
        assert raw_filepath_expected==raw_filepath

        processed_filepath_expected = __get_expected_filepath("processed", "csv.bz2")
        processed_filepath = file_processor.get_processed_filepath(filepath, common_config, {}, df)
        processed_filepath = Path(processed_filepath)
        assert processed_filepath_expected == processed_filepath
    file_utils.delete_file(filepath)

def test_parse_metadata_and_raw_filepath_with_config_content3():
    # metadata extracted from filename, in this case
    brand_client_mapping = {
        "eckrich":config_objects.SMITHFIELD,
        "nathans":config_objects.SMITHFIELD,
        "pure_farmland":config_objects.SMITHFIELD,
        "bubly":config_objects.PEPSICO,
        "lipton":config_objects.PEPSICO
    }
    brands_text = ["Eckrich", "Nathans", "Pure Farmland"]
    brands = ["eckrich", "nathans", "pure_farmland"]
    for brand, brand_text in zip(brands, brands_text):
        client = brand_client_mapping.get(brand)
        filepath = os.path.join(sales_data_variants_folder, f"test_parse_metadata_{brand}.csv")
        file_content = __get_file_content("")
        with open(filepath, 'w') as f:
            f.write(file_content)
        common_config = {
            "metadata": [{
                "id": "retail_chain",
                "value": "kroger"
            }
            ]
        }

        df = pd.read_csv(filepath, skiprows=6)
        df['date'] = pd.to_datetime(df['date'])
        metadata_expected_dict = {'filename': filepath, 'file_extension': 'csv',
                                  'retail_chain': 'kroger',
                                  'brand': brand,
                                  'client': client,
                                  'start_date': date_utils.get_date("2021-08-16"),
            'end_date': date_utils.get_date("2021-08-22")}
        metadata_expected = config_objects.FileInfo(**metadata_expected_dict)
        metadata = file_processor.parse_metadata(filepath, common_config, {}, df)

        assert metadata_expected == metadata

        def __get_expected_filepath(file_category="raw", extension="csv"):
            return Path(
            rf"/data/cac/sales_data/{file_category}/{client}/{brand}/kroger/2021/08/store_level_sales_{client}_{brand}_kroger_2021-08-16_2021-08-22_{file_category}.{extension}")
        raw_filepath_expected = __get_expected_filepath("raw")
        raw_filepath = file_processor.get_raw_filepath(filepath, common_config, {}, df)
        raw_filepath = Path(raw_filepath)
        assert raw_filepath_expected==raw_filepath

        processed_filepath_expected = __get_expected_filepath("processed", "csv.bz2")
        processed_filepath = file_processor.get_processed_filepath(filepath, common_config, {}, df)
        processed_filepath = Path(processed_filepath)
        assert processed_filepath_expected == processed_filepath
        file_utils.delete_file(filepath)

if __name__=="__main__":
    test_transform_jewel_data1()
    # https://stackoverflow.com/questions/582336/how-can-you-profile-a-python-script
    # cProfile.run('test_transform_jewel_data1()')