import pytest
import pandas as pd
import os
import json
from utils import transformations

from utils import constants
from utils.file_utils import custom_file_configs_folder

from utils.test_utils import assert_dfs_equal

sales_data_variants_folder = "resources/test/sales_data_variants"

def test_transform_data():
    multi_table_filepath = os.path.join(sales_data_variants_folder, "multi_table_file.csv")
    multi_table1_processed_filepath = os.path.join(sales_data_variants_folder, "multi_table_file_table1_processed.csv")
    common_config_path = os.path.join(sales_data_variants_folder, "multi_table_file_common_config.json")
    tables_config_path = os.path.join(sales_data_variants_folder, "multi_table_file_all_tables_config.json")
    with open(common_config_path,'r') as f:
        common_config = json.load(f)

    with open(tables_config_path,'r') as f:
        tables_config = json.load(f)

    table1_config = tables_config[0]
    columns_expected_ordered = ['store_id', 'address', 'store_id_before_banner', 'product', 'week', 'sales_dollar']
    transformation_pipeline = transformations.TransformationPipeline(common_config=common_config, table_config=table1_config)

    df = pd.read_csv(multi_table_filepath, header=[0,1], skiprows=2, nrows=3)
    df_out = transformation_pipeline.transform(df)
    df_expected = pd.read_csv(multi_table1_processed_filepath, parse_dates=[constants.WEEK_COL])
    assert_dfs_equal(df_expected, df_out, multi_table1_processed_filepath)

def test_dummy_evaluate_expression():
    out = transformations.evaluate_expression("1+2")
    assert out==3

def test_empty_config():
    df = pd.DataFrame(data={"c1": [1, 2, 3], "c2": ["a", "b", "c"]})
    transformation_pipeline = transformations.TransformationPipeline(common_config={}, table_config={})
    # ensuring that there is no exception thrown
    df_out = transformation_pipeline.transform(df)
    assert_dfs_equal(df, df_out, None)

def test_transform_metadata1():
    table1_config =     {
        "metadata": [
            {
                "id": "meta_0",
                "value": "Ad Week 2022.30",
                "row_number": 0,
                "column_number": 0
            }
        ]
    }
    common_config = {
    "transformations_metarows": {
        "meta_0": [
            {
                "function": "regex_replace",
                "text": "$x",
                "pattern": "Ad Week\\s*",
                "replacement": ""
            },
            {
                "function": "strip",
				"text":"$x"
            },
            {
                "function": "regex_replace",
                "text": "$x",
                "pattern": "(\\d{4}).*",
                "replacement": "\\1"
            },
            {
                "function": "convert_to_int",
                "text": "$x",
                "function_output": "year"
            },
            {
                "function": "evaluate_expression",
                "expression": "$year -1"
            },
            {
                "function": "add_column",
                "column_name": "year"
            }
        ]
    },
    "transformations": []}
    df = pd.DataFrame(data={"c1":[1,2,3], "c2":["a","b","c"]})
    transformation_pipeline = transformations.TransformationPipeline(common_config=common_config, table_config=table1_config)
    df_expected = df.copy()
    df_expected["year"] = 2021
    df_out = transformation_pipeline.transform(df)
    assert_dfs_equal(df_expected, df_out, "")

def test_transform_jewel_data():
    jewel_data_filepath = os.path.join(sales_data_variants_folder, "kretschmar_jewel_week_30_store_level_data.xlsx")
    jewel_processed_filepath = os.path.join(sales_data_variants_folder, "kretschmar_jewel_week_30_store_level_data_processed.csv")
    common_config_path = os.path.join(custom_file_configs_folder, "kretschmar_jewel_common_config.json")
    tables_config_path = os.path.join(custom_file_configs_folder, "kretschmar_jewel_file_config.json")
    with open(common_config_path,'r') as f:
        common_config = json.load(f)

    with open(tables_config_path,'r') as f:
        tables_config = json.load(f)

    table1_config = tables_config[0]
    columns_expected_ordered = ['store_id', 'address', 'store_id_before_banner', 'product', 'week', 'sales_dollar']
    table1_config["metadata"][0]["value"] = "Ad Week 2022.30"
    # https://stackoverflow.com/questions/31391275/using-like-inside-pandas-query
    transformation_pipeline = transformations.TransformationPipeline(common_config=common_config, table_config=table1_config)

    df = pd.read_excel(jewel_data_filepath, skiprows=8, usecols="B:M")
    df_out = transformation_pipeline.transform(df)
    # df_out.to_csv(jewel_processed_filepath, index=False)
    df_expected = pd.read_csv(jewel_processed_filepath, parse_dates=['week'])
    assert_dfs_equal(df_expected, df_out, jewel_processed_filepath)

def test_transformation_config_creator_common1():
    transformations_config_creator = transformations.TransformationsConfigCreator()
    function_input = {
			"function": "standardize_dataframe",
			"df": "$x"
		}
    transformations_config_creator.append_common_transformation(function_input)
    common_config = transformations_config_creator.get_common_config()
    common_config_expected = {"transformations": [{"function": "standardize_dataframe", "df": "$x"}]}
    assert common_config_expected == common_config
    tables_config = transformations_config_creator.get_table_configs()
    assert tables_config == []

    # even if the same dictionary for function_input is modified it should not modify the already added transformations
    function_input["function"] = "standardize_column_names"

    transformations_config_creator.append_common_transformation(function_input)
    common_config = transformations_config_creator.get_common_config()
    common_config_expected = {"transformations": [{"function": "standardize_dataframe", "df": "$x"},
                                                  {"function": "standardize_column_names", "df": "$x"}]}
    assert common_config_expected == common_config

def test_transformation_config_creator_common_meta1():
    transformations_config_creator = transformations.TransformationsConfigCreator()
    function_input = {
                "function": "regex_replace",
                "text": "$x",
                "pattern": ".*Week Ending",
                "replacement": ""
            }
    with pytest.raises(AssertionError):
        # metadata_id should be provided for metalevel transformation addition
        transformations_config_creator.append_common_transformation(function_input,
                                                                    transformations.TransformationLevel.meta_level)

    transformations_config_creator.append_common_transformation(function_input,
                                                                transformations.TransformationLevel.meta_level,
                                                                metadata_id="meta_0")
    common_config = transformations_config_creator.get_common_config()
    common_config_expected = {
        "transformations_metarows": {
            "meta_0":[{
                    "function": "regex_replace",
                    "text": "$x",
                    "pattern": ".*Week Ending",
                    "replacement": ""
                }]
            }
        }
    assert common_config_expected == common_config
    tables_config = transformations_config_creator.get_table_configs()
    assert tables_config == []

    # even if the same dictionary for function_input is modified it should not modify the already added transformations
    function_input["pattern"] = "dummy_regex"

    transformations_config_creator.append_common_transformation(function_input,
                                                                transformations.TransformationLevel.meta_level,
                                                                metadata_id="meta_0")
    common_config = transformations_config_creator.get_common_config()
    common_config_expected = {
        "transformations_metarows": {
            "meta_0":[{
                    "function": "regex_replace",
                    "text": "$x",
                    "pattern": ".*Week Ending",
                    "replacement": ""
                },
                {
                    "function": "regex_replace",
                    "text": "$x",
                    "pattern": "dummy_regex",
                    "replacement": ""
                }
            ]
            }
        }
    assert common_config_expected == common_config


def test_transformation_config_creator_table_specific1():
    # adding table specific transformation
    transformations_config_creator = transformations.TransformationsConfigCreator()
    function_input = {
			"function": "standardize_dataframe",
			"df": "$x"
		}

    transformations_config_creator.append_transformation_table_specific(function_input, 0)
    common_config = transformations_config_creator.get_common_config()
    common_config_expected = {}
    assert common_config_expected == common_config
    tables_config = transformations_config_creator.get_table_configs()
    tables_config_expected = [{"transformations": [{"function": "standardize_dataframe", "df": "$x"}]}]
    assert tables_config_expected == tables_config
    # even if the same dictionary for function_input is modified it should not modify the already added transformations
    function_input["function"] = "standardize_column_names"

    transformations_config_creator.append_transformation_table_specific(function_input, 0)
    tables_config = transformations_config_creator.get_table_configs()
    tables_config_expected = [{"transformations": [{"function": "standardize_dataframe", "df": "$x"},
                                                   {"function": "standardize_column_names", "df": "$x"}]}]
    assert tables_config_expected == tables_config

def test_transformation_config_creator_table_specific_meta1():
    # adding table specific transformation
    transformations_config_creator = transformations.TransformationsConfigCreator()
    function_input = {
                "function": "regex_replace",
                "text": "$x",
                "pattern": ".*Week Ending",
                "replacement": ""
            }
    with pytest.raises(AssertionError):
        # metadata_id should be provided for metalevel transformation addition
        transformations_config_creator.append_transformation_table_specific(function_input, 0,
                                                                    transformations.TransformationLevel.meta_level)
    transformations_config_creator.append_transformation_table_specific(function_input, 0,
                                                                        transformations.TransformationLevel.meta_level,
                                                                        "meta_0")
    common_config = transformations_config_creator.get_common_config()
    common_config_expected = {}
    assert common_config_expected == common_config
    tables_config = transformations_config_creator.get_table_configs()
    tables_config_expected = [{
        "transformations_metarows": {
            "meta_0":[{
                    "function": "regex_replace",
                    "text": "$x",
                    "pattern": ".*Week Ending",
                    "replacement": ""
                }
            ]
            }
        }]
    assert tables_config_expected == tables_config
    # even if the same dictionary for function_input is modified it should not modify the already added transformations
    function_input["pattern"] = "dummy_regex"

    transformations_config_creator.append_transformation_table_specific(function_input, 0,
                                                                        transformations.TransformationLevel.meta_level,
                                                                        "meta_0")
    tables_config = transformations_config_creator.get_table_configs()
    tables_config_expected = [{
        "transformations_metarows": {
            "meta_0":[{
                    "function": "regex_replace",
                    "text": "$x",
                    "pattern": ".*Week Ending",
                    "replacement": ""
                },
                {
                    "function": "regex_replace",
                    "text": "$x",
                    "pattern": "dummy_regex",
                    "replacement": ""
                }
            ]
            }
        }]
    assert tables_config_expected == tables_config

def test_execute_sql():
    df = pd.DataFrame(data={"c1": [1, 2, 3, 4], "c2": ["a", "b", "c", "1_b2"]})
    df_expected = pd.DataFrame(data={"c1": [1, 3], "c2": ["a", "c"]})
    common_config = {"transformations": [
        {
            "function": "read_sql",
            "sql": "select c1,c2 from dummy_table where c2 not like '%b%'",
            "df": "$x",
            "table_name": "dummy_table"

        }]}
    transformation_pipeline = transformations.TransformationPipeline(common_config=common_config, table_config={})
    # ensuring that there is no exception thrown
    df_out = transformation_pipeline.transform(df)
    assert_dfs_equal(df_expected, df_out, None)

if __name__=="__main__":
    #test_execute_sql()
    test_transform_data()