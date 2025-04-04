#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import logging
import json, copy
import time
from tqdm import tqdm
from pathlib import Path

from utils import file_loader, file_utils
from utils.transformations import TransformationPipeline
from utils.metadata_parser import MetaDataParser, METADATA, METADATA_PARSER, METADATA_UPDATER
from utils.config_objects import brand_info, remove_comments
from utils.constants import FILE_EXTENSION

"""
This class is supposed to load data from a file into a dataframe & transform it to a standard format
given a input configuration
"""


meta_data_parser = MetaDataParser(brand_info)

def __load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_common_config(config_path):
    common_table_config = __load_config(config_path)
    assert type(common_table_config)==dict
    __assert_keys_are_valid(common_table_config, "for common config")
    return  common_table_config

def load_table_specific_configs(config_path):
    table_configs = __load_config(config_path)
    assert type(table_configs)==list
    for (table_index,table_config) in enumerate(table_configs):
        __assert_keys_are_valid(table_config, f"for table specific config, table_index {table_index}")
    return table_configs

def _get_config_keys():
    keys = []
    keys.extend(TransformationPipeline._get_config_keys())
    keys.extend(file_loader._get_config_keys())
    keys.extend(MetaDataParser._get_config_keys())
    return keys

def get_common_config_template():
    common_config = {}
    common_config.update(TransformationPipeline.get_transformations_template())
    common_config.update(file_loader.get_file_loader_template())
    return common_config

def get_table_specific_config_template():
    table_specific_config = []
    table_specific_config.append(get_common_config_template())
    return table_specific_config

def __assert_keys_are_valid(config, message):
    additional_keys_in_config = set(config.keys()) - set(_get_config_keys())
    assert len(additional_keys_in_config)==0, f"additional keys present they are {additional_keys_in_config}, expected {_get_config_keys()} {message}"

def process_file(filepath, common_table_config=None, table_configs=None, **kwargs):
    """
    Process a file - csv or xlsx in the final cleaned up data
    Arguments:
        filepath:str
            path to the file
        common_table_config:dict or str
            A dictionary or path to json with common config required for all tables in this file
            This contains 3 main level keys - table_loading_config, transformations_metarows, transformations, metadata
            refer - resources/test/sales_data_variants/multi_table_file_common_config.json
            or resources/test/sales_data_variants/multi_sheet_file_common_config.json
            Eg:- {
                "table_loading_config": {
                    "header": [
                        0,
                        1
                    ]
                },
                "transformations_metarows": {
                    "0": [
                        {
                            "function": "skip"
                        }
                    ]
                },
                "transformations": [
                            {
                                "function": "transpose",
                                "df": "$x",
                                "fixed_column_indices": [
                                    0
                                ],
                                "headers_to_transpose": [
                                    0
                                ],
                                "transposed_header_names": [
                                    "store_id_address"
                                ]
                            },..]
            }
        table_configs:list or str
            Its a list of dict or path to the a json containing the same, containing table specific config for each table in the file
            refer - resources/test/sales_data_variants/multi_table_file_all_tables_config.json
            or resources/test/sales_data_variants/multi_sheet_file_config.json
            Inside the list of dict, the dict keys are same as common_table_config. i.e
            table_loading_config, transformations_metarows, transformations, metadata
            Eg:-
            [
                {
                    "metadata": [
                        {
                            "id": "meta_0",
                            "value": "Eckrich - Meijer Store Level",
                            "row_number": 0,
                            "column_number": 0
                        },
                        {
                            "id": "meta_1",
                            "value": "Time:Week Ending 05-17-20",
                            "row_number": 1,
                            "column_number": 0
                        }
                    ],
                    "table_loading_config": {
                        "start_row": 2,
                        "end_row": 6
                    }
                },..
            ]
    """
    start_time = time.time()
    assert not (common_table_config is None and table_configs is None)
    common_table_config, table_configs = get_common_and_tables_config_objects(common_table_config, table_configs)

    common_table_config = copy.deepcopy(common_table_config)
    table_configs = copy.deepcopy(table_configs)
    df_list = []
    __assert_keys_are_valid(common_table_config, "for common config")
    for (table_index,table_config) in enumerate(table_configs):
        __assert_keys_are_valid(table_config, f"for table specific config, table_index {table_index}")
        df = file_loader.load_dataframe(filepath, common_table_config, table_config)
        # loading metadata values if not already loaded based upon row_number & column_number in config
        metadata_configs = table_config.get(METADATA, [])
        metadata_updater_config = common_table_config.get(METADATA_PARSER, {}).get(METADATA_UPDATER, {})
        metadata_updater_config = remove_comments(metadata_updater_config)
        # TODO maybe add common configs metadata also & find value from file, currently there was
        # no use case for this scenario so only taking metavalues from table config for updating
        metadata_configs = file_loader.update_metacell_values(filepath, metadata_configs, **metadata_updater_config)
        table_config['metadata'] = metadata_configs

        # transforming the data
        transformation_pipeline = TransformationPipeline(common_config=common_table_config, table_config=table_config)
        df = transformation_pipeline.transform(df)
        df_list.append(df)
    df_complete = pd.concat(df_list)
    end_time = time.time()
    logging.debug(f"time taken for process_file function {end_time - start_time} for file {filepath}")
    return df_complete

def parse_metadata(input_filepath:str, common_config, table_configs, df:pd.DataFrame=None):
    """
    Get metadata from a dataframe & or from file
    This method uses the - uitls.meta_data_parser.parse_file_metadata
    It uses an initialized metadata_parser object
        Arguments:
            input_filepath:str
                filepath to the file which was processed
            common_config:str, dict
                common processing config for the file.
                Either path to the config or the actual loaded json config
            table_configs:str, list
                tables processing config for the file.
                Either path to the config or the actual loaded json config
            df:pandas.DataFrame
    Refer utils.metadata_parser for more details
    """
    return meta_data_parser.parse_file_metadata(input_filepath, common_config, table_configs, df)

def __get_or_raw_processed_filepath(original_filepath:str, common_table_config, table_configs,
                                    df_processed:pd.DataFrame=None,
                                    filename_prefix = "store_level_sales", filename_suffix="raw",
                                    data_folder:str=file_utils.get_raw_data_folder(), filename_extension=None):
    common_table_config, table_configs = get_common_and_tables_config_objects(common_table_config, table_configs)
    if df_processed is None:
        df_processed = process_file(original_filepath, common_table_config, table_configs)

    file_metadata = meta_data_parser.parse_file_metadata(original_filepath, common_table_config, table_configs, df_processed)

    if filename_extension is None:
        # in case of processed output file we want to keep it as csv even if the input file type is xlsx
        filename_extension = file_metadata.get(FILE_EXTENSION)
    raw_filepath = file_utils.get_filepath_to_store(data_folder, filename_prefix,
                                                    file_metadata, filename_suffix, filename_extension)
    return raw_filepath



def get_raw_filepath(original_filepath:str, common_table_config, table_configs, df_processed:pd.DataFrame=None,
                     raw_data_folder:str=file_utils.get_raw_data_folder(), filename_prefix = "store_level_sales"):
    """
    Get the file path where the raw data have to be copied
    Arguments:
        original_filepath:str
        common_table_config:str, dict
                common processing config for the file.
                Either path to the config or the actual loaded json config
        table_configs:str, list
            tables processing config for the file.
            Either path to the config or the actual loaded json config
        df_processed:pandas.DataFrame
            If a processed dataframe is available pass that
        raw_data_folder:str
            The main raw folder which we want to use
    """

    return __get_or_raw_processed_filepath(original_filepath, common_table_config, table_configs, df_processed,
                                           filename_prefix = filename_prefix, filename_suffix="raw",
                                           data_folder=raw_data_folder)

def get_processed_filepath(original_filepath:str, common_table_config, table_configs, df_processed:pd.DataFrame=None,
                           processed_data_folder:str=file_utils.get_processed_data_folder(),
                           filename_prefix = "store_level_sales", filename_extension="csv.bz2"):
    """
    Get the file path where the raw data have to be copied
    Arguments:
        original_filepath:str
        common_table_config:str, dict
                common processing config for the file.
                Either path to the config or the actual loaded json config
        table_configs:str, list
            tables processing config for the file.
            Either path to the config or the actual loaded json config
        df_processed:pandas.DataFrame
            If a processed dataframe is available pass that
        processed_data_folder:str
            The main raw folder which we want to use
    """
    return __get_or_raw_processed_filepath(original_filepath, common_table_config, table_configs, df_processed,
                                           filename_prefix=filename_prefix, filename_suffix="processed",
                                           data_folder=processed_data_folder, filename_extension=filename_extension)

def identify_and_copy_files(filepaths, common_table_config, table_configs, replace=True):
    """
    This method is to used to identify a file & copy it to the appropriate folder
    All of them are expected to have the same preprocessing steps
    """
    files_with_issue = []
    raw_data_folder = file_utils.get_raw_data_folder() # TODO maybe even pick it from function params
    for original_filepath in tqdm(filepaths):
        try:
            # processing raw file to a standardized form
            df_processed = process_file(original_filepath, common_table_config, table_configs)
            file_metadata = meta_data_parser.parse_file_metadata(original_filepath, common_table_config, table_configs, df_processed)
            filename_prefix = "store_level_sales"
            filename_suffix = "raw"
            filename_extension = file_metadata.get(FILE_EXTENSION)
            raw_filepath = file_utils.get_filepath_to_store(raw_data_folder, filename_prefix,
                                                            file_metadata, filename_suffix, filename_extension)
            file_utils.copy_file(original_filepath,raw_filepath, replace)
            logging.debug(f'raw_filepath {raw_filepath}')

            processed_data_folder = file_utils.get_processed_data_folder()
            filename_extension = "csv.bz2"
            processed_filepath = file_utils.get_filepath_to_store(processed_data_folder, filename_prefix, file_metadata, filename_suffix, filename_extension)
            processed_filepath = Path(processed_filepath)
            file_utils.create_parent_dirs(processed_filepath)

            if replace or not processed_filepath.exists():
                df_processed.to_csv(processed_filepath,index=False)
                logging.info(f'processed filepath {processed_filepath}')
            else:
                logging.info(f'processed filepath already exists & not copied {processed_filepath}')
        except Exception as e:
            # https://stackoverflow.com/questions/5191830/how-do-i-log-a-python-error-with-debug-information
            logging.exception(f'error occurred while processing {original_filepath} skipping & proceeding {e}')
            files_with_issue.append(original_filepath)

    return files_with_issue


def get_common_and_tables_config_objects(common_table_config, table_configs):
    """
    if python objects for common_config & tables_config if a filepath is given
    """
    # https://stackoverflow.com/questions/58647584/how-to-test-if-object-is-a-pathlib-path
    if type(common_table_config) == str or isinstance(common_table_config, Path):
        # loading the json config, if the variable is a path
        common_table_config = load_common_config(common_table_config)
    if type(table_configs) == str or isinstance(table_configs, Path):
        # loading the json config, if the variable is a path
        table_configs = load_table_specific_configs(table_configs)
    return common_table_config, table_configs