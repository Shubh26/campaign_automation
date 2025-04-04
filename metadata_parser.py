import json
import re
import logging
from pathlib import Path

import pandas as pd

from utils import constants, date_utils, file_loader, file_utils
from utils.config_objects import MetaDataCell, BrandInfo, FileInfo, TableInfo, START_ROW, START_DATE, END_DATE, RETAIL_CHAIN, BRAND, CLIENT, remove_comments
from utils.transformations import METADATA

METADATA_PARSER = "metadata_parser"
METADATA_UPDATER = "metadata_updater"

class MetaDataParser(object):
    def __init__(self, brand_info):
        self.brand_info = brand_info
        self.brand_compiled_regex = {brand: re.compile(brand_regex, flags=re.IGNORECASE) for brand, brand_regex in
                                     brand_info.get_brand_regex_mapping().items()}
    @staticmethod
    def _get_config_keys():
        return [METADATA_PARSER]

    def _identify_date_cols(self, df):
        """
        returns a list of tuple(index_no,col_name) with date in it
        Arguments:
        """
        column_names = df.columns
        date_columns = [(index, col) for index, col in enumerate(column_names)
                        if ('date' in col.lower() or date_utils.is_column_date(df,col))]
        return date_columns


    def _identify_date_col_name(self, index_date_column):
        """
        index_date_column : [(index,col_name)...]
        """
        ## raise appropriate exception
        if len(index_date_column) == 0:
            raise Exception(f"couldn't find a date column in headers -  {index_date_column}")

        if len(index_date_column) > 1:
            raise Exception(f'ambiguous date column {index_date_column}')

        if len(index_date_column) == 1:
            return index_date_column[0][1]

    def __get_required_field_level1(self):
        return [RETAIL_CHAIN, CLIENT, BRAND]

    def __get_missing_fields_level1(self, file_meta_dict):
        required_fields_level1 = self.__get_required_field_level1()
        missing_keys = set(required_fields_level1) - set(file_meta_dict.keys())
        return missing_keys

    def __is_required_fields_level1_found(self, file_meta_dict):
        missing_keys = self.__get_missing_fields_level1(file_meta_dict)
        return len(missing_keys)==0

    def parse_file_metadata(self, input_filepath:str, common_config, table_configs, df:pd.DataFrame):
        """
        Get metadata from a dataframe & or from file
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

        """
        t = df.copy()
        file_meta_dict = {}
        DATE_COLUMN="date_column"
        metadata_parser_config = common_config.get(METADATA_PARSER, {})
        metadata_parser_config = remove_comments(metadata_parser_config)
        file_meta_dict[constants.FILE_EXTENSION] = file_utils.get_file_type(input_filepath)

        ### initially iterating through the config to see if brand, client, retail chain info is present
        # going through common config for getting these values
        metadata_configs = common_config.get(METADATA, [])
        metadatas = MetaDataCell.create_list_of_metadata(metadata_configs)
        file_meta_dict = self.__update_file_metadata_from_metacell(file_meta_dict, metadatas)
        # iterating through table specific config to see if these values are present
        if not self.__is_required_fields_level1_found(file_meta_dict):
            for table_config in table_configs:
                metadata_configs = table_config.get(METADATA, [])
                metadatas = MetaDataCell.create_list_of_metadata(metadata_configs)
                file_meta_dict = self.__update_file_metadata_from_metacell(file_meta_dict, metadatas)



        ### checking if required values are present somewhere in the file
        if not self.__is_required_fields_level1_found(file_meta_dict):
            index_row_iterator = file_loader.get_index_row_iterator_complete_file(input_filepath)
            for i, row in index_row_iterator:
                row_text = ','.join(map(str,row))
                logging.debug(f'parsing data for extracting metadata {row}')
                file_meta_dict = self.__update_values_level1_from_text(file_meta_dict, row_text)
                if self.__is_required_fields_level1_found(file_meta_dict):
                    # if atleast one value is missing in a large file, then this loop would be time consuming
                    # TODO if we know that meta information is present only in the initial portion of file, then we could break early
                    break

        ### checking if required values are present in filename
        ## will update only values which were not present from previous option, i.e from common_config, tables_config & file content
        if not self.__is_required_fields_level1_found(file_meta_dict):
            filename = Path(input_filepath).name
            logging.info(f"couldn't identify brand info from file config, checking filename {filename}")
            file_meta_dict = self.__update_values_level1_from_text(file_meta_dict, filename)

        __sample_meta_values = [
                                {
                                    "id": "retail_chain",
                                    "value": "circlek"
                                },
                                {
                                    "id": "brand",
                                    "value": "bubly"
                                },
                                {
                                    "id": "client",
                                    "value": "pepsico"
                                }]

        if not self.__is_required_fields_level1_found(file_meta_dict):
            missing_keys = self.__get_missing_fields_level1(file_meta_dict)
            metadata_values_to_show = [d for d in __sample_meta_values if d["id"] in missing_keys]
            __sample_meta_config_to_show = {}
            __sample_meta_config_to_show["metadata"] = metadata_values_to_show
            raise Exception(f"Could not identify the following values, please provide them in the config {missing_keys}. Eg:- {json.dumps(__sample_meta_config_to_show)}")
        # next is to identify the start & end of week, in some files this information is present
        # in the small metadata section before the actual table
        # But here i'm extracting the date info from the actual table

        date_col = metadata_parser_config.get(DATE_COLUMN, None)
        if not date_col:
            # if date_column name is not configured in config, then try to identify it
            date_col = self._identify_date_col_name(self._identify_date_cols(t))
        if not date_utils.is_column_date(t, date_col):
            date_format = date_utils.find_date_format_df(t, date_col)
            t = date_utils.convert_column_to_date(t, date_col, date_format)

        (start_date_string,end_date_string) = date_utils.get_start_end_dates(t, date_col, return_date_obj=True)
        if start_date_string==end_date_string:
            #TODO finish functionality
            # here it would mean that data is present for a week & we have to find week start & end
            unique_dates = t[date_col].unique()
            sample_date_from_week = date_utils.get_date(unique_dates[0], date_format)
            date_utils.find_start_of_week(sample_date_from_week)
            date_utils.find_end_of_week(file_meta_dict[START_DATE])

        file_meta_dict[START_DATE] = start_date_string
        file_meta_dict[END_DATE] = end_date_string
        fileinfo = FileInfo(input_filepath, **file_meta_dict)
        return fileinfo

    def __update_if_value_not_present(self, file_meta_dict, key, value):
        if key not in file_meta_dict:
            file_meta_dict[key] = value
        return file_meta_dict

    def __update_values_level1_from_text(self, file_meta_dict, value):
        for brand_name, brand_regex in self.brand_compiled_regex.items():
            if brand_regex.search(value):
                file_meta_dict = self.__update_if_value_not_present(file_meta_dict, BRAND, brand_name)
                # updating client as well
                file_meta_dict = self.__update_if_value_not_present(file_meta_dict, CLIENT,
                                                                    self.brand_info.get_client_name(brand_name))
        return file_meta_dict

    def __update_file_metadata_from_metacell(self, file_meta_dict, metadatas):
        # initially checking if required key is present as metadata id
        for metadata in metadatas:
            for required_meta_id in self.__get_required_field_level1():
                if metadata.get_id() == required_meta_id:
                    file_meta_dict = self.__update_if_value_not_present(file_meta_dict, required_meta_id, metadata.get_value())

        # checking if any of values in metadata cell correspond to brand name or client name
        for metadata in metadatas:
            value = metadata.get_value()
            file_meta_dict = self.__update_values_level1_from_text(file_meta_dict, value)
        return file_meta_dict


def validate_file_metadata(fileinfo):
    required_fields=['client', 'brand', 'retail_chain', START_DATE, END_DATE]