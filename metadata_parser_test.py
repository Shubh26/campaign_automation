import pytest

import os, json
import pandas as pd

from utils.config_objects import BrandInfo
from utils.metadata_parser import MetaDataParser
from utils.config_objects import brand_info_file, FileInfo
from utils.file_utils import custom_file_configs_folder
from utils import constants
from utils.date_utils import get_date, DATE_FORMAT_ISO

from utils.test_utils import assert_dfs_equal, test_resources_folder, sales_data_variants_folder

def test_parse_metadata():
    brand_info = BrandInfo(brand_info_file)
    meta_data_parser = MetaDataParser(brand_info=brand_info)
    extracted_filepath = os.path.join(sales_data_variants_folder,"eckrich_kroger_store_level_sales_sample.csv")
    eckrich_processed_filepath = os.path.join(sales_data_variants_folder,
                                              "eckrich_kroger_store_level_sales_sample_processed.csv")
    common_config_path = os.path.join(custom_file_configs_folder, "smithfield_kroger_common_config.json")
    tables_config_path = os.path.join(sales_data_variants_folder, "smithfield_kroger_eckrich_sample_config.json")
    with open(common_config_path, 'r') as f:
        common_config = json.load(f)
    with open(tables_config_path, 'r') as f:
        tables_config = json.load(f)
    df_processed = pd.read_csv(eckrich_processed_filepath, parse_dates=[constants.DATE, constants.WEEK_COL],
                              dtype={constants.ZIPCODE_COL: str})
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    fileinfo = meta_data_parser.parse_file_metadata(extracted_filepath, common_config, tables_config,  df=df_processed)
    file_meta_dict = {"file_extension":"csv",
        "retail_chain": "kroger", "brand": "eckrich",
                      "client": "smithfield",
                      "start_date": get_date("2020-08-02",DATE_FORMAT_ISO),
     "end_date": get_date("2020-08-08",DATE_FORMAT_ISO)
                      }
    fileinfo_expected = FileInfo(extracted_filepath, **file_meta_dict)
    assert fileinfo_expected==fileinfo

if __name__=="__main__":
    test_parse_metadata()