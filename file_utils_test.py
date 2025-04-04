from utils import file_utils, date_utils, constants
from utils.config_objects import FileInfo
import pandas as pd
import os, json
from pathlib import Path

from utils.test_utils import test_resources_folder

standardize_file = os.path.join(test_resources_folder,'standardize_test_file.csv')
standardize_file2 = os.path.join(test_resources_folder,'standardize_test_file2.csv')

def test_get_filename():
    filename_prefix = "store_level_sales"
    client_name = "smithfield"
    brand_name = "eckrich"
    retail_chain = "kroger"
    start_date = date_utils.get_date("2021-09-19",date_utils.DATE_FORMAT_ISO)
    end_date = date_utils.get_date("2021-09-25",date_utils.DATE_FORMAT_ISO)
    filename_suffix = None
    filename_expected = "store_level_sales_smithfield_eckrich_kroger_2021-09-19_2021-09-25.csv"
    filename = file_utils.get_filename(filename_prefix, client=client_name, brand=brand_name,
                            retail_chain=retail_chain, start_date=start_date, end_date=end_date, filename_suffix=filename_suffix, filename_extension= "csv")
    # print(filename)
    assert filename_expected==filename

def test_get_filepath_to_store1():
    filename_prefix = "store_level_sales"
    client_name = "smithfield"
    brand_name = "eckrich"
    retail_chain = "kroger"
    start_date = date_utils.get_date("2021-09-19", date_utils.DATE_FORMAT_ISO)
    end_date = date_utils.get_date("2021-09-25", date_utils.DATE_FORMAT_ISO)
    filename_suffix = None
    filepath_expected = Path(r"/data/cac/sales_data/raw/smithfield/eckrich/kroger/2021/09/store_level_sales_smithfield_eckrich_kroger_2021-09-19_2021-09-25.csv")
    file_metadata = {constants.CLIENT:client_name,
                     constants.BRAND: brand_name,
                     constants.RETAIL_CHAIN: retail_chain,
                     constants.START_DATE: start_date,
                     constants.END_DATE: end_date}
    file_info = FileInfo(filename=None, **file_metadata)
    filepath = file_utils.get_filepath_to_store(file_utils.get_raw_data_folder(), filename_prefix, file_info, filename_suffix, "csv")
    filepath = Path(filepath)
    # print(filepath)
    assert filepath_expected == filepath


def test_save_json_format_for_z3():
    store_expanded_with_ave_path = os.path.join(test_resources_folder,"store_expanded_with_ave_sales.csv")
    store_expanded_with_ave_output_path = os.path.join(test_resources_folder,"store_expanded_with_ave_sales_output.json")
    store_expanded_with_ave_expected_output_path = os.path.join(test_resources_folder,"store_expanded_with_ave_sales_expected_output.json")
    df = pd.read_csv(store_expanded_with_ave_path, dtype={constants.ZIPCODE_COL:str, constants.ZIPCODE_EXPANDED:str})

    file_utils.save_json_format_for_z3(df, store_expanded_with_ave_output_path,
                            store_id_col=constants.STORE_ID_COL,
                            sales_col=constants.SALES_DOLLAR_COL,
                            zipcode_col=constants.ZIPCODE_COL,
                            zipcode_expanded_col=constants.ZIPCODE_EXPANDED,
                            radius_col=constants.RADIUS_COL,
                            validated_col=constants.VALIDATED_COL,
                            is_original_zipcode_col=constants.IS_ORIGINAL_ZIPCODE_COL)

    expected_output = {}
    with open(store_expanded_with_ave_expected_output_path, "r") as f:
        expected_output = json.load(f)

    actual_output = {}
    with open(store_expanded_with_ave_output_path, "r") as f:
        actual_output = json.load(f)

    assert expected_output==actual_output
    file_utils.delete_file(store_expanded_with_ave_output_path)

def test_get_non_duplicate_filename():
    filename = "test_file.csv"

    output_expected = "test_file.csv"
    output = file_utils.get_non_duplicate_filename(filename,[])
    assert output_expected==output

    output_expected = "test_file_v1.csv"
    output = file_utils.get_non_duplicate_filename(filename, ["test_file.csv"])
    assert output_expected == output


if __name__=="__main__":
    # test_get_filename()
    # test_get_filepath_to_store1()
    # test_save_json_format_for_z3()
    test_get_non_duplicate_filename()

