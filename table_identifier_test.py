import pytest
import os, json
from utils.table_identifier import TableIdentifier
from utils.config_objects import TableInfo

# TODO this is not the best solution change this
test_resources_folder=os.path.join('resources','test')
sales_data_variants_folder = "resources/test/sales_data_variants"

def test_identify_tables1():
    multi_table_filepath = os.path.join(sales_data_variants_folder, "multi_table_file.csv")
    table_identifier = TableIdentifier()
    out = table_identifier._identify_tables_get_json_output(multi_table_filepath)

    # print(json.dumps(out))
    # with open(os.path.join(sales_data_variants_folder,"multi_table_file_all_tables_config.json"),'w') as f:
    #     json.dump(out[0],f, indent=4)

    out_expected = json.loads('[{"metadata": [{"id": "meta_0", "value": "Eckrich - Meijer Store Level", "row_number": 0, "column_number": 0}, {"id": "meta_1", "value": "Time:Week Ending 05-17-20", "row_number": 1, "column_number": 0}], "table_loading_config": {"start_row": 2, "end_row": 6}}, {"metadata": [{"id": "meta_0", "value": "Eckrich - Meijer Store Level", "row_number": 9, "column_number": 0}, {"id": "meta_1", "value": "Time:Week Ending 05-24-20", "row_number": 10, "column_number": 0}], "table_loading_config": {"start_row": 11, "end_row": 15}}, {"metadata": [{"id": "meta_0", "value": "Eckrich - Meijer Store Level", "row_number": 18, "column_number": 0}, {"id": "meta_1", "value": "Time:Week Ending 05-31-20", "row_number": 19, "column_number": 0}], "table_loading_config": {"start_row": 20, "end_row": 24}}, {"metadata": [{"id": "meta_0", "value": "Eckrich - Meijer Store Level", "row_number": 27, "column_number": 0}, {"id": "meta_1", "value": "Time:Week Ending 06-07-20", "row_number": 28, "column_number": 0}], "table_loading_config": {"start_row": 29, "end_row": 33}}, {"metadata": [{"id": "meta_0", "value": "Eckrich - Meijer Store Level", "row_number": 36, "column_number": 0}, {"id": "meta_1", "value": "Time:Week Ending 06-14-20", "row_number": 37, "column_number": 0}], "table_loading_config": {"start_row": 38, "end_row": 42}}, {"metadata": [{"id": "meta_0", "value": "Eckrich - Meijer Store Level", "row_number": 45, "column_number": 0}, {"id": "meta_1", "value": "Time:Week Ending 06-21-20", "row_number": 46, "column_number": 0}], "table_loading_config": {"start_row": 47, "end_row": 51}}, {"metadata": [{"id": "meta_0", "value": "Eckrich - Meijer Store Level", "row_number": 54, "column_number": 0}, {"id": "meta_1", "value": "Time:Week Ending 06-28-20", "row_number": 55, "column_number": 0}], "table_loading_config": {"start_row": 56, "end_row": 60}}, {"metadata": [{"id": "meta_0", "value": "Eckrich - Meijer Store Level", "row_number": 63, "column_number": 0}, {"id": "meta_1", "value": "Time:Week Ending 07-05-20", "row_number": 64, "column_number": 0}], "table_loading_config": {"start_row": 65, "end_row": 69}}, {"metadata": [{"id": "meta_0", "value": "Eckrich - Meijer Store Level", "row_number": 72, "column_number": 0}, {"id": "meta_1", "value": "Time:Week Ending 07-12-20", "row_number": 73, "column_number": 0}], "table_loading_config": {"start_row": 74, "end_row": 78}}]')
    assert out_expected==out

def test_identify_tables2():
    multi_table_filepath = os.path.join(sales_data_variants_folder, "header_to_transpose.csv")
    table_identifier = TableIdentifier()
    out = table_identifier._identify_tables_get_json_output(multi_table_filepath)

    #print(json.dumps(out))
    out_expected = json.loads('[{"metadata": [{"id": "meta_0", "value": "Eckrich CDS Publix backdata for 24.7", "row_number": 0, "column_number": 0}, {"id": "meta_1", "value": "Time:52 WE 7-25-21", "row_number": 1, "column_number": 0}, {"id": "meta_2", "value": "Brand:ECKRICH", "row_number": 2, "column_number": 0}], "table_loading_config": {"start_row": 3, "end_row": 8}}]')
    assert out_expected==out

def test_identify_tables3():
    multi_table_filepath = os.path.join(sales_data_variants_folder, "multi_table_file.csv")
    table_identifier = TableIdentifier()
    out = table_identifier.identify_tables(multi_table_filepath)
    out_expected_json = json.loads('[{"metadata": [{"id": "meta_0", "value": "Eckrich - Meijer Store Level", "row_number": 0, "column_number": 0}, {"id": "meta_1", "value": "Time:Week Ending 05-17-20", "row_number": 1, "column_number": 0}], "table_loading_config": {"start_row": 2, "end_row": 6}}, {"metadata": [{"id": "meta_0", "value": "Eckrich - Meijer Store Level", "row_number": 9, "column_number": 0}, {"id": "meta_1", "value": "Time:Week Ending 05-24-20", "row_number": 10, "column_number": 0}], "table_loading_config": {"start_row": 11, "end_row": 15}}, {"metadata": [{"id": "meta_0", "value": "Eckrich - Meijer Store Level", "row_number": 18, "column_number": 0}, {"id": "meta_1", "value": "Time:Week Ending 05-31-20", "row_number": 19, "column_number": 0}], "table_loading_config": {"start_row": 20, "end_row": 24}}, {"metadata": [{"id": "meta_0", "value": "Eckrich - Meijer Store Level", "row_number": 27, "column_number": 0}, {"id": "meta_1", "value": "Time:Week Ending 06-07-20", "row_number": 28, "column_number": 0}], "table_loading_config": {"start_row": 29, "end_row": 33}}, {"metadata": [{"id": "meta_0", "value": "Eckrich - Meijer Store Level", "row_number": 36, "column_number": 0}, {"id": "meta_1", "value": "Time:Week Ending 06-14-20", "row_number": 37, "column_number": 0}], "table_loading_config": {"start_row": 38, "end_row": 42}}, {"metadata": [{"id": "meta_0", "value": "Eckrich - Meijer Store Level", "row_number": 45, "column_number": 0}, {"id": "meta_1", "value": "Time:Week Ending 06-21-20", "row_number": 46, "column_number": 0}], "table_loading_config": {"start_row": 47, "end_row": 51}}, {"metadata": [{"id": "meta_0", "value": "Eckrich - Meijer Store Level", "row_number": 54, "column_number": 0}, {"id": "meta_1", "value": "Time:Week Ending 06-28-20", "row_number": 55, "column_number": 0}], "table_loading_config": {"start_row": 56, "end_row": 60}}, {"metadata": [{"id": "meta_0", "value": "Eckrich - Meijer Store Level", "row_number": 63, "column_number": 0}, {"id": "meta_1", "value": "Time:Week Ending 07-05-20", "row_number": 64, "column_number": 0}], "table_loading_config": {"start_row": 65, "end_row": 69}}, {"metadata": [{"id": "meta_0", "value": "Eckrich - Meijer Store Level", "row_number": 72, "column_number": 0}, {"id": "meta_1", "value": "Time:Week Ending 07-12-20", "row_number": 73, "column_number": 0}], "table_loading_config": {"start_row": 74, "end_row": 78}}]')
    out_expected = TableInfo.create_table_infos(out_expected_json)
    assert out_expected==out

def test_identify_tables_xlsx1():
    multi_table_filepath = os.path.join(sales_data_variants_folder, "multi_sheet_file_with_transpose.xlsx")
    table_identifier = TableIdentifier()
    out = table_identifier._identify_tables_get_json_output(multi_table_filepath)

    # print(json.dumps(out))
    out_expected = json.loads('[{"metadata": [], "table_loading_config": {"start_row": 7, "end_row": 43, "sheet_name": "Bubly 16oz (Arizona)", "header": [0, 1]}}, {"metadata": [], "table_loading_config": {"start_row": 7, "end_row": 57, "sheet_name": "Bubly 20oz (Holiday States)", "header": [0, 1]}}]')
    assert out_expected==out


def test_identify_tables4_eckrich():
    multi_table_filepath = os.path.join(sales_data_variants_folder, "eckrich_kroger_store_level_sales_sample.csv")
    table_identifier = TableIdentifier()
    out = table_identifier._identify_tables_get_json_output(multi_table_filepath)

    print(json.dumps(out))
    # with open(os.path.join(sales_data_variants_folder,"multi_table_file_all_tables_config.json"),'w') as f:
    #     json.dump(out[0],f, indent=4)

    out_expected = json.loads("""[{"metadata": [{"id": "meta_0", "value": "Eckrich Store Level Sales: 8/12/2020 3:35:17 PM Eastern Standard Time", "row_number": 0, "column_number": 0}, {"id": "meta_1", "value": "Division(s)        :  'All Divisions`", "row_number": 1, "column_number": 0}, {"id": "meta_2", "value": "Days               :  'From: 8/2/2020 to 8/8/2020`", "row_number": 2, "column_number": 0}, {"id": "meta_3", "value": "Level:   'Consumer UPC`", "row_number": 3, "column_number": 0}, {"id": "meta_4", "value": "GTINs              :  '25 Items`", "row_number": 4, "column_number": 0}], "table_loading_config": {"start_row": 6, "end_row": 19}}]""")
    assert out_expected==out

# def test_identify_tables5_jewel():
#     filepath = os.path.join(sales_data_variants_folder, "kretschmar_jewel_week_30_store_level_data.xlsx")
#     table_identifier = TableIdentifier()
#     out = table_identifier._identify_tables_get_json_output(filepath)
#
#     print(json.dumps(out))
#     # with open(os.path.join(sales_data_variants_folder,"multi_table_file_all_tables_config.json"),'w') as f:
#     #     json.dump(out[0],f, indent=4)
#
#     out_expected = json.loads("""[{"metadata": [{"id": "meta_0", "value": "Eckrich Store Level Sales: 8/12/2020 3:35:17 PM Eastern Standard Time", "row_number": 0, "column_number": 0}, {"id": "meta_1", "value": "Division(s)        :  'All Divisions`", "row_number": 1, "column_number": 0}, {"id": "meta_2", "value": "Days               :  'From: 8/2/2020 to 8/8/2020`", "row_number": 2, "column_number": 0}, {"id": "meta_3", "value": "Level:   'Consumer UPC`", "row_number": 3, "column_number": 0}, {"id": "meta_4", "value": "GTINs              :  '25 Items`", "row_number": 4, "column_number": 0}], "table_loading_config": {"start_row": 6, "end_row": 19}}]""")
#     assert out_expected==out

if __name__=="__main__":
    test_identify_tables4_eckrich()