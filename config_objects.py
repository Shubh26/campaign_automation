import os, json
import pandas as pd
import copy
import logging
from enum import Enum
from pathlib import Path

from utils.constants import CLIENT, BRAND, RETAIL_CHAIN, START_DATE, END_DATE, FILE_EXTENSION
from utils.file_utils import main_resources_folder

TABLE_LOADING_CONFIG = 'table_loading_config'
METADATA = 'metadata'
SHEET_NAME = 'sheet_name'
START_ROW = 'start_row'
END_ROW = 'end_row'
HEADER = 'header'
ROW_NUMBER = "row_number"
COLUMN_NUMBER = "column_number"
COMMENT = "_comment"


#client
SMITHFIELD='smithfield'
PEPSICO = 'pepsico'

# brands
ECKRICH='eckrich'
NATHANS='nathans'
PF='pure_farmland'


# retail chains
KROGER='kroger'
#the default brand dict file, will have to maintain this in a DB or maintained in file here
brand_info_dict = [{'brand':ECKRICH,'client':SMITHFIELD,'display_name':'ECKRICH','brand_regex':'eckrich'},
    {'brand':NATHANS,'client':SMITHFIELD,'display_name':'NATHAN','brand_regex':'nathans'},
    {'brand':PF,'client':SMITHFIELD,'display_name':'PURE FARMLAND','brand_regex':'pure\s+(farmland|fl)'}]

brand_info_file = os.path.join(main_resources_folder,"brand_info.txt")

def write_brand_info(brand_info_dict, brand_info_file):
    with open(brand_info_file, 'w') as f:
        for line in brand_info_dict:
            f.write(json.dumps(line) + '\n')


# uncomment the following line to update the brand_info_fie
# write_brand_info(brand_info_dict,brand_info_file)

class BrandInfoDict(object):
    ## TODO find appropriate name for this class
    def __init__(self, **kwargs):
        super(BrandInfoDict, self).__init__()
        #         for key, value in kwargs.items():
        #             setattr(self,key,value)
        setattr(self, 'brand_client', kwargs['brand_client'])
        setattr(self, 'brand_retail_chain', kwargs['brand_retail_chain'])
        setattr(self, 'brand_display_name', kwargs['brand_display_name'])

    def get_client_name(self, brand):
        return self.brand_client.get(brand)

    def get_retail_chain(self, brand):
        return self.brand_retail_chain.get(brand)

    def get_brand_display_name(self, brand):
        return self.brand_display_name.get(brand, brand.replace('_', ' ').upper())


class BrandInfo(object):
    ## TODO find appropriate name for this class
    def __init__(self, brand_info_file):
        super(BrandInfo, self).__init__()
        # change to debug log
        logging.debug(f"Brandinfo reading file {brand_info_file}, absolute path {Path(brand_info_file).absolute()}")
        self.df_brand_info = pd.read_json(brand_info_file, lines=True)
        self.brand_regex = {d['brand']: d['brand_regex'] for d in
                            self.df_brand_info[['brand', 'brand_regex']].to_dict(orient='records')}
        # setting index as brand for faster querying
        self.df_brand_info.set_index(['brand'], inplace=True)

    def get_client_name(self, brand):
        return self.df_brand_info.loc[brand]['client']

    def get_brand_display_name(self, brand):
        return self.df_brand_info.loc[brand]['display_name']

    def get_brand_regex_mapping(self):
        """
        Returns something like -
         {'eckrich': 'eckrich',
         'nathans': 'nathans',
         'pure_farmland': 'pure\\s+farmland'}
        """
        return self.brand_regex

#TODO put it in appropriate place
brand_info = BrandInfo(brand_info_file)

class FileInfo(object):
    """
    Keep metadata related to a file
    Assumptions - data is available for a week so start of week & end of week are populated
    """

    # https://stackoverflow.com/questions/2466191/set-attributes-from-dictionary-in-python
    # https://codereview.stackexchange.com/questions/171107/python-class-initialize-with-dict
    # using slots method is more efficient but will limit if want to add more attributes in the future
    #     __slots__=['client','brand','retail_chain',START_DATE, END_DATE, FILE_EXTENSION]
    def __init__(self, filename, **kwargs):
        super(FileInfo, self).__init__()
        self.filename = filename
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, FileInfo):
            return self.__dict__ == other.__dict__
        return False

    def get_client_name(self):
        return getattr(self, CLIENT)

    def get_brand_name(self):
        return getattr(self,BRAND)

    def get_retail_chain(self):
        return getattr(self, RETAIL_CHAIN)


    def get_date_col(self):
        return getattr(self, 'date')

    def get_tables(self):
        """
        Get a list of TableInfo
        """
        return getattr(self, 'tables')

    def get(self, key):
        return getattr(self, key)

    def add(self, key, value):
        setattr(self, key, value)

class MetaDataCell(object):
    """
    Keep the metadata value available in file cell
    Generally we see that in some cases there are Metadata values for a table
    above a particular table
    """

    # https://stackoverflow.com/questions/2466191/set-attributes-from-dictionary-in-python
    # https://codereview.stackexchange.com/questions/171107/python-class-initialize-with-dict
    # using slots method is more efficient but will limit if want to add more attributes in the future
    #     __slots__=['id','value','row_number','column_number', 'sheet_name']
    def __init__(self, **kwargs):
        super(MetaDataCell, self).__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, MetaDataCell):
            return self.__dict__ == other.__dict__
        return False

    def get_id(self):
        """
        Get id for this cell
        """
        return getattr(self, 'id', None)

    def get_value(self):
        """
        Get the string value present in a cell
        """
        return getattr(self, 'value', None)

    def set_value(self, value):
        setattr(self, "value", value)

    def is_empty(self):
        return self.get_value() is None

    def get_row_number(self):
        """
        Get row number of the cell
        """
        return getattr(self, 'row_number', -1)

    def get_column_number(self):
        """
        Get column number of the  cell
        """
        return getattr(self, 'column_number', 0)

    def get_sheet_name(self):
        """
        Get sheet_name, appliable only if file is an excel type(xlsx, xls)
        If sheet_name is not specificed the default value will be 0.
        Keeping same behaviour as pd.read_excel - https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html
        """
        return getattr(self, 'sheet_name', 0)

    def get_dict_format(self):
        return self.__dict__

    @staticmethod
    def get_metadata_ids(metadata_maps_common, metadata_maps_table_specific):
        """
        Get the list of metadata ids present in both common_metadata maps & table specific metadata combined
        """
        metadatas = MetaDataCell.create_list_of_metadata_with_overriding(metadata_maps_common, metadata_maps_table_specific)
        return [metadata.get_id() for metadata in metadatas]

    @staticmethod
    def create_list_of_metadata_with_overriding(metadata_maps_common, metadata_maps_table_specific):
        """
        Get a list of metadata cells giving importance to table specific ones
        i.e If there are metadata present with the same id the appearing in table_specific one will
        be given preference. Also if same id is present multiple times in the same list the value
        which appear last will be given preference
        """
        metadata_maps = metadata_maps_common + metadata_maps_table_specific
        return MetaDataCell.create_list_of_metadata(metadata_maps)

    @staticmethod
    def create_list_of_metadata(metadata_maps):
        """
        Get a list of metadata cells, if multiple values are present for the same id,
        this will only keep the last one
        """
        metadatas = [MetaDataCell(**metadata_map) for metadata_map in metadata_maps]
        metadatas_with_id = {}
        for metadata in metadatas:
            id = metadata.get_id()
            metadatas_with_id[id] = metadata

        unique_metadatas = list(metadatas_with_id.values())
        return unique_metadatas

class TableInfo(object):
    """
    Keep metadata related to a table, a file or a sheet can contain multiple tables
    """

    # https://stackoverflow.com/questions/2466191/set-attributes-from-dictionary-in-python
    # https://codereview.stackexchange.com/questions/171107/python-class-initialize-with-dict
    # using slots method is more efficient but will limit if want to add more attributes in the future
    #     __slots__=['metadata',"table_loading_config"]
    # 'table_loading_config' will have the following keys ['start_row','end_row', 'header', 'sheet_name']
    def __init__(self, **kwargs):
        super(TableInfo, self).__init__()
        for key, value in kwargs.items():
            if key=='metadata':
                setattr(self, key, MetaDataCell.create_list_of_metadata(value))
            else:
                setattr(self, key, value)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, TableInfo):
            return self.__dict__ == other.__dict__
        return False

    @staticmethod
    def create_table_infos(table_maps):
        return [TableInfo(**table_map) for table_map in table_maps]

    def get_metadata(self):
        """
        Metadata, It will be a list of MetaDataCell with following values - Eg:-
        [{"meta_0": "Eckrich - Meijer Store Level"}, {"meta_1": "Time:Week Ending 05-17-20"}]
        [{"id": "meta_0", "value": "Eckrich - Meijer Store Level", "row_number": 9, "column_number": 0},
        {"id": "meta_1", "value": "Time:Week Ending 05-24-20", "row_number": 10, "column_number": 0}]
        """
        getattr(self, 'metadata', [])

    def get_table_config(self):
        """
        This returns config required to read the table from a csv file or a excel file.
        Possible keys in the dictionary - 'start_row','end_row', 'header', 'sheet_name'
        """
        table_config = copy.deepcopy(self.__dict__)
        # removing metadata value from output as it's not required to read table (using pandas)
        table_config.pop("metadata", [])
        return table_config


def remove_comments(config_object:dict):
    """
    remove the comments in config_object arguments, this doesn't do a nested removal
    Eg:- remove comments from transformation arguments
    argument names starting with _comment will be removed
    {"_comment":,
    "_comment_1": etc} will be removed
    Arguments:
        config_object:dict
            The config_object for which we want to remove the comments
    """
    # filtering
    return {k:v for (k,v) in config_object.items() if not k.startswith(COMMENT)}