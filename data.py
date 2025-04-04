import pandas as pd
import logging

import utils.dataframe_utils
from utils import date_utils,file_utils,zipcode_utils
from utils.constants import *

CAMPAIGN_START = 'campaign_start'
CAMPAIGN_END = 'campaign_end'

class Data(object):
    def __init__(self,**kwargs):
        """
        This is a abstract class with some functions
        """
        super(Data,self).__init__()

        # columns to use in store list
        # BANNER_COL & RETAIL_CHAIN are not present for every store list
        self._columns_expanded = copy.deepcopy(STORE_LIST_EXPANDED_COLUMNS)
        self._columns = copy.deepcopy(STORE_LIST_COLUMNS)

        # list of files with store details for each of the clients
        self.store_list_files = []
        # this have key as client id & value as dataframe with sales
        self._sales_data_dict = {}
        # this have key as client id & value as dataframe with population
        self._population_data_dict = {}

        # this have key as client id & value as a dataframe with store details
        self._store_list_dict = {}

        # this dataframe have the 5 miles expanded list of stores
        self._df_expanded = None


        # if the new sales file doesn't have this info it will help in mapping, TODO think about implementation
        # Structure {client_id : {old_id:new_id}}, keeping per client as the old numeric client id could be same between clients
        # Structure {client_id : df[[STORE_ID_COL,STORE_ID_BEFORE_BANNER_COL]]}, keeping per client as the old numeric client id could be same between clients
        self._old_store_id_to_new_mapping_per_client = {}

    def load(self):
        self.load_store_list()

    def load_store_list(self):
        df_list = []
        for client_id,filepath in enumerate(self.store_list_files):
            logging.info(f'loading store list file {filepath} client_id {client_id}')
            t = pd.read_csv(filepath, dtype={VALIDATED_COL:bool, ZIPCODE_COL:str, ZIPCODE_EXPANDED:str})
            t = zipcode_utils.get_zipcode_helper().get_dataframe_with_additional_columns(t, additional_columns=[CITY_COL, STATE_COL])

            # this standardizes column names & some of the zipcode values as well
            t = utils.dataframe_utils.StandardizeData.standardize_dataframe(t)

            df_expanded = t[self._columns_expanded]
            df_list.append(df_expanded)

            # filter out & retain only original store zipcodes
            t = utils.dataframe_utils.get_store_list_original(t, IS_ORIGINAL_ZIPCODE_COL)
            df = t
            logging.info(f'no of stores - {len(df[STORE_ID_COL].unique())} for file {filepath}')

            t = zipcode_utils.zipcode_helper.retain_only_validated_rows(t)

            df = t[self._columns]
            # this is used to maintain a list of stores per client
            self._store_list_dict[client_id] = df
            # adding population data even for the ones which have sales (eventhough it's not necessary for the current test control logic)
            self._add_population_data(client_id, df)
            logging.info(f'no of stores after removing invalid zipcode cases for dv360 - {len(df[STORE_ID_COL].unique())} for file {filepath}')
        self._df_expanded = pd.concat(df_list)
        df = self._df_expanded
        logging.info(f'total no of stores across clients including invalid for dv360 ones - {len(df[STORE_ID_COL].unique())}')

    def __old_store_id_to_new_id_mapping(self,df):
        t = df.copy()
        # if data type of store_id column is not string then definitely there is not banner in the store id
        # if t.dtypes[STORE_ID_COL] != np.str:
        return t

    # def __does_store_id_have

    def __validate_unique_store_ids(self):
        """
        This function validates whether the store ids are unique across various clients
        """


    def set_store_list_files(self,files):
        """
        Add the file path to the store lists here

        Arguments:
        files: list of strings
            List of file paths
        """
        self.store_list_files = files

    def set_control_reserved_zipcodes_files(self,files):
        """
        Add file paths to all control reserved zipcode files

        Arguments:
        files: list of string
            filepath to csv files which have zipcodes to be included in control
            Mandatory field - "zipcode"
        """
        self._control_reserved_zipcode_files = files

    def set_test_reserved_zipcodes_files(self,files):
        """
        Add file paths to all control reserved zipcode files

        Arguments:
        files: list of string
            filepath to csv files which have zipcodes to be included in test
            Mandatory field - "zipcode"

        """
        self._test_reserved_zipcode_files = files

    def set_columns_expanded(self,columns):
        """
        Set column names to be used in the expanded dataframe
        Arguments:
        columns: list of strings
            column headers to keep for expanded dataframe
            default: constants.STORE_LIST_EXPANDED_COLUMNS
        """
        self._columns_expanded = columns

    def set_columns(self,columns):
        """
        Set column names to be used in the store list dataframe
        Arguments:
        columns: list of strings
            column headers to keep for expanded dataframe
            default: constants.STORE_LIST_COLUMNS
        """
        self._columns = columns

    def add_sales_data(self, client_id, df):
        """
        This function will be used by child classes to add sales dataframes
        """
        self._sales_data_dict[client_id] = df


    def _add_population_data(self, client_id, df):
        """
        This function will be used internally to add population column to the store list file
        """
        t = df.copy()
        self._population_data_dict[client_id] = zipcode_utils.get_zipcode_helper().get_dataframe_with_population_column(t)


    def get_sales_data(self, client_id, period='baseline'):
        """
        Get the sales data for the specified client_id
        Arguments:
        client_id:int
            client_id assigned to a store for this campaign
            Eg: for a campaign Jewel stores was assigned client_id=0 & Albertsons stores client_id=1
        period:string
            This is used to decide which rows of sales will be retained according to time of the campaign
            4 possible options
            baseline - will return the sales data used for test-control generation, i.e before the campaign start
            live - will return the sales data during the campaign
            post - will return data post the campaign period
            analysis - will return data during campaign + data using the 4 week post campaign analysis period
            complete - will provide the entire data without filters

        """

        # TODO decide whether to throw exception of give a default
        df = self._sales_data_dict[client_id]
        # TODO this will not work if we are using end of sales week
        return df.copy()


    def get_population_data(self,client_id):
        # TODO decide whether to throw exception of give a default
        df = self._population_data_dict[client_id]
        return df.copy()

    def get_store_list(self, client_id):
        df = self._store_list_dict[client_id]
        return df.copy()


    def get_store_list_expanded(self, radius=2):
        """
        Returns a df with store list expanded by specified radius & filtered using is_original_zipcode & validated_col
        Args:
        radius: radius to expand to (default:2)
        """
        return utils.dataframe_utils.get_store_list_expanded(self._df_expanded, radius)
