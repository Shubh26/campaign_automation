import logging
import random
import re
import string
import copy

import numpy as np
import pandas as pd

from utils import text_utils, zipcode_utils
from utils.constants import STORE_ID_COL, STORE_ADDRESS_COL, BANNER_COL, ZIPCODE_COL, ZIPCODE_EXPANDED, \
    SALES_DOLLAR_COL, SALES_UNIT_COL, SALES_VOL_COL, STORE_ID_BEFORE_BANNER_COL, RADIUS_COL, IS_ORIGINAL_ZIPCODE_COL, VALIDATED_COL


def is_empty(df):
    """
    Check if a dataframe is empty
    Arguments:
    df: pandas dataframe

    """
    # https://stackoverflow.com/questions/39337115/testing-if-a-pandas-dataframe-exists
    return (df is None) or df.empty


def rename_headers(df, rename_dict, ignorecase=False):
    """
    This function renames the header & return a dataframe
    Arguments:
    df: pandas dataframe
    rename_dict: a dict
        {old_name:new_name} mapping
    """
    df = df.copy()
    updated_columns = [rename_dict.get(col, col) for col in df.columns]
    if ignorecase:
        rename_dict = {k.lower():v for (k,v) in rename_dict.items()}
        updated_columns = [rename_dict.get(col.lower(), col) for col in df.columns]
    df.columns = updated_columns
    return df


def split_to_columns(df, column_to_split, column_names = [STORE_ID_COL, STORE_ADDRESS_COL], separator=":", drop_original=True):
    """
    df: pandas dataframe
        The pandas dataframe with header that needs to be split.
    column_to_split:str
        Name of column to split
    column_names:list
        List of output column names
    separator:str
        String or regular expression to split on.
        separating character in the column_to_split which needs to be used for splitting a single column
        into multiple columns
    drop_original:boolean
        If we want to drop the original column
    """
    t = df.copy()
    no_of_result_columns = len(column_names)
    assert no_of_result_columns>1, "if no of result columns after splitting is not greater than 1, what is the point of splitting"
    # creating id & address columns
    t_expanded = t[column_to_split].str.split(pat=separator, n=no_of_result_columns, expand=True)
    assert len(t_expanded.columns) == len(column_names),\
        f"expanded column length & length of provided column name list should match they are {len(t_expanded.columns)}, {len(column_names)}"
    t_expanded.columns = column_names
    t = t.assign(**{col: t_expanded[col] for col in t_expanded.columns})
    if drop_original:
        # dropping the original column from dataframe
        t.drop(column_to_split, axis=1, inplace=True)
    return t


def transpose(df, fixed_column_indices=[], headers_to_transpose=[],transposed_header_names=[]):
    """
    This function is used for transposing data from headers to rows
    Check test_processor.py file for examples on how to use it
    Eg: Sample input data
    Product	store1	        store1	    store1
    Product	Dollar Sales	Unit Sales	Volume Sales
    BEEF1	1	            2	        3

    Arguments:
    df: pandas dataframe
        The pandas dataframe with header that needs to be transposed.
        This expects a dataframe with a MultiIndex column header of which atleast 1 row needs to be transposed to rows
    fixed_column_indices:list
        If certain columns have values which we want to retain as it is then add those indicies as list
        starting index = 0
        Eg:- [0] as we want to keep Product column in above example as it is
    headers_to_transpose:list
        a list of header indices that needs to be transposed to rows
        starting index = 0
        Eg:- [0] as we want to convert 1st header row into
    transposed_header_names:list
        list of column names to give to the transposed headers available as rows now
        This should have same size as headers_to_transpose
        Eg:- ["store"]

    """
    assert len(headers_to_transpose) == len(transposed_header_names), \
        f"length of headers_to_transpose & transposed_header_names should match but they are {headers_to_transpose}, {transposed_header_names}"
    t = df.copy()

    # removing the space in the beginning & end
    t.columns = pd.MultiIndex.from_tuples([tuple(c.strip() for c in col_tuple) for col_tuple in t.columns.values])
    t.columns = pd.MultiIndex.from_tuples(
        [tuple(("" if "Unnamed" in c else c) for c in col) for col in t.columns.values])
    t.columns = pd.MultiIndex.from_tuples([tuple(__get_single_header_duplicate_case(col_tuple)) for col_tuple in t.columns.values])

    if len(fixed_column_indices)>0:
        # if fixed columns are present, adding them to index
        index_columns = [v for (i, v) in enumerate(t.columns.values) if i in fixed_column_indices]
        t = t.set_index(index_columns)
        # removing additional level from index, i.e converting index to single level
        t.index.names  = [ ''.join(col) for col in t.index.names]

    # transposing headers to rows, we will we transposing everything except the last remaining header
    assert len(headers_to_transpose)==len(t.columns[0])-1, \
    f"length of headers_to_transpose should 1 less than the no of header rows, i.e it should have been {len(t.columns[0])-1}"
    level_index_start = len(index_columns)

    index_header = [(i,col) for (i,col) in zip(headers_to_transpose,transposed_header_names)]
    # sorting header index & corresponding heading name in ascending order of index
    index_header = sorted(index_header)
    for (i, (header_index, col)) in enumerate(index_header):
        t = t.stack(header_index-i)
    t = t.reset_index()
    column_rename_dict = {f"level_{i+level_index_start}":col for (i,col) in enumerate(transposed_header_names)}
    t.columns = [column_rename_dict.get(col,col) for col in t.columns]
    return t


def copy_columns(df, columns_to_copy):
    """
    Copy the values from a dataframe column to another
    Arguments:
        df: pandas dataframe
            input pandas dataframe
        columns_to_copy:dict
            dict of column names to copy as key & copied name as value
            {column_to_copy:output_column_name}
    """
    t = df.copy()
    for (column_to_copy, output_column) in columns_to_copy.items():
        t[output_column] = t[column_to_copy]
    return t


def filter_query(df, expr, inplace=False, **kwargs):
    """
    Run a filter query on the data, expression should be a boolean expression
    For more details refer - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html
    Arguments:
        df: pandas dataframe
            input pandas dataframe
        expr: str
            The query string to evaluate
        inplace : bool
            Whether the query should modify the data in place or return
            a modified copy.
    """
    return df.query(expr, inplace, **kwargs)


def strip_df(df, column_names):
    """
    Arguments:
        df: pandas dataframe
            input pandas dataframe
        column_name:list
            list of columns to strip
    """
    t = df.copy()
    for column_name in column_names:
        t[column_name] = t[column_name].str.strip()
    return t


def filter_and_order_columns(df, columns):
    """
    returns a dataframe with only the columns provided & ordered using the same list
    Arguments:
        df: pandas dataframe
            input pandas dataframe
        columns:list
            list of column names to order & filter the columns in dataframe
    """
    t = df.copy()
    validate_column_presence(df, columns)
    return t[columns]


def validate_column_presence(df, columns_to_validate):
    """
    Validate if a particular column is present in a dataframe.
    If not show the values which are similar
    Arguments:
        df: pandas dataframe
        columns_to_validate:list
            list of column names which we want to validate
    """
    error_message=''
    for column_to_validate in columns_to_validate:
        if column_to_validate not in df.columns:
            similar_columns = text_utils.identify_similar_values(column_to_validate, df.columns)
            error_message += f"{column_to_validate} is not present, "
            if len(similar_columns)>0:
                error_message += f" did you mean {','.join(similar_columns)}, "

    if len(error_message)>0:
        error_message += f".Complete list of columns {','.join(df.columns)}"
        logging.error(error_message)
        raise Exception(error_message)


def __get_single_header_duplicate_case(column_tuple):
    """
    This function takes in a column tuple & if next entry is the same keep empty space there instead & return a list
    Eg: column_tuple = ('Product', 'Product')
    output = ['Product', '']
    TODO maybe the below one needn't be handled & can just look for duplicate entries
    Eg: column_tuple = ('Product', 'Random_header','Product')
    output = ['Product', 'Random_header','Product']
    """

    out = ['' if (col.strip().lower() == column_tuple[i - 1].strip().lower() and i != 0) else col for i, col in
           enumerate(column_tuple)]
    return out

def __get_duplicated_list(l, max_header_count):
    l = copy.deepcopy(l)
    assert len(l)<=max_header_count
    value_to_duplicate = l[-1]
    if len(l)!=max_header_count:
         l.extend([value_to_duplicate for i in range(max_header_count-len(l))])
    return l


def _convert_columns_to_multi_index_split_based(columns, regex_to_split):
    mcols = [re.split(regex_to_split, col) for col in columns]
    unique_header_count = sorted(set([len(col) for col in mcols]))
    # if unique header count is [1,2] or [1,3] or [1,4] & assuming the split happened properly
    # we can duplicate the value for the case which occured only once.
    # Eg: for
    # [['Geography'],
    #  ['Product'],
    #  ['Dollar Sales (non-projected) ', ' Oct 04, 2020'],
    #  ['Dollar Sales (non-projected) ', ' Nov 01, 2020']]
    # we can duplicate & keep 2 entries i.e ['Geography', 'Geography'] as the 1st value
    assert len(unique_header_count) <= 2
    min_header_count = unique_header_count[0]
    # if min_header_count is 1 & no_of_unique header
    assert len(unique_header_count) == 1 or min_header_count == 1
    max_header_count = unique_header_count[-1]
    if len(unique_header_count) > 1:
        # if some of them have different header count compared to other columns,
        # then make all of them same
        mcols = [__get_duplicated_list(cols, max_header_count) for cols in mcols]

    return mcols


def convert_to_multi_header(df, regex_to_split, regexes_to_select=None, regexes_level=None, strip_column_names=True):
    """
    Given a dataframe with a multiple levels of headers squashed into 1
    Eg:-
    "Product"	"Dollar Sales (non-projected) 4 Weeks Ending Apr 18, 2021"
    Given regex_to_split="4 Weeks Ending", this operation will create a multi level header like below-
    "Product"   "Dollar Sales (non-projected)"
    "Product"   "Apr 18, 2021"
    Arguments:
        df: pandas dataframe
        regex_to_split:str or re.Pattern
            This is the regex based on which we want to split the data

        strip_column_names:boolean
            If we want to strip spaces at the beginning & end of the column names
    """
    # assert regex_to_split is None
    t = df.copy()
    columns = list(t.columns)
    mcols = _convert_columns_to_multi_index_split_based(columns, regex_to_split)
    if strip_column_names:
        mcols = [[col.strip() for col in cols] for cols in mcols]
    mcols = pd.MultiIndex.from_tuples(mcols)
    t.columns = mcols
    return t


class StandardizeData(object):

    def __init__(self,**kwargs):
        print('init')
    #         https://stackoverflow.com/questions/1547145/defining-private-module-functions-in-python
    __digit_regex = re.compile("\d",flags=re.I)
#     __store_regex = re.compile("([a-zA-z_\s]+)\d+",flags=re.I)
    __space_regex = re.compile("\s+")
        # 2 or more space
    __multi_space_regex = re.compile('\s{2,}')

    __sto_start_regex = re.compile('^sto_',flags=re.I)

    @staticmethod
    def standardize_dataframe(df):
        """
        Applies a set of operations on a pandas dataframe to standardize the table

        Operations included:
            1) Standardize the column names
            2) Store 'address' (constants.STORE_ADDRESS_COL) column  if present is lowercased & additional space removed
            3) Store 'banner' (constants.BANNER_COL) column if present if lowercased & spaces replaced with underscore
            4) Standardize 'store_id' (constants.STORE_ID_COL) column if present. Store banner is prepended to the store_id if
            both 'store_id' (constants.STORE_ID_COL) & 'banner' (constants.BANNER_COL) is present,
            else if no 'banner' column, store id is normalized & space replaced with underscore
            5) If column zipcode,zipcode_expanded are present, it will convert that to 5 digit.
                Note: Columns zip, zip_code will be renamed as zipcode in the first step(standardize column name)
        Arguments:
        df: pandas dataframe
            The pandas dataframe to be standardized
        """
        t = df.copy()
        t = StandardizeData.standardize_column_names(t)
        if STORE_ADDRESS_COL in t.columns:
            t = StandardizeData.standardize_store_address_column(t,store_address_col=STORE_ADDRESS_COL)
        if BANNER_COL in t.columns:
            t = StandardizeData.standardize_store_banner_column(t,BANNER_COL)
        if STORE_ID_COL in t.columns and BANNER_COL in t.columns:
            t = StandardizeData.standardize_store_id_column(t,store_id_col=STORE_ID_COL,store_banner_col=BANNER_COL)
        elif STORE_ID_COL in t.columns:
            t = StandardizeData.standardize_store_id_column(t,store_id_col=STORE_ID_COL,store_banner_col=None)

        # if we want to identify all zipcode code columns use this
        # zipcode_columns = zipcode_utils.identify_zipcode_columns(t)

        # keeping standard zipcode columns for converting to 5 digit zipcode
        zipcode_columns = set([ZIPCODE_COL, ZIPCODE_EXPANDED])
        zipcode_columns = set(t.columns).intersection(zipcode_columns)
        for zipcode_column in zipcode_columns:
            t = zipcode_utils.get_dataframe_with_5digit_zipcode(t,zipcode_actual=zipcode_column,zipcode_5digit=zipcode_column)
        # TODO ensuring sales_dollar is present, done for Jewel data
        if (SALES_DOLLAR_COL in t.columns) and t.dtypes[SALES_DOLLAR_COL].str.replace('|','') == 'O':
            t = StandardizeData.standardize_sales_dollar_column(t, sales_dollar = SALES_DOLLAR_COL)

        return t

    @staticmethod
    def remove_dollar_sign(df, column):
        """
        Remove dollar sign

        """
        t = df.copy()
        t[column] = pd.to_numeric(t[column].str.replace('$',''))
        return t

    @staticmethod
    def remove_additional_space(df,column):
        """
        Removes additional spaces given a column

        """
        t = df.copy()
        # slower
        # t[column] = t[column].apply(lambda x: StandardizeData.__multi_space_regex.sub(' ',x))
        # faster
        t[column] = t[column].str.replace(StandardizeData.__multi_space_regex,' ')
        return t

    @staticmethod
    def replace_space_with_underscore(df,column):
        """
        Replace space (single or multiple consecutive ones) with 1 underscore in the specified column
        """
        t = df.copy()
        t[column] = t[column].str.replace(StandardizeData.__space_regex,'_')
        return t

    @staticmethod
    def standardize_column_names(df):
        """
        Standardize column names to lower case & print diff
        """
        df = df.copy()
        column_names = df.columns
        # keeping for reference at the end
        column_names_original = list(column_names)
        StandardizeData.validate_column_names(column_names)
        column_names_new = [StandardizeData.__space_regex.sub("_",col.strip().lower()) for col in column_names]
        col_diff = StandardizeData._get_col_name_diff(column_names,column_names_new)
        logging.debug(f'converted table headers to lower case & removed space with _, fields changed {col_diff}')
        column_names = column_names_new


        # renaming columns to some standard names
        column_names_new = [StandardizeData.__sto_start_regex.sub("",col) for col in column_names]
        col_diff = StandardizeData._get_col_name_diff(column_names,column_names_new)
        logging.debug(f'removed sto_ from header names, changed headers {col_diff}')
        column_names = column_names_new

        rename_dict = {'zip':ZIPCODE_COL, 'zip_code':ZIPCODE_COL, 'banner':BANNER_COL}
        column_names_new = [rename_dict.get(col,col) for col in column_names]
        col_diff = StandardizeData._get_col_name_diff(column_names,column_names_new)
        logging.debug(f'replacing few more headers with standard names, changed headers {col_diff}')
        column_names = column_names_new

        ## standardize sales dollar column name
        column_names_new = [re.sub(r'(dollar(_|\s)sales|sales(_|\s)dollar).*',SALES_DOLLAR_COL, col) for col in column_names]
        col_diff = StandardizeData._get_col_name_diff(column_names,column_names_new)
        logging.debug(f'standardize sales dollar column name, changed headers {col_diff}')
        column_names = column_names_new

        ## standardize sales unit column name
        column_names_new = [re.sub(r'(unit(_|\s)sales|sales(_|\s)unit).*',SALES_UNIT_COL, col) for col in column_names]
        col_diff = StandardizeData._get_col_name_diff(column_names,column_names_new)
        logging.debug(f'standardize sales unit column name, changed headers {col_diff}')
        column_names = column_names_new

        ## standardize sales volume column name
        column_names_new = [re.sub(r'(volume(_|\s)sales|sales(_|\s)volume).*', SALES_VOL_COL, col) for col in column_names]
        col_diff = StandardizeData._get_col_name_diff(column_names, column_names_new)
        logging.debug(f'standardize sales volume column name, changed headers {col_diff}')
        column_names = column_names_new

        col_diff = StandardizeData._get_col_name_diff(column_names_original,column_names_new)
        logging.info(f'replacing headers with standard names, changed headers {col_diff}')

        StandardizeData.validate_column_names(column_names_new)
        df.columns = column_names_new
        return df

    @staticmethod
    def standardize_sales_dollar_column(df, sales_dollar = SALES_DOLLAR_COL):
        t = df.copy()
        t = StandardizeData.remove_dollar_sign(t, sales_dollar)
        return t

    @staticmethod
    def standardize_store_address_column(df,store_address_col=STORE_ADDRESS_COL, store_address_col_new=STORE_ADDRESS_COL):
        t = df.copy()
        # converting address column values to lower case
        t[store_address_col_new] = t[store_address_col].str.lower().str.strip()
        # removing additional space from address column
        t = StandardizeData.remove_additional_space(t,store_address_col_new)
        return t

    @staticmethod
    def standardize_store_banner_column(df,store_banner_col=BANNER_COL):
        t = df.copy()
        # converting address column values to lower case
        t[store_banner_col] = t[store_banner_col].str.strip().str.lower()
        t = StandardizeData.replace_space_with_underscore(t,store_banner_col)
        return t

    @staticmethod
    def standardize_store_id_column(df,store_id_col=STORE_ID_COL,store_banner_col=None):
        """
        Standardize the store_id_col & keep a store_id_before_banner column with the value before normalization
        This is useful if we want to get a old id vs new id mapping.
        Eg:- sales data generally wouldn't have banner column, so to get the new id we can create a dictionary
        with old vs new & use it

        Arguments:
        df: pandas dataframe
            A pandas dataframe with store_id column which we want to standardize
        store_id_col:String
            column header for the store_id column
            default_value:- 'store_id' (constants.STORE_ID_COL)
            store id is normalized & space replaced with underscore
        store_banner_col: String
            column header for banner column
            default_value:- None
            Store banner is prepended to the store_id if store_banner_col!=None
        """
        t = df.copy()
        t = t.astype({store_id_col:str})
        # normalizing the store_id_col column to have lower case & no spaces at both ends
        t[store_id_col] = t[store_id_col].str.lower().str.strip()
        # want the old store id to contain the stripped version, this is going to be used for replacement if a new df (like sales_df) does
        # not have banner column & if we have to get the new store ids for that
        t[STORE_ID_BEFORE_BANNER_COL] = t[store_id_col]
        if store_banner_col:
            # normalizing the store_banner_col column to have lower case & no spaces at both ends
            t[store_banner_col] = t[store_banner_col].str.lower().str.strip()
            t = StandardizeData.__prepend_store_banner_to_id(t,store_id_col,store_banner_col)
        return t

    @staticmethod
    def __prepend_store_banner_to_id(df,store_id_col=STORE_ID_COL,store_banner_col=BANNER_COL):
    # def prepend_store_banner_to_id(df,store_id_col=STORE_ID_COL,store_banner_col=BANNER_COL):
        t = df.copy()
        banner_unique = t[store_banner_col].unique()
        t = t.apply(lambda row: StandardizeData.__prepend_store_banner_to_id_per_row(row,banner_unique,store_id_col,store_banner_col),axis=1)
        return t

    @staticmethod
    def __prepend_store_banner_to_id_per_row(row,banner_list,store_id_col=STORE_ID_COL,store_banner_col=BANNER_COL):
        # the store banner standardization would be done before this step, so below one is not necessary, but just performing strip & lower
        store_banner = row[store_banner_col].strip().lower()
        store_id = row[store_id_col]
        store_id = StandardizeData.__space_regex.sub("_",store_id.strip())

        banner_unique = [banner.strip() for banner in set(banner_list)]

        store_id_new = ''
        if store_banner in store_id:
            store_id_new = store_id
        else:
            # it reaches here if current banner is not in store id
            # now check if store id mathches with any of the banner list if so throw error
            store_banner_regex = f"({'|'.join(banner_unique)})"
            if re.search(store_banner_regex,store_id,flags=re.IGNORECASE):
                raise ValueError(f"current store id {store_id} doesn't correspond to the current store banner {store_banner}")
            store_id_new = f"{store_banner}_{store_id}"

        row[store_id_col] = store_id_new
        return row

    @staticmethod
    def _get_col_name_diff(column_names,column_names_new):
        col_diff = [(col,col_new) for col,col_new in zip(column_names,column_names_new) if col!=col_new]
        return col_diff

    @staticmethod
    def validate(df):
        """
        Given a pandas dataframe perform validation on it like column name validation

        Arguments:
            df: pandas dataframe
        """
        zipcode_utils.validate_zipcode_type_pandas_df(df)
        validate_column_names(df.columns)

    @staticmethod
    def _is_unique(column_names):
        s = set(column_names)
        return len(column_names)==len(s)

    @staticmethod
    def validate_column_names(column_names):
        # https://stackoverflow.com/questions/9835762/how-do-i-find-the-duplicates-in-a-list-and-create-another-list-with-them
        if not StandardizeData._is_unique(column_names):
            dups = set()
            # using dict to get even index of 1st element
            seen = {}
            for index,col in enumerate(column_names):
                if col in seen:
                    # adding current occurrence
                    dups.add((index,col))
                    # adding 1st occurrence
                    dups.add((seen[col],col))
                else:
                    seen[col] = index
            raise Exception(f"Duplicate col names found, column name & index {dups}")


def get_store_list_expanded(df, radius=2, max_radius=10, radius_col=RADIUS_COL,
        zipcode_col=ZIPCODE_COL, zipcode_expanded_col=ZIPCODE_EXPANDED,
        is_original_zipcode_col=IS_ORIGINAL_ZIPCODE_COL, validated_col=VALIDATED_COL):
    """
    Returns a df with expanded store list filtered by specified radius
    Note:- this function doesn't do the expansion, it just filters out entries based on given radius

    Mandatory header : radius_col
    If is_original_zipcode_col is specified that is also retained in addition to the ones within a radius
    If validated_col is provided then only True ones are retained

    Args:
    df: pandas dataframe
        dataframe for which we want to filter out the radius
    radius: int or float
        radius to expand to (default:2)
    max_radius: int or float
        maximum radius available in the df, will throw an exception if user specifies a radius more than this
    radius_col: string
        radius column name (default:radius)
    is_original_zipcode_col: string
        column containing boolean as to whether this is the original zipcode
        If None or column header not present it's ignored
    validated_col: string
        column containing boolean as to whether this zipcode is valid in dv360
        If None or column header not present it's ignored

    """
    if radius>max_radius:
        raise Exception('currently loaded file is expanded only till 5')
    t = df.copy()

    # keeping an entry for additional columns so that it can be dropped at the end
    additional_columns = []
    # handle when is_original_zipcode_col is None
    if not is_original_zipcode_col:
        # generating a temp unique column name for is_original_zipcode_col & will drop this at the end
        temp_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
        is_original_zipcode_col = f'{IS_ORIGINAL_ZIPCODE_COL}_{temp_name}'

    # handle when validated_col is None
    if not validated_col:
        # generating a temp unique column name for is_original_zipcode_col & will drop this at the end
        temp_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
        validated_col = f'{VALIDATED_COL}_{temp_name}'

    # handle case when is_original_zipcode_col is not present in df
    if is_original_zipcode_col not in t.columns:
        if zipcode_col in t.columns and zipcode_expanded_col in t.columns:
            t[is_original_zipcode_col] = t[zipcode_col]==t[zipcode_expanded_col]
        else:
            # giving all stores a False value so that our selection criteria will depend only on radius
            t[is_original_zipcode_col] = False
            logging.error(f"{is_original_zipcode_col} not present ignoring it while getting the list")
        additional_columns.append(is_original_zipcode_col)

    # handle case when validated_col is not present in df
    if validated_col not in t.columns:
        t[validated_col] = True
        logging.error(f"{validated_col} not present ignoring it while getting the list")
        additional_columns.append(validated_col)

    # logic for filtering data
    t = t[( (t[radius_col]<=radius) | t[is_original_zipcode_col] ) & t[validated_col]]

    # dropping additional columns added
    if len(additional_columns)>0:
        t.drop(additional_columns,axis=1,inplace=True)
    return t


def get_store_list_original(df,is_original_zipcode_col=IS_ORIGINAL_ZIPCODE_COL):
    """
    Given a dataframe, filter out & give only rows with the original store zipcode

    Arguments:
    df: pandas dataframe
        a pandas dataframe with expanded zipcode
    is_original_zipcode_col: string
        column header for is_original_zipcode
    """
    t =  df.copy()
    return t[t[is_original_zipcode_col]]


def regex_replace_df(df, column_name, pattern, replacement,n=-1, case=None, flags=re.I, regex=True):
    """Replace each occurrence of pattern/regex in the Series/Index.

    Equivalent to :meth:`str.replace` or :func:`re.sub`, depending on
    the regex value.

    Parameters
    ----------
    df : pandas dataframe
    column_name : str
    pat : str or compiled regex
        String can be a character sequence or regular expression.
    repl : str or callable
        Replacement string or a callable. The callable is passed the regex
        match object and must return a replacement string to be used.
        See :func:`re.sub`.
    n : int, default -1 (all)
        Number of replacements to make from start.
    case : bool, default None
        Determines if replace is case sensitive:

        - If True, case sensitive (the default if `pat` is a string)
        - Set to False for case insensitive
        - Cannot be set if `pat` is a compiled regex.

    flags : int, default 0 (no flags)
        Regex module flags, e.g. re.IGNORECASE. Cannot be set if `pat` is a compiled
        regex.
    regex : bool, default True
        Determines if the passed-in pattern is a regular expression:

        - If True, assumes the passed-in pattern is a regular expression.
        - If False, treats the pattern as a literal string
        - Cannot be set to False if `pat` is a compiled regex or `repl` is
          a callable.
"""
    t = df.copy()
    #due to some bug in pandas, regex was not working correctly hence custom code to compile regex.
    #todo : remove custom changes when pandas fix this issue.
    if regex :
        if type(pattern)==str:
            pattern = re.compile(pattern,flags)
        t[column_name] = t[column_name].str.replace(pat=pattern, repl=replacement, n=n, regex=regex)
    else :
        t[column_name] = t[column_name].str.replace(pat=pattern, repl=replacement, n=n, case=case, flags=flags, regex=regex)
    return t

def _generate_random_unique_id(length=5):
    """
    Returns a unique random string
    """
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))

def concatenate_values_to_column(df, column_name:str, string_to_concatenate:str, new_column_name:str=None, sep:str="_", location:str="prefix"):
    """
    Concatenate values to a column

    Arguments:
        df:pandas.DataFrame
        column_name:str
            column to which we want to add the string as prefix or suffix
        string_to_concatenate:str
            string to concatenate
        new_column_name:str
            If we want to keep a new column after concatenation the column name
            If we want to use the same
        sep:str
            separator to use between current column data & new string
        prefix:str
            it can be either prefix or suffix
    """
    t = df.copy()
    if new_column_name is None:
        new_column_name = column_name
    location = location.lower()

    unique_id = _generate_random_unique_id()
    temp_column = f"{column_name}_{unique_id}"
    t[temp_column] = string_to_concatenate
    left = t[temp_column]
    right = t[column_name].astype('str')
    if location=="suffix":
        left = t[column_name].astype('str')
        right = t[temp_column]
    t[new_column_name] = left.str.cat(right, sep=sep)
    t = t.drop(labels=[temp_column], axis=1)
    return t

def change_datatype(df, dtype):
    """
    Cast a pandas dataframe to a specified dtype ``dtype``
    Arguments:
        df: pandas dataframe
            input pandas dataframe

        dtype :data type, or dict of column name -> data type
            Use a numpy.dtype or Python type to cast entire pandas object to
            the same type. Alternatively, use {col: dtype, ...}, where col is a
            column label and dtype is a numpy.dtype or Python type to cast one
            or more of the DataFrame's columns to column-specific types.
     """
    df = df.astype(dtype)
    return df
