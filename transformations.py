import copy
import inspect
import enum
import time
import pandas as pd
import logging

from utils import file_utils, date_utils, text_utils, dataframe_utils, sql_utils,zipcode_utils
from utils.expression_evaluator import ExpressionEvaluator
from utils.config_objects import MetaDataCell, remove_comments

TRANSFORMATIONS = "transformations"
TRANSFORMATIONS_METAROWS = "transformations_metarows"
METADATA = "metadata"

FUNCTION = 'function'
FUNCTION_OUTPUT = 'function_output'

EVALUATE_EXPRESSION="evaluate_expression"
ADD_METADATA_VALUES="add_metadata_values"
ADD_COLUMN="add_column"

"""
A tranformation pipeline object which will perform a bunch of transformations on a file to get it into the final processed state
"""
class TransformationPipeline(object):
    __DEFAULT_CACHE_KEY = "x" # this is the key stored in cache, reference for the same would be $x
    __METADATA_OUTPUT_CACHE_KEY = "$metadata_output" # this is the key stored in cache, reference would be $$metadata_output
    __PREVIOUS_OUTPUT_CACHE_KEY = "$previous_output" # this is used to store the name of previous output cache, this is required for adding columns from metadata transformations

    @staticmethod
    def _get_config_keys():
        """
        Returns the keys applicable for transformation config

        values present now
        transformations -

        transformations_metarows -

        metadata
            list of Map

        """
        return [TRANSFORMATIONS, TRANSFORMATIONS_METAROWS, METADATA]
    @staticmethod
    def get_transformations_template():
        d = {METADATA:[],
            TRANSFORMATIONS_METAROWS:{},
             TRANSFORMATIONS: []}
        return d

    def __init__(self, common_config, table_config, **kwargs):
        """
        Transformation pipeline would be able to transform 1 table in the the final processed state
        """
        # TODO, FIXME need to have some index to order when both common config & table specific config is mentioned
        # currently 1st common transformation is applied followed by table specific config
        # may have an index along with transformations & transformations can be sorted based on this index if present
        common_config = copy.deepcopy(common_config)
        table_config = copy.deepcopy(table_config)
        ### getting all the transformation configs
        # type of config is List<Transformation> = List<Map<>>
        self.transformations_config = common_config.get(TRANSFORMATIONS, [])
        self.transformations_config.extend(table_config.get(TRANSFORMATIONS, []))

        ### getting all metadata transformations
        transformations_metarows_config_common = common_config.get(TRANSFORMATIONS_METAROWS, {})
        transformations_metarows_config_specific = table_config.get(TRANSFORMATIONS_METAROWS, {})

        meta_rowids = set(transformations_metarows_config_common.keys())
        meta_rowids.update(transformations_metarows_config_specific.keys())
        transformations_metarows_config = {}
        for metarowid in meta_rowids:
            transformations_metarows_config[metarowid] = transformations_metarows_config_common.get(metarowid, []) + \
                                                         transformations_metarows_config_specific.get(metarowid, [])
        # Map<metarowIndex: List[transformations]>
        # map of list of map as a single table can have multiple metarows associated with it
        self.transformations_metarows_config = transformations_metarows_config

        ### getting all metadata values
        metadata_common = common_config.get(METADATA, [])
        metadata_specific = table_config.get(METADATA, [])

        # Eg:- {"metadata": [{"meta_0": "Eckrich - Meijer Store Level"}, {"meta_1": "Time:Week Ending 05-17-20"}
        self.metadata = MetaDataCell.create_list_of_metadata_with_overriding(metadata_common, metadata_specific)
        #TODO add a log or raise a exception for cases where transformation metadata_id doesn't match with the metadata id in metadata list


    # a dictionary to store output after each step, i.e cache
    __transformation_output = {}

    def __reset_transformation_output(self):
        self.__transformation_output = {}

    def __set_transformation_output_to_cache(self, key, value):
        self.__transformation_output[key] = value

    def __get_tranformation_output_from_cache(self, key, default=None):
        return self.__transformation_output.get(key, default)

    def __is_param_reference(self, param_value):
        return type(param_value)==str and param_value.startswith("$")

    def __get_key_from_reference(self, reference):
        return reference[1:] # parameter is stored without the initial $ symbol

    def __get_reference_from_key(self,key):
        return f"${key}"

    def __update_transformation_config_with_actual_value(self, config, function_name):
        """
        update references to variables with actual value from cache
        Eg:- if one of the config value is $x fetch the value corresponding to x & keep it instead
        similarly if config value is $df fetch the value corresponding to df & keep it as the value
        """
        output = {}
        if function_name == EVALUATE_EXPRESSION:
        # evaluate_expression function is a special case,
        # here we replace the $ from the expression & since we are passing the cached values to evaluate
        # python takes care of evaluating it
        # Eg:- "$year - 1" will be replaced with "year -1" & if variable year is present in cache it's value will be used
            EXPRESSION_ARG = "expression"
            assert EXPRESSION_ARG in config, "pass the expression to evaluate_expression"
            expression = config.get(EXPRESSION_ARG,"").replace("$","")
            output[EXPRESSION_ARG] = expression
        else:
            for (k, param_value) in config.items():
                output_value = param_value

                if self.__is_param_reference(param_value):
                    #param_key = param_value.replace("$","")
                    param_key = self.__get_key_from_reference(param_value)
                    output_value = self.__transformation_output.get(param_key)
                    if output_value is None:
                        raise Exception(f"could not find a value in cache corresponding to key `{param_key}` for function {function_name}")

                output[k] = output_value
        return output

    def transform_metadata(self):
        start_time = time.time()
        cols_output = {}
        # add metadata values is a default operation for tables & it requires this key to be present.
        # So setting empty dict as initial value, incase there is no metadata, this value will be used
        self.__set_transformation_output_to_cache(self.__METADATA_OUTPUT_CACHE_KEY, cols_output)
        for index, metaDataCell in enumerate(self.metadata):
            # row is a dictionary {"id" :"meta_0", "value": "Eckrich - Meijer Store Level"}
            metadata_id = metaDataCell.get_id()
            transformations = copy.deepcopy(self.transformations_metarows_config.get(metadata_id, []))
            # decided not to add a metadata to df unless it's explicity added, so commented out the below function, #TODO remove
            # transformations = self.__add_default_metadata_transformations(transformations, metadata_id)
            if len(transformations) == 0:
                # updating the value as it is without any transformations
                # cols_output.update({metaDataCell.get_id():metaDataCell.get_value()})
                continue

            original_value = metaDataCell.get_value()

            updated_value = original_value
            column_name = metaDataCell.get_id() # TODO check if this is the best option, if there is atleast 1 transformation we are adding that metadata
            # keeping the metadata value to x key in cache
            self.__set_transformation_output_to_cache(self.__DEFAULT_CACHE_KEY, updated_value)
            for transformation in transformations:
                function_name = transformation.pop(FUNCTION, 'skip')
                function_output_name = transformation.pop(FUNCTION_OUTPUT, self.__DEFAULT_CACHE_KEY)

                transformation = self.__update_transformation_config_with_actual_value(transformation, function_name)
                if function_name == 'skip': #TODO skip is not really required, if add_column transformation is not present metadata is not considered
                    # not using this metadata, skipping other transformation for this metadata if any
                    column_name = None
                    break
                else:
                    self.__execute_function_update_cache(function_name, function_output_name, transformation)
            # metadata values to add as columns to dataframe are also added into the cache, we have to take that value before reset &
            # set it back after the reset
            cols_output = self.__get_tranformation_output_from_cache(self.__METADATA_OUTPUT_CACHE_KEY, {})
            # resetting values to save memory
            # currently metacell values from another metacell id is not designed to be used with a different metacell id,
            # TODO resetting cache can be removed if there is cross usage of metadata values
            self.__reset_transformation_output()
            self.__set_transformation_output_to_cache(self.__METADATA_OUTPUT_CACHE_KEY, cols_output)
        end_time = time.time()
        logging.debug(f"time taken for transform_metadata {end_time-start_time}")
        return cols_output

    def __execute_function_update_cache(self, function_name, function_output_name:str, transformation:dict):
        """
        Arguments:
            function_name:str
                transformation name
            function_output_name:str
                Eg:- x, this variable would be now available in cache for future transformations to refer to
            transformation:dict
                This is a dictionary with function input

        """
        transform_function = get_transformation(function_name)
        transformation = remove_comments(transformation)
        if function_name==EVALUATE_EXPRESSION:
            # this is a special case, we have to pass the cache values for evaluation
            transformation_output = self._evaluate_expression(**transformation)
            self.__transformation_output[function_output_name] = transformation_output
        elif function_name == ADD_COLUMN:
            self._add_column(**transformation)
        elif transform_function is not None:
            transformation_output = transform_function(**transformation)
            self.__transformation_output[function_output_name] = transformation_output


        else:
            raise Exception(f"function_name {function_name} not available")

    def __add_default_metadata_transformations(self,transformations, metadata_id):
        function_names = [transformation.get(FUNCTION) for transformation in transformations]
        if not ADD_COLUMN in function_names:
            transformation = {FUNCTION:ADD_COLUMN,
                              "column_name":metadata_id}
            transformations.append(transformation)
        return transformations

    def __add_default_table_transformations(self,transformations):
        function_names = [transformation.get(FUNCTION) for transformation in transformations]
        if not ADD_METADATA_VALUES in function_names:
            transformation = {FUNCTION:ADD_METADATA_VALUES,
                              "df":self.__get_reference_from_key(self.__DEFAULT_CACHE_KEY),
                              "metadata_values":self.__get_reference_from_key(self.__METADATA_OUTPUT_CACHE_KEY)  # this value have to be updated from cache
                              }
            transformations.append(transformation)
        return transformations

    def __tranform_table(self, df):
        start_time = time.time()
        transformations = copy.deepcopy(self.transformations_config)
        transformations = self.__add_default_table_transformations(transformations)
        # the below if condition would not be required now, TODO test & remove
        if len(transformations) == 0:
            return df

        # df_original = df.copy()
        df = df.copy() # copying so that any other function using the same dataframe is not affected by the tranforms here
        # keeping the df value to x key in cache, this is before any transformations start
        function_output_name = self.__DEFAULT_CACHE_KEY
        self.__set_transformation_output_to_cache(function_output_name, df)
        for transformation in transformations:
            function_name = transformation.pop(FUNCTION, 'skip')
            function_output_name = transformation.pop('function_output', self.__DEFAULT_CACHE_KEY)

            if function_name == ADD_METADATA_VALUES:
                # adding the cache key for metadata output, the next step will fetch it's value from cache
                transformation["metadata_values"] =  self.__get_reference_from_key(self.__METADATA_OUTPUT_CACHE_KEY)  # this value have to be updated from cache

            # update references to variables with actual value from cache
            transformation = self.__update_transformation_config_with_actual_value(transformation, function_name)
            self.__execute_function_update_cache(function_name, function_output_name, transformation)
        # final function_output_name is expected to be the final value we have to use
        output = self.__get_tranformation_output_from_cache(function_output_name)
        self.__reset_transformation_output()
        end_time = time.time()
        logging.debug(f"time taken for tranform_table {end_time - start_time}")
        return output

    def _evaluate_expression(self, expression):
        expression_evaluator = ExpressionEvaluator({}, self.__transformation_output)
        return expression_evaluator.evaluate_expression(expression)

    def _add_column(self,column_name:str, column_value=None):
        """
        The add columns output
        """
        metadata_values_key = self.__METADATA_OUTPUT_CACHE_KEY
        cols_output = self.__get_tranformation_output_from_cache(metadata_values_key, {})
        if column_value is None:
            # if column value is None then we have to fetch the output value from previous transformation
            function_output_name = self.__get_tranformation_output_from_cache(self.__PREVIOUS_OUTPUT_CACHE_KEY, self.__DEFAULT_CACHE_KEY)
            column_value = self.__get_tranformation_output_from_cache(function_output_name)

        new_dict = add_column(column_name, column_value)
        cols_output.update(new_dict)
        self.__set_transformation_output_to_cache(metadata_values_key, cols_output)

    def transform(self, df):
        # this will transform & keep the values in cache
        self.transform_metadata()
        df = self.__tranform_table(df)
        return df

class TransformationLevel(enum.Enum):
    meta_level=1
    df_level=2
    invalid=3
    both=4

class TransformationsConfigCreator(object):
    def __init__(self,common_config={}, table_configs=[], **kwargs):
        """
        This is used to hold common_config, table_specific configs & update them incrementally
        """
        self.common_config = copy.deepcopy(common_config)
        self.table_configs = copy.deepcopy(table_configs)

    def append_common_transformation(self, transformation_config, transformation_level=TransformationLevel.df_level, metadata_id=None):
        transformation_config = copy.deepcopy(transformation_config)
        key_to_use = self.__get_transformation_key_to_use(metadata_id, transformation_config, transformation_level)
        transformations = self.common_config.get(key_to_use, [])
        if transformation_level == TransformationLevel.df_level:
            transformations.append(transformation_config)
        elif transformation_level == TransformationLevel.meta_level:
            # for meta level the transformations is a dictionary with metadata_id as key
            transformations = self.common_config.get(key_to_use, {})
            transformations_meta_level = transformations.get(metadata_id, [])
            transformations_meta_level.append(transformation_config)
            transformations[metadata_id] = transformations_meta_level
        self.common_config[key_to_use] = transformations


    def append_transformation_table_specific(self, transformation_config, table_number,
                                             transformation_level=TransformationLevel.df_level,
                                             metadata_id=None):
        """
        Arguments:
            transformation_config:dict
                transformation_config for a function
            table_number:int
                The table number for which  this transformation have to be applied
            transformation_level:TransformationLevel enum
            metadata_id:str
                meta
        """
        transformation_config = copy.deepcopy(transformation_config)
        key_to_use = self.__get_transformation_key_to_use(metadata_id, transformation_config, transformation_level)
        self.__assert_valid_table_number(table_number)

        table_config = {}
        is_existing_table = table_number < len(self.table_configs)
        if is_existing_table:
            table_config = self.table_configs[table_number]

        transformations = table_config.get(key_to_use, [])

        if transformation_level == TransformationLevel.df_level:
            transformations.append(transformation_config)
        elif transformation_level == TransformationLevel.meta_level:
            # for meta level the transformations is a dictionary with metadata_id as key
            transformations = table_config.get(key_to_use, {})
            transformations_meta_level = transformations.get(metadata_id, [])
            transformations_meta_level.append(transformation_config)
            transformations[metadata_id] = transformations_meta_level

        table_config[key_to_use] = transformations
        if is_existing_table:
            self.table_configs[table_number] = table_config
        else:
            self.table_configs.append(table_config)

    def __get_transformation_key_to_use(self, metadata_id, transformation_config, transformation_level):
        key_to_use = TRANSFORMATIONS
        function_name = transformation_config.get(FUNCTION, None)
        if transformation_level == TransformationLevel.meta_level:
            assert (metadata_id is not None) and len(metadata_id.strip()) > 0
            key_to_use = TRANSFORMATIONS_METAROWS
        elif transformation_level == TransformationLevel.df_level:
            key_to_use = TRANSFORMATIONS
        else:
            raise Exception(f"Not a valid transformation level {transformation_level}")
        return key_to_use

    def __assert_valid_table_number(self, table_number):
        assert table_number <= len(
            self.table_configs), f"It's only possible to add transformation for either next table or existing tables, total number of table specific configs so far is {len(self.table_configs)} & table_number is {table_number}"


    def get_common_config(self):
        return self.common_config

    def get_table_configs(self):
        return self.table_configs

def evaluate_expression(expression:str):
    """
    This method can be used to evaluate expressions
     Arguments:
        expression:str
            Expression to evaluate, variables should be name with $
            Eg:Let's say from previous tranformations a variable name - week_number is present, then expression would be
            "$week_number -1"

    """
    # This is just a placeholder function so that this function shows up in list of functions
    """
    transformation_pipeline:TransformationPipeline
            This will be automatically populated internally, don't pass anything for this variable
    """
    transformation_pipeline: TransformationPipeline = TransformationPipeline({}, {})
    return transformation_pipeline._evaluate_expression(expression)


def strip(text):
    """
    Strip a text & return the output
    """
    return text.strip()

def convert_to_int(text):
    """
    Convert string to int
    Arguments:
        text:str
            text to convert to int
    """
    return int(text)

def convert_to_float(text):
    """
    Convert string to float
    Arguments:
        text:str
            text to convert to float
    """
    return float(text)

def add_column(column_name:str, column_value):
    """
    This function is to add a column from metadata values as a
    column value for the dataframe
    Arguments:
        column_name:str
            column_name to keep in the dataframe
        column_value:
            column value can be any datatype. Eg:- string, int, float, datetime etc
    """
    # this function is mostly just a placeholder to show the documentation for the function,
    # complete actual functionality happens in the function _add_column
    return {column_name:column_value}

def __add_metadata_values(df, metadata_columns='all', metadata_values={}):
    """
    An operation to add processed metadata values to the dataframe
    Arguments:
        df: pandas dataframe
            input pandas dataframe
        metadata_columns:str or list
        'all' if we want to add all metadata values
        or list of metadata ids to add. Eg:- ['retail_chain','meta_0']
        metadata_values:dict
            a dictionary of metadata column name & value to use
            Eg:- {"retail_chain":"kroger", "date":"2020-10-20"}
            If called via transformations this value will be autopopulated
    """
    t = df.copy()
    metadata_values_to_add = copy.deepcopy(metadata_values)
    if type(metadata_columns) == str and metadata_columns == 'all':
        metadata_values_to_add = copy.deepcopy(metadata_values)
    else:
        assert type(metadata_columns) == list, "expected metadata_ids to be list of ids or string 'all' "
        metadata_values_to_add = {k:v for (k,v) in metadata_values.items() if k in metadata_columns}

    # https://stackoverflow.com/questions/29517072/add-column-to-dataframe-with-constant-value
    # https://stackoverflow.com/a/50326955
    t = t.assign(**metadata_values_to_add)
    return t


__function_mapping_df = {
    'standardize_dataframe': dataframe_utils.StandardizeData.standardize_dataframe,
    'standardize_column_names': dataframe_utils.StandardizeData.standardize_column_names,
    'split_to_columns': dataframe_utils.split_to_columns,
    'transpose': dataframe_utils.transpose,
    'convert_to_multi_header':dataframe_utils.convert_to_multi_header,
    "regex_replace_df": dataframe_utils.regex_replace_df,
    "filter_query": dataframe_utils.filter_query,
    "strip_df": dataframe_utils.strip_df,
    "find_date_format_df":date_utils.find_date_format_df,
    "convert_column_to_date":date_utils.convert_column_to_date,
    "change_datatype":dataframe_utils.change_datatype,
    "rename_headers": dataframe_utils.rename_headers,
    "add_end_of_week":date_utils.add_end_of_week,
    "filter_and_order_columns": dataframe_utils.filter_and_order_columns,
    "copy_columns": dataframe_utils.copy_columns,
    "add_metadata_values":__add_metadata_values,
    "concatenate_values_to_column":dataframe_utils.concatenate_values_to_column,
    "read_sql": sql_utils.read_sql,
    "execute_sql": sql_utils.execute_sql,
    "add_dataframe_to_db": sql_utils.add_dataframe_to_db,
    "get_dataframe_with_standard_length_zipcode":zipcode_utils.get_dataframe_with_standard_length_zipcode,
    EVALUATE_EXPRESSION:evaluate_expression # this is a special function available for both df & non_df cases
}

# functions on a non df object
__function_mapping_non_df = {
    "regex_replace": text_utils.regex_replace,
    "get_date": date_utils.get_date,
    "get_first_day_date":date_utils.get_first_day_date,
    "add_days":date_utils.add_days,
    "add_weeks":date_utils.add_weeks,
    "strip": strip,
    "convert_to_int":convert_to_int,
    "convert_to_float":convert_to_float,
    ADD_COLUMN:add_column,
    EVALUATE_EXPRESSION:evaluate_expression # this is a special function available for both df & non_df cases
}

# combining df & non df transformations into a common dictionary
__function_mapping = {}
__function_mapping.update(__function_mapping_df)
__function_mapping.update(__function_mapping_non_df)

# evaluate_expression is an exception it is available for both df & non_df level cases, hence +1
assert (len(__function_mapping_df) + len(__function_mapping_non_df))==len(__function_mapping)+1, "There are duplicates keys between df transformations & meta transformations"

def transform(df:pd.DataFrame):
    #TODO complete the functionality
    transformation_pipeline = TransformationPipeline()
    transformation_pipeline.transform()

def get_transformation(transformation_name):
    """
    Given a transformation name return the tranformation function
    Arguments:
        transformation_name:string
            name for the transformation
    """
    return __function_mapping.get(transformation_name)

def get_list_of_transformations():
    """
    Get a list of available transformations

    """
    return list(__function_mapping.keys())


def get_function_signature(func):
    """
    Returns the function signature given a function
    Arguments:
        func:python function or string
            a python function or the string name corresponding to that function
    """
    if type(func)==str:
        func = get_transformation(func)
    # https://stackoverflow.com/questions/218616/how-to-get-method-parameter-names
    return inspect.signature(func)

def get_function_arguments(func):
    """
    Given a function return it's arguments
    Arguments:
        func:python function or string
            a python function or the string name corresponding to that function
    """
    if type(func)==str:
        func = get_transformation(func)
    # https://stackoverflow.com/questions/218616/how-to-get-method-parameter-names
    return inspect.getfullargspec(func).args

def get_function_input_template(function_name):
    """
    Given a function_name return an input template to fill
    Arguments:
        function_name:string
            a python function or the string name corresponding to that function
    """
    if type(function_name)==str:
        func = get_transformation(function_name)
    function_inputs = {}
    function_inputs[FUNCTION] = function_name
    function_inputs.update({k:"" for k in get_function_arguments(func)})

    return function_inputs

def get_function_document(func):
    """
    Given a function return it's document
    Arguments:
        func:python function or string
            a python function or the string name corresponding to that function
    """
    if type(func)==str:
        func = get_transformation(func)
    return func.__doc__


def get_function_level(transformation_name):
    """
    Return an enumerator - TransformationLevel, which indicates the level at which the given transformation is applicable
    Arguments:
        transformation_name:string
            name for the transformation
    """
    level = TransformationLevel.invalid
    if (transformation_name in __function_mapping_df) and (transformation_name in __function_mapping_non_df):
        # a function like evaluate_expression is present in both levels
        level = TransformationLevel.both
    elif transformation_name in __function_mapping_df:
        level = TransformationLevel.df_level
    elif transformation_name in __function_mapping_non_df:
        level = TransformationLevel.meta_level
    return level