from sqlalchemy import create_engine
import os
import sqlite3
import pandas as pd
import logging

from utils import file_utils, text_utils, date_utils

class SQL(object):
    def __init__(self, db_name=None, engine="sqlite"):
        assert engine == "sqlite", "only sqlite is supported now"
        if db_name is None:
            db_name = "pandas_sql.db"
        db_extension = file_utils.get_file_type(db_name)
        if db_extension not in ["db","sqlite", "sqlite3", "db3"]:
            db_name = f"{db_name}.db"

        db_path = os.path.join(file_utils.get_main_resources_folder(), "sql_dbs", db_name)
        file_utils.create_parent_dirs(db_path)
        self.db_path = db_path
        self.__engine = create_engine(f'{engine}:///{db_path}')  # ,echo=True)
        self.connection = self.__engine.connect()
        self.__closed = False

    def is_closed(self):
        return self.__closed

    def close(self):
        """
        This would delete the db file created for sql lite
        """
        # https://stackoverflow.com/questions/8645250/how-to-close-sqlalchemy-connection-in-mysql
        self.connection.close()
        self.__engine.dispose()
        file_utils.delete_file(self.db_path)
        self.__closed = True

    def __enter__(self):
        # https://stackoverflow.com/questions/865115/how-do-i-correctly-clean-up-a-python-object
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # https://stackoverflow.com/questions/865115/how-do-i-correctly-clean-up-a-python-object
        self.close()

    def add_dataframe_to_db(self, df:pd.DataFrame, table_name, engine="sqlite", schema=None, if_exists='replace',
                            index=False, index_label=None, chunksize=None, dtype=None, **kwargs):
        """
        This is used to save a pandas data frame to the sql database
        This is using pandas.DataFrame.to_sql
        For more details refer - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html
        Arguments:
            df:pd.DataFrame
                dataFrame to keep in the sql database
            table_name:str
                table_name to keep for the sql table created from this DataFrame
            engine:str
                only supports sqlite now
            schema : str, optional
                Specify the schema (if database flavor supports this). If None, use
                default schema.
            if_exists : {'fail', 'replace', 'append'}, default 'fail'
                How to behave if the table already exists.

                * fail: Raise a ValueError.
                * replace: Drop the table before inserting new values.
                * append: Insert new values to the existing table.
            index : bool, default False
                Write DataFrame index as a column. Uses `index_label` as the column
                name in the table.
            index_label : str or sequence, default None
                Column label for index column(s). If None is given (default) and
                `index` is True, then the index names are used.
                A sequence should be given if the DataFrame uses MultiIndex.
            chunksize : int, optional
                Specify the number of rows in each batch to be written at a time.
                By default, all rows will be written at once.
            dtype : dict or scalar, optional
                Specifying the datatype for columns. If a dictionary is used, the
                keys should be the column names and the values should be the
                SQLAlchemy types or strings for the sqlite3 legacy mode. If a
                scalar is provided, it will be applied to all columns.
        """
        df.to_sql(table_name, con=self.connection, schema=schema, if_exists=if_exists,
                  index=index, index_label=index_label, chunksize=chunksize, dtype=dtype, **kwargs)

    def read_sql(self, sql, index_col=None, coerce_float=True, params=None,
                 parse_dates=None, columns=None, chunksize=None, **kwargs):
        """
        Run a sql query which would return a table/dataframe
        This is ysing pandas.read_sql for more info refer- https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html
        The info below is copied from their document on Nov, 2021

        sql: str
            SQL query to be executed or a table name.
        index_col : str or list of strings, optional, default: None
            Column(s) to set as index(MultiIndex).
        coerce_float : bool, default True
            Attempts to convert values of non-string, non-numeric objects (like
            decimal.Decimal) to floating point, useful for SQL result sets.
        params : list, tuple or dict, optional, default: None
            List of parameters to pass to execute method.  The syntax used
            to pass parameters is database driver dependent. Check your
            database driver documentation for which of the five syntax styles,
            described in PEP 249's paramstyle, is supported.
            Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}.
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
              to the keyword arguments of :func:`pandas.to_datetime`
              Especially useful with databases without native Datetime support,
              such as SQLite.
        columns : list, default: None
            List of column names to select from SQL table (only used when reading
            a table).
        chunksize : int, default None
            If specified, return an iterator where `chunksize` is the
            number of rows to include in each chunk.

        """
        return pd.read_sql(sql=sql, con=self.connection, index_col=index_col, coerce_float=coerce_float,
                           params=params, parse_dates=parse_dates, columns=columns, chunksize=chunksize, **kwargs)

    def execute_sql(self, sql):
        """
        This function is to execute an sql which doesn't return a dataframe
        """
        # https://stackoverflow.com/questions/26286615/how-can-pandas-read-sql-query-query-a-temp-table
        # TODO check if there is any issue if we don't close the connection

        out = self.connection.execute(sql)
        out.close()

__sql_instances = {}
def __get_sql_instance(db_name, reuse_existing=True, engine="sqlite"):
    """
    get an sql instance with a given db name
    """
    sql_instance = __sql_instances.get(db_name, None)
    if (not reuse_existing) or (sql_instance is None) or sql_instance.is_closed():
        sql_instance = SQL(db_name=db_name, engine=engine)
        logging.debug(f"created a new sql_instance for db_name {db_name}")
        __sql_instances[db_name] = sql_instance

    return sql_instance

def _generate_random_project_id():
    rand_id = text_utils.generate_random_unique_id(10)
    timestamp = date_utils.get_timestamp()
    return f"{timestamp}_{rand_id}"

def read_sql(sql, df:pd.DataFrame=None, table_name:str=None, project_name=None,
             remove_db=True, engine="sqlite",
             schema=None, if_exists='replace',
             index=False, index_label=None,
             chunksize=None, dtype=None,
             index_col=None, coerce_float=True, params=None,
             parse_dates=None, columns=None,
             **kwargs):
    """
    Run a sql query, which returns a dataFrame
    This uses pandas.DataFrame.to_sql function to copy a dataframe to the database, followed by running read_sql for executing sql query
    data frame to a sql db, for more info https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html
    For running queries it is ysing pandas.read_sql, for more info refer- https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html
    Starting with the param - if_exists, the info below is copied from their document on Nov, 2021

    Arguments:
        sql:str
            sql expression to execute
        df:pd.DataFrame | list[DataFrame], default:None
            dataFrame to keep in the sql database.
            We can also pass a list of dataFrames as argument. There is supposed to a corresponding table_name list corresponding to each dataframe
        table_name:str | list[DataFrame], default:None
            table_name(s) to keep for the sql table created from this DataFrame
        project_name:str, default: None
            project_name will be used to create a db,
            if it's None then a new one will be created
        remove_db:boolean
            If true the created db will be deleted, will be retained if false
            If project_name is None, then this variable doesn't have any effect & will be treated as True
        engine:str
            Only sqlite is supported now
        schema : str, optional
                Specify the schema (if database flavor supports this). If None, use
                default schema.
        if_exists : {'fail', 'replace', 'append'}, default 'fail'
            How to behave if the table already exists.

            * fail: Raise a ValueError.
            * replace: Drop the table before inserting new values.
            * append: Insert new values to the existing table.
        index : bool, default False
            Write DataFrame index as a column. Uses `index_label` as the column
            name in the table.
        index_label : str or sequence, default None
            Column label for index column(s). If None is given (default) and
            `index` is True, then the index names are used.
            A sequence should be given if the DataFrame uses MultiIndex.
        chunksize : int, optional
            Applicable for both df.to_sql & pd.read_sql
            Specify the number of rows in each batch to be written at a time.
            By default, all rows will be written at once.
        dtype : dict or scalar, optional
            Specifying the datatype for columns. If a dictionary is used, the
            keys should be the column names and the values should be the
            SQLAlchemy types or strings for the sqlite3 legacy mode. If a
            scalar is provided, it will be applied to all columns.
        index_col : str or list of strings, optional, default: None
                Column(s) to set as index(MultiIndex).
        coerce_float : bool, default True
            Attempts to convert values of non-string, non-numeric objects (like
            decimal.Decimal) to floating point, useful for SQL result sets.
        params : list, tuple or dict, optional, default: None
            List of parameters to pass to execute method.  The syntax used
            to pass parameters is database driver dependent. Check your
            database driver documentation for which of the five syntax styles,
            described in PEP 249's paramstyle, is supported.
            Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}.
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
              to the keyword arguments of :func:`pandas.to_datetime`
              Especially useful with databases without native Datetime support,
              such as SQLite.
        columns : list, default: None
            List of column names to select from SQL table (only used when reading
            a table).
    """
    if project_name is None:
        project_name = _generate_random_project_id()
        # if it's a new generated project_name/db_name then there is no way it can be reused as we won't know the name
        # unless someone checks the db folder manually, so removing db after query
        remove_db = True
    sql_instance = __get_sql_instance(db_name=project_name, reuse_existing=True, engine=engine)

    add_dataframe_to_db(df=df, table_name=table_name, project_name=project_name,
                        engine=engine, remove_db=False, # note that remove_db is False here
                        schema=schema, if_exists=if_exists,
                        index=index, index_label=index_label, chunksize=chunksize, dtype=dtype)
    df_out = sql_instance.read_sql(sql=sql, index_col=index_col, coerce_float=coerce_float,
                params=params, parse_dates=parse_dates, columns=columns, chunksize=chunksize, **kwargs)

    if remove_db:
        sql_instance.close()

    return df_out

def execute_sql(sql, project_name=None, remove_db=True, engine="sqlite"):
    """
    This function is to execute an sql which doesn't return a dataframe
    """
    if project_name is None:
        project_name = _generate_random_project_id()
        # if it's a new generated project_name/db_name then there is no way it can be reused as we won't know the name
        # unless someone checks the db folder manually, so removing db after query
        remove_db = True
    sql_instance = __get_sql_instance(db_name=project_name, reuse_existing=True, engine=engine)
    sql_instance.execute_sql(sql)
    if remove_db:
        sql_instance.close()

def add_dataframe_to_db(df:pd.DataFrame, table_name, project_name:str=None, engine="sqlite", remove_db=True,
                        schema=None, if_exists='replace',
                        index=False, index_label=None, chunksize=None, dtype=None, **kwargs):
    """
    This is used to save a pandas data frame to the sql database
    This is using pandas.DataFrame.to_sql
    The params from if_exists are from the to_sql function
    For more details refer - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html

    Arguments:
        df:pd.DataFrame | list[DataFrame], default:None
            dataFrame to keep in the sql database.
            We can also pass a list of dataFrames as argument. There is supposed to a corresponding table_name list corresponding to each dataframe
        table_name:str | list[DataFrame], default:None
            table_name(s) to keep for the sql table created from this DataFrame
        project_name:str, default: None
        engine:str
            only sqlite is supported now
        remove_db:boolean, default True
            If we want to remove the db after execution
        schema : str, optional
                Specify the schema (if database flavor supports this). If None, use
                default schema.
        if_exists : {'fail', 'replace', 'append'}, default 'fail'
            How to behave if the table already exists.

            * fail: Raise a ValueError.
            * replace: Drop the table before inserting new values.
            * append: Insert new values to the existing table.
        index : bool, default False
            Write DataFrame index as a column. Uses `index_label` as the column
            name in the table.
        index_label : str or sequence, default None
            Column label for index column(s). If None is given (default) and
            `index` is True, then the index names are used.
            A sequence should be given if the DataFrame uses MultiIndex.
        chunksize : int, optional
            Specify the number of rows in each batch to be written at a time.
            By default, all rows will be written at once.
        dtype : dict or scalar, optional
            Specifying the datatype for columns. If a dictionary is used, the
            keys should be the column names and the values should be the
            SQLAlchemy types or strings for the sqlite3 legacy mode. If a
            scalar is provided, it will be applied to all columns.
        kwargs:
            any additional values present in - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html
    """
    if project_name is None:
        project_name = _generate_random_project_id()
        # if it's a new generated project_name/db_name then there is no way it can be reused as we won't know the name
        # unless someone checks the db folder manually, so removing db after query
        remove_db = True

    if type(df) == list:
        assert type(table_name) == list, "if df variable is list then table_name variable is also supposed to be a list"
        assert len(df) == len(table_name), "df list & table name list are supposed to be of same size"
    sql_instance = __get_sql_instance(db_name=project_name, reuse_existing=True, engine=engine)
    if (df is not None) and (table_name is not None):
        df_list = df
        table_name_list = table_name
        if type(df) != list:
            df_list = [df]
            table_name_list = [table_name]
        for t_name, df_obj in zip(table_name_list, df_list):
            sql_instance.add_dataframe_to_db(df_obj, t_name, engine=engine, schema=schema, if_exists=if_exists,
                                             index=index, index_label=index_label,
                                             chunksize=chunksize, dtype=dtype, **kwargs)
    if remove_db:
        sql_instance.close()

def close():
    global __sql_instances
    for sql_instance in __sql_instances:
        sql_instance.close()
    __sql_instances = {}
