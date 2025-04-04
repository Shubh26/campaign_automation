#!/usr/bin/env python
# coding: utf-8

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from utils.constants import DATE, WEEK_COL

# constants
# this ISO format is just having date, can include another one for date & time
DATE_FORMAT_ISO = "%Y-%m-%d"
DATE_FORMAT_TIMESTAMP = '%Y%m%d%H%M%S'
DATE_FORMAT_TIMESTAMP_SHORT = '%Y%m%d'


def get_timestamp(date_format=DATE_FORMAT_TIMESTAMP):
    return datetime.utcnow().strftime(date_format)


def find_date_format_df(df: pd.DataFrame, date_col=DATE, max_diff=1, suppress_exceptions=False):
    date_string_list = df[date_col].unique()
    return find_date_format(date_string_list, max_diff, suppress_exceptions)


def find_date_format(date_string_list, max_diff=1, suppress_exceptions=False):
    # TODO accept even a single date entry rather than list
    """
    This script tries to identify the date format. Note this won't work if date formats are mixed
    date_string_list : a list of date strings
    max_diff : maximum permissible difference between dates given here, by default 1 day,
    this is used as another validation step
    suppress_exceptions:Boolean
        Keeping this as true will avoid an exception being thrown when the difference between days is more than the max_diff specified
        Note:- Exception will be thrown if a suitable date format is not found
    """
    unique_date_string = set(date_string_list)
    # check https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
    date_formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%Y %I:%M:%S %p",
        "%Y-%m-%dT%H:%M:%S.%f",
        # TODO keep format without spaces
        "%b %d, %Y",  # Jul 12, 2020
        '%Y-%b-%d',
        "%b %d, %Y",
        "%b%d,%Y",
        "%m-%d-%Y",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",

    ]
    matched = None
    unmatched_data = None
    error = None
    # first will check if the date parses, then will check if difference is date is one day
    for date_format in date_formats:
        try:
            l = [datetime.strptime(x, date_format) for x in unique_date_string]
            # check if difference in time is at max one day, in our use cases we get data daily
            l = sorted(l)
            diff = [(d2 - d1).days for d1, d2 in zip(l, l[1:])]
            for i, d in enumerate(diff):
                if d > max_diff and not suppress_exceptions:
                    raise ValueError("There is more ", max_diff,
                                     " days difference between dates ", l[i], l[i + 1],
                                     " while using ", date_format)
            matched = date_format
            break
        except ValueError as e:
            logging.debug(e)
            # print(e)
    if not matched:
        date_string_list = list(map(lambda x: f'"{x}"', unique_date_string))
        sample_date_string = ",".join(date_string_list[:52])
        raise ValueError(
            f'None of the predefined date formats matched, specify date format manually. Sample date provided {sample_date_string}')
    return matched


def get_date(date_string: str, date_format: str = DATE_FORMAT_ISO):
    """
    Given a date string convert it to datetime object
    Arguments:
        date_string:str
            A date string you want to covert to datetime object
            Eg:-"2021-10-19"
        date_format:str
            date format string as defined in - https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
            default-"%Y-%m-%d" (the iso format)
    """
    return datetime.strptime(date_string, date_format)


def find_start_of_week(date: datetime, start_day: str = 'sun'):
    """
    Find the start date of the week given a date & day
    Arguments:
        date:datetime.datetime object
            the date for which we want to find start of week
        start_day:str
            the week_day to consider as start_day. Possible options = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
            default:"sun"
    """
    # Eg:- if we want the week to start with Sunday, Sunday's value is 7 in ISO weekday

    week_days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    start_day = __get_standardized_week_day_month(start_day)
    __validate_week_day(start_day)
    week_days_index = {v: i + 1 for i, v in enumerate(week_days)}
    required_day_index = week_days_index[start_day]
    # 3 cases
    # 1) if required_day == actual_day, so days_diff=0
    # 2) required_day < actual_day, then we have to subtract days_dff=actual_day-required_day
    #   Eg:- if required day is tuesday & actual day is wed, then we subtract 1
    # 3) required_day > actual_day
    #   Eg:- if required_day=tue & actual_day=mon, then have to subtract 6  (i.e 7-1), i.e selecting tue from previous week
    # modulo operator (%) for negative numbers -1%7 = 6, -2%7=5 etc, for +ve numbers 1%7=1, 2%7=2 etc
    # https://stackoverflow.com/questions/1907565/c-and-python-different-behaviour-of-the-modulo-operation
    # https://stackoverflow.com/questions/3883004/the-modulo-operation-on-negative-numbers-in-python
    # https://stackoverflow.com/questions/11720656/modulo-operation-with-negative-numbers
    days_diff = (date.isoweekday() - required_day_index) % 7
    start_of_week = date - timedelta(days=days_diff)
    return start_of_week

def find_end_of_week(date, end_day: str = 'sat'):
    """
    Find the start date of the week given a date & day
    Arguments:
        date:datetime.datetime object
            the date for which we want to find start of week
        end_day:str
            the week_day to consider as start_day. Possible options = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
            default:"sat"
    """
    # Eg:- we want the end of week to be Saturday,
    # isoweekday() value starts with 1 for Mon & for Sunday = 7
    week_days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    end_day = __get_standardized_week_day_month(end_day)
    __validate_week_day(end_day)
    week_days_index = {v: i for i, v in enumerate(week_days)}
    required_day_index = week_days_index[end_day]
    # start_day will be the next day in circular list
    # Eg:- if end_day = sat, then start_day will be sun
    # find the start date for the given end date & add six to it to get end day
    start_day_index = (required_day_index + 1) % 7
    start_day = week_days[start_day_index]
    start_of_week = find_start_of_week(date, start_day)
    end_of_week = start_of_week + timedelta(days=6)
    return end_of_week


def __validate_week_day(day):
    """
    Validate that the week day mentioned is in the format we require
    """
    week_days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    assert day in week_days, f"day value {day} not valid, should be one of the following string {','.join(week_days)}"

def __get_standardized_week_day_month(day_or_month):
    """
    convert week_day (sun, mon) or months ('jan', feb, ) etc to standard form - lower case & 3 characters
    """
    return day_or_month.lower()[:3]

def add_end_of_week(df: pd.DataFrame, date_column:str=DATE, output_column:str=WEEK_COL, day:str='sat'):
    """
    Given a dataframe & it's date column add a column for week ending date
    Assumption- the entire dataframe consists of data only for 1 week & it have daily data for the week
    Arguments:
    df:pandas dataframe
        dataframe with the date_column
    date_column:str
        The date column name
    output_column:str
        Name to be given for the output column
    day:str
        end day of the week, 3 letter format of da
        default:sat, default end of week is considered to be saturday
    """
    t = df.copy()
    day = __get_standardized_week_day_month(day)
    __validate_week_day(day)
    sample_date = convert_numpy_datetime_to_datetime(t[date_column].unique()[0])
    t[output_column] = find_end_of_week(sample_date)
    return t


def is_column_date(df, column_name):
    """
    Identify whether the column is a date column
    """
    return df.dtypes[column_name].str == '<M8[ns]'


def is_date(date):
    """
    Check if the passed value is a date object
    """
    return (date is not None) and type(date) == datetime


def convert_column_to_date(df, column_name, date_format=None):
    """
    Given a particular column & date format this function will return a dataframe
    with the specified column converted to date type
    If date_format is not provided then this function will try to identify the date_format & perform the same operation

    Note- It's better to use the function - date_utils.find_date_format to identify date format validate it & provide that as input here

    Arguments :
    df: pandas dataframe
        The pandas dataframe with the column to convert to date
    column_name:String
        The column to convert to date type
    date_format:String
        Input date format string for the date provided
        If this is not provided, this function tries to identify date format. But try to provide the format whenever it's possible,
        as the identified date format maynot be correct

    """
    t = df.copy()
    if not date_format:
        date_format = find_date_format(t[column_name].unique(), suppress_exceptions=True)
    t[column_name] = pd.to_datetime(t[column_name], format=date_format)
    return t


def get_first_day_date(year, month:str='apr', day:str='sun'):
    """
    This function is used to get the first weekday of a month
    Eg: We want to check what is the 1st monday of a financial year

    Arguments:
        year:integer
            4 digit format of date. Eg:- 2020
        month:string
            3 word format of date. Eg:- apr, mar
        day:string
            3 word format for day. Eg:- sun, mon

    """
    # Sunday's value is 7 in ISO weekday
    week_days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    week_days_index = {v: i + 1 for i, v in enumerate(week_days)}
    day = day.lower()
    __validate_week_day(day)
    month = month.lower()[:3] # taking only 1st 3 characters
    valid_months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    assert month in valid_months, f"month value {month} not valid, should be one of the following string {','.join(valid_months)}"
    date_temp_str = f'{year}-{month}-01'
    date_format = '%Y-%b-%d'
    required_day_index = week_days_index[day]
    date_temp = get_date(date_temp_str, date_format)
    diff = (7 - date_temp.isoweekday()) + required_day_index
    # this is to handle if the 1st of month itself is the required day
    diff %= 7
    date = date_temp + timedelta(diff)
    return date


def convert_numpy_datetime_to_datetime(numpy_datetime_object):
    """
    Given a numpy datetime object, i.e np.datetime64, it will be converted to
    python datetime.datetime object

    Arguments:
    numpy_datetime_object: np.datetime64 object or an array/list of np.datetime64 object
    """
    # https://stackoverflow.com/questions/13703720/converting-between-datetime-timestamp-and-datetime64
    # there are other methods listed in that stackoverflow page, but since this relies on the pandas one,
    # hope is that in the future if any changes/fixes are done to the datetime objects it will by the pandas team to their function
    out = pd.to_datetime(numpy_datetime_object).to_pydatetime()
    return out


def find_diff_between_dates(dates, dates2=None):
    """
    Find the date between consecutive date objects in the 1st list
    or find difference between corresponding objects between 1st & 2nd date object/list

    Arguments:
    dates: list of dates or a date object
        dates list or date object to find the
    dates2:list of dates or a date object
        The 2nd list with which we want to compare the date difference.
        If this is None or empty then difference is calculated between consecutive values in the 1st list
        default:None

    """
    # assuming the dates are in numpy datetime format
    diff = None
    if not dates2:
        diff = [convert_timedelta_to_day((d2 - d1)) for d1, d2 in zip(dates, dates[1:])]
    else:
        # this will only take common no of elements from both lists, TODO maybe throw an exception is length is different
        diff = [convert_timedelta_to_day((d2 - d1)) for d1, d2 in zip(dates, dates2)]
    return diff


def convert_timedelta_to_day(delta: timedelta):
    """
    convert numpy timedelta in ns to day
    Arguments:
        delta:numpy.timedelta
    """
    # https://stackoverflow.com/questions/18215317/extracting-days-from-a-numpy-timedelta64-value
    return delta.astype('timedelta64[D]') / np.timedelta64(1, 'D')


def get_start_end_dates(df, date_col=DATE, date_format=DATE_FORMAT_ISO, return_date_obj=False):
    """
    Get the (start_date,end_date) strings as output from a dataframe column

    Arguments:
    df:pandas dataframe
        dataframe with the date_column
    date_col:string
        The date column in the data frame, the values in this column are expected to be of pandas(numpy) datetime object
    date_format:string
        The date format to return the date string
    return_date_obj:boolean
        Whether a datetime object needs to be returned or if a string have to be returned

    return (start_date,end_date)

    """
    dates = np.sort(df[date_col].unique())
    start_date = convert_numpy_datetime_to_datetime(dates[0])
    end_date = convert_numpy_datetime_to_datetime(dates[-1])
    if return_date_obj:
        return (start_date, end_date)

    start_date_string = start_date.strftime(date_format)
    end_date_string = end_date.strftime(date_format)
    return (start_date_string, end_date_string)


def add_days(date:datetime, days:int):
    """
    Given a date add given number of days & return the result
    Arguments:
        date:datetime.datetime object
        days:int
            number of days to add to the existing date
    """
    return date + timedelta(days)

def add_weeks(date:datetime, no_of_weeks:int):
    """
    Given a date add the specified number of weeks to it
    Arguments:
        date:datetime.datetime object
        no_of_weeks:int
            number of days to add to the existing date
    """
    return add_days(date, no_of_weeks * 7)

def get_date_string(date, date_format=DATE_FORMAT_ISO):
    """
    Given a datetime object get a string representation
    Arguments:
        date:datetime.datetime object
        date_format:str
            date format we want for the output string
            default - "%Y-%m-%d"
    """
    return date.strftime(date_format)



def __get_week_ending_date_df(df, week_number='week_number'):
    # TODO function needs to be removed
    """given dataframe and week number returns week ending date of the week"""
    t = df.copy()
    t = t.apply(lambda row: __get_calender_year_start_date(row, 'year'), axis=1)
    t['no_of_days_to_add'] = (t[week_number].astype(int)) * 7 - 1
    t = t.apply(lambda row: __add_days(row, 'cal_start_date'), axis=1)
    return t


def __get_calender_year_start_date(row, year_col, month='feb', day='sun', output_col_name='cal_start_date'):
    # TODO function needs to be removed
    """takes each row of dataframe to give start date of calender year"""
    year = int(row[year_col])
    row[output_col_name] = get_first_day_date(year, month, day)
    return row


def __add_days(row, date_col=DATE, days_to_add_col='no_of_days_to_add', output_col=DATE):
    # TODO function needs to be removed
    """takes each row of dataframe to give week ending date"""
    initial_date = row[date_col]
    no_of_days_to_add = row[days_to_add_col]
    row[output_col] = add_days(initial_date, no_of_days_to_add)
    return row
