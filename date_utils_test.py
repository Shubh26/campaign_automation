from utils import date_utils
import pandas as pd
import datetime
import numpy as np
import os

# TODO this is not the best solution change this
test_resources_folder=os.path.join('resources','test')
# test_resources_folder=os.path.join(os.getcwd(),'test','resources')

def test_get_first_day_date1():
    date_format = '%Y-%b-%d'
    date_expected = date_utils.get_date('2019-aug-07',date_format)
    date_actual = date_utils.get_first_day_date(year=2019,month='aug',day='wed')

    np.testing.assert_array_equal(date_actual,date_expected,err_msg="expected date for first day function doesn't match for date greater than 1st of month")

def test_get_first_day_date2():
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    week_days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    expected_date_dict = {
        'jan': ['2020-01-06', '2020-01-07', '2020-01-01', '2020-01-02', '2020-01-03',  '2020-01-04', '2020-01-05'],
        'feb': ['2020-02-03', '2020-02-04', '2020-02-05', '2020-02-06', '2020-02-07', '2020-02-01', '2020-02-02'],
        'mar': ['2020-03-02', '2020-03-03', '2020-03-04', '2020-03-05', '2020-03-06', '2020-03-07', '2020-03-01'],
        'apr': ['2020-04-06', '2020-04-07', '2020-04-01', '2020-04-02', '2020-04-03', '2020-04-04', '2020-04-05'],
        'may': ['2020-05-04', '2020-05-05', '2020-05-06', '2020-05-07', '2020-05-01', '2020-05-02', '2020-05-03'],
        'jun': ['2020-06-01', '2020-06-02', '2020-06-03', '2020-06-04', '2020-06-05', '2020-06-06', '2020-06-07'],
        'jul': ['2020-07-06', '2020-07-07', '2020-07-01', '2020-07-02', '2020-07-03', '2020-07-04', '2020-07-05'],
        'aug': ['2020-08-03', '2020-08-04', '2020-08-05', '2020-08-06', '2020-08-07', '2020-08-01', '2020-08-02'],
        'sep': ['2020-09-07', '2020-09-01', '2020-09-02', '2020-09-03',  '2020-09-04', '2020-09-05', '2020-09-06'],
        'oct': ['2020-10-05', '2020-10-06', '2020-10-07', '2020-10-01', '2020-10-02', '2020-10-03', '2020-10-04'],
        'nov': ['2020-11-02', '2020-11-03', '2020-11-04', '2020-11-05', '2020-11-06', '2020-11-07', '2020-11-01'],
        'dec': ['2020-12-07', '2020-12-01', '2020-12-02', '2020-12-03',  '2020-12-04', '2020-12-05', '2020-12-06']
    }
    for month in months:
        for week_index, week_day in enumerate(week_days):
            expected_date_str = expected_date_dict.get(month)[week_index]
            date_expected = date_utils.get_date(expected_date_str)
            date_actual = date_utils.get_first_day_date(year=2020, month=month, day=week_day)
            assert date_actual==date_expected, "expected date for first day function doesn't match"

def test_convert_numpy_datetime_to_datetime1():
    date_expected = '2020-08-14'
    np_datetime = np.datetime64(date_expected)

    date_actual_datetime = date_utils.convert_numpy_datetime_to_datetime(np_datetime)
    try:
        # strftime function is not present for the numpy datetime object, so it would throw an error
        # but the python datetime.datetime object have this function
        date_actual = date_actual_datetime.strftime(date_utils.DATE_FORMAT_ISO)
        np.testing.assert_array_equal(date_actual,date_expected,err_msg="expected date for numpy date to python datetime conversion function doesn't match")
    except AttributeError:
        print("numpy datetime object not converted to python datetime")

def test_convert_numpy_datetime_to_datetime2():
    date_expected = ['2020-08-14','2020-08-15']
    np_datetime = [np.datetime64(dt) for dt in date_expected]

    date_actual_datetime = date_utils.convert_numpy_datetime_to_datetime(np_datetime)
    try:
        # strftime function is not present for the numpy datetime object, so it would throw an error
        # but the python datetime.datetime object have this function
        date_actual = [dt.strftime(date_utils.DATE_FORMAT_ISO) for dt in date_actual_datetime]
        np.testing.assert_array_equal(date_actual,date_expected,err_msg="expected date for numpy date to python datetime conversion function doesn't match for array conversion")
    except AttributeError:
        print("numpy datetime not converted to python datetime for a list of numpy datetime objects")

def test_find_start_of_week1():
    date_input = date_utils.get_date("2021-10-14")
    date_expected = date_utils.get_date("2021-10-10")
    date_output = date_utils.find_start_of_week(date_input)
    assert date_output==date_expected

def test_find_start_of_week2():
    date_input = date_utils.get_date("2021-10-20")
    date_expected = date_utils.get_date("2021-10-18")
    date_output = date_utils.find_start_of_week(date_input,'mon')
    assert date_output==date_expected

    week_days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    date_expected_list = ["2021-10-18", "2021-10-19", "2021-10-20", "2021-10-14", "2021-10-15", "2021-10-16", "2021-10-17"]

    for day, date_expected_string in zip(week_days,date_expected_list):
        date_output = date_utils.find_start_of_week(date_input, day)
        date_expected = date_utils.get_date(date_expected_string)
        assert date_output==date_expected


def test_find_start_of_week3():
    # checking corner cases of mon & sun. And actual days also being the same
    date_input = date_utils.get_date("2021-10-10")
    date_expected = date_utils.get_date("2021-10-10")
    date_output = date_utils.find_start_of_week(date_input,'sun')
    assert date_output==date_expected

    date_input = date_utils.get_date("2021-10-11")
    date_expected = date_utils.get_date("2021-10-11")
    date_output = date_utils.find_start_of_week(date_input, 'mon')
    assert date_output == date_expected

def test_find_end_of_week1():
    date_input = date_utils.get_date("2021-10-14")
    date_expected = date_utils.get_date("2021-10-16")
    date_output = date_utils.find_end_of_week(date_input)
    assert date_output==date_expected

def test_find_end_of_week2():
    date_input = date_utils.get_date("2021-10-20")
    date_expected = date_utils.get_date("2021-10-23")
    date_output = date_utils.find_end_of_week(date_input, 'sat')
    assert date_output == date_expected

    week_days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    date_expected_list = ["2021-10-25", "2021-10-26", "2021-10-20", "2021-10-21", "2021-10-22", "2021-10-23", "2021-10-24"]

    for day, date_expected_string in zip(week_days, date_expected_list):
        date_output = date_utils.find_end_of_week(date_input, day)
        date_expected = date_utils.get_date(date_expected_string)
        assert date_output == date_expected

def test_add_days():
    date_input = date_utils.get_date('2021-09-20')
    date_expected = date_utils.get_date('2021-09-29')
    date_actual = date_utils.add_days(date_input, 9)
    assert date_actual==date_expected

if __name__ == '__main__':
    test_get_first_day_date2()