import os
import unittest

import numpy as np
import pandas as pd

from utils.test_utils import test_resources_folder
from utils.constants import WEEK_COL, PRODUCT_COL, SALES_DOLLAR_COL, ZIPCODE_COL
from utils.stats_helper import StatsHelper


sales_filepath = os.path.join(test_resources_folder,'sales_data_test.csv')
sales_filepath2 = os.path.join(test_resources_folder,'sales_data_test2.csv')
population_filepath = os.path.join(test_resources_folder,'population_data_test.csv')

def __sort_multidimensional_array(a, index=0):
    # https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
    return a[a[:,index].argsort()]

def test_get_average_sales1():
    df = pd.read_csv(sales_filepath)
    out_expected = np.array([['id_1', 10.0],
                            ['id_2', 10.0]],dtype=object)

    t = StatsHelper.get_average_sales(df,week_col=WEEK_COL, strategy='exclude_missing')
    out_expected = __sort_multidimensional_array(out_expected,0)
    out = __sort_multidimensional_array(t.reset_index().values,0)
    np.testing.assert_array_equal(out,out_expected, err_msg="The average sales value per week per store doesn't match, for exclude_missing case")

def test_get_average_sales2():
    df = pd.read_csv(sales_filepath)
    out_expected = np.array([['p1', 2.5],
                            ['p2', 0.625],
                            ['p3', 6.875]],dtype=object)

    t = StatsHelper.get_average_sales(df, aggregation_col=PRODUCT_COL, week_col=WEEK_COL, strategy='include_missing')
    # TODO order shouldn't matter for store_ids but for values it would be sorted
    np.testing.assert_array_equal(t.reset_index().sort_values(by='product').values,out_expected, err_msg="The average sales value per week per product doesn't match for include missing case")

def test_get_average_sales3():
    df = pd.read_csv(sales_filepath)
    out_expected = np.array([['p1', 2.5],
                            ['p2', 0.625],
                            ['p3', 9.1666]],dtype=object)

    t = StatsHelper.get_average_sales(df, aggregation_col=PRODUCT_COL, week_col=WEEK_COL, strategy='include_missing_except_oldest')
    # TODO order shouldn't matter for store_ids but for values it would be sorted
    np.testing.assert_array_almost_equal(t.reset_index().sort_values(by='product')[SALES_DOLLAR_COL].values,out_expected[:,-1], decimal =4,err_msg="The average sales value per week per product doesn't match for include missing case except oldest")

def test_get_average_sales4():
    df = pd.read_csv(sales_filepath2)
    out_expected = np.array([['p1', 2.5],
                            ['p2', 0.625],
                            ['p3', 6.875]],dtype=object)

    t = StatsHelper.get_average_sales(df, aggregation_col=PRODUCT_COL, week_col=WEEK_COL, strategy='include_missing')
    # TODO order shouldn't matter for store_ids but for values it would be sorted
    np.testing.assert_array_equal(t.reset_index().sort_values(by='product').values,out_expected, err_msg="The average sales value per week per product doesn't match for include missing case")

def test_get_average_sales_mean1():
    df = pd.read_csv(sales_filepath)
    out_expected = 10

    t = StatsHelper.get_average_sales_mean(df, week_col=WEEK_COL, strategy='exclude_missing')
    np.testing.assert_array_equal(t,out_expected, err_msg="The average sales value mean (mean of sales per week per week per store) doesn't match")


def test_get_total_sales1():
    df = pd.read_csv(sales_filepath)
    out_expected = np.array([['id_1', 50],
                            ['id_2', 30]],dtype=object)

    t = StatsHelper.get_total_sales(df)
    # TODO order shouldn't matter for store_ids but for values it would be sorted
    np.testing.assert_array_equal(t.reset_index().values,out_expected, err_msg="The total sales value per week per store doesn't match")

def test_get_total_sales_mean1():
    df = pd.read_csv(sales_filepath)
    out_expected = 40

    t = StatsHelper.get_total_sales_mean(df)
    np.testing.assert_array_equal(t,out_expected, err_msg="The total sales mean value (mean of total_sales/store) doesn't match")

def test_get_average_population1():
    df = pd.read_csv(population_filepath)
    out_expected = np.array([['id_9', 50000.0],
                    ['id_8', 50000.0],
                    ['id_7', 50000.0],
                    ['id_6', 50000.0],
                    ['id_5', 40000.0],
                    ['id_4', 40000.0],
                    ['id_3', 40000.0],
                    ['id_2', 40000.0],
                    ['id_1', 40000.0]], dtype=object)

    t = StatsHelper.get_average_population(df)
    print(t.reset_index().values)
    out_expected = __sort_multidimensional_array(out_expected)
    out = __sort_multidimensional_array(t.reset_index().values,0)
    np.testing.assert_array_equal(out,out_expected, err_msg="The population per store doesn't match")

def test_get_average_population2():
    df = pd.read_csv(population_filepath)
    out_expected = np.array([[95125, 50000],
   [95124, 40000]], dtype=np.int64)
    t = StatsHelper.get_average_population(df,ZIPCODE_COL)
    # TODO order shouldn't matter for store_ids but for values it would be sorted
    np.testing.assert_array_equal(t.reset_index().values,out_expected, err_msg="The population per zipcode doesn't match")

def test_get_average_population_mean1():
    df = pd.read_csv(population_filepath)
    out_expected = 44444.444444444445
    t = StatsHelper.get_average_population_mean(df)
    # TODO order shouldn't matter for store_ids but for values it would be sorted
    np.testing.assert_array_almost_equal(t,out_expected, err_msg="The population per store mean doesn't match")

def test_get_average_population_mean2():
    df = pd.read_csv(population_filepath)
    out_expected = 45000
    t = StatsHelper.get_average_population_mean(df,ZIPCODE_COL)
    # TODO order shouldn't matter for store_ids but for values it would be sorted
    np.testing.assert_array_equal(t,out_expected, err_msg="The population per zipcode mean doesn't match")

if __name__=="__main__":
    test_get_average_sales1()
    # test_get_average_population1()