import logging
import math

import numpy as np
import pandas as pd

from utils import date_utils
from utils.constants import STORE_ID_COL, WEEK_END_COL, SALES_DOLLAR_COL, PRODUCT_COL, WEEK_COL, POPULATION_COUNT_COL


class StatsHelper(object):
    def __init__(self,**kwargs):
        print('init stats')

#     @classmethod
#     @staticmethod
    # def get_average_sales(df,aggregation_col=STORE_ID_COL, week_col=WEEK_END_COL, sales_col=SALES_DOLLAR_COL):
    def get_average_sales(df,aggregation_col=STORE_ID_COL, week_col=WEEK_END_COL, sales_col=SALES_DOLLAR_COL, additional_columns=None, strategy='exclude_missing', suppress_exception=False):
        # https://stackoverflow.com/questions/15074821/python-passing-parameters-by-name-along-with-kwargs
        """
        This method return the average sales per aggregation col (eg:- store_id)
        Arguments :
        aggregation_col:String
            aggregation column to use. Eg: store_id, zipcode, product
        week_col:String
            week column name to use. Eg: week, {WEEK_END_COL}
        sales_col:String
            sales column name. Eg:- sales_dollar
        strategy:string
            possible_values = include_missing, exclude_missing, include_missing_except_oldest
            include_missing
                Even if a particular store (aggregation_col value) is missing from the sales for a week, that will be included in calculating average
                i.e keep a value of 0 (zero) for the missing store (aggregation_col) and calculate the average value
                Eg:- For a store for some weeks data is missing, with this strategy that store will have value of zero for that week
                i.e if the store1 data is present for 10 weeks, with a total sales of $100 but remaining stores is present for 20 weeks.
                average would be 100/20 = 5 (& not 100/10)

            exclude_missing
                Exclude missing values while calculating average
                Eg:- For a store for some weeks data is missing, with this strategy that store is not included in that weeks numbers
                i.e if the store1 data is present for 10 weeks, with a total sales of $100 but remaining stores is present for 20 weeks.
                average would be 100/10 = 10 (& not 100/20)
                Note:- If a value have an entry even if it's 0 (zero), it will be considered irrespective of this value

            include_missing_except_oldest
                Include missing values except when the missing values are for the oldest weeks
                Use cases
                    1) A product (aggregation_col) is recently/newly introduced & it would be missing for the older weeks, but we shouldn't penalize that
                    i.e we will ignore older weeks while calculating the average
                    2) Similarly if a new store is introduced
                Eg:- If a product had 10 weeks of sales data, with a total sales of $100 & other products had 20 weeks of data & all the missing weeks for this product occured at the
                    beginning i.e week1 to week10, & data is present for week11 to week20 then average is calculated considering week11 to week20
                    i.e avg = 100/10 = 10
                Eg:- If a product had 10 weeks of sales data, with a total sales of $100 & other products had 20 weeks of data & 5 of the missing weeks occured in the beginning
                    & another 5 in between, i.e for week1 to week5 data is missing & data is present for week5 to week10, missing for week11 to week15 & again present for
                    week15 to week20. i.e for avg we consider 10 available + 5 missing = 15 weeks
                    i.e avg = 100/15 = 6.67



            if aggregation_col is product, generally this value will have to be include_missing or include_missing_except_oldest
                1)When we go to product granularity, some stores may not have sold that product for that week, but should be considered as 0 for that store,
                else the average for that product will be inflated
                2)For some products it may not have featured in older weeks as it was recently introduced, so for those cases we should not include them in the average

            if aggregation_col is store_id this is debatable,
                1) if a store_id is missing from a week then the client would accidently missed this store
                2) for the product list that we have, that particular store didn't have any sales, hence it's value was zero &
                client removed it from the dataset

            default_value: False
        """
        # TODO using variables in doc strings - https://stackoverflow.com/questions/10307696/how-to-put-a-variable-into-python-docstring
#         https://stackoverflow.com/questions/735975/static-methods-in-python
        t = df.copy()
        if aggregation_col == PRODUCT_COL and (strategy == 'exclude_missing'):
            error_message = "generally if aggregation column is product we would want to include missing values for aggregate calculation"
            if suppress_exception:
                logging.error(error_message)
            else:
                raise Exception(error_message)
        if strategy=='exclude_missing':
            #there will be some stores for which there won't be a value every week
            t = t.groupby([aggregation_col,week_col])[sales_col].agg('sum').groupby(aggregation_col).agg('mean')
            t.sort_values(ascending=False,inplace=True)
        elif strategy=='include_missing':
            t = StatsHelper._get_average_sales_include_missing(df,aggregation_col, week_col, sales_col, additional_columns)
        elif strategy=='include_missing_except_oldest':
            t = StatsHelper._get_average_sales_include_missing_except_oldest(df,aggregation_col, week_col, sales_col, additional_columns)

        else:
            # FIXME temporary default
            t = t.groupby([aggregation_col,week_col])[sales_col].agg('sum').groupby(aggregation_col).agg('mean')
            t.sort_values(ascending=False,inplace=True)
        return t

    def _get_average_sales_include_missing(df,aggregation_col=STORE_ID_COL, week_col=WEEK_END_COL, sales_col=SALES_DOLLAR_COL, additional_columns=None):
        t = df.copy()
        t = StatsHelper.fill_missing(t,col_to_fill=aggregation_col,week_col=week_col,sales_col=sales_col,additional_columns=additional_columns)
        t = t.groupby([aggregation_col,week_col])[sales_col].agg('sum').groupby(aggregation_col).agg('mean')
        t.sort_values(ascending=False,inplace=True)
        return t

    def fill_missing(df, col_to_fill=STORE_ID_COL, week_col=WEEK_COL, sales_col=SALES_DOLLAR_COL, additional_columns=None):
        t = df.copy()
        # total no of unique value for the aggregation column
        total_agg_set = set(t[col_to_fill].unique())
        # creating a dataframe with column value = [missing_values]
        t = t.groupby([week_col])[col_to_fill].apply(lambda l: list(total_agg_set - set(l)))
        # getting it from index to column
        t = t.reset_index()
        #creating a new dataframe with columns = week_col,col_to_fill & values as the missing col_to_fill values for that week
        # https://medium.com/@sureshssarda/pandas-splitting-exploding-a-column-into-multiple-rows-b1b1d59ea12e
        # https://stackoverflow.com/questions/17116814/pandas-how-do-i-split-text-in-a-column-into-multiple-rows/17116976#17116976
        t1 = pd.DataFrame(t[col_to_fill].to_list(),index=t[week_col]).stack()
        t1 = t1.reset_index(0)
        t1.columns=[week_col,col_to_fill]
        # since that col_to_fill (eg:PRODUCT_COL) is missing from that week, sales for that value = 0
        t1[sales_col] = 0
        # taking the main dataframe again
        t = df.copy()
        # selecting required columns from it
        required_columns = [week_col,col_to_fill,sales_col]
        t = t[required_columns]
        # appending the dataframe with entries for missing values
        t = t.append(t1)
        return t

    def _get_average_sales_include_missing_except_oldest(df,aggregation_col=STORE_ID_COL, week_col=WEEK_END_COL, sales_col=SALES_DOLLAR_COL, additional_columns=None):
        t = df.copy()
        t = StatsHelper._fill_missing_except_oldest(t,aggregation_col=aggregation_col,week_col=week_col,sales_col=sales_col,additional_columns=additional_columns)
        # the mean calculation
        t = t.groupby([aggregation_col,week_col])[sales_col].agg('sum').groupby(aggregation_col).agg('mean')
        t.sort_values(ascending=False,inplace=True)
        return t


    def _fill_missing_except_oldest(df,aggregation_col=STORE_ID_COL, week_col=WEEK_COL, sales_col=SALES_DOLLAR_COL, additional_columns=None):
        t = df.copy()
        # if week column is not of datetime type, then converting to datetime
        if not date_utils.is_column_date(t,week_col):
            t[week_col] = pd.to_datetime(t[week_col])
        # total no of unique values for dates
        total_dates = t[week_col].unique()
        # creating a dataframe with column value = [missing_dates_after_product_launch_to_fill]
        # assuming product launch from the oldest date available for that aggregation_col (eg:- product or store_id)
        t = t.groupby([aggregation_col])[week_col].apply(lambda l : StatsHelper.__pick_dates_excluding_oldest_to_fill(l,total_dates))
        # getting aggregation_col from index to column
        t = t.reset_index()
        # creating a new dataframe with columns = aggregation_col,week_col & values as the missing week values for that aggregation_col (eg:- product or store_id)
        # https://medium.com/@sureshssarda/pandas-splitting-exploding-a-column-into-multiple-rows-b1b1d59ea12e
        # https://stackoverflow.com/questions/17116814/pandas-how-do-i-split-text-in-a-column-into-multiple-rows/17116976#17116976
        t1 = pd.DataFrame(t[week_col].to_list(),index=t[aggregation_col]).stack()
        t1 = t1.reset_index(0)
        t1.columns=[aggregation_col,week_col]
        # since that aggregation_col (eg:PRODUCT_COL) is missing from that week, sales for that value = 0
        t1[sales_col] = 0
        cols_to_pick = [week_col,aggregation_col,sales_col]
        # t1 = t1[cols_to_pick]
        # taking the main dataframe again
        t = df.copy()
        # # selecting required columns from it
        t = t[cols_to_pick]
        # appending the dataframe with entries for missing values.
        # Note- order of columns in the missing values dataframe doesn't matter, pandas uses column names to match it
        t = t.append(t1)
        return t


    def __pick_dates_excluding_oldest_to_fill(dates_groupby, total_dates):
        total_dates = np.sort(np.unique(total_dates))
        dates_current_product = np.sort(np.unique(dates_groupby))
        first_date_current_product = dates_current_product[0]
        # (oldest) dates to exclude because the product got released after this date
        dates_to_exclude = []
        for d in total_dates:
            if d < first_date_current_product:
                # keep appending till the 1st
                dates_to_exclude.append(d)
            else:
                break
        # adding current dates present for this product to the exclusion list as we
        # want to keep their original values
        dates_to_exclude.extend(dates_current_product)
        dates_to_fill = [d for d in total_dates if d not in dates_to_exclude]
        return dates_to_fill

#     @staticmethod
    def get_average_sales_mean(df,aggregation_col=STORE_ID_COL, week_col=WEEK_END_COL, sales_col=SALES_DOLLAR_COL, strategy='include_missing', suppress_exception=False):
        """
        This method return the average sales (average per aggregation col averaged again (eg:- store_id))
        Arguments :
        aggregation_col:String
            aggregation column to use. Eg: store_id, {STORE_ID_COL}, {ZIPCODE_COL}
        week_col:String
            week column name to use. Eg: week, {WEEK_END_COL}
        sales_col:String
            sales column name. Eg:- sales_dollar
        """
#         https://stackoverflow.com/questions/35459360/how-to-call-static-methods-inside-the-same-class-in-python
        t = StatsHelper.get_average_sales(df,aggregation_col, week_col, sales_col, strategy=strategy).copy()
        return t.agg('mean')

    def get_total_sales(df,aggregation_col=STORE_ID_COL, sales_col=SALES_DOLLAR_COL):
        """
        This method return the total sales per aggregation col (eg:- store_id))
        Arguments :
        aggregation_col:String
            aggregation column to use. Eg: store_id, {STORE_ID_COL}, {ZIPCODE_COL}
        sales_col:String
            sales column name. Eg:- sales_dollar
        """
        t = df.copy()
        t = t.groupby([aggregation_col])[sales_col].agg('sum')
        t.sort_values(ascending=False,inplace=True)
        return t

    def get_total_sales_mean(df,aggregation_col=STORE_ID_COL, sales_col=SALES_DOLLAR_COL):
        """
        This method return the total sales mean (total per aggregation col averaged again (eg:- store_id))
        Arguments :
        aggregation_col:String
            aggregation column to use. Eg: store_id, {STORE_ID_COL}, {ZIPCODE_COL}
        sales_col:String
            sales column name. Eg:- sales_dollar
        """
        t = StatsHelper.get_total_sales(df,aggregation_col,sales_col)
        return t.agg('mean')

    def get_average_population(df,aggregation_col=STORE_ID_COL,population_col=POPULATION_COUNT_COL):
        t = df.copy()
        t = t.groupby([aggregation_col])[population_col].agg('mean')
        t.sort_values(ascending=False,inplace=True)
        return t

    def get_average_population_mean(df,aggregation_col=STORE_ID_COL,population_col=POPULATION_COUNT_COL):
        t = StatsHelper.get_average_population(df,aggregation_col,population_col)
        return t.agg('mean')

    def get_quantile_index(quantile_percent,quantile_steps=0.05):
        return math.ceil(quantile_percent/quantile_steps)

    def greater_than_quantile(store_list,mean_dict,q,quantile_percent):
        quantile_index=StatsHelper.get_quantile_index(quantile_percent)
        return [store for store in store_list if mean_dict.get(store)>=q[quantile_index]]

    def less_than_quantile(store_list,mean_dict,q,quantile_percent):
        quantile_index=StatsHelper.get_quantile_index(quantile_percent)
        return [store for store in store_list if mean_dict.get(store)<q[quantile_index]]