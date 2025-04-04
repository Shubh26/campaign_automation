import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import date_utils, file_utils, zipcode_utils, file_processor, campaign_helper
from constants import *



class Eda(object):
    """
    this class methods can be used to perform eda and generate visualisation from processed sales data and not on raw data set.   please refer to sample data file "sales/measurement/sales_report_tool/resources/sample_data_for_eda.csv"
    """

#     def __init__(self, **kwargs):
#         print('init stats')

    def get_format_product_col(df):
        """
        given data frame it format product column
        :return:
        """
        new = df[PRODUCT_COL].str.split("-", expand=True)
        if len(new.columns) >= 2:
            df[PRODUCT_COL] = new[0]
        return df

    def get_store_product_week_count(df):
        """
        Given data frame it generate snapshot of store, product and week details
        :return:
        """
        t = pd.DataFrame(df.nunique()).reset_index()
        t.columns = ['column', 'distinct count']
        t = t[t['column'].isin(['product', 'week_ending_date', 'store_id'])].reset_index(drop=True)
        return t
    
    def get_store_avg_sales(df):
        """
        This produces store avg sales (store avg sales = sum store week sales/ no. store sales weeks)
        :return:
        """
        t = df.sort_values(WEEK_END_COL).groupby([STORE_ID_COL, WEEK_END_COL])[SALES_DOLLAR_COL].agg('sum').groupby([STORE_ID_COL]).agg('mean').reset_index()
        return t
    
    def get_product_avg_sales(df):
        """
        This produces product avg sales (product avg sales = sum product week sales/ no. product sales weeks)
        :return:
        """
        t = df.sort_values(WEEK_END_COL).groupby([PRODUCT_COL, WEEK_END_COL])[SALES_DOLLAR_COL].agg('sum').groupby([PRODUCT_COL]).agg('mean').reset_index()
        return t


class Product_eda(Eda):
    
    def get_product_plot_df(df):
        """
        Given data frame,  it produces product level aggregated data set i.e - product sales sum, week count, product sales in recent 13 weeks
        :return:
        """
        t = df.copy()
        product = t.groupby([PRODUCT_COL])[SALES_DOLLAR_COL].agg('sum').sort_values(ascending=False).reset_index()
        product['sales_sum_share'] = (round(
            (product[SALES_DOLLAR_COL] / product[SALES_DOLLAR_COL].sum()) * 100)).astype(int).astype(str) + '%'
        product_week_count = t.groupby([PRODUCT_COL])[WEEK_END_COL].nunique().sort_values(ascending=False).reset_index()
        product = pd.merge(product, product_week_count, on=PRODUCT_COL, how='inner')
        product.columns = ['product', 'sales_sum', 'sales_sum_share', 'product_week_count']
        end_date = date_utils.get_date(t[WEEK_END_COL].max(), date_utils.DATE_FORMAT_ISO)
        start_date = date_utils.add_days(end_date, -13 * 7)
        product_sales_in_recent_13wks = pd.DataFrame(
            t[t[WEEK_END_COL] > start_date.strftime("%Y-%m-%d")][PRODUCT_COL].unique(), columns=[PRODUCT_COL])
        product['recent_13wks'] = product[PRODUCT_COL].isin(product_sales_in_recent_13wks[PRODUCT_COL])
        product_sales_in_recent_13wks_metric = product.groupby(['recent_13wks'])[PRODUCT_COL].agg('count').reset_index()
        product_sales_in_recent_13wks_metric['product_share'] = round((product_sales_in_recent_13wks_metric[
                                                                           PRODUCT_COL] /
                                                                       product_sales_in_recent_13wks_metric[
                                                                           PRODUCT_COL].sum()) * 100).astype(
            int).astype(str) + '%'
        return product, product_sales_in_recent_13wks_metric

    def get_product_sales_plot(df, top_product='all'):
        """
        this takes aggregated data frame from function get_product_plot_df() to generate product vs sales plot
        :param df: product
        :param top_product: no. of product to display in plot
        """
        if top_product == 'all':
            top_product = df[PRODUCT_COL].nunique()
        plt.rcParams['figure.figsize'] = [20, 10]
        x = df[PRODUCT_COL].head(top_product)
        y = df['sales_sum'].head(top_product)
        plt.bar(x, y, color='c', label='product sales', alpha=0.4)
        for i in range(len(df['sales_sum_share'].head(top_product))):
            vertical_offset = (y[i] * 0.005).astype(int)
            plt.text(i, y[i] + vertical_offset, df['sales_sum_share'][i], ha='center', fontsize=9)
        plt.xlabel('product')
        plt.xticks(rotation=90, fontsize=8)
        plt.ylabel('product sales (USD)')
        plt.title('total sales at product level')
        plt.legend()
        plt.show()
        
    def get_product_sales_vs_week_count_plot(df):
        """
        this takes aggregated data frame from function get_product_plot_df() to generate product vs sales vs
        no. of weeks for which product have sales
        :param df: product
        """
        plt.rcParams['figure.figsize'] = [20, 10]
        x = df[PRODUCT_COL]
        y1 = df['sales_sum']
        y2 = df['product_week_count']
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        l1 = ax1.bar(x, y1, color='g', label='product sales', alpha=0.4)  # , label = 'sales dollar'
        l2 = ax2.scatter(x, y2, label='week count', alpha=0.9)  # , label = 'sales unit'
        xlabel = ax1.set_xlabel('product')
        ax1.tick_params(axis='x', labelsize=6)
        fig.autofmt_xdate(rotation=90)
        ax1.set_ylabel('product sales (USD)', color='g')
        ax2.set_ylabel('product week count ', color='b')
        ax1.set_title('product sales vs product week count')
        plt.show()

    def get_product_vs_recent_13wks_plot(df1, df2):
        """
        this takes data frames from get_product_plot_df() to display ratio of product count that have sales in recent 13 weeks
        :param df1: product
        :param df2: product_sales_in_recent_13wks_metric
        """
        chart = sns.countplot(df1['recent_13wks'], color='firebrick', alpha=0.8)
        for i in range(len(df2['recent_13wks'])):
            vertical_offset = (df2[PRODUCT_COL][i] * 0.009).astype(int)
            chart.text(i, df2[PRODUCT_COL][i] + vertical_offset, df2['product_share'][i], ha='center', fontsize=12)
        chart.set(ylabel='product count')
        chart.set_title('product that have sales in recent 13 weeks')
        plt.show()


class Store_eda(Eda):
        
    def get_store_plot_df(df):
        """
        Given data frame it produces store level aggregated data set i.e - store sales sum, week count, store sales in recent 13 weeks
        :return:
        """
        t = df.copy()
        # creating store sales, avg sales, count of week store has sales df
        store_sales = t.groupby([STORE_ID_COL])[SALES_DOLLAR_COL].agg('sum').sort_values(ascending=False).reset_index()
        store_sales['sales_sum_share'] = (round(
            (store_sales[SALES_DOLLAR_COL] / store_sales[SALES_DOLLAR_COL].sum()) * 100)).astype(int).astype(str) + '%'
        store_avg_sales = round(
            t.groupby([STORE_ID_COL, WEEK_END_COL])[SALES_DOLLAR_COL].agg('sum').groupby([STORE_ID_COL]).agg(
                'mean').reset_index(), 2)
        store_sales.rename(columns={SALES_DOLLAR_COL: 'sales_sum'}, inplace=True)
        store_avg_sales.rename(columns={SALES_DOLLAR_COL: 'sales_avg'}, inplace=True)
        store_sales = pd.merge(store_sales, store_avg_sales, on=STORE_ID_COL, how='inner')
        #
        store_sales_week_count = pd.DataFrame(
            t.groupby([STORE_ID_COL])[WEEK_END_COL].nunique().sort_values(ascending=False)).reset_index()
        store_sales_week_count.rename(columns={WEEK_END_COL: 'week_count'}, inplace=True)
        #
        store_sales_week_count_reverse = store_sales_week_count.groupby(['week_count'])[
            STORE_ID_COL].count().reset_index()
        store_sales_week_count_reverse.rename(columns={STORE_ID_COL: 'store_count'}, inplace=True)
        store_sales_week_count_reverse['store_count_share'] = round((store_sales_week_count_reverse['store_count'] /
                                                                     store_sales_week_count_reverse[
                                                                         'store_count'].sum()) * 100).astype(
            int).astype(str) + '%'
        # store have sales in recent 13 weeks
        end_date = date_utils.get_date(t[WEEK_END_COL].max(), date_utils.DATE_FORMAT_ISO)
        start_date = date_utils.add_days(end_date, -13 * 7)
        store_sales_in_recent_13wks = pd.DataFrame(
            t[t[WEEK_END_COL] > start_date.strftime("%Y-%m-%d")][STORE_ID_COL].unique(), columns=[STORE_ID_COL])
        store_sales_week_count['recent_13wks'] = store_sales_week_count[STORE_ID_COL].isin(
            store_sales_in_recent_13wks[STORE_ID_COL])
        store_sales = pd.merge(store_sales, store_sales_week_count, on=STORE_ID_COL, how='inner')
        #
        store_sales_in_recent_13wks_metric = store_sales_week_count.groupby(['recent_13wks'])[STORE_ID_COL].agg(
            'count').reset_index()
        store_sales_in_recent_13wks_metric.rename(columns={STORE_ID_COL: 'store_count'}, inplace=True)
        store_sales_in_recent_13wks_metric['store_count_share'] = round((store_sales_in_recent_13wks_metric[
                                                                             'store_count'] /
                                                                         store_sales_in_recent_13wks_metric[
                                                                             'store_count'].sum()) * 100).astype(
            int).astype(str) + '%'

        return store_sales, store_sales_week_count_reverse, store_sales_in_recent_13wks_metric
    
    def get_store_count_vs_week_count_plot(df1, df2):
        """
        Given aggregated data frames generated from get_store_plot_df() it generate plot for store count vs week count
        :param df1: store_sales
        :param df2: store_sales_week_count_reverse
        """
        chart = sns.countplot(df1['week_count'], color='green', alpha=0.6)
        for i in range(len(df2['week_count'])):
            vertical_offset = (df2['store_count'][i] * 0.01).astype(int)
            chart.text(i, df2['store_count'][i] + vertical_offset, df2['store_count'][i], ha='center', fontsize=10)
        chart.set(xlabel='no of weeks')
        chart.set(ylabel='store count')
        chart.set_title('store count vs week count')
        plt.show()
        
    def get_store_vs_recent_13wks_plot(df1, df3):
        """
        this takes data frames from get_product_plot_df() to display ratio of store count that have sales in recent 13 weeks
        :param df1: store_sales
        :param df3: store_sales_in_recent_13wks_metric
        """
        chart = sns.countplot(df1['recent_13wks'], color='orange', alpha=0.8)
        for i in range(len(df3['recent_13wks'])):
            chart.text(i, df3['store_count'][i], df3['store_count_share'][i], ha='center', fontsize=10)
        chart.set(ylabel='store_count')
        chart.set_title('no of store that have sales in recent 12 wks')
        plt.show()


class Week_eda(Eda):
    
    def get_sales_vs_unit_vs_store_count_weekly_plot(df):
        """
        Provided data frame this generate twp weekly plots
        1. weekly sales dollar vs sales unit (no. of product unit)
        2. weekly sales dollar vs store count plot (no. of store that reported sales in respective weeks)
        """
        sales = df.sort_values(WEEK_END_COL).groupby([WEEK_END_COL])[SALES_DOLLAR_COL].agg('sum').reset_index()
        units = df.sort_values(WEEK_END_COL).groupby([WEEK_END_COL])[SALES_UNIT_COL].agg('sum').reset_index()
        store_count = df.sort_values(WEEK_END_COL).groupby([WEEK_END_COL])[STORE_ID_COL].nunique().reset_index()
        t = pd.merge(pd.merge(sales, units, on=WEEK_END_COL, how='inner'), store_count, on=WEEK_END_COL, how='inner')
        plt.rcParams['figure.figsize'] = [20, 10]
        x = t[WEEK_END_COL]
        y1 = t[SALES_DOLLAR_COL]
        y2 = t[SALES_UNIT_COL]
        y3 = t[STORE_ID_COL]

        plt.rcParams['figure.figsize'] = [20, 10]
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        l1 = ax1.plot(x, y1, color='g', label='sales dollar')
        l2 = ax2.plot(x, y2, linestyle='--', label='sales unit')
        xlabel = ax1.set_xlabel('week ending')
        fig.autofmt_xdate(rotation=90)
        ax1.set_ylabel('total sales (USD)', color='g')
        ax2.set_ylabel('total units', color='b')
        ax1.set_title('sales dollar vs sales unit')

        leg = l1 + l2
        labs = [l.get_label() for l in leg]
        ax1.legend(leg, labs, loc=0)

        plt.rcParams['figure.figsize'] = [20, 10]
        fig, ax3 = plt.subplots()
        ax4 = ax3.twinx()
        l3 = ax3.plot(x, y1, color='g', label='sales dollar')
        l4 = ax4.plot(x, y3, linestyle='--', label='store count')
        xlabel = ax3.set_xlabel('week ending')
        fig.autofmt_xdate(rotation=90)
        ax3.set_ylabel('total sales (USD)', color='g')
        ax4.set_ylabel('total stores', color='b')
        ax3.set_title('sales dollar vs store count')
        leg = l3 + l4
        labs = [l.get_label() for l in leg]
        ax3.legend(leg, labs, loc=0)
        plt.show()
    
    def get_product_vs_store_count_weekly_plot(df):
        """
        Provided data frame this generate product count vs store count  weekly plots (no. of product and store reported sales in respective weeks)
        """
        store_count = df.sort_values(WEEK_END_COL).groupby([WEEK_END_COL])[STORE_ID_COL].nunique().reset_index()
        weekly_product_count = df.sort_values(WEEK_END_COL).groupby([WEEK_END_COL])[PRODUCT_COL].nunique().reset_index()
        weekly_store_product_count = pd.merge(store_count, weekly_product_count, on=WEEK_END_COL, how='inner')

        plt.rcParams['figure.figsize'] = [20, 10]
        x = weekly_store_product_count[WEEK_END_COL]
        y1 = weekly_store_product_count[STORE_ID_COL]
        y2 = weekly_store_product_count[PRODUCT_COL]

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        l1 = ax1.plot(x, y1, color='g', label='store count')
        for i in range(len(y1)):
            vertical_offset = (y1[i] * 0.0002).astype(int)
            ax1.text(i, y1[i] + vertical_offset, y1[i], ha='center', fontsize=8)
        l2 = ax2.bar(x, y2, label='product count', alpha=0.3)
        ax2.axhline(y2.mean(), linestyle='--', color='blue', linewidth=2)
        ax2.text(x.max(), y2.mean() + 1, 'product count avg.: {}'.format(y2.mean().astype(int)), ha='right',
                 fontsize=11)

        xlabel = ax1.set_xlabel('week ending')
        ax1.tick_params(axis='x', labelsize=10)
        fig.autofmt_xdate(rotation=90)
        ax1.set_ylabel('store count', color='g')
        ax2.set_ylabel('product count', color='b')
        ax1.set_title('weekly store count vs product count')
        plt.show()
        
    def get_store_vs_product_avg_sales(df):
        """
        Given data frame this generate store and product avg. sales plot and mean value marked as green triangle
        """
        store_avg_sales = get_store_avg_sales(df)
        product_avg_sales = get_product_avg_sales(df)

        plt.figure(1, figsize=(10, 7))
        plt.subplot(121)
        plt.boxplot(store_avg_sales[SALES_DOLLAR_COL], showmeans=True)
        vertical_offset = store_avg_sales['sales_dollar'].mean() * 0.05
        plt.text(1, store_avg_sales['sales_dollar'].mean() + vertical_offset,
                 store_avg_sales['sales_dollar'].mean().round(1), ha='center', fontsize=10)
        plt.title('store avg sales')
        plt.xlabel('store')
        plt.ylabel('sales dollar (USD)')
        plt.subplot(122)
        plt.boxplot(product_avg_sales[SALES_DOLLAR_COL], showmeans=True)
        vertical_offset = store_avg_sales['sales_dollar'].mean() * 0.9
        plt.text(1, product_avg_sales['sales_dollar'].mean() + vertical_offset,
                 product_avg_sales['sales_dollar'].mean().round(1), ha='center', fontsize=10)
        plt.title('product avg sales')
        plt.xlabel('product')
        plt.axis([0, 2, 0, 10000])
        plt.show()