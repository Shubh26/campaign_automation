import pandas as pd
import os
from utils.file_utils import main_package_folder

"""
This file contains util functions for test cases
"""

test_resources_folder=os.path.join(main_package_folder, 'resources', 'test')
sales_data_variants_folder = os.path.join(test_resources_folder,"sales_data_variants")


def assert_dfs_equal(df_expected, df_out, expected_df_filepath):
    """
    Assert the the given dataframes are equal, this also provides helpful messages on why they differ
    """
    columns_expected = sorted(df_expected.columns)
    columns_actual = sorted(df_out.columns)
    assert columns_expected == columns_actual, f"expected columns & output columns not matching they are \n{columns_expected}\n{columns_actual}"
    df_out = df_out[columns_actual]
    df_out = df_out.reset_index(
        drop=True)  # doing this to reset index to start with 0, since there is filteration logic the indexes would have changed
    df_expected = df_expected[columns_expected]
    pd.testing.assert_frame_equal(df_expected,
                                  df_out)  # pd.testing.assert_frame_equal will show the difference between dataframes as well
    # dataframe equals method was giving mismatch in some cases where a dataframe with float values
    # noticed this in a case where 1 of the dataframes was loaded from a file
    # assert df_expected.equals(
    #    df_out), "output after transformations have changed, output should have been like - %s " % expected_df_filepath
