from utils.zipcode_utils import *
import pandas as pd
import numpy as np
import os
from utils.test_utils import test_resources_folder

# TODO this is not the best solution change this
zipcodes_filepath = os.path.join(test_resources_folder,'zipcode_data_test.csv')
def test_get_dataframe_with_5digit_zipcode1():
    df = pd.read_csv(zipcodes_filepath)

    out_expected = ['00501', '01001', '00501', '01001', '10001', '10001']
    out = get_dataframe_with_5digit_zipcode(df,zipcode_actual=ZIPCODE_COL,zipcode_5digit=ZIPCODE_COL)[ZIPCODE_COL]
    np.testing.assert_array_equal(out,out_expected,err_msg="zipcodes not converted to 5 digit properly")

def test_get_dataframe_with_5digit_zipcode2():
    df = pd.read_csv(zipcodes_filepath)

    out_expected = ['00501', '01001', '50155', '10011', '10001', '10001']
    out = get_dataframe_with_5digit_zipcode(df,zipcode_actual='zipcode_numeric',zipcode_5digit=ZIPCODE_COL)[ZIPCODE_COL]
    np.testing.assert_array_equal(out,out_expected,err_msg="zipcodes not converted to 5 digit properly")

if __name__ == '__main__':
    test_get_dataframe_with_5digit_zipcode2()
