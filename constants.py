#!/usr/bin/env python
# coding: utf-8
import copy


# constants used in metadata info
START_DATE='start_date'
END_DATE='end_date'
TABLE_START='table_start'
BRAND = 'brand'
CLIENT = 'client'
# retail chain eg:- Kroger, Albertsons
RETAIL_CHAIN='retail_chain'
FILE_EXTENSION="file_extension"

# START_OF_WEEK='start_of_week'
# END_OF_WEEK='end_of_week'
DATE='date'

WEEK_END_COL='week_ending_date'
WEEK_COL='week'
STORE_ID_COL = 'store_id'
STORE_ID_BEFORE_BANNER_COL = f'{STORE_ID_COL}_before_banner'
STORE_ADDRESS_COL = 'address'
# STORE_ADDRESS_COL = 'store_address'
STORE_CATEGORY_COL = 'store_category' # Eg:- for circle k - holiday vs circle_k
DISTRICT_COL='district'
DIVISION_col='division'
CITY_COL='city'
COUNTY_COL='county'
STATE_COL='state'
PHONE_COL='store_phone_num'
SALES_DOLLAR_COL='sales_dollars'
SALES_UNIT_COL = 'sales_units'
SALES_VOL_COL = 'sales_volume'
SALES_UNIT_WEIGHT_COL="unit_weight"

#column to use to indicate whether test/control
GROUP_COL = 'group'

ZIPCODE_COL='zipcode'
#TODO remove col from here in the value
# ZIPCODE_5DIGIT_COL="zipcode_5digit_col"
# @deprecated(reason="use zipcode")
ZIPCODE_5DIGIT_COL="zipcode_5digit"
# @deprecated(reason="use zipcode_expanded")
ZIPCODE_5DIGIT_EXPANDED='zipcode_5digit_expanded'
ZIPCODE_EXPANDED='zipcode_expanded'
ZIPCODE_5DIGIT_ACTUAL='zipcode_5digit_actual'

RADIUS_COL='radius'
DISTANCE_COL='distance'
IS_ORIGINAL_ZIPCODE_COL='is_original_zipcode'
LATITUDE_COL='latitude'
LONGITUDE_COL='longitude'
VALIDATED_COL='validated'
URL_COL='url'

POPULATION_COUNT_COL="population_count"

# Banner eg:- Jewel, Kroger market place etc
BANNER_COL='store_banner'
STORE_STATUS='status'

# product category
# CATEGORY_COL='category'
PRODUCT_CATEGORY='product_category' # the top level product group, level 1
PRODUCT_SUB_CATEGORY='product_sub_category' # level 2 product group
PRODUCT_CATEGORY_LEVEL3='product_category_level3' # level3 of product group
PRODUCT_CATEGORY_LEVEL4='product_category_level4' # level3 of product group
PRODUCT_COL='product' # the final granular level product name

RANK_COL='rank'
SCORE_COL='score'

# required in campaign data
GROUP_GENERATION_PARAMETER='group_generation_parameter'
GROUP_GENERATION_PARAMETER_SALES='sales'
GROUP_GENERATION_PARAMETER_POPULATION='population'
GROUP_GENERATION_PARAMETER_TEST_ONLY='test_only'

RETAIN_STRATEGY_PARAMETER = 'retain_stragegy'
RETAIN_STRATEGY_PARAMETER_RATIO = 'ratio'
RETAIN_STRATEGY_PARAMETER_CUSTOM = 'custom'
RETAIN_PERCENTAGE='retain_percentage'

# ignoring BANNER_COL, RETAIL_CHAIN as they might be missing for some cases
STORE_LIST_COLUMNS = [STORE_ID_COL, STORE_ADDRESS_COL, ZIPCODE_COL, CITY_COL, STATE_COL, URL_COL, LATITUDE_COL, LONGITUDE_COL ]
STORE_LIST_EXPANDED_COLUMNS = copy.deepcopy(STORE_LIST_COLUMNS)
STORE_LIST_EXPANDED_COLUMNS.extend([ZIPCODE_EXPANDED, IS_ORIGINAL_ZIPCODE_COL, VALIDATED_COL, DISTANCE_COL, RADIUS_COL])

SUCCESS = 'SUCCESS'