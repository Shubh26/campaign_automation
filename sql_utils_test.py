import os
import pandas as pd
from utils import sql_utils
from utils.test_utils import sales_data_variants_folder, assert_dfs_equal

def test_with_instance_creation_loading_and_simple_query():
    inp_data = os.path.join(sales_data_variants_folder, "header_to_transpose_processed.csv")
    df = pd.read_csv(inp_data)
    sql = sql_utils.SQL("test_db")
    sql.add_dataframe_to_db(df, "header_to_transpose_processed", index=True)
    df_out = sql.read_sql("select product, store, sales_dollar, sales_unit, volume_sales from header_to_transpose_processed")
    assert_dfs_equal(df, df_out, inp_data)

    df_out = sql.read_sql("select * from header_to_transpose_processed")
    df_out = df_out.drop("index", axis=1)
    assert_dfs_equal(df, df_out, inp_data)

    df_expected = pd.DataFrame({"count(*)":[20]})
    df_out = sql.read_sql("select count(*) from header_to_transpose_processed")
    assert_dfs_equal(df_expected, df_out, "")

    # loading data without index
    sql.add_dataframe_to_db(df, "header_to_transpose_processed", index=False)
    df_out = sql.read_sql("select * from header_to_transpose_processed")
    assert_dfs_equal(df, df_out, inp_data)

    sql.close()

def test_with_instance_creation_loading_and_execute_read_sql1():
    inp_data = os.path.join(sales_data_variants_folder, "header_to_transpose_processed.csv")
    df = pd.read_csv(inp_data)
    sql = sql_utils.SQL("test_db")
    sql.add_dataframe_to_db(df, "header_to_transpose_processed")
    df_out = sql.read_sql("select product, store, sales_dollar, sales_unit, volume_sales from header_to_transpose_processed")
    assert_dfs_equal(df, df_out, inp_data)

    sql.execute_sql("CREATE TEMP TABLE tmptable AS SELECT * from header_to_transpose_processed")
    df_out = sql.read_sql("select * from tmptable")
    assert_dfs_equal(df, df_out, inp_data)
    sql.close()

def test_read_sql():
    inp_data = os.path.join(sales_data_variants_folder, "header_to_transpose_processed.csv")
    df = pd.read_csv(inp_data)
    sql_expression = "select product, store, sales_dollar, sales_unit, volume_sales from header_to_transpose_processed"
    df_out = sql_utils.read_sql(sql=sql_expression, df=df, table_name="header_to_transpose_processed", project_name="test_db")
    assert_dfs_equal(df, df_out, inp_data)

    # executing without project_name
    df_out = sql_utils.read_sql(sql=sql_expression, df=df, table_name="header_to_transpose_processed")
    assert_dfs_equal(df, df_out, inp_data)

    sql_expression = "select * from header_to_transpose_processed"
    df_out = sql_utils.read_sql(sql=sql_expression, df=df, table_name="header_to_transpose_processed", project_name="test_db")
    assert_dfs_equal(df, df_out, inp_data)

    df_expected = pd.DataFrame({"count(*)": [20]})
    sql_expression = "select count(*) from header_to_transpose_processed"
    df_out = sql_utils.read_sql(sql=sql_expression, df=df, table_name="header_to_transpose_processed", project_name="test_db")
    assert_dfs_equal(df_expected, df_out, "")

def test_read_sql2():
    df = pd.DataFrame(data={"c1": [1, 2, 3, 4], "c2": ["a", "b", "c", "1_b2"]})
    df_expected = pd.DataFrame(data={"c1": [1, 3], "c2": ["a", "c"]})

    sql_expression = "select c1,c2 from dummy_table where c2 not like '%b%'"
    df_out = sql_utils.read_sql(sql=sql_expression, df=df, table_name="dummy_table")
    assert_dfs_equal(df_expected, df_out, None)

    # changing column name
    sql_expression = "select c1,c2 as c2_new from dummy_table where c2_new not like '%b%'"
    df_out = sql_utils.read_sql(sql=sql_expression, df=df, table_name="dummy_table")
    df_expected = df_expected.copy()
    df_expected.columns = ["c1", "c2_new"]
    assert_dfs_equal(df_expected, df_out, None)

def test_read_sql3():
    # test joining of tables & passing of 2 tables to read_sql
    df = pd.DataFrame(data={"c1": [1, 2, 3], "c2": ["a", "b", "c"]})
    df2 = pd.DataFrame(data={"c3": [4, 5, 6], "c2": ["a", "b", "c"]})
    df_expected = pd.DataFrame(data={"c1": [1, 2, 3], "c2": ["a", "b", "c"], "c3":[4, 5, 6]})

    sql_expression = "select t1.c1, t1.c2, t2.c3 from t1 INNER JOIN t2 ON t1.c2=t2.c2"
    df_out = sql_utils.read_sql(sql=sql_expression, df=[df, df2], table_name=["t1", "t2"])
    assert_dfs_equal(df_expected, df_out, None)

def test_df_to_db_excute_and_read_sql1():
    project_name = "test_case_tmp"
    # test joining of tables using 3 steps -
    # 1) table addition
    # 2) execute_sql query
    # 3) followed by read_sql query
    df = pd.DataFrame(data={"c1": [1, 2, 3], "c2": ["a", "b", "c"]})
    df2 = pd.DataFrame(data={"c3": [4, 5, 6], "c2": ["a", "b", "c"]})
    df_expected = pd.DataFrame(data={"c1": [1, 2, 3], "c2": ["a", "b", "c"], "c3":[4, 5, 6]})

    sql_utils.add_dataframe_to_db([df, df2], ["table1", "t2"], project_name=project_name, remove_db=False)
    sql_expression = "CREATE TEMP TABLE t1 AS SELECT * from table1"
    sql_utils.execute_sql(sql_expression, project_name=project_name, remove_db=False)
    sql_expression = "select t1.c1, t1.c2, t2.c3 from t1 INNER JOIN t2 ON t1.c2=t2.c2"
    df_out = sql_utils.read_sql(sql=sql_expression, project_name=project_name)
    assert_dfs_equal(df_expected, df_out, None)

if __name__=="__main__":
    test_read_sql()
