import os
from pathlib import Path
import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import beta

from utils import date_utils


def process_dsrows(files, keys=['ad_id', 'dsrow_id'], is_parse_date=True):
    df_list = []
    for filepath in tqdm(files):
        filepath = Path(filepath)
        # eg:filename - dsrow_bubly_2021-05-20.json
        filename = filepath.name
        date_str=''
        if is_parse_date:
            s = filename.rfind("_") + 1
            e = filename.rfind(".json")
            date_str = filename[s:e]
        sep="||"
        with open(filepath) as f:
            rows = []
            for line in f:
                # line eg:-
                # main_rowds = 6378_10932956||{"imp": 979231, "clicks": 3708, "ctr": 0.38}
                # metadata_rowds =  6378_10932956_10930826||{"imp": 133, "clicks": 1, "ctr": 0.75}

                uniq_id, agg_json = line.strip().split(sep)
                ids = uniq_id.split('_')
                assert len(ids) == len(keys), f"keys {keys} values {ids}, file {filename}"
                id_dict = dict(zip(keys,ids))
                agg_dict = json.loads(agg_json)
                agg_dict.update(id_dict)
                agg_dict['date'] = date_str
                rows.append(agg_dict)
            df = pd.DataFrame(rows)
            df_list.append(df)
    df = pd.concat(df_list)
    del df_list
    columns = keys + ['clicks', 'ctr', 'date', 'imp']
    df = df[columns]
    if is_parse_date:
        df['date'] = pd.to_datetime(df['date'])
    def get_per_day_metrics(df, keys=['ad_id','dsrow_id']):
        start_date, end_date = date_utils.get_start_end_dates(df,'date')
        # ignoring this data, campaign might have started few days prior to this start_date value
        df_1st_day = df[df['date']==start_date]
        del df_1st_day

        t = df.copy()
        # let's  start_date = '2021-05-18'
        # in date_join column '2021-05-18' will be shown as '2021-05-19' & we will merge based on this
        t['date_join'] = t['date']+pd.DateOffset(1) # could have kept the name of new column as date, but that might cause confusions
        t = t.drop(['date'],axis=1)
        # will take '2021-05-19' from column 2 then
        t = pd.merge(df, t, left_on=keys+['date'],right_on=keys+['date_join'],suffixes=('','_2'), how='inner')
        t['imp_1day'] = t['imp'] - t['imp_2']
        t['clicks_1day'] = t['clicks'] - t['clicks_2']
        t = t.drop(['date_join','clicks_2','ctr_2','imp_2'],axis=1)
        return t
    t = df
    if is_parse_date and df['date'].nunique()>1 :
        t = get_per_day_metrics(df,keys)
    return t

def process_main_dsrows(mab_aggregated_data_folder, is_parse_date=True):
    main_ds_pattern = f'{mab_aggregated_data_folder}/dsrow_*'
    main_ds_files = sorted(glob.glob(main_ds_pattern))
    print(f'no of main ds files {len(main_ds_files)}')
    return process_dsrows(main_ds_files, keys=['ad_id', 'dsrow_id'], is_parse_date=is_parse_date)

def process_metadata_dsrows(mab_aggregated_data_folder, is_parse_date=True):
    meta_ds_pattern = f'{mab_aggregated_data_folder}/metadsrow_*'
    meta_ds_files = sorted(glob.glob(meta_ds_pattern))
    print(f'no of metadata ds files {len(meta_ds_files)}')
    return process_dsrows(meta_ds_files, keys=['ad_id', 'dsrow_id', 'metadata_dsrow_id'], is_parse_date=is_parse_date)

def simulate_mab_run(clicks, impressions, ids=None, num_iterations=1000, a_prior=1, b_prior=1, num_simulations=100):
    """
    Arguments:
    clicks:numpy array
        a numpy array with impression count per beta distribution
    impressions:numpy array
        a numpy array with impression count per beta distribution
    num_impressions:int
        num of impressions to simulate
    a_prior:double
        alpha prior for beta distribution
    b_prior:double
        beta prior for beta distribution
    num_simulations:int
        number of simulations to run to get the value
    """
    a = clicks + a_prior
    b = (impressions - clicks)+ b_prior
    no_of_betas = len(clicks)
    simulation_results = np.zeros((num_simulations, no_of_betas))

    a = np.repeat(np.array([a]),num_iterations,axis=0)
    b = np.repeat(np.array([b]),num_iterations,axis=0)
    for i in range(num_simulations):
        samples = beta.rvs(a,b)
        served_size = np.bincount(np.argmax(samples,axis=1),minlength=no_of_betas)
        simulation_results[i:] = served_size
    served_size = simulation_results[0]
    deviation = np.std(simulation_results,axis=0)
    if ids is not None:
        served_size = {k:v for k,v in zip(ids,served_size)}
        deviation = {k:v for k,v in zip(ids,deviation)}
    return (served_size, deviation)


if __name__ == '__main__':
#     ## uncomment following lines to load main table json files from dsgpool
#     mab_aggregated_data_folder = '/data/cac/expt_data/aggregated_data_for_mab/bubly'
#     df = process_main_dsrows(mab_aggregated_data_folder)
#     main_ds_output_file = os.path.join(mab_aggregated_data_folder,'main_ds_combined.csv')
#     df.to_csv(main_ds_output_file,index=False)
#     print(f"main_dsrows combined file saved to {main_ds_output_file}")

#     ## uncomment following lines to load metadata table json files from dsgpool
#     df_meta = process_metadata_dsrows(mab_aggregated_data_folder)
#     metadata_ds_output_file = os.path.join(mab_aggregated_data_folder,'metadata_ds_combined.csv.bz2')
#     df_meta.to_csv(metadata_ds_output_file,index=False)
#     print(f"metadata_dsrows combined file saved to {metadata_ds_output_file}")

    mainds_files = [r'resources/test/dsrow_offline_testing_2021-07-06.json']
    df_main = process_dsrows(mainds_files,is_parse_date=False)
    served_count, deviation = simulate_mab_run(df_main['clicks'], df_main['imp'] )
    print(f"served count {served_count} deviation {deviation}")

    # metads_files = [r'resources/test/metadsrow_offline_testing1_2021-07-06.json']
    metads_files = [r'resources/test/metadsrow_bubly_2021-07-08.json']
    df_meta = process_dsrows(metads_files, keys=['ad_id', 'dsrow_id', 'metadata_dsrow_id'], is_parse_date=False)
    # metads rowids (afternoon heading) for bubly circle k campaign 2021, campaign id - 935
    metaid_subset = ["10905817","10905818","10905819","10905820","10905821","10905822"]
    ad_ids=['6378' '6377' '6376']
    ad_ids = ['6377']
    t = df_meta.copy()
    t = t[t['metadata_dsrow_id'].isin(metaid_subset)]
    t = t[t['ad_id'].isin(ad_ids)]
    print(t['ad_id'].unique())

    df_meta = t
    served_count, deviation = simulate_mab_run(df_meta['clicks'], df_meta['imp'], ids= df_meta['ad_id']+"_"+df_meta['dsrow_id']+"_"+df_meta['metadata_dsrow_id'])
    print(f"served count {served_count} deviation {deviation}")
