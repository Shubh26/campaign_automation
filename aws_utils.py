import os
from pathlib import Path
import boto3
from tqdm import tqdm

from utils import date_utils

def download_mab_aggregate_files(download_folder='/data/cac/expt_data/aggregated_data_for_mab', client_name='bubly',
                                 start_date='2021-05-20', days=60):
    """
    This function downloads mab aggregate files from s3  bucket to the mentioned folder
    Args:
    download_folder:str 
        folder to download the aggregated json files
    client_name:str
    start_date:str 
        date in YYYY-MM-DD format, Eg:-2021-06-14
    days:int 
        number of days from the starting day to download
    """
    s3 = boto3.resource('s3')
    s3_client = boto3.client('s3')
    start_date_obj = date_utils.get_date(start_date, date_utils.DATE_FORMAT_ISO)
    client_specific_output_folder = Path(os.path.join(download_folder,client_name))
    client_specific_output_folder.mkdir(mode=0o777, parents=True,exist_ok=True)
    for i in tqdm(range(days+1)):
        date_str = date_utils.get_date_string(date_utils.add_days(start_date_obj,i))
        filename = f'{client_name}_{date_str}.json'
        current_date_str = date_utils.get_timestamp(date_format=date_utils.DATE_FORMAT_ISO)
        
        try:
            aggregate_data_dsrow_path = 'processed/dsTrainingData/dsrow/'+filename
            aggregate_data_metadsrow_path = 'processed/dsTrainingData/metadsrow/'+filename

            aggregate_data_dsrow_download_path = os.path.join(client_specific_output_folder,f'dsrow_{filename}')
            aggregate_data_metadsrow_download_path = os.path.join(client_specific_output_folder,f'metadsrow_{filename}')

            # downloading dsrow file
            s3_client.download_file(Bucket='ec-reports',Key=aggregate_data_dsrow_path,Filename=aggregate_data_dsrow_download_path)
            # downloading metads row file
            s3_client.download_file(Bucket='ec-reports',Key=aggregate_data_metadsrow_path,Filename=aggregate_data_metadsrow_download_path)
        except Exception as e:
            print(f'error while downloading {filename} {e}')
        # if file download happens after the aggregation for that day is complete then files would be available, hence keeping
        # the break statement after current date 
        if date_str==current_date_str:
            print(f"Done. reached current date {current_date_str}, stopping further download, as these files won't be there")
            break
    
if __name__ == '__main__':
    download_mab_aggregate_files(download_folder='/data/cac/expt_data/aggregated_data_for_mab', client_name='eckrich',
                                 start_date='2021-09-28', days=20)