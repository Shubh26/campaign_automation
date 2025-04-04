from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote.errorhandler import NoSuchElementException, InvalidSessionIdException
from selenium.common.exceptions import WebDriverException

# from bs4 import BeautifulSoup
import random
import pandas as pd
import numpy as np
import time
import csv
import os, sys
from pathlib import Path
from multiprocessing.dummy import Pool
from tqdm import tqdm
import glob
import re

from utils.constants import STORE_ID_COL, URL_COL, ZIPCODE_COL, STORE_ADDRESS_COL, LATITUDE_COL, LONGITUDE_COL

def process_batchwise(df, index, output_folder, driver_options, chrome_driver_path):
    store_url = df.set_index(STORE_ID_COL)[URL_COL].to_dict()
    store_address = df.set_index(STORE_ID_COL)[STORE_ADDRESS_COL].to_dict()
    output_folder = Path(output_folder)
    output_folder.mkdir(mode=0o777, parents=True,exist_ok=True)

    csv_file = os.path.join(output_folder,f"stores_with_zipcode_{index}.csv")
    exception_file = os.path.join(output_folder,f"exceptions_{index}.csv")
    def shutdown_driver(driver):
        if driver:
            try:
                print(f"driver session id {driver.session_id}")
                driver.quit()
            except Exception as e:
                print(f"error occured while closing driver {e}")

    def reinitialize_driver(driver):
        shutdown_driver(driver)
        time.sleep(30)
        driver = webdriver.Chrome(chrome_driver_path,options=driver_options)
        driver.implicitly_wait(30)
        return driver

    def clear_cache_cookies(driver):
        """
        clearing cache & cookies in this session, this is required to save space,
        otherwise the history will take up lot of space
        """
        # this doesn't work for headless run
        if not options.headless:
            # clearing cache
            # https://stackoverflow.com/a/56765601/2352424
            clearButton = driver.execute_script("""return document.querySelector('settings-ui').
            shadowRoot.querySelector('settings-main').
            shadowRoot.querySelector('settings-basic-page').
            shadowRoot.querySelector('settings-section > settings-privacy-page').
            shadowRoot.querySelector('settings-clear-browsing-data-dialog').
            shadowRoot.querySelector('#clearBrowsingDataDialog').querySelector('#clearBrowsingDataConfirm')""")
            clearButton.click()

        # clearning cookies
        driver.delete_all_cookies()
        time.sleep(2)

    #
    with open(csv_file, 'w') as f, open(exception_file, 'w') as f2, webdriver.Chrome(chrome_driver_path,options=driver_options) as driver:
        driver.implicitly_wait(30)
        writer = csv.writer(f)
        writer.writerow([STORE_ID_COL, URL_COL, ZIPCODE_COL, STORE_ADDRESS_COL, LATITUDE_COL, LONGITUDE_COL])
        writer_exception = csv.writer(f2)
        max_tries = 2
        for index, (store_id, url) in tqdm( enumerate(store_url.items()) ):
            # clearing cache after 20 url fetches
            if index%20==0:
                clear_cache_cookies(driver)
            if len(str(url))==0:
                driver.get("https://www.google.com/maps")
                inputElement = driver.find_element_by_id("searchboxinput")
                inputElement.clear()
                address = store_address.get(store_id)
                inputElement.send_keys(address)
                # for address = "7 eleven 944 portion rd ronkonkoma ny 117791976" if we directly hit the search
                # button it will return multiple values, but if we wait for the suggestion & click that i gives the
                # correct suggestion (maybe not always), so clicking that
                # https://www.google.com/maps/search/7+eleven+944+portion+rd+ronkonkoma+ny+117791976/@40.8323058,-73.0902787,15z/data=!3m1!4b1
                inputElement.click()
                suggestion_element = driver.find_element_by_class_name("suggest-text-layout")
                time.sleep(2)
                suggestion_element.click()
                # driver.find_element_by_xpath('//*[@id="searchbox-searchbutton"]').click()
                # time.sleep(2)
            else:
                driver.get(url)
            address_element = None
            retry = True
            try_count = 0
            while(retry and (try_count<max_tries) ):
                sleep_time = random.random()*10
                time.sleep(sleep_time)
                try_count+=1
                try:
                    # https://www.google.com/maps/place/3720+Fort+Union+Blvd+%237Eleven,+Cottonwood+Heights,+UT+84121,+USA/@40.618815,-111.7920693,17z/data=!3m1!4b1!4m5!3m4!1s0x87526315f2f2951f:0x2e730408c7eb5e23!8m2!3d40.618815!4d-111.7898806
                    # xpath = '//span[@class="section-info-text"]'
                    xpath = '//div[@data-tooltip="Copy address"]'
                    address_element = driver.find_element_by_xpath(xpath)
                    retry = False
                except NoSuchElementException as e1:
                    try:
                        # https://www.google.com/maps/place/7-Eleven/@45.4845626,-122.6401414,17z/data=!3m1!4b1!4m5!3m4!1s0x54950a9a7b453603:0x5ce9b2ec8a159e58!8m2!3d45.4845626!4d-122.6379527
                        # xpath = '//div[@class="QSFF4-text gm2-body-2"]'
                        # xpath = '//div[contains(@class,"QSFF4-text") and contains(@class,"gm2-body-2")]'
                        # xpath = '//div[contains(@class,"QSFF4-text")]'
                        xpath = '//button[@data-tooltip="Copy address"]'
                        address_element = driver.find_element_by_xpath(xpath)
                        retry = False
                    except NoSuchElementException as e2:
                        writer_exception.writerow([store_id,url,e2])
                    except (InvalidSessionIdException, WebDriverException) as e3:
                        print(f'following session id failed {driver.session_id}, restarting session')
                        retry, try_count,address_element = True, 0, None
                        driver = reinitialize_driver(driver)
                except (InvalidSessionIdException, WebDriverException) as e3:
                    print(f'following session id failed {driver.session_id}, restarting session')
                    retry, try_count,address_element = True, 0, None
                    driver = reinitialize_driver(driver)
                except Exception as e:
                    writer_exception.writerow([store_id,url,e])

                address_line='NA'
                zipcode='NA'
                try:
                    if address_element:
                        # sometimes the text element is empty when we don't scroll to that element
                        # (i.e element is not visible in the current small window)
                        driver.execute_script("arguments[0].scrollIntoView(true);",address_element)
                        time.sleep(.5)
                        address_line = address_element.text.strip()
#                         print(f'url {url} address_line {address_line}')
                        zipcode = get_zipcode_from_address(address_line)
                        url = str(driver.current_url)
                        lat_long = url.split("@")[1]
                        latitude = lat_long.split(",")[0]
                        longitude = lat_long.split(",")[1]
                        writer.writerow([store_id, url ,zipcode, address_line,latitude, longitude ])
                        f.flush() # flushing contents so that data is written to file right away
                except (InvalidSessionIdException, WebDriverException) as e3:
                    print(f'following session id failed {driver.session_id}, restarting session')
                    retry, try_count,address_element = True, 0, None
                    driver = reinitialize_driver(driver)
                except Exception as e:
                    writer.writerow([store_id, zipcode, address_line])
                    writer_exception.writerow([store_id,url,e])
                    f2.flush() # flushing contents so that data is written to file right away
    # keeping this quit statement as we are creating a new session if an existing session fails
    shutdown_driver(driver)

# Eg: 54321, 54321-123, 54321_123 etc
# this doesn't match canada zipcodes
zipcode_regex = re.compile(r'\b[\d+_-]+\b')

def is_zipcode(zipcode):
    return zipcode_regex.search(zipcode) is not None

def get_zipcode_from_address(address):
    # Address Eg:- '680 Strander Blvd #7eleven, Tukwila, WA 98188, USA', zipcode = 98188
    zipcode = address.split(",")[-2].split(" ")[-1].strip()
    if not is_zipcode(zipcode):
        # Address Eg:- '16998 W Bernardo Dr, San Diego, CA 92127', zipcode = 92127
        zipcode = address.split(",")[-1].split(" ")[-1].strip()

    if not is_zipcode(zipcode):
        # Address Eg:- '1000 Gerrard St E, Toronto, ON M4M 3G6, Canada', zipcode = M4M 3G6
        temp = address.split(",")[-2].strip()
        space_index = temp.index(" ")
        zipcode = temp[space_index:].strip()
    return zipcode

if __name__ == '__main__':
    chrome_driver_path = "/opt/custom/webdriver/chromedriver/91/chromedriver"
#     chrome_installation_location = "/usr/bin/"
#     sys.path.extend([chrome_installation_location])
    run_headless = True
    #output_folder = '/data/cac/sales_data/stores/7eleven/crawled_output_jul_19th'
    #output_folder = '/data/users/shubhamg/campaign/pepsico/bubly/7eleven'
    output_folder = '/data/users/shubhamg/campaign/lays'
    options = Options()
    # running without opening a GUI version of browser
    options.headless = run_headless
    num_parallel_executions = 2
    pool = Pool(int(num_parallel_executions))

    # reading input file
    #input_path = Path("/data/cac/sales_data/stores/7eleven/7eleven_stores_with_zips_jul_19th.csv")
    #input_path = Path("/data/users/shubhamg/campaign/pepsico/bubly/7eleven/stores_missing_zip.csv"
    input_path = Path("/data/users/shubhamg/campaign/lays/lays_stores_remaining.csv")
    df = pd.read_csv(input_path).fillna("")
    if URL_COL not in df.columns:
        df[URL_COL] = ""

    start = np.ceil(np.linspace(-1,len(df), num=num_parallel_executions)) + 1
    end = np.ceil(np.linspace(-1,len(df), num=num_parallel_executions))
    end = end[1:]
    batch_processing_input = []
    for i, (start_index, end_index) in enumerate(zip(start,end)):
        print(f"index {i} start_index {start_index} end_index {end_index}")
        df_subset = df.loc[start_index:end_index]
        batch_processing_input.append( (df_subset, i, output_folder, options, chrome_driver_path) )

    # starting crawling using multiple threads
    pool.starmap(process_batchwise, batch_processing_input)

    # combining output files from various threads
    #csv_file_pattern = os.path.join(output_folder,f"stores_with_zipcode_*.csv")
    csv_file_pattern = os.path.join(output_folder,f"lays_stores_with_details*.csv")
    df_list = []
    for filepath in glob.glob(csv_file_pattern):
        df_temp = pd.read_csv(filepath,dtype={ZIPCODE_COL:str})
        df_list.append(df_temp)

    df_generated = pd.concat(df_list)
    df_output = pd.merge(df, df_generated[[STORE_ID_COL,ZIPCODE_COL, URL_COL, LATITUDE_COL, LONGITUDE_COL]], on=STORE_ID_COL, suffixes=('_original',''))
    output_path = os.path.join(output_folder, f"{input_path.stem}_output{input_path.suffix}")
    print(f"output filepath {output_path}")
    df_output.to_csv(output_path, index=False)
