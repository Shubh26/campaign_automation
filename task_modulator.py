import sys, ast, json, glob, os
import pandas as pd
from IPython.display import clear_output
from utils import ab_group_utils as ab
from utils import measurement_utils as meas
from utils import optimization_utils as opt



def dummy_config_generator(output_path):
    config_dict = {"output_path": "",
                   "campaign": "smithfield_fall_ai",
                   "client": "smithfield",
                   "brand": "smithfield",
                   "store_sales_folder_path": {"kroger": "",
                                               "giant": ""},
                   "inclusion": {"kroger": {"product" : [], "weeks": []},
                                 "giant": {"product": [], "weeks": []}
                                 },
                   
                   "exclusion": {"kroger": {"product" : [], "weeks": []},
                                 "giant": {"product": [], "weeks": []}
                                 },
                   "recent_sales_week": {"kroger": 13,
                                         "giant": 24
                                 },
                   "process": {"ab_group":
                    					 # {"activate": False,
                    					 # "mode": "single",
                    					 # "parallelization": True,
                    					 # "cluster": True,
                    					 # "split": [{"control":0.20,"test":0.80}, {"control":0.30,"test":0.70}],
                    					 # "groups": [[100], [200]],
                    					 # "avg_tol": [[10], [20]],
                    					 # "size_tol": [[100], [300]],
                    					 # "retailers": {"kroger":{"split_folder":None},
                    					 # "giant":{"split_folder": None}}
                    					 # },
                                       {"activate": False,
                                        "mode": "multi",
                                        "parallelization": True,
                                        "cluster": True,
                                        "split": [{"control":0.20,"test":0.80}, {"control":0.30,"test":0.70}],
                                        "groups": [[100, 300], [200, 400]],
                                        "avg_tol": [[10, 5], [20, 15]],
                                        "size_tol": [[100, 200], [300, 400]],
                                        "retailers": {"kroger":{"split_folder":None},
                                                    "giant":{"split_folder": None}}
                                       },
                                    "measurement": {"activate": False,
                                                   "kroger": {"list_of_metrics":[None]},
                                                   "giant": {"list_of_metrics":["store_division", "product", "product_category", "product_sub_category"]}
                                        },
                                    "optimization": {"activate": False,
                                                     "kroger": {"lift_type": "incremental_lift",
                                                                "optimization": True,
                                                                "lower_limit": 0.8,
                                                                "upper_limit": 1.0,
                                                                "lift_plot": True},

                                                     "giant": {"lift_type": "incremental_lift",
                                                               "optimization": True,
                                                               "lower_limit": 0.8,
                                                               "upper_limit": 1.0,
                                                               "lift_plot": True}
                                         }
                                    }
                  }
    out_file = open(os.path.join(output_path, "task_modulator_config.json"), "w")
    json.dump(config_dict, out_file, indent = 4)
    out_file.close()
    

def task_modulator_config_generator(config_json_path):
    with open(os.path.realpath(config_json_path), 'r') as f:
            config_json = json.load(f)
    stores = config_json['store_sales_folder_path']
    for task in config_json['process']:
        if task == "ab_group":
            if config_json["process"]["ab_group"]["activate"]:
                temp = {}
                for store, folder_path in stores.items():
                    temp = {"historical_files": glob.glob(os.path.join(folder_path,'*'), recursive = False)}
                    config_json['process']['ab_group']['retailers'][store]['input_files'] = temp
        elif task == "measurement":
            if config_json["process"]["measurement"]["activate"]:
                temp = {}
                for store, folder_path in stores.items():
                    temp = {"campaign_files": glob.glob(os.path.join(folder_path,'*'), recursive = False)}
                    config_json['process']['measurement'][store]['input_files'] = temp
        elif task == "optimization":
            if config_json["process"]["optimization"]["activate"]:
                temp = {}
                for store, folder_path in stores.items():
                    temp = {"historical_files": glob.glob(os.path.join(folder_path,'*'), recursive = False),
                            "campaign_files": glob.glob(os.path.join(folder_path,'*'), recursive = False)}
                    config_json['process']['optimization'][store]['input_files'] = temp
    out_file = open(config_json_path, "w")
    json.dump(config_json, out_file, indent = 4)
    out_file.close()
    
def run_task_modulator(config_json_path):
    with open(config_json_path, 'r') as f:
        config_dict = json.load(f)
    campaign_details = {'campaign': config_dict['campaign'], 'client': config_dict['client'], 'brand': config_dict['brand'], 'output_path': config_dict['output_path']}
    if 'inclusion' in config_dict:
        campaign_details['inclusion'] = config_dict['inclusion']
    if 'exclusion' in config_dict:
        campaign_details['exclusion'] = config_dict['exclusion']
    if 'recent_sales_week' in config_dict:
        campaign_details['recent_sales_week'] = config_dict['recent_sales_week']
    for task, param in config_dict['process'].items():
        if task == 'ab_group':
            if config_dict["process"]["ab_group"]["activate"]:
                print("Starting ab group creation process........")
                campaign_details['task'] = task
                campaign_details['z3_param'] = param
                ab.generate_ab_group(campaign_details)
        elif task == 'measurement':
            if config_dict["process"]["measurement"]["activate"]:
                print("Starting measurement process........")
                for store, args in param.items():
                    if store == 'activate':
                        continue
                    campaign_details['task'] = task
                    campaign_details['store'] = store
                    campaign_files = args['input_files']['campaign_files']
                    meas.generate_measurement_file_v3(campaign_details, campaign_files, args['list_of_metrics'])
        elif task == 'optimization':
            if config_dict["process"]["optimization"]["activate"]:
                print("Starting optimization process........")
                for store, args in param.items():
                    if store == 'activate':
                        continue
                    campaign_details['task'] = task
                    campaign_details['store'] = store
                    historical_files = args['input_files']['historical_files']
                    campaign_files = args['input_files']['campaign_files']
                    opt.calculate_lift_v2(campaign_details, historical_files, campaign_files, args['lift_type'], args['optimization'], args['lower_limit'], args['upper_limit'], args['lift_plot'])