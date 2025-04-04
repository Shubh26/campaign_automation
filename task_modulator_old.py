import sys
import ast
sys.path.extend(['D:\work\project\cac\sales_measurment_service'])
from utils import ab_group_utils as ab
from utils import measurement_utils as meas
from utils import optimization_utils as opt
from IPython.display import clear_output
#print("packages loaded")
import importlib
importlib.reload(ab)
importlib.reload(meas)
importlib.reload(opt)



def task_modulator():
    print("Please press 'y' to run task modulator and 'n' to exit")
    result = input()
    if result == 'y':
        print("Please enter the folder path that has campaign sales files")
        input_folder = input()
        print("You have entered this location:  %s" %input_folder)
        input_folder = r_string = r'{}'.format(input_folder)
        print("Would you like to continue, press y/n")
        result = input()
        if result == 'y':
            print("Please enter one of below option for further process\nab group creation\nmeasurement\noptimization")
            result = input()
            if result == 'ab group creation':
                print("You have selected:  %s" %result)
                print("Would you like to continue, press y/n")
                result = input()
                if result == 'y':
                    print("Please enter store name, split ratio, avg tol, size tol each value separated by space. i.e 7eleven {'control':0.2,'test':0.8} 15 30")
                    result = input()
                    result = result.split()
                    store = result[0]
                    split = ast.literal_eval(result[1])
                    avg_tol = int(result[2])
                    size_tol = int(result[3])
                    print("You have entered:\n %s %s %s %s" %(store, split, avg_tol, size_tol))
                    print("Would you like to continue, press y/n")
                    result = input()
                    if result == 'y':
                        print("Starting ab group generation process")
                        ab.generate_ab_group(input_folder, store, split, avg_tol, size_tol)
                    else:
                        raise SystemExit("Exiting execution!")
                else:
                    raise SystemExit("Exiting execution!")

            elif result == 'measurement':
                print("You have selected:  %s" %result)
                print("Would you like to continue, press y/n")
                result = input()
                if result == 'y':
                    print("Please provide list of measurement you would like to generate or you can pass [None] to get default metrics i.e [store_division, product, product_category, product_sub_category]")
                    result = input()
                    list_of_metrics = ast.literal_eval(result)
                    print("You have entered: %s" %result)
                    print("Would you like to continue, press y/n")
                    result = input()
                    if result == 'y':
                        print("Starting measurement process")
                        meas.generate_measurement_file(input_folder, list_of_metrics)
                    else:
                        raise SystemExit("Exiting execution!")
                else:
                    raise SystemExit("Exiting execution!")

            else:
                print("You have selected:  %s" %result)
                print("Would you like to continue, press y/n")
                result = input()
                if result == 'y':
                    print("Please enter lift type, optimization, lower limit, upper limit, lift plot with space separated for each value i.e incremental_lift True 0.8 1.0 True")
                    result = input()
                    result = result.split()
                    lift_type = result[0].replace('_', ' ')
                    optimization = ast.literal_eval(result[1])
                    lower_limit = float(result[2])
                    upper_limit = float(result[3])
                    lift_plot = ast.literal_eval(result[4])
                    print("You have entered:\n %s %s %s %s %s" %(lift_type, optimization, lower_limit, upper_limit, lift_plot))
                    print("Would you like to continue, press y/n")
                    result = input()
                    if result == 'y':
                        print("Starting optimization process")
                        opt.calculate_lift(input_folder, lift_type, optimization, lower_limit, upper_limit, lift_plot)
                    else:
                        raise SystemExit("Exiting execution!")
                else:
                    raise SystemExit("Exiting execution!")

        else:
            raise SystemExit("Exiting execution!")
        print("Would you like to clear output, press y/n")
        result = input()
        if result == 'y':
            clear_output(wait=True)
        return task_modulator()
    else:
        raise SystemExit("Exiting execution!")