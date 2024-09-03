import pandas as pd
import functions as func
import yaml

#function to import yaml file
def import_yaml():
    try:
        with open("../config.yaml", "r") as file:
            config = yaml.safe_load(file)
    except:
        print("The config.yaml file was not found in the main folder!")

    return config



def tukeys_test_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for the outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify the outliers
    outliers = data[(data < lower_bound) | (data > upper_bound)]

    return outliers

