import pandas as pd
import functions as func
import yaml
from statsmodels.stats.proportion import proportions_ztest

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



# Function to calculate completion rate
def calculate_completion_rate_by_group(df, final_step):
    # Calculate total number of unique users by group
    total_users_by_group = df.groupby('Variation')['client_id'].nunique()
    
    # Calculate the number of users who completed the process (reached the final step) by group
    completed_by_group = df[df['process_step'] == final_step].groupby('Variation')['client_id'].nunique()
    
    # Calculate the completion rate by group
    completion_rate_by_group = (completed_by_group / total_users_by_group) * 100
    
    return completion_rate_by_group

#function to calculate error rate
def calculate_error_rate(df):

    # Sort by client_id and date_time to ensure steps are in order
    df = df.sort_values(by=['client_id', 'date_time'])
    
    # Convert process_step to numeric if it isn't already (assuming steps are ordered numerically)
    df['process_step'] = pd.to_numeric(df['process_step'], errors='coerce')
    
    # Calculate the difference between the current step and the previous step for each user
    df['step_diff'] = df.groupby('client_id')['process_step'].diff()
    
    # Identify errors: step_diff < 0 indicates a backward movement (error)
    df['error'] = df['step_diff'] < 0
    
    # Calculate the error rate for each group (Test and Control)
    error_rate_by_group = df.groupby('Variation')['error'].mean() * 100
    
    return error_rate_by_group

#function to calculate time spent on each taks
def calculate_time_spent(df):

    # Convert the date_time column to datetime format if it isn't already
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    # Sort by client_id and date_time to ensure steps are in order
    df = df.sort_values(by=['client_id', 'date_time'])
    
    # Calculate the time difference between each step for each user
    df['time_diff'] = df.groupby('client_id')['date_time'].diff()
    
    # Calculate the average time spent on each step overall
    avg_time_spent = df.groupby('process_step')['time_diff'].mean()
    
    # Calculate the average time spent on each step by group (control/test)
    avg_time_spent_by_group = df.groupby(['Variation', 'process_step'])['time_diff'].mean().reset_index()
    
    return avg_time_spent, avg_time_spent_by_group




#function to calculate completion rate hypoteses
def completion_rate_hypothesis_test(df, final_step):
    
    # Count completed users in the Control group
    completed_control = df[(df['process_step'] == final_step) & (df['Variation'] == 'Control')]['client_id'].nunique()
    total_control = df[df['Variation'] == 'Control']['client_id'].nunique()
    
    # Count completed users in the Test group
    completed_test = df[(df['process_step'] == final_step) & (df['Variation'] == 'Test')]['client_id'].nunique()
    total_test = df[df['Variation'] == 'Test']['client_id'].nunique()
    
    # Successes (completions) and total users in both groups
    successes = [completed_control, completed_test]
    totals = [total_control, total_test]
    
    # Print to debug and check for zero values
    print(f"Successes (Control, Test): {successes}")
    print(f"Totals (Control, Test): {totals}")
    
    # Check for zero totals or completions
    if 0 in totals or 0 in successes:
        print("Error: One of the groups has no users or no completions.")
        return None, None
    
    # Perform z-test for two proportions
    z_stat, p_value = proportions_ztest(successes, totals, alternative='larger')
    
    return z_stat, p_value