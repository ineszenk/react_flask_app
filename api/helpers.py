from configparser import ConfigParser
from datetime import datetime, timedelta
import os 
import pandas as pd
import numpy as np
import requests
import json





def read_config(ini_file):
    """
    read configuration file for the reference period
    """
    # Initialize the parser
    parser = ConfigParser()
    parser.read(os.path.abspath(ini_file))
    
    return dict(parser.items( "inputs" ))

def write_output_file(ini_file,**kwargs):
    """
    Function to
    
    ----------
    Inputs :

    ----------
    Outputs :

    """
    reg_method = kwargs.get('reg_method')
    
    # Initialize the parser
    parser = ConfigParser()
    parser.read(ini_file)

    # Create new section in the .ini file to store the parameters
    try : 
        parser.add_section(reg_method)
    except :
        pass
        
    # Regression output
    parser.set(reg_method, 'intercept', str(kwargs.get('intercept'))) # save intercept
    parser.set(reg_method, 'slope', str(kwargs.get('slope'))) # save slope
    
    # Parameters output
    parser.set(reg_method, 'ref_period', kwargs.get('ref_period')) # save reference period
    parser.set(reg_method, 'days_tot', str(kwargs.get('days_tot'))) # save total number of days
    parser.set(reg_method, 'days_kept', str(kwargs.get('days_kept'))) # save number of days considered for the regression
    parser.set(reg_method, 'r_square', str(kwargs.get('r_square'))) # save R^2 parameters
    parser.set(reg_method, 'RMSE', str(kwargs.get('RMSE'))) # save RMSE
    parser.set(reg_method, 'precision', str(kwargs.get('precision'))) # save RMSE
    parser.set(reg_method, 'delta_t', str(kwargs.get('delta_t'))) # save delta t (temperature range)
    parser.set(reg_method, 'number_of_points', str(kwargs.get('number_of_points'))) # Total number of points (15 min interval)
    parser.set(reg_method, 'temp_min_15min', str(kwargs.get('temp_min_15min')))
    parser.set(reg_method, 'temp_max_15min', str(kwargs.get('temp_max_15min')))
    parser.set(reg_method, 'temp_min_daily', str(kwargs.get('temp_min_daily')))
    parser.set(reg_method, 'temp_max_daily', str(kwargs.get('temp_max_daily')))
    
    
    # save to a file
    with open(ini_file, 'w') as configfile:
        parser.write(configfile)
        
def write_output_opti(ini_file,**kwargs):
    """
    Function to
    
    ----------
    Inputs :

    ----------
    Outputs :

    """
    # Initialize the parser
    parser = ConfigParser()
    parser.read(ini_file)
    
    # Add optimisation output
    try : 
        parser.add_section(str(kwargs.get('reg_method')))
    except :
        pass
    
    parser.set(str(kwargs.get('reg_method')),
               'percentage', str(kwargs.get('percentage'))) # save percentage
    
    parser.set(str(kwargs.get('reg_method')),
               'consumption', str(kwargs.get('consumption'))) # save consumption
    
    parser.set(str(kwargs.get('reg_method')),
               'tot_normalised', str(kwargs.get('tot_normalised'))) # save consumption
    
    parser.set(str(kwargs.get('reg_method')),
               'tot_consolidated', str(kwargs.get('tot_consolidated'))) # save consumption
    
    parser.set(str(kwargs.get('reg_method')),
               'points_opti', str(kwargs.get('points_opti')))
    
    parser.set(str(kwargs.get('reg_method')),
               'points_validated', str(kwargs.get('points_validated')))
    
    parser.set(str(kwargs.get('reg_method')),
               'ratio', str(kwargs.get('ratio')))
    
    parser.set(str(kwargs.get('reg_method')),
               'co2', str(kwargs.get('co2')))
    
    # save to a file
    with open(ini_file, 'w') as configfile:
        parser.write(configfile)
        
def write_config_file(ini_file,**kwargs):
    """
    Function to
    
    ----------
    Inputs :

    ----------
    Outputs :

    """
    # Get regresion method
    reg_method = kwargs.get('reg_method')
    
    # Initialize the parser
    parser = ConfigParser()
    parser.read(ini_file)
        
    # Create new section in the .ini file to store the parameters
    try : 
        parser.add_section(reg_method)
    except :
        pass
    
    parser.set(reg_method, 'intercept', str(kwargs.get('intercept'))) # save intercept
    parser.set(reg_method, 'slope', str(kwargs.get('slope'))) # save slope
    
    # save to a file
    with open(ini_file, 'w') as configfile:
        parser.write(configfile)
        
def aggregate_df(filename,agg_period='1D',**kwargs):
    """
    Function to read the file and and aggregate the values. The consumption is summed and the temperature is averaged
    
    ----------
    Inputs :
        - filename : string with the file containing the values
        - agg_period : '1D' or '1H' 
        
    ----------
    Outputs :
        - df_agg : pandas dataframe with all the values aggregated
    """
    #Read CSV file
    df_temp = pd.read_csv(filename,index_col=0)
    #Convert index to datetime
    df_temp.index = pd.to_datetime(df_temp.index)
    
    # Drop duplicates
    df_temp = df_temp[~df_temp.index.duplicated(keep='first')]
    df_temp.index = pd.to_datetime(df_temp.index)

    if kwargs.get('weeks') is None: # No value for ref_weeks was entered hence we take the end_date
        df_temp = df_temp.loc[kwargs.get('start').strftime("%Y-%m-%d %H:%M:%S"):\
           kwargs.get('end').strftime("%Y-%m-%d %H:%M:%S")]
        
    else : # A value for ref_weeks was entered, it bypasses the end_date
        duration = kwargs.get('start') + timedelta(weeks=kwargs.get('end'),hours=-0.25) # Compute the duration
        df_temp = df_temp.loc[kwargs.get('start').strftime("%Y-%m-%d %H:%M:%S"):\
           duration.strftime("%Y-%m-%d %H:%M:%S")]
        
    # Extract the paramters at 15min interval : number of points, min and max
    parameters = {'number_of_points' : df_temp.shape[0],
                 'temp_min_15min' : df_temp['TT'].min(),
                 'temp_max_15min' : df_temp['TT'].max()}
    
    #Aggregation
    df_agg = df_temp.resample(agg_period).agg({'TT':'mean',
                                  'conso':'sum'})
    return df_agg, parameters

def read_optimisation(ini_file):
    """
    read optimisation configuration file
    """
    # Initialize the parser
    parser = ConfigParser()
    parser.read(os.path.abspath(ini_file))
    
    return dict(parser.items( "optimisation" ))

def read_regression(ini_file,reg_number):
    """
    read regression parameters computed from the reference period
    """
    # Initialize the parser
    parser = ConfigParser()
    parser.read(os.path.abspath(ini_file))
    
    return dict(parser.items(reg_number))

def convert_config(inputs):
    if inputs.get('end') is None : 
        inputs['end'] = int(inputs.get('weeks'))
    else :
        inputs['end'] = pd.to_datetime(inputs.get('end'))+timedelta(hours=23,minutes=45)
    
    inputs['t_lim'] = float(inputs.get('t_lim'))
    inputs['t_int'] = float(inputs.get('t_int'))
    inputs['start'] = pd.to_datetime(inputs.get('start'))
    inputs['reg_method'] = inputs.get('reg_method').split(',')
    inputs['reg_method'] = [inputs.get('reg_method')[i].strip() for i in range(len(inputs.get('reg_method')))]
    return inputs

def convert_opti(inputs):
    inputs['start'] = pd.to_datetime(inputs.get('start'))
    inputs['end'] = pd.to_datetime(inputs['end'])+timedelta(hours=23,minutes=45)
    return inputs

def convert_reg(inputs):
    inputs['intercept'] = float(inputs.get('intercept'))
    inputs['slope'] = float(inputs['slope'])
    return inputs

def get_token():
    """
    Get token for connection.

    Parameters
    ----------
    username : string

    password : string

    Returns
    ----------
    token : string
        Token for authentication to the API.
    """
    LOGIN_URL = "https://enno-api.herokuapp.com/user/login"

    obj     = {"username": "test", "password": "123456"}
    headers = {"content-type": "application/json"}
    req     = requests.post(LOGIN_URL,
                            data=json.dumps(obj), headers=headers)
        
    print(json.loads(req.content)['accessToken'])
    return json.loads(req.content)['accessToken']


def get_api_data(enno_serial, startDate, weeks):
    print('ENNO SERIAL',enno_serial)
    print('START DATE',startDate)
    print('WEEKS',weeks)


    '''
    Get history of the command by id

    Parameters
    ----------
    cmd_id : int
        Command id

    range_hour : int
        Size of the history bucket in hours

    Returns
    ----------
    values : list of dict
        Value of all the measurements in the bucket
    '''
    CMD_HISTORY_URL = "https://enno-api.herokuapp.com/device/%s" %enno_serial

    print(CMD_HISTORY_URL)

    days = int(weeks)*7
    start_date = pd.to_datetime(startDate)
    start_date = datetime(year = start_date.year, 
                    month = start_date.month, 
                    day = start_date.day,
                    hour = start_date.hour,
                     minute = start_date.minute)
    end_date = start_date + timedelta(days=days)


    print(start_date, "start date")
    print(end_date, "end_date")


    body       = {"startDate":start_date, "endDate": end_date}
    headers = {"accept": "application/json",
               "token": get_token(),
               "content-type": "application/json"}
    res     = requests.get(CMD_HISTORY_URL ,
                            headers=headers,
                            data=json.dumps(body,default=str))

    # Parse api call response
    df = pd.DataFrame()
    for item in list(json.loads(res.content).keys())[1:-1]:
        content_item = json.loads(res.content)[item]
        for i in range(len(content_item)):
            tmp = content_item[i]['tmp']
            value = content_item[i]['value']
            cmd_name = content_item[i]['cmd_name']
            df.loc[tmp,cmd_name] = value
    # Prep for csv extraction by removing unused values
    del df['Backward Temperature']
    del df['Forward Temperature']
    del df['gh_globalRadiationHorizPlan_wm2']
    del df['Optimisation Values']
    del df['State']

    df.columns = ['conso', 'TT']
    # Csv extraction
    df.to_csv('ref.csv', index=True)




# get_api_data('2019.09.01.002', '2020-10-01', 12)