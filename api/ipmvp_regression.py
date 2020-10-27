import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

#Remove warnings 
import warnings
warnings.filterwarnings("ignore")

from configparser import ConfigParser
import os 
import math

from helpers import *

from scipy import stats # for the second approach
from sklearn import linear_model # for the third approach
from sklearn.metrics import r2_score,mean_squared_error # for coefficient of determination (R^2) and RMSE


def fit_regression(x,y):
    # Function to fit a linear regression to some data
    reg = np.poly1d(np.polyfit(x,y,1))
    return reg

def reg_ransac(x,y):
    # Fit ransac method and return the inliers, outliers and the model
    ransac = linear_model.RANSACRegressor(random_state=0).\
            fit(x.reshape(-1,1),y.reshape(-1,1))
    
    # Regression fitted on the inliers
    reg = fit_regression(x[ransac.inlier_mask_],y[ransac.inlier_mask_])
    
    # Inliers
    x_inliers = x[ransac.inlier_mask_]
    y_inliers = y[ransac.inlier_mask_]
    
    # Outliers
    x_outliers = x[~ransac.inlier_mask_]
    y_outliers = y[~ransac.inlier_mask_]
    
    return x_inliers, y_inliers, x_outliers, y_outliers, reg

from sklearn.decomposition import PCA

def threshold_pca(x,y,threshold=2):
    X = np.asarray([x,y]).T

    # Rescale the data
    X_pca = (X - np.mean(X,axis=0))/np.std(X,axis=0)

    # Fit PCA
    pca = PCA(2)
    _ = pca.fit_transform(X_pca)

    # Rotate the data 
    rotated = np.dot(pca.components_,X_pca.T).T
    
    # Remove outliers from the rotated data (threshold = 2 means 95 % confidence interval)
    x_new = rotated[:,0][(np.abs(stats.zscore(rotated[:,1])) < float(threshold))]
    y_new = rotated[:,1][(np.abs(stats.zscore(rotated[:,1])) < float(threshold))]

    # Inverse transform to obtain the original data without the outliers
    X_new = pca.inverse_transform(np.asarray([x_new,y_new]).T)*np.std(X,axis=0)+np.mean(X,axis=0)

    # Inverse transform to obtain the outliers 
    X_out = pca.inverse_transform(np.asarray([rotated[:,0][~(np.abs(stats.zscore(rotated[:,1])) < float(threshold))],
                                          rotated[:,1][~(np.abs(stats.zscore(rotated[:,1])) < float(threshold))]]).T)*np.std(X,axis=0)+np.mean(X,axis=0)

    # Inliers
    x_in = X_new[:,0]
    y_in = X_new[:,1]

    # Outliers
    x_out = X_out[:,0]
    y_out = X_out[:,1]
    
    # Return the outliers 
    return x_in, y_in, x_out, y_out

def prepare_metrics(reg,x,y,x_inliers,y_inliers,**kwargs):
    """
    Function to prepare the metrics needed to be saved. The metrics are the following : 
    - Intercept : intercept of the linear regression
    - Slope : slope of the linear regression
    - Ref Period :  String with the reference period written
    - Days tot : total number of days in the reference period
    - Days kept : total number of days kept in the reference period (outliers excluded)
    - R square : (coefficient of determination) regression score function link: https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score 
    - RMSE (Root mean square error) : metrics to measure the difference between the real and the predicted values
    - Delta T : maximum range of temperature based on the daily averaged temperature of the inliers
    - Min daily temp : min daily averaged temperature
    - Max daily temp : max daily averaged temperature
    
    INPUTS : 
    - reg : regression model
    - x and y : all the data points
    - x_inliers and y_inliers : only the data kept to plot the regression
    
    OUTPUTS :
    - parameters : dictionnary with all the metrics
    
    """
    if kwargs.get('weeks') is None:
        time_period = kwargs.get('start').strftime("%d %b %Y") +' - ' + kwargs.get('end').strftime("%d %b %Y") 
    else :
        time_period = kwargs.get('start').strftime("%d %b %Y") +' - ' + (kwargs.get('start')+ timedelta(weeks=kwargs.get('end'),hours=-1)).strftime("%d %b %Y")
    # Prepare parameters to export
    
    parameters  = {'intercept':reg[0],
                  'slope':reg[1],
                  'ref_period':time_period,
                  'days_tot':x.shape[0],
                  'days_kept':x_inliers.shape[0],
                  'r_square':r2_score(y_inliers,reg(x_inliers)).round(2),
                  'RMSE':np.sqrt(mean_squared_error(y_inliers,reg(x_inliers))).round(2),
                  'delta_t':round(x_inliers.max()- x_inliers.min(),2),
                  'temp_min_daily':x_inliers.min(),
                  'temp_max_daily':x_inliers.max()}
    return parameters

def compute_precision(x):
    """
    Compute the confidence interval for the mean in percentage for a regression period
    
    ----------
    INPUTS : 
        - x : data coming from the reference period
        
    ----------
    OUTPUTS : 
        - relative_error : percentage of the confidence interval. 
    """
    # Compute the mean
    mean = x.mean()
    
    # Compute the standard error
    std_err = x.std()/math.sqrt(x.shape[0])

    #Student, n=number of samples - p(1) - 1, p <0.05, 2-tail
    t = stats.t.ppf(1-0.025, x.shape[0]-1-1)
    
    # Compute the precision error in percent
    precision_error = t*std_err/mean*100 
    
    return precision_error

def regression_IPMVP(ini_file,output_file,csv_file):
    
    # Read the config file
    inputs = read_config(ini_file)
    inputs = convert_config(inputs)
    
    # Aggregate the dataframe in order to work with daily aggregated points 
    df_agg, raw_params = aggregate_df(csv_file,agg_period='1D',**inputs)
    
    
    df_agg['JDC'] = (df_agg['TT'] - inputs.get('t_lim') < 0).astype(int)

    x = df_agg.loc[df_agg['JDC']==1,'TT'].values
    y = df_agg.loc[df_agg['JDC']==1,'conso'].values
    
    filepath = os.getcwd()+'/'
    filepath = filepath.replace("api", "src/assets/figures")
    print("FILE PATH",filepath)

    
    
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        
    # LOCAL : iterate through all the considered method given in the input file
    print("REG METHOD",inputs.get('reg_method'))
    final_params =[]
    for method_ in inputs.get('reg_method') :
        # # Figure plot
        fig, ax = plt.subplots(figsize=(10,7)) 
        
        if method_ =='regression_1' : # Ransac algorithm

            x_inliers, y_inliers, x_outliers, y_outliers, reg = reg_ransac(x,y) # call regression method

             # Plot scatter plots
            ax.scatter(x_inliers,y_inliers,c='forestgreen')
            ax.scatter(x_outliers,y_outliers,c='r',marker='x')


            # Compute parameters
            param_to_save = prepare_metrics(reg,x,y,x_inliers,y_inliers,**inputs)
            param_to_save.update({'reg_method':'regression_1'})
            
        if method_ =='regression_2' : # Threshold + RANSAC algorithm
            print("2",len(inputs.get('reg_method')))
            threshold=2
            x_new = x[(np.abs(stats.zscore(y)) < float(threshold))]
            y_new = y[(np.abs(stats.zscore(y)) < float(threshold))]
    
            x_inliers, y_inliers, x_outliers, y_outliers, reg = reg_ransac(x_new,y_new) # call regression method

            ax.scatter(x_inliers,y_inliers,c='forestgreen')
            ax.scatter(x_outliers,y_outliers,c='r',marker='x')
            ax.scatter(x[~(np.abs(stats.zscore(y)) < float(threshold))],y[~(np.abs(stats.zscore(y)) < float(threshold))],c='b',marker='x')
    
            # Compute parameters
            param_to_save = prepare_metrics(reg,x,y,x_inliers,y_inliers,**inputs)
            param_to_save.update({'reg_method':'regression_2'})
            
        if method_ =='regression_3' : # PCA + Threshold + RANSAC algorithm
            print("3",len(inputs.get('reg_method')))

            
            x_in, y_in, x_out, y_out = threshold_pca(x,y) # Apply PCA
    
            x_inliers, y_inliers, x_outliers, y_outliers, reg = reg_ransac(x_in,y_in) # Apply RANSAC on the inliers from PCA

             # Plot scatter plots
            ax.scatter(x_inliers,y_inliers,c='forestgreen') # Inliers
            ax.scatter(x_outliers,y_outliers,c='r',marker='x') # Outliers from RANSAC
            ax.scatter(x_out,y_out,c='b',marker='x') # Outliers from PCA


            # Compute parameters
            param_to_save = prepare_metrics(reg,x,y,x_inliers,y_inliers,**inputs)
            param_to_save.update({'reg_method':'regression_3'})

                # GLOBAL :
        # Plot regression line
        line, = ax.plot(np.linspace(x.min(),x.max()),
                    reg(np.linspace(x.min(),x.max()).reshape(-1,1)),'-k')
    
        # x and y labels 
        plt.xlabel('$T_{moy}$',fontsize=20)
        plt.ylabel('Consommation [$kWh$]',fontsize=20)
        # Legend to display the equation of the graph
        plt.legend([line], 
           ['Reg : {:.2f}'.format(reg[1])+'$\cdot T_{moy} + $' +'{:.2f}'.format(reg[0]) +' \n $ R^2$ : {:.2f}'.format(param_to_save.get('r_square'))])  
    
        # Add precision
        param_to_save.update({'precision':round(compute_precision(y_inliers),2)})
        
        # Add parameters     
        param_to_save.update({'number_of_points':raw_params.get('number_of_points'),
                              'temp_min_15min':raw_params.get('temp_min_15min'),
                              'temp_max_15min':raw_params.get('temp_max_15min')})

        final_params.append({method_ : param_to_save})

        # Save figures
        plt.savefig(filepath+method_+'_'+str(inputs.get('start').year)+'.png',dpi=600,bbox_inches='tight')

 

    print(final_params)
    return final_params
