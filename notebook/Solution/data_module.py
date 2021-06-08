#
#################################################################################
#
#  Copyright (c) 2020-2021 CertifAI Sdn. Bhd.
#
#  This program is part of OSRFramework. You can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#################################################################################
#
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import pandas as pd
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib.lines import Line2D

# ------------------------Data Sequencing Function---------------------------------------
# Data sequencing function for  univariate input , univariate output , single step forecast
def univariate_single_step(sequence, window_size):
    x, y = list(), list()
    for i in range(len(sequence)):
    # find the end of this pattern
        end_ix = i + window_size
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
    # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)

# Data sequencing function for univariate input , univariate output , multi step forecast
def univariate_multi_step(sequence,window_size,n_multistep):
    x, y = list(), list()
    for i in range(len(sequence)):
    # find the end of this pattern
        end_ix = i + window_size
        out_ix = end_ix+n_multistep
        # check if we are beyond the sequence
        if out_ix > len(sequence):
            break
    # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_ix]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)

# Data sequencing function for multivariate input , univariate output , single step forecast
def multivariate_univariate_single_step(sequence,window_size):
    x, y = list(), list()
    for i in range(len(sequence)):
    # find the end of this pattern
        end_ix = i + window_size
        # check if we are beyond the sequence
        if end_ix > len(sequence):
            break
    # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix,:-1], sequence[end_ix-1,-1]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)

# Data sequencing function for multivariate input , univariate output , multi step forecast
def multivariate_univariate_multi_step(sequence,window_size,n_multistep):
    x, y = list(), list()
    for i in range(len(sequence)):
    # find the end of this pattern
        end_ix = i + window_size
        out_ix = end_ix + n_multistep -1
        # check if we are beyond the sequence
        if out_ix > len(sequence):
            break
    # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix,:-1], sequence[end_ix-1:out_ix,-1]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)

# ------------------------------------Learning curve function-------------------------------------------
# Plot Learning curve
def learning_curve(num_epochs,train_loss,val_loss):
    plt.figure(figsize=(10,6))
    plt.plot(train_loss, label="Training")
    plt.plot(val_loss, label="Testing")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.title("Learning Curve")
    plt.show()
    for i in range(num_epochs):
        print(f'Epoch : {i} , training loss : {train_loss[i]} , validation loss : {val_loss[i]}')

# Zoom specific epoch in learning curve
def zoom_learning_curve(start_epoch,end_epoch,training_loss,validation_loss):
    plt.figure(figsize=(10,6))
    plt.plot(training_loss[start_epoch:end_epoch], label="Training loss")
    plt.plot(validation_loss[start_epoch:end_epoch], label="Testing loss")
    plt.title("Losses")
    plt.xlabel("Epoch_Demand")
    plt.ylabel("MSE_Demand")
    position=range(end_epoch-start_epoch)
    labels=range(start_epoch,end_epoch)
    plt.xticks(position, labels)
    plt.legend()

#--------------------------------------- Data flow function-----------------------------------------------------    
def key_assign(trainingX,testingX,trainingY,testingY):
    """ 
    Use to assgin the key to create the train_data_dict and test_data_dict
    
    Arguments:
    trainingX -- x-feature for traning data 
    testingX -- x-feature for testing data
    trainingY -- y-label for traning data
    testingY -- y-label for testing data
    
    Returns: 
    train_data_dict -- dictionary of trainingX and trainingY
    test_data_dict -- dictionary of testingX and testingY
    """
    # Create dictionary that can store the train set x-feature and y-label
    train_data_dict = {"train_data_x_feature" : trainingX, "train_data_y_label" : trainingY}
    
    # Create dictionary that can store the test set x-feature and y-label
    test_data_dict  = {"test_data_x_feature" : testingX , "test_data_y_label" : testingY }
    
    return train_data_dict , test_data_dict


def key_assign_evaluation(y_train_prediction,
                          y_test_prediction,
                          train_data_dictionary,
                          test_data_dictionary):
    """ 
    Assign key for prediction and output data dictionary 
    
    Arguments:
    y_train_prediction -- (tensor) prediction for training data
    y_test_prediction -- (tensor) prediction for test data
    train_data_dictionary -- (dict) train data dictionary
    test_data_dictionary -- (dict) test data dictionary
    
    
    Returns: 
    prediction -- (dict) dictionary that consist prediction of train data and test data
    output_data -- (dict) dictionary that consist output(y-label) from train data and test data
    """
    
    prediction ={"train_data_prediction" : y_train_prediction,
            "test_data_prediction" :y_test_prediction }
    output_data ={"train_data_output" : train_data_dictionary["train_data_y_label"] ,
               "test_data_output" : test_data_dictionary["test_data_y_label"]}
    return prediction , output_data

# Transform the numpy data to torch tensor
def transform(train_data_dict, test_data_dict):
    """ 
    Transform the numpy data to torch tensor
    
    Arguments:
    train_data_dict -- train data dictionary 
    test_data_dict -- test data dictionary
    
    Returns: 
    train_data_dict -- train data dictionary 
    test_data_dict -- test data dictionary
    """
    for train_datapoint in train_data_dict:
        train_data_dict[train_datapoint] =  torch.from_numpy(train_data_dict[train_datapoint]).type(torch.Tensor)
        
    for test_datapoint in test_data_dict:
        test_data_dict[test_datapoint] = torch.from_numpy(test_data_dict[test_datapoint]).type(torch.Tensor)
    return train_data_dict,test_data_dict

# Check Shape
def sanity_check(data_1,data_2):
    """ 
    Print the shape of data_1 and data_2
    Arguments:
    data_1 -- (dict) type of data
    data_2 -- (dict) type of data 
    """
    
    for key_1 in data_1:
        print(key_1 +" shape : " + str(data_1[key_1].shape))
    for key_2 in data_2:
        print(key_2 +" shape : " + str(data_2[key_2].shape))

# Create Iterator
def iterator(train_data_dict,test_data_dict,batch_size):
    """ 
    Create iterator for train data and test data 
    
    Arguments:
    train_data_dict -- (dict)train data dictionary 
    test_data_dict -- (dict)test data dictionary
    batch_size -- the number of the batch size
    
    Returns: 
    train_iter -- train data iterator 
    test_iter -- test data iterator 
    """
    train_dataset = TensorDataset(train_data_dict["train_data_x_feature" ],
                                  train_data_dict["train_data_y_label"])
    train_iter = DataLoader(train_dataset,batch_size=batch_size,shuffle=False)

    test_dataset = TensorDataset(test_data_dict["test_data_x_feature"],
                                 test_data_dict["test_data_y_label"])
    test_iter = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    
    return train_iter , test_iter

# Reshape both to the original data dimension
def squeeze_dimension(output):
    """ 
    Squeeze the dimension of output data
    
    Arguments:
    output -- (dict) output_data
    
    Returns: 
    output_data -- (dict) output_data
    """
    for key in output:
        output[key] = torch.squeeze(output[key],2)
    return output

# Transpose function
def transpose(train_data_dict,test_data_dict):
    train_data_dict['train_data_x_feature'] = torch.transpose(train_data_dict['train_data_x_feature'],1,2)
    test_data_dict['test_data_x_feature'] = torch.transpose(test_data_dict['test_data_x_feature'],1,2)
    return train_data_dict , test_data_dict

# Data Scaling for multivariate data 
def multi_data_scaler(train_data,test_data,scale_mode = "Standardize"):   
    """ 
    Perform Data Scalling 
    
    Arguments:
    train_data: Train Data
    test_data: Test Data
    scale_mode: Types of scaler ["Normalize": MinMaxScaler(),"Standardize": StandardScaler()]
    
    Returns: 
    scaler, scaled train data. scaled test data
    """
    scalers = dict()
    data_dict = {"train_data":train_data,
                "test_data":test_data}
    
    scale_type ={"Normalize": MinMaxScaler(),
                "Standardize": StandardScaler()}
    
    if scale_mode not in scale_type:
        print("Invalid scale mode. Expected one of: %s" %scale_type.keys())
    
    for columns in train_data.columns:
        scaler= scale_type[scale_mode].fit(train_data[columns].values.reshape(-1,1))
        scalers['scaler+'+columns] = scaler
        
    for key in data_dict:
        data_scaled = list()
        for columns in data_dict[key].columns:
            data_scaled.append(scaler.transform(data_dict[key][columns].values.reshape(-1,1)))
        standard_data = np.array(data_scaled)
        data_dict[key] = np.transpose(np.squeeze(standard_data))
        
    return scalers,data_dict["train_data"],data_dict["test_data"]

# Invert the scaling back to orignal data value
def inverse_scaler(scaled_data,scaler):
    """ 
    Inverse the scaled data
    
    Arguments:
    scaled_data -- (dict) data that being scaled 
    scaler -- scaler 
    
    Returns: 
    scaled_data -- (dict) data after inverse scale
    """
    
    for item in scaled_data:
        scaled_data[item] =  scaler.inverse_transform(scaled_data[item].detach().numpy())    
    return scaled_data

# list the test output and prediction output side by side   
def list_forecast_value(output_data,prediction):
    """ 
    To list the test output and prediction output side by side
    
    Arguments:
    output_data --  (dict) output data dictionary
    prediction -- (dict) prediction output dictionary
    """
    ### BEGIN SOLUTION
    print("Test Data\t\t\tForecast")
    for test, forecast in zip(output_data["test_data_output"],prediction["test_data_prediction"]):   
        print(f"{test}\t\t{forecast}")
    ### END SOLUTION

# Calculate the RMSE of train and test data
def rmse(prediction,output_data):
    """ 
    Calculate RMSE between output data and prediction data 
    
    Arguments:
    prediction -- (dict) prediction output dictionary
    output_data --  (dict) output data dictionary
    
    Returns:
    trainScore - RMSE of train dataset
    testScore - RMSE of test dataset
    """
    trainScore = math.sqrt(mean_squared_error(prediction["train_data_prediction"], output_data["train_data_output"]))
    testScore = math.sqrt(mean_squared_error(prediction["test_data_prediction"], output_data["test_data_output"]))
    return trainScore,testScore

# ------------------------------------------Forecast Plot-----------------------------------------------------
# Plot forecast plot for single-step
def single_step_plot(original_test_data,sequence_test_data,forecast_data,test_time,window_size,
                     original_plot =False,multivariate = False):
    
    plt.figure(figsize=(10,6))
    
    if multivariate:
        sequence_test_time = test_time[window_size-1:]
    else: 
        sequence_test_time = test_time[window_size:]
                                 
    if original_plot:
        plt.plot(test_time,original_test_data,color="blue",label = 'Test Data')
        
    plt.plot(sequence_test_time,sequence_test_data,color="green", label = 'Test Data After Sequence')
    plt.plot(sequence_test_time,forecast_data,color="red", label = 'Forecast')
    plt.xticks(rotation = 45)
    plt.ylabel("Value")
    plt.title("Forecast plot")
    plt.legend()

# Plot forecast plot for multi-step
def multi_step_plot(original_test_data,
                    after_sequence_test_data ,
                    forecast_data,test_time,window_size,
                    n_step ,
                    details = {},
                    original_plot = False,
                    multivariate = False):
    
    """ 
    Plot the result of multi-step forecast 
    
    Arguments:
    original_test_data -- test data before sequence
    after_sequence_test_data -- (dict) output data dictionary
    forecast_data -- (dict) prediction data dictionary
    test_time -- time index for test data before sequence
    window_size -- window size for the data sequence
    n_step -- the number of future step , 1 -> single >1 -> multi-step
    details -- (dict) details for plot such as "x-axis" ,"y-axis", "title"
    original_plot -- (boolean) True ->observe how sliding window (data sequence) take place in the test data
    
    """
    
    after_sequence_test_data = after_sequence_test_data['test_data_output'] 
    forecast_data = forecast_data["test_data_prediction"]
    
    # Plot Setting
    plt.figure(figsize=(10,6))
    plt.xticks(rotation=45)    
    
    # Store test and forecast data into DataFrame type 
    column_names = ["timestep_" + str(i) for i in range(after_sequence_test_data.shape[1])]
    y_test_dataframe = pd.DataFrame(after_sequence_test_data,columns = column_names)
    y_test_pred_dataframe =pd.DataFrame(forecast_data,columns = column_names)
    
    # Create time index for data after sequence
    if multivariate:
        time_index_after_sequence = test_time[window_size-1:]
        
    else:
        time_index_after_sequence = test_time[window_size:]
    
    # Test Data plot before sliding window(data sequencing)
    if original_plot:
        plt.plot(test_time,original_test_data,marker='x',color="blue")

    # For loop to plot the data step by step base on time index    
    start_idx = 0 
    for row in range(len(y_test_dataframe)):
        
               
        # Iterate the time index after sequence
        time_index = time_index_after_sequence[start_idx:start_idx+n_step]
        
        
        
        # Plot the test data
        plt.plot(time_index,y_test_dataframe.iloc[row],color="green",marker='o')
        
        # Plot the forecast data
        plt.plot(time_index,y_test_pred_dataframe.iloc[row],color="red",marker='o')
        
        # Pointer for time_index_after_sequence
        start_idx += 1
        
    # Customize the legend
    custom_lines = [Line2D([0], [0], color="green", lw=4),
                Line2D([0], [0], color="red", lw=4),
                Line2D([0], [0], color="blue", lw=4)]
    plt.legend(custom_lines, ['Test Data After Sequencing', 'Forecast Data', 'Test Data Before Sequencing'])
    
    # Extra details - Optional function
    if details != {}:
        plt.xlabel(details["x-axis"])
        plt.ylabel(details["y-axis"])
        plt.title(details["title"])





