#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt


# In[1]:


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


# In[2]:


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


# In[4]:


def learning_curve(train_loss,val_loss):
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


# In[5]:


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

def single_step_plot(original_test_data,sequence_test_data,forecast_data,test_time,window_size,
                     original_plot =False):
    sequence_test_time = test_time[window_size:]
    plt.figure(figsize=(10,6))
    
    if original_plot:
        plt.plot(test_time,original_test_data,color="blue",label = 'Test Data')
        
    plt.plot(sequence_test_time,sequence_test_data,color="green", label = 'Test Data After Sequence')
    plt.plot(sequence_test_time,forecast_data,color="red", label = 'Forecast')
    plt.xticks(rotation = 45)
    plt.ylabel("Value")
    plt.title("Forecast plot")
    plt.legend()
# In[ ]:




