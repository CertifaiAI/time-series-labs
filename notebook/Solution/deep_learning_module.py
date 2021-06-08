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

# In[2]:


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class LSTM(nn.Module):

        def __init__(self, n_feature, hidden_dim, num_layers,n_step):
            super(LSTM, self).__init__()
            ### BEGIN SOLUTION
            
            # Number of feature of data
            self.n_feature = n_feature
            
            # Hidden unit dimensions
            self.hidden_dim = hidden_dim

            # Number of hidden layers
            self.num_layers = num_layers
            
            # Number of step ,step =1 -> single step forecast
            self.n_step = n_step 
          

            # Building your LSTM
            # batch_first=True causes input/output tensors to be of shape
            # Configuration -> (number of feature , number of hidden unit,number of layer)
            self.lstm = nn.LSTM(n_feature, hidden_dim, num_layers, batch_first=True)

            # Readout layer
            self.fc = nn.Linear(hidden_dim, n_step)


        def forward(self, x):
            # Initialize hidden state with zeros
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)

            # Initialize cell state with zeros
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)

            # We need to detach as we are doing truncated backpropagation through time (BPTT)
            # If we don't, we'll backprop all the way to the start even after going through another batch
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

            # Index hidden state of last time step
            # we just want last time step hidden states(output)
            out = out[:, -1, :]
            out = self.fc(out)
            
            ### END SOLUTION
            return out


# In[4]:


class BidirectionalLSTM(nn.Module):

    def __init__(self, n_feature, hidden_dim, num_layers,n_step):
        super(BidirectionalLSTM, self).__init__()
        ### BEGIN SOLUTION

        # Number of feature of data
        self.n_feature = n_feature
        
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers
        
        # Number of step ,step =1 -> single step forecast
        self.n_step = n_step 

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # Configuration -> (number of feature , number of hidden unit,number of layer)
        self.lstm = nn.LSTM(n_feature, hidden_dim, num_layers, batch_first=True,bidirectional=True)

        # Readout layer *2 for bidirectional LSTM
        self.fc = nn.Linear(hidden_dim*2, n_step)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim)

        # Initialize cell state with zeros
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Index hidden state of last time step
        # we just want last time step hidden states(output)
        out = out[:, -1, :]

        # Index hidden state of last time step
        out = self.fc(out)
        
        ### END SOLUTION
        return out


# In[6]:


def training(num_epochs,train_iter,test_iter,optimizer,loss_fn,model):
    #seed
    torch.manual_seed(123)
    
    train_loss = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    for t in range(num_epochs):
        # Initialise hidden state
    #     Don't do this if you want your LSTM to be stateful
    #     model.hidden = model.init_hidden()
        running_loss_train = 0
        running_loss_valid = 0
        for _,(train_X,train_Y) in enumerate(train_iter):


            # Forward pass
            y_train_pred = model(train_X)

            # Reshape to ensure the predicted output (y_train_pred) same size with train_Y shape 
            y_train_pred=torch.reshape(y_train_pred,(train_Y.shape[0],train_Y.shape[1],train_Y.shape[2]))

            #Compare the value using MSE
            loss_train = loss_fn(y_train_pred, train_Y)

            # Zero out gradient, else they will accumulate between epochs
            optimizer.zero_grad()

            # Backward pass
            loss_train.backward()

            # Update parameters
            optimizer.step()
            
            # Summing up the loss over each epoch
            running_loss_train += loss_train.item()*train_X.size(0)

        # Average the loss base of the batch size 
        epoch_loss_train = running_loss_train /len(train_iter.dataset)
        
        # Store the averaged value
        train_loss[t] = epoch_loss_train

        # Validate the test data loss
        with torch.no_grad():
            for _,(test_X,test_Y) in enumerate(test_iter):
                y_test_pred = model(test_X)

                #Reshape to perform MSE 
                y_test_pred=torch.reshape(y_test_pred,(test_Y.shape[0],test_Y.shape[1],test_Y.shape[2]))
                
                # Calculate the loss
                loss_test = loss_fn(y_test_pred, test_Y)
                
                # Summing up the loss over each epoch
                running_loss_valid += loss_test.item()*test_X.size(0)

        # Average the loss base of the batch size
        epoch_loss_test =running_loss_valid /len(test_iter.dataset)

        # Store the averaged value
        val_loss[t] = epoch_loss_test
    
    return train_loss,val_loss


# In[ ]:
class CNN(nn.Module):

    def __init__(self,n_feature,n_step):
        super(CNN, self).__init__()
        
        self.n_feature = n_feature
        self.n_step = n_step

        # Conv1d in_channels is base on num time series
        # Input:(N,C,Lin) Output : (N,C,Lout)
        self.conv1 = nn.Conv1d(in_channels = n_feature, out_channels = 30, kernel_size = 3)
        
        # For example Input:(N,C,Lin) Output : (N,C,Lout)
        self.poo1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels = 30, out_channels = 20, kernel_size = 2)
        
        # AdaptiveMaxPool1d use to make sure it always will output  = 1 ,to make sure return the correct batch size 
        self.pool2 = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10,n_step)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.poo1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1,20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x