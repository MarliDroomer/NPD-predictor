import torch
from torch import nn
import torch.nn.init as init
from torch.autograd import Variable

class two_feature_RNN(nn.Module):
    def __init__(self, hidden_size = 10, num_layers = 1):
        
        super(two_feature_RNN, self).__init__()
        
        #define the properties
        self.num_feat = 2
        self.num_batch = 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        output_size = 1
        
        #define the rnn
        self.rnn = nn.RNN(input_size = self.num_feat, 
                          hidden_size = self.hidden_size, 
                          num_layers = self.num_layers,
                          nonlinearity = 'relu')
        self.out_layer = nn.Linear(self.hidden_size, output_size)
        
    def forward(self, inputs):
    
        #initialize the hidden layer
        h0 = torch.zeros(self.num_layers, self.num_batch, self.hidden_size)
        
        #forward pass
        output_rnn, hn = self.rnn(inputs, h0)
        output_out_layer = self.out_layer(output_rnn)
        
        return output_out_layer, hn

    def train(self,data, n_epochs, lr = 0.01, weight_decay = 0.01):
        '''
        Call this to train on a series of data
        '''
        seq_length = data.shape[0]-1

        inputs = Variable(torch.from_numpy(data[:-1,:]).float())
        inputs = torch.reshape(inputs,(seq_length,1,self.num_feat))
        targets = Variable(torch.from_numpy(data[1:,0]).float())
        targets = torch.reshape(targets,(seq_length,1,1))
        
        #convert the input data to tensors
#         inputs = Variable( torch.from_numpy(data[:-1]).float())
#         inputs = torch.reshape(inputs,(seq_length,1,1))
#         targets = Variable(torch.from_numpy(data[1:]).float())
#         targets = torch.reshape(targets,(seq_length,1,1))

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01 , weight_decay = weight_decay)
        
        for epoch in range(n_epochs):
    
            outputs, hidden = self(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if epoch%50 == 0:
                print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
                print("Loss: {:.4f}".format(loss.item()))
                
    def predict(self,data,n=4):
        '''
        Send a series in and receive the prediction
        '''
        
        inputs = Variable( torch.from_numpy(data).float())
        inputs = torch.reshape(inputs,(data.shape[0],1,self.num_feat))
        
        out, hidden = self.forward(inputs)
        
        return out.detach().numpy()[:,0,0]


class SimpleRNN(nn.Module):
    def __init__(self):
        
        super(SimpleRNN, self).__init__()
        
        #define the properties
        num_feat = 1
        self.num_batch = 1
        self.hidden_size = 10
        self.num_layers = 1

        output_size = 1
        
        #define the rnn
        self.rnn = nn.RNN(input_size = num_feat, 
                          hidden_size = self.hidden_size, 
                          num_layers = self.num_layers,
                          nonlinearity = 'relu')
        self.out_layer = nn.Linear(self.hidden_size, output_size)
        
    def forward(self, inputs):
    
        #initialize the hidden layer
        h0 = torch.zeros(self.num_layers, self.num_batch, self.hidden_size)
        
        #forward pass
        output_rnn, hn = self.rnn(inputs, h0)
        output_out_layer = self.out_layer(output_rnn)
        
        return output_out_layer, hn

    def train(self,data, n_epochs, lr = 0.01, weight_decay = 0.1):
        '''
        Call this to train on a series of data
        '''
        seq_length = data.shape[0]-1
        #convert the input data to tensors
        inputs = Variable( torch.from_numpy(data[:-1]).float())
        inputs = torch.reshape(inputs,(seq_length,1,1))
        targets = Variable(torch.from_numpy(data[1:]).float())
        targets = torch.reshape(targets,(seq_length,1,1))

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01 , weight_decay = weight_decay)
        
        for epoch in range(n_epochs):
    
            outputs, hidden = self(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if epoch%500 == 0:
                print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
                print("Loss: {:.4f}".format(loss.item()))
                
    def predict(self,data,n=4):
        '''
        Send a series in and receive the prediction
        '''
        
        inputs = Variable( torch.from_numpy(data).float())
        inputs = torch.reshape(inputs,(data.shape[0],1,1))
        
        out, hidden = self.forward(inputs)
        
        return out.detach().numpy()[:,0,0]
    
    