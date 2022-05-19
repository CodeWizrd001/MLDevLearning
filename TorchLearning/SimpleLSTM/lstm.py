import torch

class LSTM(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(LSTM,self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size,hidden_size)
        self.linear = torch.nn.Linear(hidden_size,output_size)
        self.out = torch.nn.Sigmoid()
    def forward(self,x):
        x,_ = self.lstm(x)
        x = self.linear(x)
        x = self.out(x)
        return x
    def __call__(self,x):
        return self.forward(x)

# Big LSTM
class LSTM_Big(torch.nn.Module):
    def __init__(self,input_size,output_size):
        super(LSTM_Big,self).__init__()
        self.lstm = torch.nn.LSTM(input_size,1024)
        self.l1 = torch.nn.Linear(1024,2048)
