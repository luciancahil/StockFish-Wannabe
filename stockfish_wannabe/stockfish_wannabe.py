import chess

import torch
import torch.nn as nn
import pathlib
path = pathlib.Path(__file__).parent.resolve()

# Hyper-parameters 
input_size = 64 # number of squares
hidden_size = 200 # number of pixels in hidden layer
num_classes = 230
device = torch.device('cpu')


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)  # connects the input layer to the first hidden layer
        self.relu = nn.ReLU() # relu function
        self.l2 = nn.Linear(hidden_size, hidden_size) # connects the first hidden layer to the second
        self.l3 = nn.Linear(hidden_size, num_classes) # connects the second hidden layer to the output layer


    def forward(self, x):
        out = self.l1(x) #runs input into hidden layer
        out = self.relu(out) # runs output of layer 1 through relu
        out = self.l2(out) # runs relu'd function through to the second hidden layer
        out = self.relu(out)
        out = self.l3(out) # runs the relu'd output of the 2nd hidden layer into the output layer
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

paramPath = str(path) + "\\RandomParam.txt"
model.load_state_dict(torch.load(paramPath))

class Chess:
    def multiply(num1, num2):
        return num1 * num2