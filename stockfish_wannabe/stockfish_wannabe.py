import chess as c

import torch
import torch.nn as nn
import pathlib
path = pathlib.Path(__file__).parent.resolve()
from torch.utils.data import Dataset, DataLoader, TensorDataset


# Hyper-parameters 
input_size = 64 # number of squares
hidden_size = 200 # number of pixels in hidden layer
num_classes = 230
device = torch.device('cpu')

# Turns a board into a string with no spaces or newlines
def getBoardString(board):
    boardString = str(board)
    boardString = boardString.replace(" ", "")
    boardString = boardString.replace("\n", "")
    return boardString

# accepts a chessboardObject as an input, and outputs the input layer of our neural netwrok
def getInputArray(board):
    boardString = getBoardString(board)
    array = [0] * 64
    valueDict = {"." : 0, "P": 1/12, "p" : 2/12, "N": 3/12, "n": 4/12, "B" : 5/12, "b" : 6/12, "R": 7/12, "r" : 8/12, "Q": 9/12, "q" : 10/12, "K": 11/12, "k" : 12/12}

    for i in range(64):
        key = boardString[i]
        array[i] = valueDict[key]
    
    return array


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
    
class Chess:
    def __init__(self):
        print("Initalizing")
        self.model = NeuralNet(input_size, hidden_size, num_classes).to(device)
        paramPath = str(path) + "\\RandomParam.txt"
        self.model.load_state_dict(torch.load(paramPath))
        self.chessBoard = c.Board()
        print("Done Initializing")
    
    
    

    def largestIndex(self, arr):
        """
        Finds the index of the largest value in an array
        Arguments:
            arr: Array we are parsing
        Returns:
            The index of the largest value, or first occurence
            if the largest occurs more than once
        """
        largest = 0

        for i in range(1, len(arr)):
            if(arr[i] > arr[largest]):
                largest = i
        

        return largest

    
    def __str__(self):
        print(str(self.chessBoard))
        