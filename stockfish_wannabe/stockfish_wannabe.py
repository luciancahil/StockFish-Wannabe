from logging import exception
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
        self.isBlack = True
        self.depth = 2
        print("Done Initializing")
    
    def play(self):
        """
        Allows the user to play against this program's chess engine from the current position.
        """
        self.reset()
        done = False

        while(not done):
            print("Hello!")
            colorSet = False

            while(not colorSet):
                print("Would you like to play as (w)hite or (b)lack?")
                color = input()
                try:
                    self.playAs(color)
                except(ValueError):
                    print("ERROR! Invalid Color.")
                    continue

                
                colorSet = True

            engineTurn = not self.isBlack

            print("Begin!")

            


            while(not self.chessBoard.is_game_over()):
                print(self.chessBoard)
                nextMove = None

                if(engineTurn):
                    print("Thinking...")
                    nextMove, eval = self.findBestMove(self.depth, self.isBlack)
                    nextMove
                    print(self.chessBoard.san(nextMove))
                    self.chessBoard.push(nextMove)
                else:
                    print("please enter your next move:")
                    
                    moveSet = False

                    while(not moveSet):
                        nextMove = input()

                        try:
                            self.chessBoard.push_san(nextMove)
                        except(ValueError):
                            print("Error! Invalid move entered")
                            continue

                        moveSet = True
                
                engineTurn = not engineTurn
   
            
            done = True
            


    

    def makeMove(self, move):
        """Inputs a move to the engine

        Args:
            move: A move to make in Standard Notateion (Ex: NF3)

        Raises:
            Value Error: If any string does not represent a legal series of ches moves
        """
        self.chessBoard.push_san(move)
    


    def setup(self, moveString: str):
        """Sets the board up to a starting position

        Args:
            moveString: A series of moves in SAN seperated by commas. For example, "e4,e5" will cause the position
            the board to make the moves e4, then e5

        Raises:
            The best move, the evaluation if said move is made.
        """
        moveArr = moveString.split(",")

        for move in moveArr:
            self.makeMove(move)

    def takeBack(self):
        """
        Undoes the last move made
        """
        self.chessBoard.pop()

    def findBestMove(self, depth, isBlack):
        """Find the best move in the current position.

        Args:
            depth: How many moves deep it should evaluate
            isBlack: If true, the engien will try to find the best move for black. Otherwise, 
            it will try to find the best move for white.
        

        Returns:
            The best move in the current position, the evaluation if said move is played
        """

        moveList = list(self.chessBoard.legal_moves)
        

        if (depth == 0 or self.chessBoard.is_game_over()): # at a depth of zero, we return the eval, and no move
            return None, self.eval()
        
        # start by assuming that the first move in the list is the best
        bestMove = moveList[0]
        self.chessBoard.push(moveList[0])
        enemyMove, bestEval = self.findBestMove(depth - 1, not isBlack) # find the best enemy response to the first move, and the eval it creates
        self.chessBoard.pop()

        for i in range(1, len(moveList)):
            self.chessBoard.push(moveList[i])
            nextEnemyMove, newEval = self.findBestMove(depth - 1, not isBlack)  # find the best enemy response to the first move, and the eval it creates
            self.chessBoard.pop()

            if(isBlack) : # for black, find the lowest eval
                    if(newEval < bestEval):
                        bestMove = moveList[i]
                        bestEval = newEval
            else:
                if(newEval > bestEval): # for white, find the highest eval
                    bestMove = moveList[i]
                    bestEval = newEval
            
        return bestMove, bestEval
    

    def eval(self):
        """
        Returns what the engines neural network thinks the evaluation of the position is
        Returns:
            A number between 0 and 229 inclusive. An evaluation n between 0 and 14 means 
            mate for black in n moves. An evalation n between 15 and 214 inclusive means an 
            evaluation of (n - 115) * 10 centipawns. An evaluation n between 215 and 229 inclusive
            means mate for white in 229 - n moves. This evaluation sadly isn't very accurate.
        """
        input = torch.tensor(getInputArray(self.chessBoard))
        ouput = self.model.forward(input)
        return self.largestIndex(ouput)

    
    def reset(self):
        """
        Resets the chessboard to the original starting position
        """
        self.chessBoard.reset()
    

    def isGameOver(self):
        """
        Checks if the game is over.
        Returns:
            True if the game has ended, False otherwise
        """
        
        return self.chessBoard.is_game_over()
    
    def resultString(self):
        return None
    
    def playAs(self, color: str):
        """
        Sets the color that the player will be playing
        Args:
            color, either "w" for white or "b" for black
        Raises:
            ValueError: When the provided color isn't a valid chess color.
        """
        if(color.lower() != "b" and color.lower() != "w"):
            raise ValueError("Invalid color")
        
        self.isBlack = color.lower() == "w" 
    
    def gameMessage(self):
        """
        Returns a message describing how the game ended, or None if the game isn't over
        
        Returns:
            "WHITE won" if white has won, "BLACK won" if black has won, or "DRAW" if the game is drawn
        """
        if(not self.chessBoard.is_game_over()):
            return None
        
        result = self.chessBoard.outcome().result()

        if(result == "1/2-1/2"):
            return "DRAW"
        elif(result == "1-0"):
            return "WHITE won"
        else:
            return "BLACK won"

        


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
        return str(self.chessBoard)
        