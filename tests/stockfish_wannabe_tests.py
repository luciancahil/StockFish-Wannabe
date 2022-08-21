import unittest
import chess as c
from stockfish_wannabe import Chess

class StockFishWannabeTestCase(unittest.TestCase):
    def setUp(self):
        self.refrence = c.Board()
        self.wannnabe = Chess()
    
    def tearDown(self) -> None:
        self.reset()
    
    
    def reset(self):
        self.refrence.reset()
        self.wannnabe.reset()

    def testMoves(self):
        self.refrence.push_san("e4")

        self.wannnabe.makeMove("e4")

        assert(str(self.wannnabe) == str(self.refrence))
    
    
    
    def testTakeBack(self):
        self.refrence.push_san("e4")

        self.wannnabe.makeMove("e4")

        assert(str(self.wannnabe) == str(self.refrence))

        self.refrence.pop()
        self.wannnabe.takeBack()

        assert(str(self.wannnabe) == str(self.refrence))
    
    def testSetup(self):
        moves = "e4,e5,Nf3,Nc6,d4,d5";

        moveArr = moves.split(",")

        for move in moveArr:
            self.refrence.push_san(move)
        
        self.wannnabe.setup(moves)

        assert(str(self.wannnabe) == str(self.refrence))

    
    def testGameOver(self):
        moves = "e4,e5,Bc4,Nc6,Qf3,b6,Qf7"
        
        self.wannnabe.setup(moves)

        assert(self.wannnabe.isGameOver)
    
    def testResultString(self):
        assert(self.wannnabe.gameMessage() == None)

        moves = "e4,e5,Bc4,Nc6,Qf3,b6,Qf7"
        
        self.wannnabe.setup(moves)

        assert(self.wannnabe.gameMessage() == "WHITE won")

        self.reset()

        moves = "f3,e5,g4,Qh4"

        self.wannnabe.setup(moves)

        assert(self.wannnabe.gameMessage() == "BLACK won")

        self.reset()

        moves = "e4,Nf6,Qf3,Nc6,Qxf6,Rb8,Qxc6,Rg8,Qxb7,Ra8,Qxa7,Rb8,Qxb8,Rh8,Qxc8,f6,Qxc7,Kf7,Qxd7,Kg8,Qxd8,Kf7,Qxe7+,Kg6,Qxf8,Kh6,Qxh8,Kg6,Qxh7+,Kf7,Qxg7+,Ke8,Qxf6,Kd7,Qc3,Ke8,Qd3,Kf8,Qd4,Kg8,Qd7,Kh8,Qf7"

        self.wannnabe.setup(moves)

        assert(self.wannnabe.gameMessage() == "DRAW")




if __name__ == '__main__':
    unittest.main()


#python -m unittest -v tests/stockfish_wannabe_tests.py