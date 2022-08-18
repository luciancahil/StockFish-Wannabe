import unittest
from stockfish_wannabe import Chess as chess

class StockFishWannabeTestCase(unittest.TestCase):
    def setUp(self):
        self.number = 3

    def test_multiply(self):
        ans = chess.multiply(self.number, 6)
        self.assertEquals(ans, 18)
    

if __name__ == '__main__':
    unittest.main()


#python -m unittest -v tests/stockfish_wannabe_tests.py