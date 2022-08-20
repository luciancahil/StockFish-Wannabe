from stockfish_wannabe import Chess


c = Chess()
c.makeMove("e4")


print(c.findBestMove(2, False))  
