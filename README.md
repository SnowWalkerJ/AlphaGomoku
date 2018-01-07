AlphaGomoku is to use the similar algorithm as AlphaZero to train an AI player in Gomoku.
The main idea is from the paper [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270). We also reference to the Github repository [junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku) for realization of applying this algorithem on Gomoku.

# What we do

1. We modified the program into multiprocessing version so that the training process should be much faster.
2. We rewrite some of the game logic with [Cython](http://cython.org) so that the time spent on CPU is reduced.


# Experiments

We start with a simple game where the board size is 7*7, and four-in-a-row is considered winning. The configuration of this AlphaGomoku is listed in the table.

|Set               | Value   |
|:----------------:|:--------|
|processes         |4        |
|replay memory size|5000     |
|batch size        |256      |
|learning rate     |1e-3     |
|momentum          |0.3      |
|num_playouts      |400      |

The opponent is a pure MCTS player with num_playouts=2000.
At first AlphaGomoku has a winning rate of only 0.3. But with the benefit of multiprocessing, it reaches over 0.9 winnging rate in half an hour on a PC with GTX1070 GPU.
