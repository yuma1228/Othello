from othello import Othello
from play.random_play import random_play
from play.terminal_play import terminal_play
from gui_othello import GUI_Othello
from play.alpha_beta_play import alpha_beta_play
from evaluate import evaluate_board_ngn,evaluate_board_nkmr
from play.mcts_agent import mcts_agent

nk = alpha_beta_play(3,evaluate_board_nkmr)
ng = alpha_beta_play(3,evaluate_board_ngn)
mc = mcts_agent
o = GUI_Othello()
win = o.play(mc)
print(win)

