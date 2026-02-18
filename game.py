from play.alphazero_agent import load_alphazero_agent
from gui_sized_othello import GUI_SizedOthello
from play import random_play, alpha_beta_play, mcts_agent
mcts_agent = mcts_agent
agent = load_alphazero_agent("checkpoints/model_final.pt")
game = GUI_SizedOthello(6)
# 黒=AI, 白=あなた (Noneで人間操作)
winner = game.play(black=agent, white=None)
print(f"Winner: {['Draw', 'Black', 'White'][winner]}")
