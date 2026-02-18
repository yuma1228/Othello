from play.alphazero_agent import load_alphazero_agent
from SizedOthello import SizedOthello

agent = load_alphazero_agent("checkpoints/model_final.pt")
game = SizedOthello(6)
game.play(black=agent, do_print=True)