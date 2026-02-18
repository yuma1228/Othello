from SizedOthello import SizedOthello
from play.alphazero_agent import AlphaZeroAgent
from play.random_play import random_play
from config import AlphaZeroConfig


def play_eval_game(black_agent, white_agent, board_size=6):
    othello = SizedOthello(board_size)
    return othello.play(black=black_agent, white=white_agent, do_print=False)


def evaluate_agents(agent_a, agent_b, num_games, board_size=6):
    """半分黒・半分白で公平に対戦"""
    wins, losses, draws = 0, 0, 0
    half = num_games // 2
    for i in range(num_games):
        if i < half:
            winner = play_eval_game(agent_a, agent_b, board_size)
            if winner == 1:
                wins += 1
            elif winner == 2:
                losses += 1
            else:
                draws += 1
        else:
            winner = play_eval_game(agent_b, agent_a, board_size)
            if winner == 2:
                wins += 1
            elif winner == 1:
                losses += 1
            else:
                draws += 1
    return {'wins': wins, 'losses': losses, 'draws': draws,
            'win_rate': wins / max(num_games, 1)}


def evaluate_against_random(network, config, device, num_games=None):
    if num_games is None:
        num_games = config.num_eval_games
    az_agent = AlphaZeroAgent(network, config, device, temperature=0.01)
    return evaluate_agents(az_agent, random_play, num_games, config.board_size)
