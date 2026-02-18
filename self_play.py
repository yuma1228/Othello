import numpy as np
import torch
from collections import deque
from SizedOthello import SizedOthello
from OthelloEnv import encode_state, action_to_xy
from play.alphazero_mcts import run_mcts, select_action_with_temperature
from config import AlphaZeroConfig


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, policy, value):
        self.buffer.append((state, policy, value))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        states, policies, values = [], [], []
        for i in indices:
            s, p, v = self.buffer[i]
            states.append(s)
            policies.append(p)
            values.append(v)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(policies), dtype=torch.float32),
            torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


def play_one_game(network, config, device):
    """1ゲーム自己対戦してデータを返す"""
    othello = SizedOthello(config.board_size)
    examples = []
    move_number = 0

    while othello.winner() is None:
        color = othello.color
        if not othello.is_possible_to_put_anywhere(color):
            othello.color = 3 - color
            continue

        state_array = encode_state(othello, color).numpy()
        action_probs = run_mcts(
            root_othello=othello,
            root_color=color,
            network=network,
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            add_noise=True,
            device=device,
        )
        examples.append((state_array, action_probs, color))

        temperature = 1.0 if move_number < config.temperature_threshold else 0.01
        action = select_action_with_temperature(action_probs, temperature)
        x, y = action_to_xy(action)
        othello.put(x, y, color)
        othello.color = 3 - color
        move_number += 1

    winner = othello.winner()
    training_examples = []
    for state_array, policy_array, color in examples:
        if winner == 0:
            value = 0.0
        elif winner == color:
            value = 1.0
        else:
            value = -1.0
        training_examples.append((state_array, policy_array, value))

    return training_examples


def generate_self_play_data(network, config, replay_buffer, device):
    """自己対戦データ生成 (回転によるデータ拡張付き)"""
    network.eval()
    for game_idx in range(config.num_self_play_games):
        examples = play_one_game(network, config, device)
        for state, policy, value in examples:
            policy_2d = policy.reshape(config.board_size, config.board_size)
            for rotation in range(4):
                rotated_state = np.rot90(state, k=rotation, axes=(1, 2)).copy()
                rotated_policy = np.rot90(policy_2d, k=rotation).copy()
                replay_buffer.push(rotated_state, rotated_policy.flatten(), value)
        if (game_idx + 1) % 10 == 0:
            print(f"  Self-play {game_idx + 1}/{config.num_self_play_games} done. Buffer: {len(replay_buffer)}")
