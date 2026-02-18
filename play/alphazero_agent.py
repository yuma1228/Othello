import torch
from network import AlphaZeroNetwork
from play.alphazero_mcts import run_mcts, select_action_with_temperature
from OthelloEnv import action_to_xy
from config import AlphaZeroConfig


class AlphaZeroAgent:
    """既存エージェントインターフェース互換の AlphaZero エージェント"""
    def __init__(self, network, config, device, num_simulations=None, temperature=0.01):
        self.network = network
        self.config = config
        self.device = device
        self.num_simulations = num_simulations or config.num_simulations
        self.temperature = temperature

    def __call__(self, othello, color):
        action_probs = run_mcts(
            root_othello=othello,
            root_color=color,
            network=self.network,
            num_simulations=self.num_simulations,
            c_puct=self.config.c_puct,
            add_noise=False,
            device=self.device,
        )
        action = select_action_with_temperature(action_probs, self.temperature)
        return action_to_xy(action)


def load_alphazero_agent(checkpoint_path, config=None, device=None):
    """チェックポイントからエージェントをロード"""
    if config is None:
        config = AlphaZeroConfig()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = AlphaZeroNetwork(
        board_size=config.board_size,
        num_res_blocks=config.num_res_blocks,
        num_filters=config.num_filters,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()
    return AlphaZeroAgent(network, config, device)
