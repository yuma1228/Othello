import math
import numpy as np
import torch
from SizedOthello import SizedOthello
from OthelloEnv import encode_state, get_legal_moves_mask, action_to_xy


class AZNode:
    """AlphaZero MCTSノード。エッジ(行動)にN,W,P統計量を持つ。"""
    def __init__(self, othello, color, parent=None, parent_action=None):
        self.othello = othello
        self.color = color
        self.parent = parent
        self.parent_action = parent_action
        self.children = {}          # action -> AZNode
        self.N = {}                 # 訪問回数
        self.W = {}                 # 累計価値
        self.P = {}                 # 事前確率 (NNのpolicy出力)
        self.is_expanded = False

    def q_value(self, action):
        if self.N[action] == 0:
            return 0.0
        return self.W[action] / self.N[action]

    def select_action(self, c_puct):
        """PUCT式で行動を選択"""
        total_visits = sum(self.N.values())
        sqrt_total = math.sqrt(total_visits + 1)
        best_action = -1
        best_score = -float('inf')
        for action in self.P:
            q = self.q_value(action)
            u = c_puct * self.P[action] * sqrt_total / (1 + self.N[action])
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def expand(self, policy, legal_mask):
        """NNのpolicy出力で展開"""
        self.is_expanded = True
        masked = policy * legal_mask
        s = masked.sum()
        if s > 0:
            masked = masked / s
        else:
            masked = legal_mask / legal_mask.sum()
        for action in range(len(masked)):
            if legal_mask[action] > 0:
                self.P[action] = masked[action].item()
                self.N[action] = 0
                self.W[action] = 0.0


def run_mcts(root_othello, root_color, network, num_simulations, c_puct,
             dirichlet_alpha=0.3, dirichlet_epsilon=0.25, add_noise=True,
             device=torch.device('cpu')):
    """
    AlphaZero MCTSを実行。訪問回数分布とルート価値を返す。
    """
    board_size = root_othello.size
    action_size = board_size * board_size

    root = AZNode(root_othello.copy(), root_color)

    # ルート展開
    state_t = encode_state(root.othello, root.color).to(device)
    legal_mask = get_legal_moves_mask(root.othello, root.color).to(device)
    policy, _ = network.predict(state_t)
    root.expand(policy.cpu(), legal_mask.cpu())

    # ルートにDirichletノイズ追加
    if add_noise and len(root.P) > 0:
        noise = np.random.dirichlet([dirichlet_alpha] * len(root.P))
        for i, action in enumerate(root.P):
            root.P[action] = (1 - dirichlet_epsilon) * root.P[action] + dirichlet_epsilon * noise[i]

    for _ in range(num_simulations):
        node = root
        search_path = [node]

        # SELECT: 葉まで降りる
        while node.is_expanded and not node.othello.winner() is not None:
            action = node.select_action(c_puct)
            if action not in node.children:
                child_othello = node.othello.copy()
                x, y = action_to_xy(action)
                child_othello.put(x, y, node.color)
                child_color = 3 - node.color
                if not child_othello.is_possible_to_put_anywhere(child_color):
                    child_color = 3 - child_color
                child_othello.color = child_color
                child = AZNode(child_othello, child_color, parent=node, parent_action=action)
                node.children[action] = child
            node = node.children[action]
            search_path.append(node)

        # EVALUATE
        if node.othello.winner() is not None:
            winner = node.othello.winner()
            if winner == 0:
                leaf_value = 0.0
            elif winner == node.color:
                leaf_value = 1.0
            else:
                leaf_value = -1.0
        else:
            state_t = encode_state(node.othello, node.color).to(device)
            legal_mask = get_legal_moves_mask(node.othello, node.color).to(device)
            policy, value = network.predict(state_t)
            node.expand(policy.cpu(), legal_mask.cpu())
            leaf_value = value

        # BACKUP: 符号を反転しながら遡る
        current_value = leaf_value
        for i in range(len(search_path) - 1, 0, -1):
            child_node = search_path[i]
            parent_node = search_path[i - 1]
            action = child_node.parent_action
            parent_node.N[action] += 1
            parent_node.W[action] += (-current_value)
            current_value = -current_value

    # 訪問回数分布を返す
    action_probs = np.zeros(action_size, dtype=np.float32)
    for action, count in root.N.items():
        action_probs[action] = count
    total = action_probs.sum()
    if total > 0:
        action_probs /= total

    return action_probs


def select_action_with_temperature(action_probs, temperature):
    """温度付き行動選択。temp→0でgreedy。"""
    if temperature < 1e-6:
        best = np.where(action_probs == np.max(action_probs))[0]
        return int(np.random.choice(best))
    probs = action_probs ** (1.0 / temperature)
    total = probs.sum()
    if total == 0:
        legal = np.where(action_probs > 0)[0]
        return int(np.random.choice(legal))
    probs /= total
    return int(np.random.choice(len(probs), p=probs))
