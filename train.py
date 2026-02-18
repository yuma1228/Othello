import os
import torch
import torch.nn as nn
import torch.optim as optim
from network import AlphaZeroNetwork
from self_play import ReplayBuffer, generate_self_play_data
from arena import evaluate_against_random
from config import AlphaZeroConfig


def train_network(network, optimizer, replay_buffer, config, device):
    network.train()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    # バッファ全体を何周か回す (エポック数 × バッファ/バッチサイズ 分のバッチ)
    batches_per_epoch = max(len(replay_buffer) // config.batch_size, 1)
    for epoch in range(config.num_epochs):
        for _ in range(batches_per_epoch):
            states, target_policies, target_values = replay_buffer.sample(config.batch_size)
            states = states.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)

            policy_logits, value_pred = network(states)

            log_probs = torch.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.mean(torch.sum(target_policies * log_probs, dim=1))
            value_loss = nn.functional.mse_loss(value_pred, target_values)

            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

    return {
        'policy_loss': total_policy_loss / num_batches,
        'value_loss': total_value_loss / num_batches,
        'total_loss': (total_policy_loss + total_value_loss) / num_batches,
    }


def save_checkpoint(network, optimizer, iteration, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    torch.save({
        'iteration': iteration,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def main():
    config = AlphaZeroConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    network = AlphaZeroNetwork(
        board_size=config.board_size,
        num_res_blocks=config.num_res_blocks,
        num_filters=config.num_filters,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in network.parameters()):,}")

    optimizer = optim.Adam(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    replay_buffer = ReplayBuffer(config.max_replay_buffer_size)

    # 前回のチェックポイントがあれば再開
    start_iteration = 1
    resume_path = "checkpoints/model_final.pt"
    if os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device, weights_only=True)
        network.load_state_dict(ck['model_state_dict'])
        if 'optimizer_state_dict' in ck:
            optimizer.load_state_dict(ck['optimizer_state_dict'])
        start_iteration = ck['iteration'] + 1
        print(f"Resumed from iteration {ck['iteration']}")

    for iteration in range(start_iteration, start_iteration + config.num_iterations):
        print(f"\n{'='*50}")
        print(f"Iteration {iteration}")
        print(f"{'='*50}")

        # 1. Self-play
        print("\n[Self-play]")
        generate_self_play_data(network, config, replay_buffer, device)
        print(f"Buffer size: {len(replay_buffer)}")

        # 2. Train
        print("\n[Training]")
        if len(replay_buffer) >= config.batch_size:
            stats = train_network(network, optimizer, replay_buffer, config, device)
            print(f"  Policy loss: {stats['policy_loss']:.4f}")
            print(f"  Value loss:  {stats['value_loss']:.4f}")

        # 3. Save (毎回)
        save_checkpoint(network, optimizer, iteration, "checkpoints/model_final.pt")
        print(f"  Saved iteration {iteration}.")

        # 4. Evaluate (checkpoint_interval毎)
        if iteration % config.checkpoint_interval == 0:
            print("\n[Evaluation vs Random]")
            results = evaluate_against_random(network, config, device, num_games=20)
            print(f"  W/L/D = {results['wins']}/{results['losses']}/{results['draws']} "
                  f"(win rate: {results['win_rate']:.1%})")
            save_checkpoint(network, optimizer, iteration, f"checkpoints/model_iter_{iteration}.pt")

    save_checkpoint(network, optimizer, iteration, "checkpoints/model_final.pt")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
