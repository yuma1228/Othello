class AlphaZeroConfig:
    # Board
    board_size = 6
    action_size = 36  # 6 * 6

    # Neural Network
    num_res_blocks = 5
    num_filters = 64
    input_channels = 3  # black_plane, white_plane, color_plane
    learning_rate = 0.001
    weight_decay = 1e-4

    # MCTS
    num_simulations = 200
    c_puct = 1.5
    dirichlet_alpha = 0.3
    dirichlet_epsilon = 0.25
    temperature_threshold = 12

    # Self-Play
    num_self_play_games = 100
    max_replay_buffer_size = 50000

    # Training
    num_iterations = 50
    num_epochs = 10
    batch_size = 64
    checkpoint_interval = 5

    # Evaluation
    num_eval_games = 40
