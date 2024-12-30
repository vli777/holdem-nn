Inspired by Amortized Planning with Large-Scale Transformers: A Case Study on Chess
[arXiv:2402.04494](https://arxiv.org/abs/2402.04494)

Using the latest transformer models, the team was able to approach the leading poker AI performance using Monte Carlo Tree Search (MCTS)

This project uses generated data instead of actual game play data for training. The recommended training data sizes were provided as:
# simulated games : estimated usage capability
10 - 20k : prototype
100 - 200k: practical use
500k - 1m: production optimal

For the model, I utilized the Linformer model [ arXiv:2006.04768v3](https://arxiv.org/abs/2006.04768) for the transformer model and encoded several features for the card states of multiple players and their actions. 

Ideally after training the baseline model with transformers, if this approach can approximate SOTA performance, we can update the model with reinforcement learning to cover the remainder of the explorer space with faster convergence.* 

Instructions

1. Install requirements
2. Generate or place game data .npy files in data/ (see generate_data.py encoding methods to view format)
3. Run train.py
4. Test out a sample hand by running main.py


