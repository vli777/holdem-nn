WIP: to-do: sequence data training

Inspired by Amortized Planning with Large-Scale Transformers: A Case Study on Chess
[arXiv:2402.04494](https://arxiv.org/abs/2402.04494)

Using the latest transformer models, the team was able to approach the leading chess AI performance using Monte Carlo Tree Search (MCTS)

This project uses generated data instead of actual game play data for training. The recommended training data sizes were provided as:
# simulated games : estimated usage capability
10 - 20k : prototype
100 - 200k: practical use
500k - 1m: production optimal

Features being included: Game context, player tendencies, temporal dynamics, opponent modeling, etc.

