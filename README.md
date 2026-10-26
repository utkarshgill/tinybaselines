# tinybaselines

Small reinforcement learning implementations built on tinygrad.

The first supported task is `LunarLanderContinuous-v3`, with PPO.
Development starts from the existing LunarLander trainer and its PyTorch reference.
We verify behavior before extracting reusable interfaces.

This project is in early development. No package has been published to PyPI.

`lander.py` is the tinygrad trainer. `reference_lander.py` is the PyTorch reference.
Both were copied unchanged from
[`utkarshgill/lunar_lander` at `26a7232`](https://github.com/utkarshgill/lunar_lander/tree/26a7232dc58a46522a7e1ddab6808addd949430f).
They retain the known rollout-boundary issues until those receive focused fixes.
