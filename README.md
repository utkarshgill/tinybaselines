# tinybaselines

Reinforcement learning in tinygrad. PPO on `LunarLanderContinuous-v3`.

`lander.py` is the tinygrad trainer. `reference_lander.py` is the PyTorch reference.
Both were copied unchanged from
[`utkarshgill/lunar_lander` at `26a7232`](https://github.com/utkarshgill/lunar_lander/tree/26a7232dc58a46522a7e1ddab6808addd949430f).

Python 3.11 and clang are required.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
NUM_ENVS=4 .venv/bin/python lander.py
```

For the PyTorch version, install `requirements-reference.txt` and run `reference_lander.py`.
