"""Compatibility shim — primary trainer is RL.train_ppo."""

from RL.train_ppo import *  # noqa: F401,F403
from RL.train_ppo import TorchResidualPolicy, main, train

__all__ = ["TorchResidualPolicy", "main", "train"]

if __name__ == "__main__":
    main()
