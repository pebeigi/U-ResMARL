"""Compatibility shim — optional RLlib trainer lives in RL.train_rllib."""

from RL.train_rllib import *  # noqa: F401,F403
from RL.train_rllib import main, train

__all__ = ["main", "train"]

if __name__ == "__main__":
    main()
