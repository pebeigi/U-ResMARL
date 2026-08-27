"""Compatibility shim — implementation lives in RL.eval_rllib."""

from RL.eval_rllib import *  # noqa: F401,F403
from RL.eval_rllib import find_latest_checkpoint, load_algorithm, main, record_rollout

__all__ = ["find_latest_checkpoint", "load_algorithm", "main", "record_rollout"]

if __name__ == "__main__":
    main()
