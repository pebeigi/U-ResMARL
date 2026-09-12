"""Versioned experiment contract; pre-fix checkpoints remain historical artifacts."""

PROTOCOL_VERSION = 3


def validate_checkpoint(blob, obs_dim, path):
    if blob.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"{path}: checkpoint predates traffic protocol v{PROTOCOL_VERSION}; retrain before benchmarking")
    if int(blob.get("obs_dim", -1)) != int(obs_dim):
        raise ValueError(f"{path}: observation dimensions differ; retrain with the current observation layout")
