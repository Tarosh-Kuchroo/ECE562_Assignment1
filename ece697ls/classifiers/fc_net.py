"""Wrapper module to expose fc_net via the ece697ls namespace."""

from ece662.classifiers.fc_net import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
