import re

from pgx1.chess import Chess
from pgx1.connect_four import ConnectFour
from pgx1.go import Game, GameState, Go, State
from pgx1.hexnoswap import Hexnoswap
from pgx1.tic_tac_toe import TicTacToe

__all__ = ["Chess", "ConnectFour", "Game", "GameState", "Go", "Hexnoswap", "State", "TicTacToe", "make"]


def make(env_id: str, **kwargs):
    """Load the specified environment, pgx-`make`-style.

    Unlike `pgx.make`, extra keyword arguments are forwarded to the env's
    constructor (e.g. `pgx1.make("chess", use_bitmask=True)`), since pgx1 envs
    expose construction knobs (bitmask legality, observation toggling, ...)
    that pgx's don't.
    """
    if env_id == "tic_tac_toe":
        return TicTacToe(**kwargs)
    if env_id == "connect_four":
        return ConnectFour(**kwargs)
    if env_id == "chess":
        return Chess(**kwargs)
    m = re.fullmatch(r"(go|hexnoswap)_(\d+)x(\2)", env_id)
    if m:
        size = int(m.group(2))
        if m.group(1) == "go":
            return Go(size=size, **kwargs)
        return Hexnoswap(size=size, **kwargs)
    raise ValueError(f"Unknown env_id {env_id!r}")
