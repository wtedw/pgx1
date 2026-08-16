"""Replay real professional games through the engine.

`tests/data/pro_games_128.json` and `tests/data/pro_games_1024.json` each
vendor real games converted to pgx1 mover-relative action labels by
`scripts/convert_pro_games.py` (see its docstring for the exact source
archives -- World Championship 2023 / Candidates 2024 / Wijk aan Zee 2024
event files for the 128 set; those three plus the Fischer/Kasparov/Karpov/
Carlsen player archives for the 1024 set), together with each game's
expected final board derived *independently* via python-chess at
conversion time.

Parametrized over both vendored sets, the full game batch replays through
`jax.vmap(Chess.step)`: every professionally played move must be legal in
the env's own mask at its ply, and each game's final board must equal the
python-chess-derived position -- real opening/middlegame/endgame traffic,
including castling both ways, en passant, promotions, checks, and mates.

The full 1024-game batch also replays through the compiled pipeline on
real hardware via `scripts/replay_pro_games_tpu.py`
(`just replay-pro-games --data tests/data/pro_games_1024.json`), which
this suite mirrors.
"""

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pgx1.chess import Chess

jax.config.update("jax_platform_name", "cpu")

DATA_DIR = pathlib.Path(__file__).parent / "data"
DATA_FILES = ["pro_games_128.json", "pro_games_1024.json"]


def _load(data_file):
    with open(DATA_DIR / data_file) as fh:
        games = json.load(fh)["games"]
    plies = np.array([g["plies"] for g in games])
    max_plies = int(plies.max())
    # [games, max_plies] int32, -1 beyond each game's end.
    labels = np.full((len(games), max_plies), -1, np.int32)
    for i, g in enumerate(games):
        labels[i, : g["plies"]] = g["labels"]
    final_boards = np.array([g["final_board"] for g in games], np.int32)
    final_colors = np.array([g["final_color"] for g in games], np.int32)
    return labels, plies, final_boards, final_colors


def _replay(step_fn, state, labels, plies, game_indices):
    """Drive `state` (a batched State) through the vendored games at
    `game_indices` with `step_fn`, asserting each professional move is
    legal in the env's mask at its ply. Games shorter than the longest
    keep stepping on an arbitrary legal action (argmax of the mask) purely
    to stay in lockstep; their boards are snapshotted at their own final
    ply. Returns ``(state, final board rows, final colors)`` aligned with
    `game_indices`."""
    sub_labels = jnp.asarray(labels[game_indices])
    sub_plies = plies[game_indices]
    max_plies = int(sub_plies.max())
    boards = np.zeros((len(game_indices), 64), np.int32)
    colors = np.zeros(len(game_indices), np.int32)
    for t in range(max_plies):
        active = t < sub_plies
        mask = np.asarray(state.legal_action_mask)
        pro_action = np.asarray(sub_labels[:, t])
        rows = np.flatnonzero(active)
        assert mask[rows, pro_action[rows]].all(), (
            f"ply {t}: professional move illegal in env mask for games "
            f"{[int(game_indices[r]) for r in rows[~mask[rows, pro_action[rows]]]]}"
        )
        action = jnp.where(jnp.asarray(active), sub_labels[:, t], jnp.argmax(mask, axis=-1).astype(jnp.int32))
        state = step_fn(state, action)
        ended = sub_plies == t + 1
        if ended.any():
            boards[ended] = np.asarray(state._x.board, np.int32)[ended]
            colors[ended] = np.asarray(state._x.color, np.int32)[ended]
    return state, boards, colors


@pytest.mark.parametrize("data_file", DATA_FILES)
def test_pro_games_replay_full_batch_scalar_env(data_file):
    labels, plies, final_boards, final_colors = _load(data_file)
    env = Chess()
    keys = jax.random.split(jax.random.PRNGKey(0), len(plies))
    state = jax.jit(jax.vmap(env.init))(keys)
    step_fn = jax.jit(jax.vmap(env.step))
    all_games = np.arange(len(plies))
    _, boards, colors = _replay(step_fn, state, labels, plies, all_games)
    np.testing.assert_array_equal(colors, final_colors, err_msg="side to move after last ply")
    np.testing.assert_array_equal(boards, final_boards, err_msg="final boards vs python-chess")
