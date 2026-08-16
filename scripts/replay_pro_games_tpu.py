"""Replay vendored professional games through the compiled engine ON DEVICE.

`tests/test_chess_pro_games.py` runs the same replay on CPU; this script
replays a full vendored set (default `tests/data/pro_games_128.json`: World
Championship 2023, Candidates 2024, Wijk aan Zee 2024; pass
`--data tests/data/pro_games_1024.json` for the 1024-game set of
Fischer/Karpov/Kasparov/Carlsen plus those three events -- see
`scripts/convert_pro_games.py`) through `jax.vmap(Chess.step)` compiled on
the TPU, checking at every ply that

- each professionally played move is legal in the device-computed mask, and
- each game's final board equals the python-chess-derived position
  vendored at conversion time.

Games shorter than the longest keep stepping on an arbitrary legal action
(argmax of the mask) purely to stay in lockstep; their boards are compared
at their own final ply. Exits nonzero on any failure. Run via
`just replay-pro-games` (or `just replay-pro-games --data tests/data/pro_games_1024.json`).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

import pgx1.chess as chess

DEFAULT_DATA_PATH = pathlib.Path(__file__).resolve().parent.parent / "tests" / "data" / "pro_games_128.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=pathlib.Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()

    device = jax.devices()[0]

    with open(args.data) as fh:
        games = json.load(fh)["games"]
    num_games = len(games)
    plies = np.array([g["plies"] for g in games])
    max_plies = int(plies.max())
    labels = np.full((num_games, max_plies), -1, np.int32)
    for i, g in enumerate(games):
        labels[i, : g["plies"]] = g["labels"]
    final_boards = np.array([g["final_board"] for g in games], np.int32)

    env = chess.Chess()
    step_fn = jax.jit(jax.vmap(env.step))
    keys = jax.random.split(jax.random.PRNGKey(0), num_games)
    state = jax.jit(jax.vmap(env.init))(keys)

    labels_dev = jnp.asarray(labels)
    boards = np.zeros((num_games, 64), np.int32)
    failures = 0
    for t in range(max_plies):
        mask = np.asarray(state.legal_action_mask)

        active = t < plies
        rows = np.flatnonzero(active)
        legal = mask[rows, labels[rows, t]]
        if not legal.all():
            bad = rows[~legal]
            print(f"ILLEGAL ply={t}: professional moves rejected for games {bad.tolist()[:8]}")
            failures += 1

        action = jnp.where(
            jnp.asarray(active), labels_dev[:, t], jnp.argmax(jnp.asarray(mask), axis=-1).astype(jnp.int32)
        )
        state = step_fn(state, action)
        ended = plies == t + 1
        if ended.any():
            boards[ended] = np.asarray(state._x.board, np.int32)[ended]

    if not np.array_equal(boards, final_boards):
        bad = np.unique(np.argwhere(boards != final_boards)[:, 0])
        print(f"FINAL BOARD MISMATCH for games {bad.tolist()[:8]}")
        failures += 1

    total = int(plies.sum())
    print(
        f"replayed {num_games} professional games ({total} plies, longest {max_plies}) "
        f"through vmap(Chess.step) on {device.platform}:{device.device_kind}"
    )
    if failures:
        print(f"FAILED: {failures} check(s)")
        sys.exit(1)
    print("OK: every move legal, final boards match python-chess")


if __name__ == "__main__":
    main()
