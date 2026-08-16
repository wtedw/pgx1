"""Cross-check legal-move *counts* (not just legality) between pgx1 and
python-chess across the 128 vendored professional games.

`scripts/convert_pro_games.py` already proves every professionally played
move is legal in pgx1's mask and that final boards match; this script goes
one step further and, at every ply of every game, compares
``int(state.legal_action_mask.sum())`` (pgx1) against
``len(list(board.legal_moves))`` (python-chess) for the *same* position.
The two engines encode moves very differently (pgx1: from-square x 73
AlphaZero planes, queen promotions sharing the ordinary-move plane;
python-chess: one Move object per legal (from, to, promotion) triple) so
this is a real independent check that neither engine is over- or
under-counting -- e.g. a promoting pawn should contribute exactly 4 to
both counts (queen via the normal plane + 3 underpromotion planes, vs.
python-chess's 4 explicit Move objects).

Reproduces the identical deterministic game selection as
scripts/convert_pro_games.py (same PGN files, same order, same filters,
same PRNGKey(0) env) rather than reading the vendored JSON, so it only
needs the two engines to agree at generation time -- it doesn't depend on
that JSON having been written first.

Needs the `chess` package (python-chess), which is NOT a project
dependency -- run it as

    uv run --with chess python scripts/compare_legal_move_counts.py \\
        WorldChamp2023.pgn Candidates2024.pgn WijkaanZee2024.pgn
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import chess
import chess.pgn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np

from convert_pro_games import convert_game, verify_with_env, _JITTED  # noqa: E402

import pgx1.chess as pgx_chess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pgn_files", nargs="+", type=pathlib.Path)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--max-plies", type=int, default=200)
    args = parser.parse_args()

    env = pgx_chess.Chess()
    _JITTED["init"] = jax.jit(env.init)
    _JITTED["step"] = jax.jit(env.step)

    total_plies = 0
    mismatches = []
    checked_games = 0

    for path in args.pgn_files:
        with open(path, encoding="latin-1") as fh:
            while checked_games < args.games:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                converted = convert_game(game)
                if converted is None or converted["plies"] > args.max_plies:
                    continue
                reason = verify_with_env(converted)
                if reason is not None:
                    continue  # same skip criteria as convert_pro_games.py

                # Re-replay this (now known-good) game, this time counting
                # legal moves on both sides at every ply.
                state = _JITTED["init"](jax.random.PRNGKey(0))
                board = game.board()
                game_mismatches = []
                for t, (move, label) in enumerate(zip(game.mainline_moves(), converted["labels"])):
                    pgx_count = int(np.asarray(state.legal_action_mask).sum())
                    chess_count = len(list(board.legal_moves))
                    if pgx_count != chess_count:
                        game_mismatches.append((t, pgx_count, chess_count))
                    total_plies += 1
                    state = _JITTED["step"](state, jnp.int32(label))
                    board.push(move)

                checked_games += 1
                status = "OK" if not game_mismatches else f"{len(game_mismatches)} MISMATCH(ES)"
                print(
                    f"  {checked_games}/{args.games}: {converted['white']} - {converted['black']} "
                    f"({converted['plies']} plies): {status}",
                    flush=True,
                )
                if game_mismatches:
                    mismatches.append((converted, game_mismatches))
                    for t, pgx_count, chess_count in game_mismatches[:5]:
                        print(f"      ply {t}: pgx1={pgx_count} python-chess={chess_count}")

    print()
    print(f"checked {checked_games} games, {total_plies} plies")
    if mismatches:
        print(f"FAILED: {len(mismatches)} game(s) had a legal-move-count mismatch")
        sys.exit(1)
    print("OK: legal-move counts agree at every ply, every game")


if __name__ == "__main__":
    main()
