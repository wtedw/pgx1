"""Convert professional-game PGNs into vendored pgx1 replay test data.

Reads PGN files, converts each game's mainline moves into pgx1's
mover-relative AlphaZero action labels (`pgx1.chess.Action`), derives the
expected final board position *independently* via python-chess, verifies
the whole thing by replaying every game through `pgx1.chess.Chess.step`
(each converted label must be legal in the env's own mask at its ply, and
the env's final board must equal the python-chess-derived one), and writes
`tests/data/pro_games_128.json` for `tests/test_chess_pro_games.py` and
`scripts/replay_pro_games_tpu.py`.

This is a one-time generation tool: it needs the `chess` package
(python-chess), which is NOT a project dependency -- run it as

    uv run --with chess python scripts/convert_pro_games.py \
        WorldChamp2023.pgn Candidates2024.pgn WijkaanZee2024.pgn

The vendored JSON is the artifact; tests never import python-chess.

Coordinate conversion ("board convention"):
pgx1 squares are file-major (``sq = file*8 + rank``; a1=0, a2=1, ..., h8=63)
and mover-relative (the side to move always looks "up" the ranks; `_flip`
mirrors ranks within each file). python-chess squares are rank-major
(a1=0, b1=1, ...). A white-to-move square maps as ``file*8 + rank``; a
black-to-move square as ``file*8 + (7 - rank)``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import chess
import chess.pgn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

import pgx1.chess as pgx_chess

# python-chess piece types PAWN..KING = 1..6 match pgx1's constants exactly.
assert (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING) == (
    pgx_chess.PAWN,
    pgx_chess.KNIGHT,
    pgx_chess.BISHOP,
    pgx_chess.ROOK,
    pgx_chess.QUEEN,
    pgx_chess.KING,
)

UNDERPROMOTION_PIECE = {chess.ROOK: 0, chess.BISHOP: 1, chess.KNIGHT: 2}
# Mover-relative (to - from) offset -> underpromotion direction index d
# (label plane = u*3 + d), matching Action._from_label's up_off casework.
UNDERPROMOTION_DIRECTION = {1: 0, 9: 1, -7: 2}

_JITTED: dict = {}  # populated in main(); shared by verify_with_env


def to_pgx_square(square: int, white_to_move: bool) -> int:
    file, rank = chess.square_file(square), chess.square_rank(square)
    return file * 8 + (rank if white_to_move else 7 - rank)


def move_to_label(move: chess.Move, white_to_move: bool) -> int:
    from_ = to_pgx_square(move.from_square, white_to_move)
    to = to_pgx_square(move.to_square, white_to_move)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        u = UNDERPROMOTION_PIECE[move.promotion]
        d = UNDERPROMOTION_DIRECTION[to - from_]
        return from_ * 73 + u * 3 + d
    # Queen promotions are ordinary moves (pgx1 auto-queens); everything
    # else (castling included: the king's two-square hop is a slider plane)
    # goes through the real label arithmetic.
    label = pgx_chess.Action(from_=jnp.int32(from_), to=jnp.int32(to))._to_label()
    return int(label)


def final_board_pgx(board: chess.Board) -> list[int]:
    """The position after the last move, in pgx1 mover-relative encoding."""
    white_to_move = board.turn == chess.WHITE
    out = [0] * 64
    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == board.turn else -1
        out[to_pgx_square(square, white_to_move)] = sign * piece.piece_type
    return out


def convert_game(game: chess.pgn.Game) -> dict | None:
    if game.headers.get("SetUp") == "1" or "FEN" in game.headers:
        return None  # non-standard start
    if game.headers.get("Variant", "Standard") != "Standard":
        return None
    board = game.board()
    labels = []
    for move in game.mainline_moves():
        labels.append(move_to_label(move, board.turn == chess.WHITE))
        board.push(move)
    if len(labels) < 20:
        return None
    return {
        "event": game.headers.get("Event", "?"),
        "date": game.headers.get("Date", "?"),
        "white": game.headers.get("White", "?"),
        "black": game.headers.get("Black", "?"),
        "result": game.headers.get("Result", "?"),
        "plies": len(labels),
        "labels": labels,
        "final_board": final_board_pgx(board),
        "final_color": 0 if board.turn == chess.WHITE else 1,
    }


def verify_with_env(game: dict) -> str | None:
    """Replay one game through the scalar env. Returns None if every
    converted label is legal in the env's own mask at its ply, the env
    never auto-terminates before the game's last move (it enforces
    threefold-repetition/50-move draws that players are free not to
    claim), and the env's final board matches the python-chess-derived
    one; otherwise a reason string (the game should then be skipped)."""
    env = pgx_chess.Chess()
    state = _JITTED["init"](jax.random.PRNGKey(0))
    for t, label in enumerate(game["labels"]):
        if bool(state.terminated):
            return f"env auto-terminated at ply {t} (unclaimed draw rule?)"
        if not bool(state.legal_action_mask[label]):
            return f"ply {t} label {label} illegal in env mask"
        state = _JITTED["step"](state, jnp.int32(label))
    env_board = np.asarray(state._x.board, np.int32).tolist()
    if env_board != game["final_board"]:
        return "env final board != python-chess final board"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pgn_files", nargs="+", type=pathlib.Path)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent / "tests" / "data" / "pro_games_128.json",
    )
    args = parser.parse_args()

    env = pgx_chess.Chess()
    _JITTED["init"] = jax.jit(env.init)
    _JITTED["step"] = jax.jit(env.step)

    games: list[dict] = []
    for path in args.pgn_files:
        with open(path, encoding="latin-1") as fh:
            while len(games) < args.games:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                converted = convert_game(game)
                if converted is None or converted["plies"] > args.max_plies:
                    continue
                reason = verify_with_env(converted)
                if reason is not None:
                    print(f"  skipping {converted['white']} - {converted['black']}: {reason}", flush=True)
                    continue
                games.append(converted)
                print(
                    f"  verified {len(games)}/{args.games}: {converted['white']} - "
                    f"{converted['black']} ({converted['plies']} plies)",
                    flush=True,
                )
    if len(games) < args.games:
        raise SystemExit(f"only {len(games)} usable games found, need {args.games}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "pgnmentor.com event files (WorldChamp2023, Candidates2024, WijkaanZee2024)",
        "generator": "scripts/convert_pro_games.py (see its docstring)",
        "games": games,
    }
    args.out.write_text(json.dumps(payload) + "\n")
    total = sum(g["plies"] for g in games)
    print(f"wrote {args.out} ({len(games)} games, {total} plies, max {max(g['plies'] for g in games)})")


if __name__ == "__main__":
    main()
