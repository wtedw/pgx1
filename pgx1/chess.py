# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified from the original for pgx1.
"""Single-file port of `pgx.chess`

Main differences from the original
- uses a strictly-legal move generation algorithm (faster to execute)
- allows for bitmask representation of legal actions (saves space)
- allows for State to have None for observation (saves space)
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

# Everything is kept here so the complete environment can be read from top to
# bottom without following other implementation modules.

EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = tuple(range(7))

NUM_ACTIONS = 4672
NUM_ACTION_WORDS = (NUM_ACTIONS + 31) // 32
_BIT_OFFSETS = jnp.arange(32, dtype=jnp.uint32)
_BIT_VALUES = jnp.left_shift(jnp.uint32(1), _BIT_OFFSETS)


def pack_mask(mask: Array) -> Array:
    """Pack a bool legal-action mask into little-endian uint32 words."""
    n = mask.shape[-1]
    num_words = (n + 31) // 32
    pad = num_words * 32 - n
    padded = jnp.pad(mask.astype(jnp.uint32), [(0, 0)] * (mask.ndim - 1) + [(0, pad)])
    words = padded.reshape(mask.shape[:-1] + (num_words, 32))
    return jnp.sum(words * _BIT_VALUES, axis=-1, dtype=jnp.uint32)


def unpack_bitmask(bitmask: Array, *, length: int = NUM_ACTIONS) -> Array:
    """Unpack little-endian uint32 words into a bool mask."""
    mask = ((bitmask[..., :, None] & _BIT_VALUES) != 0).reshape(bitmask.shape[:-1] + (-1,))
    return mask[..., :length]


def bitmask_has_action(bitmask: Array, action: Array) -> Array:
    """Return whether `action` is set in a packed legal-action bitmask."""
    word = action // jnp.int32(32)
    bit = (action % jnp.int32(32)).astype(jnp.uint32)
    return ((jnp.take_along_axis(bitmask, word[..., None], axis=-1)[..., 0] >> bit) & jnp.uint32(1)).astype(jnp.bool_)


def bitmask_any(bitmask: Array) -> Array:
    return jnp.any(bitmask != jnp.uint32(0), axis=-1)


def full_legal_action_bitmask(shape_prefix=()) -> Array:
    words = jnp.full(shape_prefix + (NUM_ACTION_WORDS,), jnp.iinfo(jnp.uint32).max, dtype=jnp.uint32)
    extra = NUM_ACTION_WORDS * 32 - NUM_ACTIONS
    if extra:
        valid_last = jnp.uint32((1 << (32 - extra)) - 1)
        words = words.at[..., -1].set(valid_last)
    return words

PADDED_SQUARES = 128
NUM_ACTION_PLANES = 73

# Import-time geometry uses `(rank_delta, file_delta)` and the board encoding
# `square = file * 8 + rank` throughout.
RAY_DELTAS = (
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
)
KNIGHT_DELTAS = (
    (1, 2), (1, -2), (-1, 2), (-1, -2),
    (2, 1), (2, -1), (-2, 1), (-2, -1),
)
KING_DELTAS = (
    (-1, -1), (-1, 0), (-1, 1), (0, -1),
    (0, 1), (1, -1), (1, 0), (1, 1),
)
ENEMY_PAWN_ATTACK_DELTAS = ((-1, -1), (-1, 1))
FRIEND_PAWN_CAPTURE_DELTAS = ((1, -1), (1, 1))
FRIEND_PAWN_PUSH1_DELTAS = ((1, 0),)

_RAY_DR = jnp.int32([dr for dr, _ in RAY_DELTAS])
_RAY_DC = jnp.int32([dc for _, dc in RAY_DELTAS])
_RAY_IS_DIAGONAL = jnp.bool_([dr != 0 and dc != 0 for dr, dc in RAY_DELTAS])

_KNIGHT_PLANES = {
    (-1, -2): 65, (1, -2): 66, (-2, -1): 67, (2, -1): 68,
    (-1, 2): 69, (1, 2): 70, (-2, 1): 71, (2, 1): 72,
}


def _square_coords(square: int) -> tuple[int, int]:
    """Return `(rank, file)` for the mover-relative square encoding."""
    return square % 8, square // 8


def _square_index(rank: int, file: int) -> int:
    return file * 8 + rank


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _aligned_step(source: int, target: int) -> tuple[int, int] | None:
    """Return the unit line step, or `None` when the squares are unaligned."""
    source_rank, source_file = _square_coords(source)
    target_rank, target_file = _square_coords(target)
    rank_delta = target_rank - source_rank
    file_delta = target_file - source_file
    if source == target:
        return None
    if rank_delta == 0 or file_delta == 0 or abs(rank_delta) == abs(file_delta):
        return _sign(rank_delta), _sign(file_delta)
    return None


def _squares_between(source: int, target: int) -> tuple[int, ...]:
    """Return the strictly intermediate squares on an aligned line."""
    step = _aligned_step(source, target)
    if step is None:
        return ()
    rank, file = _square_coords(source)
    target_coords = _square_coords(target)
    rank += step[0]
    file += step[1]
    squares = []
    while (rank, file) != target_coords:
        squares.append(_square_index(rank, file))
        rank += step[0]
        file += step[1]
    return tuple(squares)


def _action_plane(source: int, target: int) -> int | None:
    """Return the ordinary AlphaZero plane for an import-time move pair."""
    if source == target:
        return None
    source_rank, source_file = _square_coords(source)
    target_rank, target_file = _square_coords(target)
    dr = target_rank - source_rank
    dc = target_file - source_file
    if (dr, dc) in _KNIGHT_PLANES:
        return _KNIGHT_PLANES[(dr, dc)]
    if not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        return None
    distance = max(abs(dr), abs(dc))
    forward, reverse = distance - 1, 7 - distance
    if dr < 0 and dc == 0:
        return 9 + reverse
    if dr > 0 and dc == 0:
        return 9 + 7 + forward
    if dr == 0 and dc < 0:
        return 9 + 2 * 7 + reverse
    if dr == 0 and dc > 0:
        return 9 + 3 * 7 + forward
    if dr < 0 and dc < 0:
        return 9 + 4 * 7 + reverse
    if dr > 0 and dc > 0:
        return 9 + 5 * 7 + forward
    if dr > 0 and dc < 0:
        return 9 + 6 * 7 + reverse
    return 9 + 7 * 7 + forward


def _between_incidence_by_source() -> np.ndarray:
    """Build ``[source, between-square, padded-target]`` ray incidence."""
    table = np.zeros((64, PADDED_SQUARES, PADDED_SQUARES), np.float32)
    for source in range(64):
        for target in range(64):
            for between in _squares_between(source, target):
                table[source, between, target] = 1
    return table


def _slider_geometry() -> np.ndarray:
    """Build bishop, rook, and queen destination geometry per source."""
    geometry = np.zeros((64, 3, PADDED_SQUARES), np.int32)
    for source in range(64):
        source_rank, source_file = _square_coords(source)
        for target in range(64):
            target_rank, target_file = _square_coords(target)
            rook = source_rank == target_rank or source_file == target_file
            bishop = abs(source_rank - target_rank) == abs(source_file - target_file)
            geometry[source, 0, target] = bishop and source != target
            geometry[source, 1, target] = rook and source != target
            geometry[source, 2, target] = (bishop or rook) and source != target
    return geometry


BETWEEN_BY_SOURCE = jnp.asarray(
    _between_incidence_by_source(), dtype=jnp.bfloat16
)
SLIDER_GEOMETRY = jnp.asarray(_slider_geometry())



def _fixed_offset_table(deltas: tuple[tuple[int, int], ...]) -> np.ndarray:
    """``[64,128]`` float32: ``table[source,target] = 1`` iff ``target =
    source + delta`` (in bounds) for some ``delta`` in ``deltas``. Uses
    pgx1.chess's ``sq = file*8 + rank`` convention: ``r = sq%8`` (rank),
    ``c = sq//8`` (file)."""
    table = np.zeros((64, PADDED_SQUARES), np.float32)
    for source in range(64):
        r0, c0 = _square_coords(source)
        for dr, dc in deltas:
            r1, c1 = r0 + dr, c0 + dc
            if 0 <= r1 < 8 and 0 <= c1 < 8:
                table[source, _square_index(r1, c1)] = 1.0
    return table


# An enemy pawn attacks `pos` from one rank *above* it (mover-relative board:
# enemy pawns advance toward decreasing rank.
# Friendly pawns advance toward increasing rank.
KNIGHT_DEST = jnp.asarray(_fixed_offset_table(KNIGHT_DELTAS))
KING_DEST = jnp.asarray(_fixed_offset_table(KING_DELTAS))
ENEMY_PAWN_ATTACK_DEST = jnp.asarray(_fixed_offset_table(ENEMY_PAWN_ATTACK_DELTAS))
FRIEND_PAWN_CAPTURE_DEST = jnp.asarray(_fixed_offset_table(FRIEND_PAWN_CAPTURE_DELTAS))
FRIEND_PAWN_PUSH1_DEST = jnp.asarray(_fixed_offset_table(FRIEND_PAWN_PUSH1_DELTAS))

def _between_squares_arithmetic(source: Array, target: Array) -> Array:
    """``[128]`` bool: squares strictly between scalar `source` and `target`
    on a straight/diagonal line (all-zero if unaligned, adjacent, or equal).

    Pure per-board arithmetic, not a table lookup: used only for the at-most-
    one actual checking slider on a given board, where materializing a
    per-board `(64,128)` or `(128,128)` relation across the whole batch
    (as `BETWEEN_BY_SOURCE` would require, since the king's square is a
    per-board dynamic value) would be exactly the wasteful materialization
    this design otherwise avoids.
    """
    r0, c0 = source % 8, source // 8
    r1, c1 = target % 8, target // 8
    dr, dc = r1 - r0, c1 - c0
    aligned = (dr == 0) | (dc == 0) | (jnp.abs(dr) == jnp.abs(dc))
    step_dr, step_dc = jnp.sign(dr), jnp.sign(dc)
    delta = step_dc * 8 + step_dr
    path_len = jnp.maximum(jnp.abs(dr), jnp.abs(dc))
    step_multipliers = jnp.arange(1, 7)
    intermediate_squares = source + step_multipliers * delta
    valid = aligned & (step_multipliers < path_len)
    safe_idx = jnp.where(valid, intermediate_squares, -1)
    return jax.nn.one_hot(safe_idx, PADDED_SQUARES, dtype=jnp.bool_).any(axis=0)


# -----------------------------------------------------------------------
# Static move geometry
# -----------------------------------------------------------------------

MOVE_PAWN_PUSH, MOVE_PAWN_CAPTURE, MOVE_KNIGHT, MOVE_BISHOP, MOVE_ROOK, MOVE_QUEEN, MOVE_KING = range(7)


def _pawn_push_geometry() -> np.ndarray:
    """``[64,128]``: single push always, plus the double-push target only
    from rank index 1 (the starting rank in this mover-relative board, same
    convention `friendly_near_pseudo_moves` uses for `pawn_at_rank1`). The
    double push's intermediate-square emptiness is not encoded here -- it
    falls out for free from the same incidence-table blocker dot that the
    slider pipeline uses, since ``BETWEEN_BY_SOURCE`` is piece-agnostic
    geometry: a two-squares-same-file target already has its one
    intermediate square marked in that table.
    """
    table = np.zeros((64, PADDED_SQUARES), np.float32)
    for source in range(64):
        r0, c0 = _square_coords(source)
        r1 = r0 + 1
        if r1 < 8:
            table[source, _square_index(r1, c0)] = 1.0
            if r0 == 1:
                r2 = r0 + 2
                if r2 < 8:
                    table[source, _square_index(r2, c0)] = 1.0
    return table


def _move_geometry() -> np.ndarray:
    """``[64,7,128]``: per-source destination geometry for each of the seven
    move classes (see ``MOVE_PAWN_PUSH`` .. ``MOVE_KING``), ignoring
    blockers/occupancy (those are separate per-board filters applied in the
    emission kernel). Reuses the already-validated fixed-offset and slider
    tables rather than recomputing geometry from scratch.
    """
    geometry = np.zeros((64, 7, PADDED_SQUARES), np.float32)
    geometry[:, MOVE_PAWN_PUSH, :] = _pawn_push_geometry()
    geometry[:, MOVE_PAWN_CAPTURE, :] = _fixed_offset_table(FRIEND_PAWN_CAPTURE_DELTAS)
    geometry[:, MOVE_KNIGHT, :] = _fixed_offset_table(KNIGHT_DELTAS)
    slider_geometry = _slider_geometry()
    geometry[:, MOVE_BISHOP, :] = slider_geometry[:, 0, :]
    geometry[:, MOVE_ROOK, :] = slider_geometry[:, 1, :]
    geometry[:, MOVE_QUEEN, :] = slider_geometry[:, 2, :]
    geometry[:, MOVE_KING, :] = _fixed_offset_table(KING_DELTAS)
    return geometry


def _lines_by_source() -> np.ndarray:
    """``[64,4,128]``: for each source, the four full lines through it (same
    rank, same file, main diagonal, anti-diagonal), each a whole-board
    bitboard clipped to the 64 real squares -- not rays restricted to one
    direction or to "between", and not blocker-aware. Used by the pass-2 pin
    overlay's collinear trick: a pinned piece may move only within whichever
    of these four lines the king currently sits on, computable purely from a
    king one-hot with no per-(board,source) data (see the completion plan's
    "Pin overlay" step). The source's own square is (trivially) a member of
    all four lines; harmless, since no piece's move geometry ever targets its
    own square.
    """
    table = np.zeros((64, 4, PADDED_SQUARES), np.float32)
    for source in range(64):
        r0, c0 = _square_coords(source)
        for target in range(64):
            r1, c1 = _square_coords(target)
            if r1 == r0:
                table[source, 0, target] = 1.0
            if c1 == c0:
                table[source, 1, target] = 1.0
            if (r1 - r0) == (c1 - c0):
                table[source, 2, target] = 1.0
            if (r1 - r0) == -(c1 - c0):
                table[source, 3, target] = 1.0
    return table


def _pin_mask_by_source() -> np.ndarray:
    """``[64,128,128]``: ``table[source, king_square, target] = 1`` iff
    ``target`` lies on some line through ``source`` (rank/file/diagonal/
    anti-diagonal) that also passes through ``king_square`` -- the pass-2
    pin overlay's collinear-trick result, precomputed here rather than
    combined from `_lines_by_source()` inside the kernel.

    The in-kernel version of this (``king_onehot @ lines_by_source[source].T``
    then ``lines_by_source[source] @ that_result``, contracting over the
    4-element line-index axis) measured as an unsupported Mosaic TPU
    lowering ("failed to legalize arith.cmpi" on an oddly-shaped
    ``vector<8x128x4xi8>`` intermediate) -- the size-4 axis is far narrower
    than the native 128-lane tile width, and letting Mosaic's vectorizer
    fuse a comparison across it produced a packed layout it couldn't
    legalize. Precomputing the union directly, at build time in plain numpy
    with an explicit `king_square` axis, lets the kernel apply it as one
    ordinary ``[king_onehot] @ [128,128]`` MXU dot -- the same shape and
    pattern `BETWEEN_BY_SOURCE`/`PLANE_PERM` already use successfully.
    """
    lines = _lines_by_source()  # [64,4,128]
    table = np.zeros((64, PADDED_SQUARES, PADDED_SQUARES), np.float32)
    for source in range(64):
        for king_square in range(64):
            axes_with_king = lines[source, :, king_square] > 0  # [4]
            if not axes_with_king.any():
                continue
            table[source, king_square, :] = lines[source][axes_with_king].any(axis=0)
    return table


def _plane_permutation_by_source() -> np.ndarray:
    """``[64,128,128]``: ``table[source,target,plane] = 1`` iff a move from
    ``source`` to ``target`` maps to that AlphaZero ``plane`` under
    `Action._to_label` (planes 9..72; underpromotion planes 0..8 are label
    surgery applied afterward in JAX, exactly as `_to_label` itself never
    produces them for a plain from/to pair). Built directly from the same
    arithmetic `Action._to_label` uses, restricted to one fixed `source` at a
    time so it reduces to plain per-target casework -- not a reimplementation
    of different logic, just that logic unrolled into a lookup table so the
    kernel can apply it as one MXU dot instead of scalar control flow.
    """
    table = np.zeros((64, PADDED_SQUARES, PADDED_SQUARES), np.float32)
    for source in range(64):
        for target in range(64):
            plane = _action_plane(source, target)
            if plane is not None:
                table[source, target, plane] = 1.0
    return table


MOVE_GEOMETRY = jnp.asarray(_move_geometry())
PIN_MASK_BY_SOURCE = jnp.asarray(_pin_mask_by_source(), dtype=jnp.bfloat16)
_plane_perm_table = _plane_permutation_by_source()
PLANE_PERM = jnp.asarray(_plane_perm_table, dtype=jnp.bfloat16)
# [64,128] int32 `argmax(PLANE_PERM, axis=-1)`, precomputed so
# `apply_en_passant`'s per-board (from,to)->plane lookup is a tiny one-hot
# contraction instead of a per-board [128]-lane gather from the
# [64,128,128] table (two such gathers -- one per en passant `from_`
# candidate -- measured ~0.5 ms/step combined at batch 4096 on TPU v4).
PLANE_INDEX = jnp.asarray(np.argmax(_plane_perm_table, axis=-1).astype(np.int32))



def _underpromotion_ordinary_planes(plane_permutation: np.ndarray) -> np.ndarray:
    """``[3]`` int32: the ordinary (non-underpromotion) plane index that a
    one-step pawn push (index 0), a capture toward the higher file (index
    1), and a capture toward the lower file (index 2) map to under
    `Action._to_label` -- the same plane any of that (from,to) pair's
    underpromotions (rook/bishop/knight) share. Constant across files (only
    depends on the fixed ``(dr,dc)`` of each direction), derived directly
    from ``PLANE_PERM`` (not hand-computed) using source ``b7`` (14) so all
    three directions stay in-bounds."""
    source = 14  # b7: file 1, rank index 6
    targets = [source + 1, source + 8 + 1, source - 8 + 1]  # push, capture-right, capture-left
    return np.array(
        [int(np.argmax(plane_permutation[source, target])) for target in targets],
        np.int32,
    )


UNDERPROMOTION_ORDINARY_PLANES = jnp.asarray(
    _underpromotion_ordinary_planes(_plane_perm_table)
)
del _plane_perm_table
RANK6_SOURCES = jnp.arange(8) * 8 + 6  # a7,b7,...,h7: the only sources underpromotion can originate from


_CASTLE_QUEEN_SIDE_SQUARES = jnp.array([0, 8, 16, 24, 32])
_CASTLE_QUEEN_SIDE_EXPECTED = jnp.array([ROOK, 0, 0, 0, KING])
_CASTLE_KING_SIDE_SQUARES = jnp.array([32, 40, 48, 56])
_CASTLE_KING_SIDE_EXPECTED = jnp.array([KING, 0, 0, ROOK])
_CASTLE_QUEEN_SIDE_SAFE_SQUARES = jnp.array([16, 24, 32])
_CASTLE_KING_SIDE_SAFE_SQUARES = jnp.array([32, 40, 48])

CASTLE_QUEEN_SIDE_LABEL = 2364
CASTLE_KING_SIDE_LABEL = 2367


def near_enemy_attacked(board: Array) -> Array:
    """Squares attacked by enemy knights, kings, and pawns.

    The leading dimensions are shape-polymorphic: scalar callers pass
    ``[64]`` and the batch-native pipeline passes ``[batch,64]``.
    """
    knight = (board == -KNIGHT).astype(jnp.float32)
    king = (board == -KING).astype(jnp.float32)
    pawn = (board == -PAWN).astype(jnp.float32)
    total = knight @ KNIGHT_DEST + king @ KING_DEST + pawn @ ENEMY_PAWN_ATTACK_DEST
    return total > 0


def full_attacked_squares(board: Array, slider_attacked: Array) -> Array:
    """Combine slider and fixed-offset enemy attacks for any leading shape."""
    return (slider_attacked != 0) | near_enemy_attacked(board)


def apply_underpromotions(planes: Array, board: Array) -> Array:
    """Fill underpromotion planes for scalar or explicitly batched inputs."""
    is_pawn_here = board[..., RANK6_SOURCES] == PAWN
    ordinary = planes[..., RANK6_SOURCES, :][..., UNDERPROMOTION_ORDINARY_PLANES]
    multiples = (1,) * (ordinary.ndim - 1) + (3,)
    underpromo = jnp.tile(ordinary, multiples) & is_pawn_here[..., None]
    return planes.at[..., RANK6_SOURCES, :9].set(underpromo)


def apply_castling(
    mask4672: Array,
    board: Array,
    castling_rights: Array,
    attacked_full: Array,
) -> Array:
    """Add castling labels for scalar or explicitly batched inputs."""
    q_pieces = board[..., _CASTLE_QUEEN_SIDE_SQUARES]
    can_castle_q = castling_rights[..., 0, 0] & (
        q_pieces == _CASTLE_QUEEN_SIDE_EXPECTED
    ).all(axis=-1)
    k_pieces = board[..., _CASTLE_KING_SIDE_SQUARES]
    can_castle_k = castling_rights[..., 0, 1] & (
        k_pieces == _CASTLE_KING_SIDE_EXPECTED
    ).all(axis=-1)

    not_attacked_q = ~attacked_full[..., _CASTLE_QUEEN_SIDE_SAFE_SQUARES].any(axis=-1)
    not_attacked_k = ~attacked_full[..., _CASTLE_KING_SIDE_SAFE_SQUARES].any(axis=-1)
    mask4672 = mask4672.at[..., CASTLE_QUEEN_SIDE_LABEL].set(
        mask4672[..., CASTLE_QUEEN_SIDE_LABEL] | (can_castle_q & not_attacked_q)
    )
    return mask4672.at[..., CASTLE_KING_SIDE_LABEL].set(
        mask4672[..., CASTLE_KING_SIDE_LABEL] | (can_castle_k & not_attacked_k)
    )



# -----------------------------------------------------------------------
# Scalar legal-move generation
# -----------------------------------------------------------------------


class _PositionFacts(NamedTuple):
    king: Array
    king_square: Array
    occupancy: Array
    occupancy_without_king: Array
    empty: Array
    friendly_occupied: Array
    enemy_occupied: Array
    piece_class: Array
    enemy_slider_class: Array


def _friendly_king_square(board: Array) -> Array:
    """Locate the single friendly king required by the legal-move pipeline."""
    return jnp.argmin(jnp.abs(board.astype(jnp.int32) - KING))


def _analyze_position(board: Array) -> _PositionFacts:
    """Compute the padded occupancy and piece channels used by both passes."""
    if board.ndim != 1 or board.shape[0] != 64:
        raise ValueError(f"expected board [64], got {board.shape}")
    king = jnp.pad((board == KING).astype(jnp.int32), (0, PADDED_SQUARES - 64))
    occupied = jnp.pad(board != 0, (0, PADDED_SQUARES - 64))
    enemy_slider_class = (
        (board == -BISHOP).astype(jnp.int32)
        + 2 * (board == -ROOK).astype(jnp.int32)
        + 3 * (board == -QUEEN).astype(jnp.int32)
    )
    return _PositionFacts(
        king=king,
        king_square=_friendly_king_square(board),
        occupancy=occupied.astype(jnp.bfloat16),
        occupancy_without_king=(occupied & (king == 0)).astype(jnp.bfloat16),
        empty=jnp.pad(
            (board == 0).astype(jnp.int32), (0, PADDED_SQUARES - 64)
        ),
        friendly_occupied=jnp.pad(
            (board > 0).astype(jnp.int32), (0, PADDED_SQUARES - 64)
        ),
        enemy_occupied=jnp.pad(
            (board < 0).astype(jnp.int32), (0, PADDED_SQUARES - 64)
        ),
        piece_class=jnp.maximum(board, 0),
        enemy_slider_class=enemy_slider_class,
    )


def _board_at(board: Array, idx: Array) -> Array:
    """``board[idx]`` via one-hot contraction; invalid indices read as empty."""
    valid = (idx >= 0) & (idx < 64)
    safe = jnp.where(valid, idx, 0)
    one_hot = jax.nn.one_hot(safe, 64, dtype=board.dtype)
    pieces = jnp.einsum("...q,q->...", one_hot, board)
    return jnp.where(valid, pieces, 0)


def _board_set(board: Array, idx: Array, val: Array) -> Array:
    """``board.at[idx].set(val)`` via one-hot mask; invalid indices no-op."""
    mask = jax.nn.one_hot(idx, 64, dtype=board.dtype)
    return board * (1 - mask) + mask * val


def _slider_checkers_and_pins_from_king(
    board: Array, king_square: Array
) -> tuple[Array, Array]:
    r0, c0 = king_square % 8, king_square // 8
    distance = jnp.arange(1, 8, dtype=jnp.int32)
    ray_r = r0 + _RAY_DR[:, None] * distance[None, :]
    ray_c = c0 + _RAY_DC[:, None] * distance[None, :]
    in_bounds = (ray_r >= 0) & (ray_r < 8) & (ray_c >= 0) & (ray_c < 8)
    ray_square = jnp.where(in_bounds, ray_c * 8 + ray_r, -1)
    pieces = _board_at(board, ray_square)

    blocker_number = jnp.cumsum((pieces != 0).astype(jnp.int32), axis=1)
    first_mask = (pieces != 0) & (blocker_number == 1)
    second_mask = (pieces != 0) & (blocker_number == 2)
    first_piece = (pieces * first_mask).sum(axis=1)
    second_piece = (pieces * second_mask).sum(axis=1)
    first_square = (jnp.where(first_mask, ray_square, 0)).sum(axis=1)

    def compatible_enemy_slider(piece: Array) -> Array:
        abs_piece = jnp.abs(piece)
        compatible = jnp.where(
            _RAY_IS_DIAGONAL,
            (abs_piece == BISHOP) | (abs_piece == QUEEN),
            (abs_piece == ROOK) | (abs_piece == QUEEN),
        )
        return (piece < 0) & compatible

    checker_square = jnp.where(compatible_enemy_slider(first_piece), first_square, -1)
    pinned_square = jnp.where(
        (first_piece > 0) & compatible_enemy_slider(second_piece), first_square, -1
    )
    checker = jax.nn.one_hot(
        checker_square, PADDED_SQUARES, dtype=jnp.bool_
    ).any(axis=0)
    pinned = jax.nn.one_hot(
        pinned_square, PADDED_SQUARES, dtype=jnp.bool_
    ).any(axis=0)
    return checker, pinned


def _slider_state(
    board: Array,
    facts: _PositionFacts,
) -> tuple[Array, Array, Array]:
    """Scalar pass 1, returning ``attacked/checker/pinned`` as ``[128]``."""
    geometry_mask = SLIDER_GEOMETRY != 0

    blockers_nk = jnp.einsum(
        "j,sjt->st",
        facts.occupancy_without_king,
        BETWEEN_BY_SOURCE,
        preferred_element_type=jnp.bfloat16,
    )
    geo_mask = (
        ((facts.enemy_slider_class == 1)[:, None] & geometry_mask[:, 0, :])
        | ((facts.enemy_slider_class == 2)[:, None] & geometry_mask[:, 1, :])
        | ((facts.enemy_slider_class == 3)[:, None] & geometry_mask[:, 2, :])
    )
    attacked = (geo_mask & (blockers_nk == 0)).any(axis=0)
    checker, pinned = _slider_checkers_and_pins_from_king(
        board, facts.king_square
    )
    return (
        attacked.astype(jnp.int32),
        checker.astype(jnp.int32),
        pinned.astype(jnp.int32),
    )


def _check_evasion_targets(
    board: Array, checker: Array, king_square: Array
) -> Array:
    """Return the destinations that non-king moves may use during check."""
    one_hot_king = jax.nn.one_hot(king_square, 64, dtype=jnp.float32)

    knight_reaches_king = (
        jnp.einsum("st,t->s", KNIGHT_DEST[:, :64], one_hot_king) > 0
    )
    knight_checker_sources = (board == -KNIGHT) & knight_reaches_king
    pawn_reaches_king = (
        jnp.einsum(
            "st,t->s", ENEMY_PAWN_ATTACK_DEST[:, :64], one_hot_king
        )
        > 0
    )
    pawn_checker_sources = (board == -PAWN) & pawn_reaches_king
    slider_checker_sources = checker[:64] > 0

    n_check = (
        slider_checker_sources.sum(axis=-1)
        + knight_checker_sources.sum(axis=-1)
        + pawn_checker_sources.sum(axis=-1)
    )
    in_check = n_check > 0
    single_check = n_check == 1
    slider_single = slider_checker_sources.sum(axis=-1) == 1

    slider_source = jnp.argmax(slider_checker_sources.astype(jnp.int32))
    between_bitboard = jnp.where(
        slider_single,
        _between_squares_arithmetic(slider_source, king_square),
        jnp.zeros((PADDED_SQUARES,), jnp.bool_),
    )

    evasion_from_slider = (
        jnp.pad(slider_checker_sources & slider_single, (0, PADDED_SQUARES - 64))
        | between_bitboard
    )
    evasion_from_knight = jnp.pad(knight_checker_sources, (0, PADDED_SQUARES - 64))
    evasion_from_pawn = jnp.pad(pawn_checker_sources, (0, PADDED_SQUARES - 64))
    evasion_union = evasion_from_slider | evasion_from_knight | evasion_from_pawn

    all_ones = jnp.ones((PADDED_SQUARES,), jnp.bool_)
    all_zeros = jnp.zeros((PADDED_SQUARES,), jnp.bool_)
    return jnp.where(
        ~in_check,
        all_ones,
        jnp.where(single_check, evasion_union, all_zeros),
    )


def _legal_move_planes(
    board: Array,
    facts: _PositionFacts,
    evasion_targets: Array,
    attacked_full: Array,
    pinned: Array,
) -> Array:
    """Scalar pass 2, returning ``[64,73]`` AlphaZero planes."""
    blockers = jnp.einsum(
        "j,sjt->st",
        facts.occupancy,
        BETWEEN_BY_SOURCE,
        preferred_element_type=jnp.bfloat16,
    )
    blockers_clear = blockers == 0
    geometry = MOVE_GEOMETRY != 0

    def class_targets(class_index: int, move_class: int) -> Array:
        return (facts.piece_class == class_index)[:, None] & geometry[
            :, move_class, :
        ]

    empty_target = (facts.empty != 0)[None, :]
    enemy_target = (facts.enemy_occupied != 0)[None, :]
    not_friendly = (facts.friendly_occupied == 0)[None, :]
    legal_targets = (
        (class_targets(PAWN, MOVE_PAWN_PUSH) & empty_target)
        | (class_targets(PAWN, MOVE_PAWN_CAPTURE) & enemy_target)
        | (class_targets(KNIGHT, MOVE_KNIGHT) & not_friendly)
        | (class_targets(BISHOP, MOVE_BISHOP) & not_friendly)
        | (class_targets(ROOK, MOVE_ROOK) & not_friendly)
        | (class_targets(QUEEN, MOVE_QUEEN) & not_friendly)
        | (class_targets(KING, MOVE_KING) & not_friendly)
    ) & blockers_clear

    legal_targets &= jnp.where(
        (facts.piece_class == KING)[:, None],
        (attacked_full == 0)[None, :],
        (evasion_targets != 0)[None, :],
    )

    pinned_here = (pinned[:64] != 0)[:, None]
    pin_line = (
        jnp.einsum(
            "k,skt->st",
            facts.king.astype(jnp.bfloat16),
            PIN_MASK_BY_SOURCE,
            preferred_element_type=jnp.bfloat16,
        )
        > 0
    )
    legal_targets &= ~pinned_here | pin_line

    planes = jnp.einsum(
        "st,stp->sp",
        legal_targets.astype(jnp.bfloat16),
        PLANE_PERM,
        preferred_element_type=jnp.bfloat16,
    )
    return planes[:, :NUM_ACTION_PLANES] > 0


def _slider_attacks_square(board: Array, target_sq: Array) -> Array:
    r0, c0 = target_sq % 8, target_sq // 8
    distances = jnp.arange(1, 8, dtype=jnp.int32)
    new_r = r0 + _RAY_DR[:, None] * distances[None, :]
    new_c = c0 + _RAY_DC[:, None] * distances[None, :]
    in_bounds = (new_r >= 0) & (new_r < 8) & (new_c >= 0) & (new_c < 8)
    ray_squares = jnp.where(in_bounds, new_c * 8 + new_r, -1)

    pieces_on_rays = _board_at(board, ray_squares)
    is_blocker = pieces_on_rays != 0
    is_first_blocker = is_blocker & (
        jnp.cumsum(is_blocker.astype(jnp.int32), axis=1) == 1
    )
    first_blocker = (pieces_on_rays * is_first_blocker).sum(axis=1)
    is_opponent = first_blocker < 0
    abs_piece = jnp.abs(first_blocker)
    can_attack = jnp.where(
        _RAY_IS_DIAGONAL,
        (abs_piece == BISHOP) | (abs_piece == QUEEN),
        (abs_piece == ROOK) | (abs_piece == QUEEN),
    )
    return (is_opponent & can_attack).any(axis=0)


def _offset_piece_attacks_square(
    board: Array,
    target_sq: Array,
    piece: int,
    deltas: tuple[tuple[int, int], ...],
) -> Array:
    """Whether an enemy fixed-offset piece attacks one target square."""
    target_rank, target_file = target_sq % 8, target_sq // 8
    rank_delta = jnp.asarray([dr for dr, _ in deltas], dtype=jnp.int32)
    file_delta = jnp.asarray([dc for _, dc in deltas], dtype=jnp.int32)
    source_rank = target_rank - rank_delta
    source_file = target_file - file_delta
    in_bounds = (
        (source_rank >= 0)
        & (source_rank < 8)
        & (source_file >= 0)
        & (source_file < 8)
    )
    source = jnp.where(in_bounds, source_file * 8 + source_rank, -1)
    return (_board_at(board, source) == -piece).any()


def _is_in_check(board: Array) -> Array:
    """Targeted attack query for the friendly king on a valid chess board."""
    if board.ndim != 1 or board.shape[0] != 64:
        raise ValueError(f"expected board [64], got {board.shape}")
    target = _friendly_king_square(board)
    by_near = (
        _offset_piece_attacks_square(board, target, KNIGHT, KNIGHT_DELTAS)
        | _offset_piece_attacks_square(board, target, KING, KING_DELTAS)
        | _offset_piece_attacks_square(
            board, target, PAWN, ENEMY_PAWN_ATTACK_DELTAS
        )
    )
    return by_near | _slider_attacks_square(board, target)


def apply_en_passant(
    mask4672: Array,
    board: Array,
    en_passant: Array,
    king_square: Array,
) -> Array:
    to = en_passant.astype(jnp.int32)
    zero = jnp.asarray(0, board.dtype)
    pawn_val = jnp.asarray(PAWN, board.dtype)

    for from_ in (to - 9, to + 7):
        ok = (
            (from_ >= 0)
            & (from_ < 64)
            & (to >= 0)
            & (_board_at(board, from_) == PAWN)
            & (_board_at(board, to - 1) == -PAWN)
        )
        safe_from = jnp.where(ok, from_, 64)
        safe_capture = jnp.where(ok, to - 1, 64)
        safe_to = jnp.where(ok, to, 64)
        modified = _board_set(board, safe_from, zero)
        modified = _board_set(modified, safe_capture, zero)
        modified = _board_set(modified, safe_to, pawn_val)
        ok = ok & ~_slider_attacks_square(modified, king_square)

        safe_from_idx = jnp.clip(from_, 0, 63)
        safe_to_idx = jnp.clip(to, 0, 63)
        one_hot_from = jax.nn.one_hot(safe_from_idx, 64, dtype=jnp.float32)
        one_hot_to = jax.nn.one_hot(safe_to_idx, PADDED_SQUARES, dtype=jnp.float32)
        plane = jnp.einsum(
            "s,st,t->", one_hot_from, PLANE_INDEX.astype(jnp.float32), one_hot_to
        ).astype(jnp.int32)
        label = jnp.where(ok, from_ * NUM_ACTION_PLANES + plane, -1)
        mask4672 = mask4672 | jax.nn.one_hot(label, 64 * NUM_ACTION_PLANES, dtype=jnp.bool_)
    return mask4672


class _LegalResult(NamedTuple):
    mask: Array
    in_check: Array


def _legal_action_result(
    board: Array, en_passant: Array, castling_rights: Array
) -> _LegalResult:
    """Generate the scalar legal mask and reusable position metadata."""
    facts = _analyze_position(board)

    attacked, checker, pinned = _slider_state(board, facts)
    attacked_full = full_attacked_squares(board, attacked)
    in_check = (
        attacked_full & facts.king.astype(jnp.bool_)
    ).any()
    # Evasion counting deliberately excludes an adjacent enemy king, while
    # checkmate metadata must include every attacked-square class. Adjacent
    # kings are unreachable in legal play but are useful synthetic test input.
    evasion_targets = _check_evasion_targets(
        board, checker, facts.king_square
    )

    planes = _legal_move_planes(
        board,
        facts,
        evasion_targets,
        attacked_full,
        pinned,
    )
    planes = planes & (board > 0)[:, None]
    planes = apply_underpromotions(planes, board)

    mask4672 = planes.reshape(64 * NUM_ACTION_PLANES)
    mask4672 = apply_castling(mask4672, board, castling_rights, attacked_full)
    mask4672 = apply_en_passant(
        mask4672, board, en_passant, facts.king_square
    )
    return _LegalResult(mask=mask4672, in_check=in_check)


def legal_action_mask(board: Array, en_passant: Array, castling_rights: Array) -> Array:
    """Scalar ``[4672]`` legal-action mask with no singleton batch axis."""
    return _legal_action_result(board, en_passant, castling_rights).mask

MAX_TERMINATION_STEPS = 512  # from AlphaZero paper

# index: a1: 0, a2: 1, ..., h8: 63 (sq = file*8 + rank)
INIT_BOARD = jnp.int8([4, 1, 0, 0, 0, 0, -1, -4, 2, 1, 0, 0, 0, 0, -1, -2, 3, 1, 0, 0, 0, 0, -1, -3, 5, 1, 0, 0, 0, 0, -1, -5, 6, 1, 0, 0, 0, 0, -1, -6, 3, 1, 0, 0, 0, 0, -1, -3, 2, 1, 0, 0, 0, 0, -1, -2, 4, 1, 0, 0, 0, 0, -1, -4])  # fmt: skip

# Same key/split/shape convention as pgx so hashes match bit-for-bit.
_zobrist_keys = jax.random.split(jax.random.PRNGKey(12345), 4)
ZOBRIST_BOARD = jax.random.randint(_zobrist_keys[0], shape=(64, 13, 2), minval=0, maxval=2**31 - 1, dtype=jnp.uint32)
ZOBRIST_SIDE = jax.random.randint(_zobrist_keys[1], shape=(2,), minval=0, maxval=2**31 - 1, dtype=jnp.uint32)
ZOBRIST_CASTLING = jax.random.randint(_zobrist_keys[2], shape=(4, 2), minval=0, maxval=2**31 - 1, dtype=jnp.uint32)
ZOBRIST_EN_PASSANT = jax.random.randint(_zobrist_keys[3], shape=(65, 2), minval=0, maxval=2**31 - 1, dtype=jnp.uint32)


def _zobrist_hash_raw(board: Array, color: Array, castling_rights: Array, en_passant: Array) -> Array:
    hash_ = lax.select(color == 0, ZOBRIST_SIDE, jnp.zeros_like(ZOBRIST_SIDE))
    one_hot_board = jax.nn.one_hot(board.astype(jnp.int32) + 6, 13, dtype=jnp.uint32)
    to_reduce = jnp.einsum("sp,sph->sh", one_hot_board, ZOBRIST_BOARD, preferred_element_type=jnp.uint32)
    hash_ ^= lax.reduce(to_reduce, jnp.uint32(0), lax.bitwise_xor, (0,))
    to_reduce = jnp.where(castling_rights.reshape(-1, 1), ZOBRIST_CASTLING, jnp.uint32(0))
    hash_ ^= lax.reduce(to_reduce, jnp.uint32(0), lax.bitwise_xor, (0,))
    safe_idx = en_passant.astype(jnp.int32) % 65
    one_hot_en_passant = jax.nn.one_hot(safe_idx, 65, dtype=jnp.uint32)
    hash_ ^= one_hot_en_passant @ ZOBRIST_EN_PASSANT
    return hash_


INIT_ZOBRIST_HASH = _zobrist_hash_raw(INIT_BOARD, jnp.int16(0), jnp.ones((2, 2), jnp.bool_), jnp.int16(-1))


class GameState(NamedTuple):
    color: Array = jnp.int16(0)  # w: 0, b: 1
    board: Array = INIT_BOARD  # (64,) int8, side-to-move relative: mine>0, opp<0
    castling_rights: Array = jnp.ones([2, 2], dtype=jnp.bool_)  # [mine, opp] x [queenside, kingside]
    en_passant: Array = jnp.int16(-1)
    halfmove_count: Array = jnp.int16(0)  # since last capture/pawn move (50-move rule uses >=100)
    fullmove_count: Array = jnp.int16(1)  # increases every black move
    hash_history: Array = jnp.zeros((MAX_TERMINATION_STEPS + 1, 2), dtype=jnp.uint32).at[0].set(INIT_ZOBRIST_HASH)
    board_history: Array = jnp.zeros((8, 64), dtype=jnp.int8).at[0, :].set(INIT_BOARD)
    step_count: Array = jnp.int32(0)


class Action(NamedTuple):
    """AlphaZero-style label = from_ * 73 + plane (4672 = 64 x 73).

    plane in [0,9): underpromotion (plane//3: 0 rook/1 bishop/2 knight; plane%3:
    0 straight/1 capture-right/2 capture-left), only valid from rank index 6.
    plane in [9,73): 56 queen-style slider planes (8 directions x 7 distances)
    followed by 8 fixed knight-offset planes. Both directions are pure
    arithmetic (no gather/table) so they vmap/jit efficiently.
    """

    from_: Array = jnp.int32(-1)
    to: Array = jnp.int32(-1)
    underpromotion: Array = jnp.int32(-1)  # 0: rook, 1: bishop, 2: knight

    @staticmethod
    def _from_label(label: Array) -> "Action":
        from_ = label // 73
        plane = label % 73
        r0, c0 = from_ % 8, from_ // 8

        underpromotion = lax.select(plane >= 9, -1, plane // 3)

        # underpromotion (planes 0..8): +1 straight, +9 capture-right, -7 capture-left
        up_off = jnp.where(plane % 3 == 0, 1, 0) + jnp.where(plane % 3 == 1, 9, 0) + jnp.where(plane % 3 == 2, -7, 0)
        up_to = from_ + up_off
        up_dr = 1
        up_dc = jnp.where(plane % 3 == 0, 0, jnp.where(plane % 3 == 1, 1, -1))

        # slider (planes 9..64): 8 directions x 7 distances
        p = plane - 9
        dir_code = p // 7
        offset = p % 7
        distance = jnp.where(dir_code % 2 == 0, 7 - offset, offset + 1)
        dr_sign = (
            jnp.where(dir_code == 1, 1, 0)
            + jnp.where(dir_code == 0, -1, 0)
            + jnp.where(dir_code == 5, 1, 0)
            + jnp.where(dir_code == 4, -1, 0)
            + jnp.where(dir_code == 6, 1, 0)
            + jnp.where(dir_code == 7, -1, 0)
        )
        dc_sign = (
            jnp.where(dir_code == 3, 1, 0)
            + jnp.where(dir_code == 2, -1, 0)
            + jnp.where(dir_code == 5, 1, 0)
            + jnp.where(dir_code == 4, -1, 0)
            + jnp.where(dir_code == 7, 1, 0)
            + jnp.where(dir_code == 6, -1, 0)
        )
        slider_to = (c0 + dc_sign * distance) * 8 + (r0 + dr_sign * distance)

        # knight (planes 65..72): 8 fixed offsets
        kn_dr = (
            jnp.where(plane == 65, -1, 0)
            + jnp.where(plane == 66, 1, 0)
            + jnp.where(plane == 67, -2, 0)
            + jnp.where(plane == 68, 2, 0)
            + jnp.where(plane == 69, -1, 0)
            + jnp.where(plane == 70, 1, 0)
            + jnp.where(plane == 71, -2, 0)
            + jnp.where(plane == 72, 2, 0)
        )
        kn_dc = (
            jnp.where((plane == 65) | (plane == 66), -2, 0)
            + jnp.where((plane == 67) | (plane == 68), -1, 0)
            + jnp.where((plane == 69) | (plane == 70), 2, 0)
            + jnp.where((plane == 71) | (plane == 72), 1, 0)
        )
        knight_to = (c0 + kn_dc) * 8 + (r0 + kn_dr)

        is_knight = plane >= 65
        is_underpromo = plane < 9
        to = jnp.where(is_underpromo, up_to, jnp.where(is_knight, knight_to, slider_to))

        dr = jnp.where(is_knight, kn_dr, jnp.where(is_underpromo, up_dr, dr_sign * distance))
        dc = jnp.where(is_knight, kn_dc, jnp.where(is_underpromo, up_dc, dc_sign * distance))
        new_r, new_c = r0 + dr, c0 + dc
        in_bounds = (new_r >= 0) & (new_r < 8) & (new_c >= 0) & (new_c < 8)
        valid_underpromo = r0 == 6
        is_valid = jnp.where(is_underpromo, in_bounds & valid_underpromo, in_bounds)
        to = jnp.where(is_valid, to, -1)

        return Action(from_=from_, to=to, underpromotion=underpromotion)

    def _to_label(self) -> Array:
        r0, c0 = self.from_ % 8, self.from_ // 8
        r1, c1 = self.to % 8, self.to // 8
        dr, dc = r1 - r0, c1 - c0
        abs_dr, abs_dc = jnp.abs(dr), jnp.abs(dc)
        distance = jnp.maximum(abs_dr, abs_dc)

        dr_neg, dr_pos = dr < 0, dr > 0
        dc_neg, dc_pos = dc < 0, dc > 0
        dr_zero, dc_zero = ~(dr_neg | dr_pos), ~(dc_neg | dc_pos)

        is_down, is_up = dr_neg & dc_zero, dr_pos & dc_zero
        is_left, is_right = dr_zero & dc_neg, dr_zero & dc_pos
        is_down_left, is_up_right = dr_neg & dc_neg, dr_pos & dc_pos
        is_up_left, is_down_right = dr_pos & dc_neg, dr_neg & dc_pos

        fwd, rev = distance - 1, 7 - distance
        slider_plane = (
            jnp.where(is_down, 9 + 0 * 7 + rev, 0)
            + jnp.where(is_up, 9 + 1 * 7 + fwd, 0)
            + jnp.where(is_left, 9 + 2 * 7 + rev, 0)
            + jnp.where(is_right, 9 + 3 * 7 + fwd, 0)
            + jnp.where(is_down_left, 9 + 4 * 7 + rev, 0)
            + jnp.where(is_up_right, 9 + 5 * 7 + fwd, 0)
            + jnp.where(is_up_left, 9 + 6 * 7 + rev, 0)
            + jnp.where(is_down_right, 9 + 7 * 7 + fwd, 0)
        )

        knight_plane = (
            jnp.where((dr == -1) & (dc == -2), 65, 0)
            + jnp.where((dr == +1) & (dc == -2), 66, 0)
            + jnp.where((dr == -2) & (dc == -1), 67, 0)
            + jnp.where((dr == +2) & (dc == -1), 68, 0)
            + jnp.where((dr == -1) & (dc == +2), 69, 0)
            + jnp.where((dr == +1) & (dc == +2), 70, 0)
            + jnp.where((dr == -2) & (dc == +1), 71, 0)
            + jnp.where((dr == +2) & (dc == +1), 72, 0)
        )

        is_knight = (abs_dr * abs_dc) == 2
        plane = jnp.where(is_knight, knight_plane, slider_plane)
        return self.from_ * 73 + plane


class _Outcome(NamedTuple):
    terminated: Array
    rewards: Array


class Game:
    def __init__(self, auto_terminate: bool = True):
        self.auto_terminate = auto_terminate

    def init(self) -> GameState:
        return GameState()

    def step(self, state: GameState, action: Array) -> GameState:
        state = _apply_move(state, Action._from_label(action))
        state = _flip(state)
        state = _update_history(state)
        return state._replace(step_count=state.step_count + 1)

    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        if color is None:
            color = state.color
        ones = jnp.ones((1, 8, 8), dtype=jnp.float32)

        def make(board_history_slice, hash_history_slice):
            board = jnp.rot90(board_history_slice.reshape((8, 8)), k=1)

            def piece_feat(p):
                return (board == p).astype(jnp.float32)

            my_pieces = jax.vmap(piece_feat)(jnp.arange(1, 7))
            opp_pieces = jax.vmap(piece_feat)(-jnp.arange(1, 7))

            h = hash_history_slice
            rep = (state.hash_history == h).all(axis=1).sum() - 1
            rep = lax.select((h == 0).all(), 0, rep)
            rep0 = ones * (rep == 0)
            rep1 = ones * (rep >= 1)
            return jnp.vstack([my_pieces, opp_pieces, rep0, rep1])

        board_features = jax.vmap(make)(state.board_history, state.hash_history[:8]).reshape(-1, 8, 8)
        return jnp.vstack(
            [
                board_features,
                color * ones,
                jnp.minimum(state.step_count / MAX_TERMINATION_STEPS, 1.0) * ones,
                state.castling_rights.flatten()[:, None, None] * ones,
                (state.halfmove_count.astype(jnp.float32) / 100.0) * ones,
            ]
        ).transpose((1, 2, 0))

    def legal_action_mask(self, state: GameState) -> Array:
        return legal_action_mask(
            state.board.astype(jnp.int32),
            state.en_passant.astype(jnp.int32),
            state.castling_rights,
        )

    def _terminated(
        self,
        state: GameState,
        has_legal_moves: Array,
    ) -> Array:
        terminated = ~has_legal_moves
        terminated |= state.halfmove_count >= 100
        terminated |= has_insufficient_pieces(state)
        rep = (state.hash_history == _zobrist_hash(state)).all(axis=1).sum() - 1
        terminated |= rep >= 2
        if self.auto_terminate:
            terminated |= MAX_TERMINATION_STEPS <= state.step_count
        return terminated

    def _rewards(
        self,
        state: GameState,
        has_legal_moves: Array,
        in_check: Array,
    ) -> Array:
        is_checkmate = (~has_legal_moves) & in_check
        return lax.select(
            is_checkmate,
            jnp.ones(2, dtype=jnp.float32).at[state.color].set(-1),
            jnp.zeros(2, dtype=jnp.float32),
        )

    def _outcome(
        self,
        state: GameState,
        has_legal_moves: Array,
        in_check: Array,
    ) -> _Outcome:
        return _Outcome(
            terminated=self._terminated(state, has_legal_moves),
            rewards=self._rewards(state, has_legal_moves, in_check),
        )

    def is_terminal(self, state: GameState, legal_action_mask: Array) -> Array:
        return self._terminated(state, legal_action_mask.any())

    def rewards(
        self,
        state: GameState,
        legal_action_mask: Array,
        in_check: Optional[Array] = None,
    ) -> Array:
        if in_check is None:
            in_check = _is_in_check(state.board.astype(jnp.int32))
        return self._rewards(
            state,
            legal_action_mask.any(),
            in_check,
        )


# -----------------------------------------------------------------------
# Gather-free board access helpers
# -----------------------------------------------------------------------


def _pieces_at(board: Array, idx: Array) -> Array:
    """board[idx] via one-hot matmul (no gather); idx == -1 reads as EMPTY.

    Returns int32 (not `board.dtype`): the `jnp.int32(EMPTY)` sentinel below
    promotes the result via `jnp.where`, and `_apply_move`'s `lax.select`
    calls on the resulting `piece` value rely on that int32 promotion to
    stay dtype-compatible with piece-code constants.
    """
    idx = jnp.asarray(idx, jnp.int32)
    valid = idx >= 0
    safe = jnp.where(valid, idx, 0)
    pieces = jax.nn.one_hot(safe, 64, dtype=board.dtype) @ board
    return jnp.where(valid, pieces, jnp.int32(EMPTY))


def _set_pieces(board: Array, idx: Array, val: Array) -> Array:
    """board.at[idx].set(val) via one-hot mask (no scatter); idx/val may be scalar or 1-D."""
    idx = jnp.atleast_1d(idx).astype(jnp.int32)
    val = jnp.atleast_1d(val).astype(board.dtype)
    mask = jax.nn.one_hot(idx, 64, dtype=board.dtype)
    return board * (1 - mask.max(axis=0)) + (mask.T @ val)


# -----------------------------------------------------------------------
# Move application, flipping, and history/hash bookkeeping
# -----------------------------------------------------------------------


def _flip_pos(x: Array) -> Array:  # e.g., 37 <-> 34, -1 <-> -1
    return lax.select(x == -1, x, (x // 8) * 8 + (7 - (x % 8)))


def _flip(state: GameState) -> GameState:
    return state._replace(
        board=-jnp.flip(state.board.reshape(8, 8), axis=1).flatten(),
        color=(state.color + 1) % 2,
        en_passant=_flip_pos(state.en_passant),
        castling_rights=state.castling_rights[::-1],
        board_history=-jnp.flip(state.board_history.reshape(-1, 8, 8), axis=-1).reshape(-1, 64),
    )


def _update_history(state: GameState) -> GameState:
    board_history = jnp.roll(state.board_history, 64)
    board_history = board_history.at[0].set(state.board)
    hash_hist = jnp.roll(state.hash_history, 2)
    hash_hist = hash_hist.at[0].set(_zobrist_hash(state))
    return state._replace(board_history=board_history, hash_history=hash_hist)


def _zobrist_hash(state: GameState) -> Array:
    return _zobrist_hash_raw(state.board, state.color, state.castling_rights, state.en_passant)


def has_insufficient_pieces(state: GameState) -> Array:
    # Same condition as OpenSpiel: KvK, Kv(K+minor), or same-color-bishops-only.
    num_pieces = (state.board != EMPTY).sum()
    num_pawn_rook_queen = ((jnp.abs(state.board) >= ROOK) | (jnp.abs(state.board) == PAWN)).sum() - 2  # two kings
    num_bishop = (jnp.abs(state.board) == BISHOP).sum()
    coords = jnp.arange(64).reshape((8, 8))
    black_coords = jnp.hstack((coords[::2, ::2].ravel(), coords[1::2, 1::2].ravel()))
    pieces_on_black = _pieces_at(state.board, black_coords)
    num_bishop_on_black = (jnp.abs(pieces_on_black) == BISHOP).sum()

    is_insufficient = num_pieces <= 2
    is_insufficient |= (num_pieces == 3) & (num_pawn_rook_queen == 0)
    is_bishop_all_on_black = num_bishop_on_black == num_bishop
    is_bishop_all_on_white = num_bishop_on_black == 0
    is_insufficient |= (num_pieces == num_bishop + 2) & (is_bishop_all_on_black | is_bishop_all_on_white)
    return is_insufficient


def _apply_move(state: GameState, a: Action) -> GameState:
    piece = _pieces_at(state.board, a.from_)

    is_en_passant = (state.en_passant >= 0) & (piece == PAWN) & (state.en_passant == a.to)
    removed_pawn_pos = a.to - 1
    state = state._replace(
        board=lax.select(
            is_en_passant,
            _set_pieces(state.board, jnp.array([removed_pawn_pos], jnp.int32), jnp.array([EMPTY], jnp.int32)),
            state.board,
        )
    )
    is_double_push = (piece == PAWN) & (jnp.abs(a.to - a.from_) == 2)
    state = state._replace(en_passant=jnp.int16(lax.select(is_double_push, (a.to + a.from_) // 2, -1)))

    captured = (_pieces_at(state.board, a.to) < 0) | is_en_passant
    state = state._replace(
        halfmove_count=lax.select(
            captured | (piece == PAWN), jnp.zeros_like(state.halfmove_count), state.halfmove_count + 1
        ),
        fullmove_count=state.fullmove_count + jnp.int16(state.color == 1),
    )

    board = state.board
    is_queen_side_castling = (piece == KING) & (a.from_ == 32) & (a.to == 16)
    board_q = _set_pieces(_set_pieces(board, 0, EMPTY), 24, ROOK)
    board = lax.select(is_queen_side_castling, board_q, board)

    is_king_side_castling = (piece == KING) & (a.from_ == 32) & (a.to == 48)
    board_k = _set_pieces(_set_pieces(board, 56, EMPTY), 40, ROOK)
    board = lax.select(is_king_side_castling, board_k, board)
    state = state._replace(board=board)

    cond = jnp.bool_(
        [[(a.from_ != 32) & (a.from_ != 0), (a.from_ != 32) & (a.from_ != 56)], [a.to != 7, a.to != 63]]
    )
    state = state._replace(castling_rights=state.castling_rights & cond)

    piece = lax.select((piece == PAWN) & (a.from_ % 8 == 6) & (a.underpromotion < 0), QUEEN, piece)
    is_underpromotion = a.underpromotion >= 0
    promoted_piece = piece
    promoted_piece = jnp.where(a.underpromotion == 0, ROOK, promoted_piece)
    promoted_piece = jnp.where(a.underpromotion == 1, BISHOP, promoted_piece)
    promoted_piece = jnp.where(a.underpromotion == 2, KNIGHT, promoted_piece)
    piece = jnp.where(is_underpromotion, promoted_piece, piece)

    board = _set_pieces(board, a.from_, EMPTY)
    board = _set_pieces(board, a.to, piece)
    return state._replace(board=board)



def _field(factory):
    return dataclasses.field(default_factory=factory)



@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class State:
    current_player: Array = _field(lambda: jnp.int32(0))
    rewards: Array = _field(lambda: jnp.float32([0.0, 0.0]))
    terminated: Array = _field(lambda: jnp.bool_(False))
    truncated: Array = _field(lambda: jnp.bool_(False))
    legal_action_mask: Array = _field(lambda: jnp.ones(4672, dtype=jnp.bool_))
    legal_action_bitmask: Array = _field(lambda: full_legal_action_bitmask())
    observation: Array = _field(lambda: jnp.zeros((8, 8, 119), dtype=jnp.float32))
    _step_count: Array = _field(lambda: jnp.int32(0))
    _player_order: Array = _field(lambda: jnp.int32([0, 1]))  # column 0 is always Player 1
    _x: GameState = _field(GameState)

    def replace(self, **kwargs) -> "State":
        return dataclasses.replace(self, **kwargs)

    @property
    def env_id(self) -> str:
        return "chess"





# -----------------------------------------------------------------------
# Public legal-move entry points
# -----------------------------------------------------------------------


def legal_action_bitmask(
    board: Array,
    en_passant: Array,
    castling_rights: Array,
) -> Array:
    """Packed scalar legal-action mask."""
    return pack_mask(legal_action_mask(board, en_passant, castling_rights))


def legal_action_mask_batch(
    board: Array,
    en_passant: Array,
    castling_rights: Array,
) -> Array:
    """Batch convenience wrapper; equivalent to ``jax.vmap(legal_action_mask)``."""
    return jax.vmap(legal_action_mask)(board, en_passant, castling_rights)


def legal_action_bitmask_batch(
    board: Array,
    en_passant: Array,
    castling_rights: Array,
) -> Array:
    """Packed batch convenience wrapper using the same scalar implementation."""
    return jax.vmap(legal_action_bitmask)(board, en_passant, castling_rights)

INITIAL_LEGAL_ACTIONS = jnp.int32(
    [
        89,
        90,
        652,
        656,
        673,
        674,
        1257,
        1258,
        1841,
        1842,
        2425,
        2426,
        3009,
        3010,
        3572,
        3576,
        3593,
        3594,
        4177,
        4178,
    ]
)
INITIAL_LEGAL_ACTION_MASK = (
    jnp.zeros((NUM_ACTIONS,), dtype=jnp.bool_)
    .at[INITIAL_LEGAL_ACTIONS]
    .set(True)
)
INITIAL_LEGAL_ACTION_BITMASK = pack_mask(INITIAL_LEGAL_ACTION_MASK)
FULL_LEGAL_ACTION_BITMASK = full_legal_action_bitmask()


class Chess:
    """Standalone scalar-first chess env; batch externally with ``jax.vmap``."""

    def __init__(
        self,
        *,
        auto_terminate: bool = True,
        use_bitmask: bool = False,
        return_observation: bool = True,
    ):
        if not isinstance(use_bitmask, bool):
            raise TypeError("use_bitmask must be a Python bool so it is static under jit.")
        if not isinstance(return_observation, bool):
            raise TypeError("return_observation must be a Python bool so it is static under jit.")
        self._game = Game(auto_terminate=auto_terminate)
        self.use_bitmask = use_bitmask
        self.return_observation = return_observation

    def observe(self, state: State, player_id: Optional[Array] = None) -> Array:
        if player_id is None:
            player_id = state.current_player

        def observe_one(x: GameState, current_player: Array, pid: Array) -> Array:
            current_view = current_player == pid
            color = lax.select(current_view, x.color, 1 - x.color)
            x_view = lax.cond(current_view, lambda: x, lambda: _flip(x))
            return self._game.observe(x_view, color)

        if state.current_player.ndim == 0:
            return lax.stop_gradient(observe_one(state._x, state.current_player, player_id))
        if jnp.ndim(player_id) == 0:
            player_id = jnp.broadcast_to(player_id, state.current_player.shape)
        return lax.stop_gradient(jax.vmap(observe_one)(state._x, state.current_player, player_id))

    def _check_legality(self, state: State, action: Array) -> Array:
        if self.use_bitmask:
            return bitmask_has_action(state.legal_action_bitmask, action)
        mask = state.legal_action_mask.astype(jnp.int32)
        return jnp.dot(jax.nn.one_hot(action, mask.shape[0], dtype=jnp.int32), mask).astype(jnp.bool_)

    def _step_with_illegal_action(self, state: State, loser: Array) -> State:
        rewards = jnp.where(jnp.arange(2) == loser, -1.0, 1.0).astype(jnp.float32)
        return state.replace(rewards=rewards, terminated=jnp.bool_(True))

    def _with_full_legal_actions(
        self,
        state: State,
        when: Optional[Array] = None,
    ) -> State:
        """Store the terminal-state full mask, optionally under a predicate."""
        if self.use_bitmask:
            bitmask = FULL_LEGAL_ACTION_BITMASK
            if when is not None:
                bitmask = lax.select(
                    when, bitmask, state.legal_action_bitmask
                )
            return state.replace(
                legal_action_mask=None,
                legal_action_bitmask=bitmask,
            )

        mask = jnp.ones_like(state.legal_action_mask)
        if when is not None:
            mask = lax.select(when, mask, state.legal_action_mask)
        return state.replace(
            legal_action_mask=mask,
            legal_action_bitmask=None,
        )

    def _with_legal_actions(self, state: State, mask: Array) -> State:
        """Store a computed legal mask in the configured representation."""
        return state.replace(
            legal_action_mask=None if self.use_bitmask else mask,
            legal_action_bitmask=pack_mask(mask) if self.use_bitmask else None,
        )

    def _terminal_noop(self, state: State) -> State:
        state = state.replace(rewards=jnp.zeros_like(state.rewards))
        return self._with_full_legal_actions(state)

    @property
    def version(self) -> str:
        return "v5-single-source"

    @property
    def num_players(self) -> int:
        return 2

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    def init(self, key: Optional[Array] = None) -> State:
        del key
        x = self._game.init()
        player_order = jnp.int32([0, 1])
        return State(
            current_player=player_order[x.color],
            legal_action_mask=None if self.use_bitmask else INITIAL_LEGAL_ACTION_MASK,
            legal_action_bitmask=INITIAL_LEGAL_ACTION_BITMASK if self.use_bitmask else None,
            observation=self._game.observe(x, x.color) if self.return_observation else None,
            _player_order=player_order,
            _x=x,
        )

    def step(
        self,
        state: State,
        action: Array,
        key: Optional[Array] = None,
    ) -> State:
        del key
        is_illegal = ~self._check_legality(state, action)
        current_player = state.current_player

        advanced = self._advance(
            state.replace(_step_count=state._step_count + 1),
            action,
        )
        zeroed = state.replace(rewards=jnp.zeros_like(state.rewards))
        state_out = lax.cond(
            state.terminated | state.truncated,
            lambda: zeroed,
            lambda: advanced,
        )
        state_out = lax.cond(
            is_illegal,
            lambda: self._step_with_illegal_action(state_out, current_player),
            lambda: state_out,
        )
        state_out = lax.cond(
            state_out.terminated,
            lambda: self._with_full_legal_actions(state_out),
            lambda: state_out,
        )
        return state_out

    def step_batch(
        self,
        state: State,
        action: Array,
    ) -> State:
        """Convenience only: native ``jax.vmap(step)``, not a batch fast path."""
        return jax.vmap(self.step)(state, action)

    def _advance(
        self,
        state: State,
        action: Array,
    ) -> State:
        x = state._x
        a = Action._from_label(action)
        x = _apply_move(x, a)
        x = _flip(x)
        x = _update_history(x)
        x = x._replace(step_count=x.step_count + 1)

        legal = _legal_action_result(
            x.board.astype(jnp.int32),
            x.en_passant.astype(jnp.int32),
            x.castling_rights,
        )
        mask = legal.mask
        bitmask = pack_mask(mask) if self.use_bitmask else None
        outcome = self._game._outcome(x, mask.any(), legal.in_check)
        return state.replace(
            _x=x,
            legal_action_mask=None if self.use_bitmask else mask,
            legal_action_bitmask=bitmask if self.use_bitmask else None,
            observation=self._game.observe(x, x.color) if self.return_observation else None,
            terminated=outcome.terminated,
            rewards=outcome.rewards,
            current_player=x.color.astype(jnp.int32),
        )

    def _replay_prefix(
        self,
        x: GameState,
        actions: Array,
        replay_depth: Optional[Array],
    ) -> tuple[GameState, Array]:
        """Apply the active, non-padding part of a fixed-capacity path."""
        capacity = actions.shape[0]
        if replay_depth is None:
            replay_depth = jnp.asarray(capacity, dtype=jnp.int32)
        else:
            replay_depth = jnp.asarray(replay_depth, dtype=jnp.int32)
        replay_depth = jnp.clip(replay_depth, 0, capacity)

        def body(carry):
            i, x = carry
            action = actions[i]
            stepped = self._game.step(x, jnp.maximum(action, 0))
            x = jax.tree.map(
                lambda new, old: lax.select(action >= 0, new, old),
                stepped,
                x,
            )
            return i + 1, x

        if capacity:
            _, x = lax.while_loop(
                lambda carry: carry[0] < replay_depth,
                body,
                (jnp.asarray(0, dtype=jnp.int32), x),
            )
        active = jnp.arange(capacity, dtype=jnp.int32) < replay_depth
        steps_taken = jnp.sum((active & (actions >= 0)).astype(jnp.int32))
        return x, steps_taken

    def _reached_replay_state(
        self,
        state: State,
        x: GameState,
        steps_taken: Array,
        terminated: Array,
    ) -> State:
        """Rebuild replay metadata, leaving legal-action fields unchanged."""
        return state.replace(
            _x=x,
            observation=(
                self._game.observe(x, x.color)
                if self.return_observation
                else None
            ),
            terminated=terminated,
            current_player=x.color.astype(jnp.int32),
            _step_count=state._step_count + steps_taken,
        )

    def replay(
        self,
        state: State,
        actions: Array,
        final_action: Array,
        already_terminal: Optional[Array] = None,
        final_action_is_legal: bool = False,
        replay_depth: Optional[Array] = None,
    ) -> State:
        """Fast-forward known-good actions, then ``step`` the final action.

        Scalar-first like ``step``; batch externally with ``jax.vmap``. The
        ``actions`` prefix runs only ``Game.step`` (mutate + flip + history)
        per entry -- no legal-move generation, no termination checks -- so a
        depth-``D`` replay costs ``D`` cheap steps, one legality evaluation
        for the reached position, and one full ``step``. Because the final
        action goes through the real ``step``, its semantics (legality check
        with the illegal-loser branch, terminal no-op guard, forced full
        mask) hold by construction, and the result is bit-identical to
        iterating ``step`` over ``actions + [final_action]``.

        Args:
          state: the starting :class:`State` (e.g. an MCTS root).
          actions: ``int32[D]`` action labels; each must be legal in the
            position it is applied to and must not be applied to a terminal
            position -- neither is checked. Entries ``< 0`` are padding
            no-ops, so rows with different path lengths can share one ``D``.
            ``D`` may be zero.
          final_action: the action for the concluding ``step``. May be
            illegal (handled exactly as ``step`` handles it).
          already_terminal: optional bool. Set when the caller knows the
            position reached by ``actions`` is terminal (an MCTS path
            truncated at a terminal node): the reconstructed position is
            marked terminated with a full mask -- the invariant ``step``
            maintains for the states it returns -- so the concluding
            ``step`` reproduces the terminal no-op (zeroed rewards).
          final_action_is_legal: static Python bool. When true, the caller
            proves the final action legal for every nonterminal row, allowing
            replay to skip legality generation at the reached position and
            apply the final move directly. Terminal rows still take the exact
            zero-reward no-op path. False preserves the general illegal-action
            semantics above.
          replay_depth: optional scalar number of leading entries in
            ``actions`` that may contain real actions. When supplied, replay
            uses a dynamic ``while_loop`` bounded by this value, so callers
            can keep one fixed-capacity action array without executing its
            unused suffix. Entries before the bound may still be ``-1`` for
            ragged batches. Defaults to the static action-array width.
        """
        if not isinstance(final_action_is_legal, bool):
            raise TypeError("final_action_is_legal must be a static Python bool")
        x, steps_taken = self._replay_prefix(
            state._x,
            actions,
            replay_depth,
        )

        if already_terminal is None:
            already_terminal = jnp.bool_(False)
        if final_action_is_legal:
            reached = self._reached_replay_state(
                state,
                x,
                steps_taken,
                already_terminal,
            )

            advanced = self._advance(
                reached.replace(_step_count=reached._step_count + 1),
                final_action,
            )
            advanced = self._with_full_legal_actions(
                advanced,
                when=advanced.terminated,
            )
            zeroed = self._terminal_noop(reached)
            return jax.tree.map(
                lambda terminal, normal: lax.select(
                    already_terminal, terminal, normal),
                zeroed, advanced)

        mask = legal_action_mask(
            x.board.astype(jnp.int32),
            x.en_passant.astype(jnp.int32),
            x.castling_rights,
        )
        reached = self._reached_replay_state(
            state,
            x,
            steps_taken,
            already_terminal,
        )
        reached = self._with_legal_actions(reached, mask)
        # Terminal states are stored with a full mask (see ``step``'s guard);
        # reproduce that so the concluding step's legality check no-ops.
        reached = self._with_full_legal_actions(
            reached,
            when=already_terminal,
        )
        return self.step(reached, final_action)

    def replay_batch(
        self,
        state: State,
        actions: Array,
        final_action: Array,
        already_terminal: Optional[Array] = None,
        final_action_is_legal: bool = False,
        replay_depth: Optional[Array] = None,
    ) -> State:
        """Convenience only: native ``jax.vmap(replay)``, not a batch fast path."""
        if already_terminal is None:
            return jax.vmap(lambda s, a, f: self.replay(
                s, a, f, final_action_is_legal=final_action_is_legal,
                replay_depth=replay_depth))(
                state, actions, final_action
            )
        return jax.vmap(lambda s, a, f, t: self.replay(
            s, a, f, t, final_action_is_legal=final_action_is_legal,
            replay_depth=replay_depth))(
                state, actions, final_action, already_terminal)

    @property
    def id(self) -> str:
        return "chess"


__all__ = [
    "Action",
    "Chess",
    "Game",
    "GameState",
    "State",
    "INITIAL_LEGAL_ACTION_BITMASK",
    "INITIAL_LEGAL_ACTION_MASK",
    "legal_action_mask",
    "legal_action_mask_batch",
    "legal_action_bitmask",
    "legal_action_bitmask_batch",
    "pack_mask",
    "unpack_bitmask",
    "bitmask_has_action",
]
