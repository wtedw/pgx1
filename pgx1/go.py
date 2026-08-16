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
"""Speed-optimized Go environment in a single, portable file.

Drop-in replacement for `pgx.go` (same rules, same observations, same
rewards) that only depends on JAX. Optimizations over upstream pgx:

- Chain statistics (pseudo-liberty sums) are computed once per step and
  carried in the state; upstream recomputed the O(N^4) `_count` twice per
  step (once in `_apply_action`, once in `legal_action_mask`).
- The per-chain aggregation is a single one_hot einsum (dot_general)
  instead of an N^2-way vmapped full-board reduction.
- Neighbor sums use pad+shift on the (N, N) board — no adjacency gathers.
- Terminal territory scoring floods a zeroed board on non-terminal steps,
  so the while_loop converges in one iteration mid-game; upstream ran the
  full flood fill every step.
- Zobrist hashing uses a one_hot einsum instead of a table gather.
- All small scatters (`.at[ko].set`, `.at[action].set`, `.at[color].add`,
  `.at[step].set`) and `jnp.nonzero` are replaced with dense masked ops.
- `board_history` is int8 (values in {-1, 0, 1, 2}), 4x less roll traffic.
"""

import dataclasses
from typing import NamedTuple, Optional

import jax
from jax import Array, lax
from jax import numpy as jnp

# Same key/shape/dtype as pgx so hash histories match bit-for-bit.
ZOBRIST_BOARD = jax.random.randint(jax.random.PRNGKey(12345), (3, 19 * 19, 2), 0, 2**31 - 1, jnp.uint32)

# Komi giving fair-ish games per size; boards <= 8x8 use `fair komi - 0.5`
# (mirrors pgx.core.make registrations).
DEFAULT_KOMI = {3: 8.5, 4: 1.5, 5: 24.5, 6: 3.5, 7: 8.5, 8: 9.5}

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


class GameState(NamedTuple):
    step_count: Array = jnp.int32(0)
    # ids of representative stone (smallest idx + 1) in connected chains
    board: Array = jnp.zeros(19 * 19, dtype=jnp.int32)  # b > 0, w < 0, empty = 0
    board_history: Array = jnp.full((8, 19 * 19), 2, dtype=jnp.int8)  # for obs
    num_captured: Array = jnp.zeros(2, dtype=jnp.int32)  # (b, w)
    consecutive_pass_count: Array = jnp.int32(0)
    ko: Array = jnp.int32(-1)  # by SSK
    is_psk: Array = FALSE
    hash_history: Array = jnp.zeros((19 * 19 * 2, 2), dtype=jnp.uint32)
    # Chain stats of `board`, computed once per step and reused by both the
    # next `_apply_action` (same board) and `legal_action_mask`.
    # Per point: whether the chain occupying it has exactly one liberty, and
    # where that liberty is. Zero/False (and unused) at empty points.
    chain_in_atari: Array = jnp.zeros(19 * 19, dtype=jnp.bool_)
    chain_single_liberty: Array = jnp.zeros(19 * 19, dtype=jnp.int32)

    @property
    def color(self) -> Array:
        return self.step_count % 2


class Game:
    def __init__(
        self, size: int = 19, komi: float = 7.5, history_length: int = 8, max_termination_steps: Optional[int] = None
    ):
        self.size = size
        self.komi = komi
        self.history_length = history_length
        self.max_termination_steps = size * size * 2 if max_termination_steps is None else max_termination_steps

    def init(self) -> GameState:
        board = jnp.zeros(self.size**2, dtype=jnp.int32)
        in_atari, single_liberty = _chain_stats(board, self.size)
        return GameState(
            board=board,
            board_history=jnp.full((self.history_length, self.size**2), 2, dtype=jnp.int8),
            hash_history=jnp.zeros((self.max_termination_steps, 2), dtype=jnp.uint32),
            chain_in_atari=in_atari,
            chain_single_liberty=single_liberty,
        )

    def step(self, state: GameState, action: Array) -> GameState:
        size = self.size
        state = state._replace(ko=jnp.int32(-1))
        # update state
        state = lax.cond(
            (action < size * size),
            lambda: _apply_action(state, action, size),
            lambda: _apply_pass(state),
        )
        # update board history (int8; row 0 is the most recent board)
        board_history = jnp.roll(state.board_history, 1, axis=0)
        board_history = board_history.at[0].set(jnp.clip(state.board, -1, 1).astype(jnp.int8))
        # check PSK. The hash is written densely (no scatter) and reused
        # directly instead of being gathered back out of the history.
        hash_ = _compute_hash(state.board)
        step_ixs = jnp.arange(state.hash_history.shape[0])
        hash_history = jnp.where((step_ixs == state.step_count)[:, None], hash_[None, :], state.hash_history)
        not_passed = state.consecutive_pass_count == 0
        is_psk = not_passed & ((hash_ == hash_history).all(axis=-1).sum() > 1)
        # chain stats of the new board, shared by legal_action_mask now and
        # by _apply_action at the next step
        in_atari, single_liberty = _chain_stats(state.board, size)
        return state._replace(
            board_history=board_history,
            hash_history=hash_history,
            is_psk=is_psk,
            chain_in_atari=in_atari,
            chain_single_liberty=single_liberty,
            step_count=state.step_count + 1,
        )

    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        if color is None:
            color = state.color
        my_sign = (1 - 2 * color).astype(jnp.int8)
        # plane 2*i is history[i] == my color, plane 2*i+1 is history[i] == opp
        sign_pair = jnp.stack([my_sign, -my_sign])  # (2,)
        log = (state.board_history[:, None, :] == sign_pair[None, :, None]).reshape(
            2 * self.history_length, -1
        )  # (2H, N2)
        color_plane = jnp.full_like(log[:1], color)  # b = 0, w = 1
        return jnp.vstack([log, color_plane]).transpose().reshape((self.size, self.size, -1))

    def legal_action_mask(self, state: GameState) -> Array:
        # some logic is inspired by OpenSpiel's Go implementation
        size = self.size
        board = state.board
        is_empty = board == 0
        my_sign = 1 - 2 * state.color
        has_liberty = (board * my_sign > 0) & ~state.chain_in_atari
        can_kill = (board * -my_sign > 0) & state.chain_in_atari
        # a point is playable if empty and any neighbor is empty, killable,
        # or a friendly chain with a spare liberty (neighbor-any via shifts)
        ok = (is_empty | can_kill | has_liberty).astype(jnp.int32)
        neighbor_ok = _neighbor_sum(ok.reshape(size, size)).reshape(-1) > 0
        mask = is_empty & neighbor_ok
        mask &= jnp.arange(size * size) != state.ko  # no-op when ko == -1
        return jnp.append(mask, TRUE)  # pass is always legal

    def is_terminal(self, state: GameState) -> Array:
        two_consecutive_pass = state.consecutive_pass_count >= 2
        timeover = self.max_termination_steps <= state.step_count
        return two_consecutive_pass | state.is_psk | timeover

    def rewards(self, state: GameState) -> Array:
        terminated = self.is_terminal(state)
        # Zero the board on non-terminal steps so the scoring flood fill
        # converges immediately; the result is masked out below anyway.
        board = jnp.where(terminated, state.board, 0)
        scores = _count_scores(board, self.size)
        is_black_win = scores[0] - self.komi > scores[1]
        rewards = lax.select(is_black_win, jnp.float32([1, -1]), jnp.float32([-1, 1]))
        to_play = state.color
        psk_rewards = jnp.where(jnp.arange(2) == to_play, 1.0, -1.0).astype(jnp.float32)
        rewards = lax.select(state.is_psk, psk_rewards, rewards)
        rewards = lax.select(terminated, rewards, jnp.zeros(2, dtype=jnp.float32))
        return rewards


def _apply_pass(state: GameState) -> GameState:
    return state._replace(consecutive_pass_count=state.consecutive_pass_count + 1)


def _apply_action(state: GameState, action, size) -> GameState:
    state = state._replace(consecutive_pass_count=jnp.int32(0))
    my_sign = (1 - 2 * state.color).astype(jnp.int32)
    opp_sign = -my_sign
    board = state.board

    # neighbors of the played point; all reads at those 4 points go through
    # a tiny (4, N2) one_hot matmul instead of gathers
    adj_ixs = _adj_ixs(action, size)  # (4,), -1 if off-board
    on_board = adj_ixs != -1
    oh4 = adj_ixs[:, None] == jnp.arange(size * size)[None, :]  # (4, N2); off-board rows all False
    oh4i = oh4.astype(jnp.int32)

    # remove killed stones, using the chain stats carried from the previous
    # step (they describe exactly this pre-move board)
    adj_ids = oh4i @ board  # (4,)
    adj_in_atari = (oh4 & state.chain_in_atari[None, :]).any(axis=-1)
    adj_single_liberty = oh4i @ state.chain_single_liberty
    is_killed = on_board & (adj_ids * opp_sign > 0) & adj_in_atari & (adj_single_liberty == action)
    surrounded_stones = (board[:, None] == adj_ids[None, :]) & is_killed[None, :]  # (N2, 4)
    num_captured = jnp.count_nonzero(surrounded_stones)
    ko_ix = jnp.argmax(is_killed)  # first killed neighbor (0 if none; guarded below)
    ko_pos = jnp.sum(jnp.where(jnp.arange(4) == ko_ix, adj_ixs, 0))
    ko_may_occur = (~on_board | (adj_ids * opp_sign > 0)).all()
    board = jnp.where(surrounded_stones.any(axis=-1), 0, board)
    state = state._replace(
        num_captured=state.num_captured + jnp.where(jnp.arange(2) == state.color, num_captured, 0),
        ko=lax.select(ko_may_occur & (num_captured == 1), ko_pos, jnp.int32(-1)),
    )

    # set stone (dense, no scatter)
    new_id = (action + 1) * my_sign
    board = jnp.where(jnp.arange(size * size) == action, new_id, board)

    # merge adjacent chains
    tgt_ids = oh4i @ board  # neighbor ids after capture removal
    should_merge = on_board & (tgt_ids * my_sign > 0)
    smallest_id = jnp.min(jnp.where(should_merge, jnp.abs(tgt_ids), size * size + 1))
    smallest_id = jnp.minimum(action + 1, smallest_id) * my_sign
    mask = (board == new_id) | (should_merge[None, :] & (board[:, None] == tgt_ids[None, :])).any(axis=-1)
    board = jnp.where(mask, smallest_id, board)

    return state._replace(board=board)


def _neighbor_sum(x: Array) -> Array:
    """Sum of the 4 orthogonal neighbors at every point, via pad+shift.

    `x` is (N, N) or (N, N, C); off-board neighbors contribute 0.
    """
    pad = ((1, 1), (1, 1)) + ((0, 0),) * (x.ndim - 2)
    p = jnp.pad(x, pad)
    return p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]


def _chain_stats(board: Array, size: int):
    """Per-point atari status and single-liberty position of each chain.

    Uses the pseudo-liberty trick: a chain is in atari iff the sum of its
    pseudo-liberty indices squared equals the sum of squares times the
    count (all pseudo-liberties are the same point). Aggregation across a
    chain and broadcast back to its points are one_hot einsums, and the
    per-point neighbor sums are pad+shift — no gathers or scatters.
    """
    size2 = size * size
    is_empty = board == 0
    idx1 = jnp.arange(1, size2 + 1, dtype=jnp.int32)
    lib = jnp.where(is_empty, idx1, 0)
    # per point: (#empty neighbors, sum of their idx+1, sum of (idx+1)^2)
    vals = jnp.stack([is_empty.astype(jnp.int32), lib, lib * idx1], axis=-1)
    nvals = _neighbor_sum(vals.reshape(size, size, 3)).reshape(size2, 3)
    # aggregate per chain id: oh[p, i] == (point p belongs to chain i+1)
    oh = (jnp.abs(board)[:, None] == idx1[None, :]).astype(jnp.int32)  # (N2, N2)
    per_id = jnp.einsum("pi,pc->ic", oh, nvals)  # (N2, 3)
    num_pseudo, idx_sum, idx_squared_sum = per_id[:, 0], per_id[:, 1], per_id[:, 2]
    in_atari = (idx_sum**2 == idx_squared_sum * num_pseudo).astype(jnp.int32)
    single_liberty = idx_squared_sum // jnp.maximum(idx_sum, 1) - 1
    # broadcast per-chain values back to the points of each chain
    per_point = jnp.einsum("pi,ic->pc", oh, jnp.stack([in_atari, single_liberty], axis=-1))
    return per_point[:, 0] > 0, per_point[:, 1]


def _adj_ixs(xy, size):
    dx, dy = jnp.int32([-1, +1, 0, 0]), jnp.int32([0, 0, -1, +1])
    xs, ys = xy // size + dx, xy % size + dy
    on_board = (0 <= xs) & (xs < size) & (0 <= ys) & (ys < size)
    return jnp.where(on_board, xs * size + ys, -1)  # -1 if out of board


def _compute_hash(board: Array):
    """Zobrist hash via one_hot einsum; identical values to pgx's gather."""
    size2 = board.shape[-1]
    # {-1 -> 2, 0 -> 0, 1 -> 1} matches pgx's wrapped negative indexing
    one_hot_board = jax.nn.one_hot(jnp.clip(board, -1, 1) % 3, 3, dtype=jnp.uint32)  # (N2, 3)
    to_reduce = jnp.einsum(
        "pc,cph->ph", one_hot_board, ZOBRIST_BOARD[:, :size2, :], preferred_element_type=jnp.uint32
    )
    return lax.reduce(to_reduce, jnp.uint32(0), lax.bitwise_xor, (0,))


def _count_scores(board: Array, size):
    def calc_point(c):
        return _count_ji(board, c, size) + jnp.count_nonzero(board * c > 0)

    return jax.vmap(calc_point)(jnp.int32([1, -1]))


def _count_ji(board: Array, color, size: int):
    b = jnp.clip(board * color, -1, 1).reshape(size, size)  # my stone: 1, opp: -1

    def fill_opp(x):
        b, _ = x
        # true if empty and adjacent to opponent's stone (neighbor-any via shifts)
        mask = (b == 0) & (_neighbor_sum((b == -1).astype(jnp.int32)) > 0)
        return jnp.where(mask, -1, b), mask.any()

    b, _ = lax.while_loop(lambda x: x[1], fill_opp, (b, TRUE))
    return (b == 0).sum()


# -----------------------------------------------------------------------
# pgx-compatible env wrapper (self-contained; mirrors pgx.core.Env.step)
# -----------------------------------------------------------------------


def _field(factory):
    return dataclasses.field(default_factory=factory)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class State:
    current_player: Array = _field(lambda: jnp.int32(0))
    rewards: Array = _field(lambda: jnp.float32([0.0, 0.0]))
    terminated: Array = _field(lambda: FALSE)
    truncated: Array = _field(lambda: FALSE)
    legal_action_mask: Array = _field(lambda: jnp.ones(19 * 19 + 1, dtype=jnp.bool_))
    observation: Array = _field(lambda: jnp.zeros((19, 19, 17), dtype=jnp.bool_))
    _step_count: Array = _field(lambda: jnp.int32(0))
    _x: GameState = _field(GameState)

    # pgx State API compatibility
    def replace(self, **kwargs) -> "State":
        return dataclasses.replace(self, **kwargs)

    @property
    def env_id(self) -> str:
        size = int(self._x.board.shape[-1] ** 0.5)
        return f"go_{size}x{size}"


class Go:
    def __init__(
        self,
        *,
        size: int = 19,
        komi: Optional[float] = None,
        history_length: int = 8,
        max_terminal_steps: Optional[int] = None,
    ):
        assert isinstance(size, int)
        if komi is None:
            komi = DEFAULT_KOMI.get(size, 7.5)
        self._game = Game(
            size=size, komi=komi, history_length=history_length, max_termination_steps=max_terminal_steps
        )

    def init(self, key: Optional[Array] = None) -> State:
        del key
        x = self._game.init()
        size = self._game.size
        return State(
            current_player=jnp.int32(0),  # First player always starts
            legal_action_mask=jnp.ones(size * size + 1, dtype=jnp.bool_),
            observation=jnp.zeros((size, size, 2 * self._game.history_length + 1), dtype=jnp.bool_),
            _x=x,
        )

    def step(self, state: State, action: Array, key: Optional[Array] = None) -> State:
        del key
        is_illegal = ~self._check_legality(state, action)
        current_player = state.current_player

        # If already terminated/truncated, return the same state with zero rewards
        state = lax.cond(
            (state.terminated | state.truncated),
            lambda: state.replace(rewards=jnp.zeros_like(state.rewards)),
            lambda: self._step(state.replace(_step_count=state._step_count + 1), action),
        )

        # Taking an illegal action leads to immediate terminal with negative reward
        state = lax.cond(
            is_illegal,
            lambda: self._step_with_illegal_action(state, current_player),
            lambda: state,
        )

        # All legal_action_mask elements are True at terminal states
        state = lax.cond(
            state.terminated,
            lambda: state.replace(legal_action_mask=jnp.ones_like(state.legal_action_mask)),
            lambda: state,
        )
        return state

    def observe(self, state: State, player_id: Optional[Array] = None) -> Array:
        if player_id is None:
            player_id = state.current_player
        curr_color = state._x.color
        my_turn = lax.select(player_id == state.current_player, curr_color, 1 - curr_color)
        return lax.stop_gradient(self._game.observe(state._x, my_turn))

    def _step(self, state: State, action: Array) -> State:
        x = self._game.step(state._x, action)
        return state.replace(
            current_player=1 - state.current_player,
            legal_action_mask=self._game.legal_action_mask(x),
            rewards=self._game.rewards(x),
            terminated=self._game.is_terminal(x),
            _x=x,
        )

    def _check_legality(self, state: State, action: Array) -> Array:
        mask_i32 = state.legal_action_mask.astype(jnp.int32)
        one_hot_a = jax.nn.one_hot(action, mask_i32.shape[0], dtype=jnp.int32)
        return jnp.dot(one_hot_a, mask_i32).astype(jnp.bool_)

    def _step_with_illegal_action(self, state: State, loser: Array) -> State:
        rewards = jnp.where(jnp.arange(2) == loser, -1.0, 1.0).astype(jnp.float32)
        return state.replace(rewards=rewards, terminated=TRUE)

    @property
    def id(self) -> str:
        return f"go_{int(self._game.size)}x{int(self._game.size)}"

    @property
    def version(self) -> str:
        return "v1-opt"

    @property
    def num_players(self) -> int:
        return 2

    @property
    def num_actions(self) -> int:
        return self._game.size * self._game.size + 1
