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
"""Single-file port of `pgx.connect_four`.

Drop-in replacement for `pgx.connect_four` (same rules, same observations,
same rewards) that only depends on JAX. The game logic is copied verbatim
from `pgx._src.games.connect_four`; only the `pgx.core` machinery (State
dataclass base and the `Env.step` wrapper) is inlined below, mirroring
`pgx.core.Env.step` semantics exactly (illegal actions, terminal handling,
terminal mask).
"""

import dataclasses
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import Array, lax

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


class GameState(NamedTuple):
    color: Array = jnp.int32(0)
    # 6x7 board
    # [[ 0,  1,  2,  3,  4,  5,  6],
    #  [ 7,  8,  9, 10, 11, 12, 13],
    #  [14, 15, 16, 17, 18, 19, 20],
    #  [21, 22, 23, 24, 25, 26, 27],
    #  [28, 29, 30, 31, 32, 33, 34],
    #  [35, 36, 37, 38, 39, 40, 41]]
    board: Array = -jnp.ones(42, jnp.int32)  # -1 (empty), 0, 1
    winner: Array = jnp.int32(-1)


class Game:
    def init(self) -> GameState:
        return GameState()

    def step(self, state: GameState, action: Array) -> GameState:
        board2d = state.board.reshape(6, 7)
        # Read the target column via a one-hot matmul instead of a dynamic gather.
        col = board2d @ jax.nn.one_hot(action, 7, dtype=board2d.dtype)
        num_filled = (col >= 0).sum()
        # Place the stone via a one-hot add instead of a scatter (empty=-1 -> color).
        flat_idx = (5 - num_filled) * 7 + action
        board = state.board + jax.nn.one_hot(flat_idx, 42, dtype=state.board.dtype) * (state.color + 1)
        # Win check via a precomputed [69, 42] line-mask matmul instead of a gather.
        owned = (board == state.color).astype(jnp.float32)
        won = (WIN_MASKS @ owned == 4).any()
        winner = jax.lax.select(won, state.color, -1)
        return state._replace(  # type: ignore
            color=1 - state.color,
            board=board,
            winner=winner,
        )

    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        board = state.board.reshape(6, 7)
        my_board = board == color
        opp_board = board == (1 - color)
        ones = jnp.ones_like(my_board)
        return jnp.stack([my_board, opp_board, ones], 2, dtype=jnp.bool_)

    def legal_action_mask(self, state: GameState) -> Array:
        board2d = state.board.reshape(6, 7)
        return (board2d >= 0).sum(axis=0) < 6

    def is_terminal(self, state: GameState) -> Array:
        board2d = state.board.reshape(6, 7)
        return (state.winner >= 0) | jnp.all((board2d >= 0).sum(axis=0) == 6)

    def rewards(self, state: GameState) -> Array:
        return jax.lax.select(
            state.winner >= 0,
            jnp.float32([-1, -1]).at[state.winner].set(1),
            jnp.zeros(2, jnp.float32),
        )


def _make_win_cache():
    idx = []
    # Vertical
    for i in range(3):
        for j in range(7):
            a = i * 7 + j
            idx.append([a, a + 7, a + 14, a + 21])
    # Horizontal
    for i in range(6):
        for j in range(4):
            a = i * 7 + j
            idx.append([a, a + 1, a + 2, a + 3])

    # Diagonal
    for i in range(3):
        for j in range(4):
            a = i * 7 + j
            idx.append([a, a + 8, a + 16, a + 24])
    for i in range(3):
        for j in range(3, 7):
            a = i * 7 + j
            idx.append([a, a + 6, a + 12, a + 18])
    return jnp.int32(idx)


IDX = _make_win_cache()
# [69, 42] line-membership matrix: row i has 1s at the 4 cells of win line i.
WIN_MASKS = jax.nn.one_hot(IDX, 42, dtype=jnp.float32).sum(axis=1)


# -----------------------------------------------------------------------
# pgx-compatible env wrapper (self-contained; mirrors pgx.core.Env.step)
# -----------------------------------------------------------------------


def _field(factory):
    return dataclasses.field(default_factory=factory)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class State:
    current_player: Array = _field(lambda: jnp.int32(0))
    observation: Array = _field(lambda: jnp.zeros((6, 7, 3), dtype=jnp.bool_))
    rewards: Array = _field(lambda: jnp.float32([0.0, 0.0]))
    terminated: Array = _field(lambda: FALSE)
    truncated: Array = _field(lambda: FALSE)
    legal_action_mask: Array = _field(lambda: jnp.ones(7, dtype=jnp.bool_))
    _step_count: Array = _field(lambda: jnp.int32(0))
    _x: GameState = _field(GameState)

    # pgx State API compatibility
    def replace(self, **kwargs) -> "State":
        return dataclasses.replace(self, **kwargs)

    @property
    def env_id(self) -> str:
        return "connect_four"


class ConnectFour:
    def __init__(self):
        self._game = Game()

    def init(self, key: Optional[Array] = None) -> State:
        del key
        current_player = jnp.int32(0)  # First player always starts
        return State(current_player=current_player, _x=self._game.init())  # type:ignore

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
        my_color = jax.lax.select(player_id == state.current_player, curr_color, 1 - curr_color)
        return lax.stop_gradient(self._game.observe(state._x, my_color))

    def _step(self, state: State, action: Array) -> State:
        x = self._game.step(state._x, action)
        state = state.replace(  # type: ignore
            current_player=1 - state.current_player,
            _x=x,
        )
        legal_action_mask = self._game.legal_action_mask(state._x)
        terminated = self._game.is_terminal(state._x)
        rewards = self._game.rewards(state._x)
        should_flip = state.current_player != state._x.color
        rewards = jax.lax.select(should_flip, jnp.flip(rewards), rewards)
        rewards = jax.lax.select(terminated, rewards, jnp.zeros(2, jnp.float32))
        return state.replace(  # type: ignore
            legal_action_mask=legal_action_mask,
            rewards=rewards,
            terminated=terminated,
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
        return "connect_four"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2

    @property
    def num_actions(self) -> int:
        return 7
