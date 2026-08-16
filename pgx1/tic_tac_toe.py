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
"""Single-file port of `pgx.tic_tac_toe`.

Drop-in replacement for `pgx.tic_tac_toe` (same rules, same observations,
same rewards) that only depends on JAX. The game logic is copied verbatim
from `pgx._src.games.tic_tac_toe`; only the `pgx.core` machinery (State
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
    color: Array = jnp.int32(0)  # 0 = X, 1 = O
    # 0 1 2
    # 3 4 5
    # 6 7 8
    board: Array = -jnp.ones(9, jnp.int32)  # -1 (empty), 0, 1
    winner: Array = jnp.int32(-1)


class Game:
    def init(self) -> GameState:
        return GameState()

    def step(self, state: GameState, action: Array) -> GameState:
        board = state.board.at[action].set(state.color)
        idx = jnp.int32([[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]])  # type: ignore
        won = (board[idx] == state.color).all(axis=1).any()
        winner = jax.lax.select(won, state.color, -1)
        return state._replace(  # type: ignore
            board=state.board.at[action].set(state.color),
            color=(state.color + 1) % 2,
            winner=winner,
        )

    def observe(self, state: GameState, color: Optional[Array] = None) -> Array:
        if color is None:
            color = state.color

        # 1. Get the grid shape (3x3)
        grid = state.board.reshape((3, 3))

        # 2. Create the My/Opponent planes
        my_board = (grid == color)
        opp_board = (grid == (1 - color))

        # 3. Create the Color plane (All 0s for Player 0, All 1s for Player 1)
        # This tells the network "Which player am I?" (First or Second)
        color_plane = jnp.full((3, 3), color, dtype=jnp.bool_)

        # 4. Create the Ones plane (Always all 1s)
        # This helps CNNs handle borders and provides a constant bias
        ones_plane = jnp.ones((3, 3), dtype=jnp.bool_)

        # Stack them to get shape (3, 3, 4)
        return jnp.stack([my_board, opp_board, color_plane, ones_plane], -1)


    def legal_action_mask(self, state: GameState) -> Array:
        return state.board < 0

    def is_terminal(self, state: GameState) -> Array:
        return (state.winner >= 0) | jnp.all(state.board != -1)

    def rewards(self, state: GameState) -> Array:
        return jax.lax.select(
            state.winner >= 0,
            jnp.float32([-1, -1]).at[state.winner].set(1),
            jnp.zeros(2, jnp.float32),
        )


# -----------------------------------------------------------------------
# pgx-compatible env wrapper (self-contained; mirrors pgx.core.Env.step)
# -----------------------------------------------------------------------


def _field(factory):
    return dataclasses.field(default_factory=factory)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class State:
    current_player: Array = _field(lambda: jnp.int32(0))
    observation: Array = _field(lambda: jnp.zeros((3, 3, 4), dtype=jnp.bool_))
    rewards: Array = _field(lambda: jnp.float32([0.0, 0.0]))
    terminated: Array = _field(lambda: FALSE)
    truncated: Array = _field(lambda: FALSE)
    legal_action_mask: Array = _field(lambda: jnp.ones(9, dtype=jnp.bool_))
    _step_count: Array = _field(lambda: jnp.int32(0))
    _x: GameState = _field(GameState)

    # pgx State API compatibility
    def replace(self, **kwargs) -> "State":
        return dataclasses.replace(self, **kwargs)

    @property
    def env_id(self) -> str:
        return "tic_tac_toe"


class TicTacToe:
    def __init__(self):
        self._game = Game()

    def init(self, key: Optional[Array] = None) -> State:
        del key
        current_player = jnp.int32(0)
        x = self._game.init()
        return State(current_player=current_player, _x=x)  # type:ignore

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
        state = state.replace(  # type: ignore
            current_player=(state.current_player + 1) % 2, _x=self._game.step(state._x, action)
        )
        legal_action_mask = self._game.legal_action_mask(state._x)
        terminated = self._game.is_terminal(state._x)
        rewards = self._game.rewards(state._x)
        should_flip = state.current_player != state._x.color
        rewards = jax.lax.select(should_flip, jnp.flip(rewards), rewards)
        rewards = jax.lax.select(terminated, rewards, jnp.zeros(2, jnp.float32))
        return state.replace(  # type: ignore
            legal_action_mask=legal_action_mask, rewards=rewards, terminated=terminated
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
        return "tic_tac_toe"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2

    @property
    def num_actions(self) -> int:
        return 9
