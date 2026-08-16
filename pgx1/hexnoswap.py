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
"""Single-file port of `pgx.hexnoswap` (Hex without the swap rule).

Drop-in replacement for `pgx.hexnoswap` (same rules, same observations,
same rewards) that only depends on JAX. The game logic is copied verbatim
from pgx; only the `pgx.core` machinery (State dataclass base and the
`Env.step` wrapper) is inlined below, mirroring `pgx.core.Env.step`
semantics exactly (illegal actions, terminal handling, terminal mask).
"""

import dataclasses
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array, lax

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


def _field(factory):
    return dataclasses.field(default_factory=factory)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class State:
    current_player: Array = _field(lambda: jnp.int32(0))
    observation: Array = None  # Will be set based on size
    rewards: Array = _field(lambda: jnp.float32([0.0, 0.0]))
    terminated: Array = _field(lambda: FALSE)
    truncated: Array = _field(lambda: FALSE)
    legal_action_mask: Array = None  # Will be set based on size
    _step_count: Array = _field(lambda: jnp.int32(0))
    # --- Hex specific ---
    _size: Array = None  # Will be set based on size
    _turn: Array = _field(lambda: jnp.int32(1))
    _board: Array = None  # Will be set based on size

    # pgx State API compatibility
    def replace(self, **kwargs) -> "State":
        return dataclasses.replace(self, **kwargs)

    @property
    def env_id(self) -> str:
        return f"hexnoswap_{self._size}x{self._size}"


class Hexnoswap:
    def __init__(self, *, size: int = 11):
        assert isinstance(size, int)
        assert 3 <= size <= 17, "Hex board size must be between 3 and 17"
        self.size = size

    def init(self, key: Optional[Array] = None) -> State:
        del key
        current_player = jnp.int32(0)  # First player always starts
        size = self.size
        return State(
            current_player=current_player,
            observation=jnp.zeros((size, size, 3), dtype=jnp.bool_),  # Removed swap channel
            legal_action_mask=jnp.ones(size * size, dtype=jnp.bool_),  # Removed swap action
            _size=jnp.int32(size),
            _board=jnp.zeros(size * size, jnp.int32)
        )

    def step(self, state: State, action: Array, key: Optional[Array] = None) -> State:
        del key
        is_illegal = ~self._check_legality(state, action)
        current_player = state.current_player

        # If already terminated/truncated, return the same state with zero rewards
        state = lax.cond(
            (state.terminated | state.truncated),
            lambda: state.replace(rewards=jnp.zeros_like(state.rewards)),
            lambda: _step(state.replace(_step_count=state._step_count + 1), action, size=self.size),
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
        return lax.stop_gradient(_observe(state, player_id, size=self.size))

    def _check_legality(self, state: State, action: Array) -> Array:
        mask_i32 = state.legal_action_mask.astype(jnp.int32)
        one_hot_a = jax.nn.one_hot(action, mask_i32.shape[0], dtype=jnp.int32)
        return jnp.dot(one_hot_a, mask_i32).astype(jnp.bool_)

    def _step_with_illegal_action(self, state: State, loser: Array) -> State:
        rewards = jnp.where(jnp.arange(2) == loser, -1.0, 1.0).astype(jnp.float32)
        return state.replace(rewards=rewards, terminated=TRUE)

    @property
    def id(self) -> str:
        return f"hexnoswap_{self.size}x{self.size}"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2

    @property
    def num_actions(self) -> int:
        return self.size * self.size


@partial(jax.jit, inline=True)
def _fast_gather1d(v: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    """Gather v[idx] via a small dot_general instead of a TPU-unfriendly gather."""
    one_hot = jax.nn.one_hot(idx, v.shape[0], dtype=v.dtype)  # [6, N]
    return jax.lax.dot_general(
        one_hot, v,
        dimension_numbers=(
            ((1,), (0,)),  # contract one_hot’s axis-1 with v’s axis-0
            ((), ())       # no batch dims
        )
    )  # → [6]

def _step(state: State, action: Array, size: int) -> State:
    N            = state._board.size
    set_place_id = action + 1

    # 1) Place the new stone
    board = state._board + jax.nn.one_hot(action, N, dtype=state._board.dtype) * set_place_id  # [N]

    # 2) Fast-gather the 6 neighbour labels
    neigh_idx = _neighbour(action, size)                                         # [6]
    neigh_val = _fast_gather1d(board, jnp.maximum(neigh_idx, 0))                 # [6]
    # zero-out off-board or opponent stones
    neigh_val = jnp.where((neigh_idx >= 0) & (neigh_val > 0), neigh_val, 0)      # [6]

    # 3) Build a [N,6] “match matrix” and reduce to [N] mask
    pos       = neigh_val > 0                                                    # [6] static shape
    matches   = (board[:, None] == neigh_val[None, :]) & pos[None, :]            # [N,6]
    touched   = jnp.any(matches, axis=1)                                         # [N]

    # 4) Merge in one shot
    board     = jnp.where(touched, set_place_id, board)                          # [N]
    won    = _is_game_end(board, size, state._turn)
    reward = jax.lax.cond(
        won,
        lambda: jnp.float32([-1.0, -1.0]).at[state.current_player].set(1.0),
        lambda: jnp.zeros(2, jnp.float32),
    )

    return state.replace(
        current_player    = 1 - state.current_player,
        _turn             = 1 - state._turn,
        _board            = board * -1,
        rewards           = reward,
        terminated        = won,
        legal_action_mask = state.legal_action_mask.at[:].set(board == 0),
    )

def _observe(state: State, player_id: Array, size) -> Array:
    board = jax.lax.select(
        player_id == state.current_player,
        state._board.reshape((size, size)),
        -state._board.reshape((size, size)),
    )

    my_board = board * 1 > 0
    opp_board = board * -1 > 0
    ones = jnp.ones_like(my_board)
    color = jax.lax.select(player_id == state.current_player, state._turn, 1 - state._turn)
    color = color * ones

    return jnp.stack([my_board, opp_board, color, ones], 2, dtype=jnp.bool_)

def _neighbour(xy, size):
    """
        (x,y-1)   (x+1,y-1)
    (x-1,y)    (x,y)    (x+1,y)
       (x-1,y+1)   (x,y+1)
    """
    x = xy // size
    y = xy % size
    xs = jnp.array([x, x + 1, x - 1, x + 1, x - 1, x])
    ys = jnp.array([y - 1, y - 1, y, y, y + 1, y + 1])
    on_board = (0 <= xs) & (xs < size) & (0 <= ys) & (ys < size)
    return jnp.where(on_board, xs * size + ys, -1)

def _is_game_end(board, size, turn):
    top, bottom = jax.lax.cond(
        turn == 1,
        lambda: (board[:size], board[-size:]),
        lambda: (board[::size], board[size - 1 :: size]),
    )

    def check_same_id_exist(_id):
        return (_id > 0) & (_id == bottom).any()

    return jax.vmap(check_same_id_exist)(top).any()
