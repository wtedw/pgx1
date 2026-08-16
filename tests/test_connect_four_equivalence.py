"""Step-by-step equivalence of pgx1.connect_four against upstream pgx.connect_four.

Plays random games and asserts every observable piece of state matches the
reference implementation exactly at every step: legal masks, boards, colors,
winners, rewards, termination, and observations. Two intentional divergences
are normalized: upstream's 2-plane observation (my/opp stones) is compared
against the first two of pgx1's 3 planes with the extra all-ones plane
checked directly, and upstream's init-time first-player randomization is
pinned to pgx1's fixed current_player = 0 (see `_ref_init`).
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pgx.connect_four as ref_connect_four
from pgx1.connect_four import ConnectFour

# Default to CPU; set PGX1_TEST_PLATFORM=tpu to validate on device.
jax.config.update("jax_platform_name", os.environ.get("PGX1_TEST_PLATFORM", "cpu"))


def _ref_init(ref_env, key):
    # Upstream randomizes which player id moves first from the init key;
    # pgx1 (following the fork it was ported from) always starts with
    # current_player = 0. Pin upstream to the same convention -- everything
    # downstream is driven by the current_player field -- so trajectories
    # are comparable. This is the second intentional divergence.
    s = ref_env.init(key)
    return s.replace(current_player=jnp.int32(0))


def _envs():
    return ref_connect_four.ConnectFour(), ConnectFour()


def _assert_states_equal(s_ref, s_new, t):
    ctx = f"step={t}"
    assert int(s_ref.current_player) == int(s_new.current_player), ctx
    np.testing.assert_array_equal(
        np.asarray(s_ref.legal_action_mask), np.asarray(s_new.legal_action_mask), err_msg=ctx
    )
    assert bool(s_ref.terminated) == bool(s_new.terminated), ctx
    np.testing.assert_array_equal(np.asarray(s_ref.rewards), np.asarray(s_new.rewards), err_msg=ctx)
    x_ref, x_new = s_ref._x, s_new._x
    assert int(x_ref.color) == int(x_new.color), ctx
    np.testing.assert_array_equal(np.asarray(x_ref.board), np.asarray(x_new.board), err_msg=ctx)
    assert int(x_ref.winner) == int(x_new.winner), ctx


@pytest.mark.parametrize("seed", range(20))
def test_random_playout_equivalence(seed):
    ref_env, new_env = _envs()
    ref_step = jax.jit(ref_env.step)
    new_step = jax.jit(new_env.step)
    ref_obs = jax.jit(ref_env.observe)
    new_obs = jax.jit(new_env.observe)

    key = jax.random.PRNGKey(seed)
    s_ref = _ref_init(ref_env, key)
    s_new = new_env.init(key)
    rng = np.random.default_rng(seed)

    for t in range(42 + 2):
        _assert_states_equal(s_ref, s_new, t)
        for pid in (0, 1):
            octx = f"step={t} obs pid={pid}"
            o_ref = np.asarray(ref_obs(s_ref, pid))
            o_new = np.asarray(new_obs(s_new, pid))
            # pgx1's my/opp planes must match upstream's 2-plane observation.
            np.testing.assert_array_equal(o_ref, o_new[..., :2], err_msg=octx)
            # Extra plane 2 is all ones.
            np.testing.assert_array_equal(o_new[..., 2], np.ones((6, 7), bool), err_msg=octx)
        if bool(s_ref.terminated):
            break
        legal = np.flatnonzero(np.asarray(s_ref.legal_action_mask))
        action = int(rng.choice(legal))
        s_ref = ref_step(s_ref, action)
        s_new = new_step(s_new, action)
    else:
        pytest.fail(f"game did not terminate (seed={seed})")


def test_illegal_action_equivalence():
    ref_env, new_env = _envs()
    ref_step = jax.jit(ref_env.step)
    new_step = jax.jit(new_env.step)
    key = jax.random.PRNGKey(0)
    s_ref = _ref_init(ref_env, key)
    s_new = new_env.init(key)
    # fill column 0 (6 legal drops), then drop into the full column (illegal)
    for t in range(7):
        s_ref = ref_step(s_ref, 0)
        s_new = new_step(s_new, 0)
        _assert_states_equal(s_ref, s_new, f"column-fill t={t}")
    assert bool(s_new.terminated)
    assert set(np.asarray(s_new.rewards).tolist()) == {-1.0, 1.0}


def test_step_after_terminal_equivalence():
    ref_env, new_env = _envs()
    ref_step = jax.jit(ref_env.step)
    new_step = jax.jit(new_env.step)
    key = jax.random.PRNGKey(0)
    s_ref = _ref_init(ref_env, key)
    s_new = new_env.init(key)
    # player 0 wins with four in column 0 (players alternate columns 0/1)
    for action in (0, 1, 0, 1, 0, 1, 0):
        s_ref = ref_step(s_ref, action)
        s_new = new_step(s_new, action)
    assert bool(s_new.terminated)
    # keep stepping after terminal: same state, zero rewards
    for action in (2, 3):
        s_ref = ref_step(s_ref, action)
        s_new = new_step(s_new, action)
        _assert_states_equal(s_ref, s_new, f"post-terminal action={action}")
    np.testing.assert_array_equal(np.asarray(s_new.rewards), np.zeros(2, np.float32))


def test_env_api_equivalence():
    ref_env, new_env = _envs()
    assert ref_env.id == new_env.id
    assert ref_env.version == new_env.version
    assert ref_env.num_players == new_env.num_players
    assert ref_env.num_actions == new_env.num_actions
    s_ref = _ref_init(ref_env, jax.random.PRNGKey(0))
    s_new = new_env.init(jax.random.PRNGKey(0))
    assert s_ref.env_id == s_new.env_id
