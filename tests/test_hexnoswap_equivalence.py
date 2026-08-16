"""Step-by-step equivalence of pgx1.hexnoswap against the reference pgx.hexnoswap.

Plays random games and asserts every observable piece of state matches the
reference implementation exactly at every step: legal masks, boards, turns,
rewards, termination, and observations.

hexnoswap only exists in the local pgx fork the env was ported from (upstream
sotetsuk/pgx has only hex-with-swap), so this module skips unless that fork
is the installed `pgx`. pgx1 is the source of truth; this suite exists to
re-verify the port if the fork is ever installed again.
"""

import os

import jax
import numpy as np
import pytest

ref_hexnoswap = pytest.importorskip(
    "pgx.hexnoswap", reason="reference pgx.hexnoswap only exists in the local pgx fork"
)
from pgx1.hexnoswap import Hexnoswap

# Default to CPU; set PGX1_TEST_PLATFORM=tpu to validate on device.
jax.config.update("jax_platform_name", os.environ.get("PGX1_TEST_PLATFORM", "cpu"))


def _envs(size):
    ref = ref_hexnoswap.Hexnoswap(size=size)
    new = Hexnoswap(size=size)
    return ref, new


def _assert_states_equal(s_ref, s_new, size, t):
    ctx = f"size={size} step={t}"
    assert int(s_ref.current_player) == int(s_new.current_player), ctx
    np.testing.assert_array_equal(
        np.asarray(s_ref.legal_action_mask), np.asarray(s_new.legal_action_mask), err_msg=ctx
    )
    assert bool(s_ref.terminated) == bool(s_new.terminated), ctx
    np.testing.assert_array_equal(np.asarray(s_ref.rewards), np.asarray(s_new.rewards), err_msg=ctx)
    np.testing.assert_array_equal(np.asarray(s_ref._board), np.asarray(s_new._board), err_msg=ctx)
    assert int(s_ref._turn) == int(s_new._turn), ctx
    assert int(s_ref._size) == int(s_new._size), ctx


@pytest.mark.parametrize("size", [3, 4, 5, 7, 11])
@pytest.mark.parametrize("seed", range(8))
def test_random_playout_equivalence(size, seed):
    ref_env, new_env = _envs(size)
    ref_step = jax.jit(ref_env.step)
    new_step = jax.jit(new_env.step)
    ref_obs = jax.jit(ref_env.observe)
    new_obs = jax.jit(new_env.observe)

    key = jax.random.PRNGKey(seed)
    s_ref = ref_env.init(key)
    s_new = new_env.init(key)
    rng = np.random.default_rng(seed)

    # Hex has no draws: a full board always contains a winning chain.
    for t in range(size * size + 2):
        _assert_states_equal(s_ref, s_new, size, t)
        for pid in (0, 1):
            o_ref = ref_obs(s_ref, pid)
            o_new = new_obs(s_new, pid)
            np.testing.assert_array_equal(
                np.asarray(o_ref), np.asarray(o_new), err_msg=f"size={size} step={t} obs pid={pid}"
            )
        if bool(s_ref.terminated):
            break
        legal = np.flatnonzero(np.asarray(s_ref.legal_action_mask))
        action = int(rng.choice(legal))
        s_ref = ref_step(s_ref, action)
        s_new = new_step(s_new, action)
    else:
        pytest.fail(f"game did not terminate (size={size} seed={seed})")


@pytest.mark.parametrize("size", [5])
def test_illegal_action_equivalence(size):
    ref_env, new_env = _envs(size)
    key = jax.random.PRNGKey(0)
    s_ref = ref_env.init(key)
    s_new = new_env.init(key)
    # play one stone, then play the same point again (illegal)
    for action in (12, 12):
        s_ref = jax.jit(ref_env.step)(s_ref, action)
        s_new = jax.jit(new_env.step)(s_new, action)
    _assert_states_equal(s_ref, s_new, size, "illegal")
    assert bool(s_new.terminated)
    assert set(np.asarray(s_new.rewards).tolist()) == {-1.0, 1.0}


@pytest.mark.parametrize("size", [5])
def test_step_after_terminal_equivalence(size):
    ref_env, new_env = _envs(size)
    ref_step = jax.jit(ref_env.step)
    new_step = jax.jit(new_env.step)
    key = jax.random.PRNGKey(1)
    s_ref = ref_env.init(key)
    s_new = new_env.init(key)
    rng = np.random.default_rng(1)
    # play a random game to termination
    while not bool(s_ref.terminated):
        legal = np.flatnonzero(np.asarray(s_ref.legal_action_mask))
        action = int(rng.choice(legal))
        s_ref = ref_step(s_ref, action)
        s_new = new_step(s_new, action)
    # keep stepping after terminal: same state, zero rewards
    for action in (0, 3):
        s_ref = ref_step(s_ref, action)
        s_new = new_step(s_new, action)
        _assert_states_equal(s_ref, s_new, size, f"post-terminal action={action}")
    assert bool(s_new.terminated)
    np.testing.assert_array_equal(np.asarray(s_new.rewards), np.zeros(2, np.float32))


@pytest.mark.parametrize("size", [3, 11])
def test_env_api_equivalence(size):
    ref_env, new_env = _envs(size)
    assert ref_env.id == new_env.id
    assert ref_env.version == new_env.version
    assert ref_env.num_players == new_env.num_players
    assert ref_env.num_actions == new_env.num_actions
    s_ref = ref_env.init(jax.random.PRNGKey(0))
    s_new = new_env.init(jax.random.PRNGKey(0))
    assert s_ref.env_id == s_new.env_id
