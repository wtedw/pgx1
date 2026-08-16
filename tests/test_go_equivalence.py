"""Step-by-step equivalence of pgx1.go against upstream pgx.go.

Plays random games and asserts every observable piece of state matches the
reference implementation exactly at every step: legal masks, boards, ko,
captures, PSK, hashes, rewards, termination, and observations. The one
intentional divergence, upstream's init-time first-player randomization, is
pinned to pgx1's fixed current_player = 0 (see `_ref_init`).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pgx.go as ref_go
from pgx1.go import Go

jax.config.update("jax_platform_name", "cpu")



def _ref_init(ref_env, key):
    # Upstream randomizes which player id moves first from the init key;
    # pgx1 (following the fork it was ported from) always starts with
    # current_player = 0. Pin upstream to the same convention -- everything
    # downstream is driven by the current_player field -- so trajectories
    # are comparable. Upstream go threads the randomization through the
    # stored _player_order (used to map colors to players every step), so
    # that must be pinned to the identity order too.
    s = ref_env.init(key)
    order = jnp.int32([0, 1])
    return s.replace(current_player=order[s._x.color], _player_order=order)


def _envs(size):
    komi = 7.5  # fixed for both so the comparison is exact
    ref = ref_go.Go(size=size, komi=komi)
    new = Go(size=size, komi=komi)
    return ref, new


def _assert_states_equal(s_ref, s_new, size, t):
    ctx = f"size={size} step={t}"
    assert int(s_ref.current_player) == int(s_new.current_player), ctx
    np.testing.assert_array_equal(
        np.asarray(s_ref.legal_action_mask), np.asarray(s_new.legal_action_mask), err_msg=ctx
    )
    assert bool(s_ref.terminated) == bool(s_new.terminated), ctx
    np.testing.assert_array_equal(np.asarray(s_ref.rewards), np.asarray(s_new.rewards), err_msg=ctx)
    x_ref, x_new = s_ref._x, s_new._x
    np.testing.assert_array_equal(np.asarray(x_ref.board), np.asarray(x_new.board), err_msg=ctx)
    np.testing.assert_array_equal(
        np.asarray(x_ref.board_history), np.asarray(x_new.board_history), err_msg=ctx
    )
    assert int(x_ref.ko) == int(x_new.ko), ctx
    np.testing.assert_array_equal(np.asarray(x_ref.num_captured), np.asarray(x_new.num_captured), err_msg=ctx)
    assert bool(x_ref.is_psk) == bool(x_new.is_psk), ctx
    assert int(x_ref.consecutive_pass_count) == int(x_new.consecutive_pass_count), ctx
    np.testing.assert_array_equal(np.asarray(x_ref.hash_history), np.asarray(x_new.hash_history), err_msg=ctx)


@pytest.mark.parametrize("size", [3, 4, 5, 6, 7, 8, 9])
@pytest.mark.parametrize("seed", range(8))
def test_random_playout_equivalence(size, seed):
    ref_env, new_env = _envs(size)
    ref_step = jax.jit(ref_env.step)
    new_step = jax.jit(new_env.step)
    ref_obs = jax.jit(ref_env.observe)
    new_obs = jax.jit(new_env.observe)

    key = jax.random.PRNGKey(seed)
    s_ref = _ref_init(ref_env, key)
    s_new = new_env.init(key)
    rng = np.random.default_rng(seed)

    for t in range(2 * size * size + 4):
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
    s_ref = _ref_init(ref_env, key)
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
    key = jax.random.PRNGKey(0)
    s_ref = _ref_init(ref_env, key)
    s_new = new_env.init(key)
    pass_action = size * size
    # two passes -> terminal, then keep stepping
    for action in (pass_action, pass_action, 0, 3):
        s_ref = jax.jit(ref_env.step)(s_ref, action)
        s_new = jax.jit(new_env.step)(s_new, action)
        _assert_states_equal(s_ref, s_new, size, f"post-terminal action={action}")
    assert bool(s_new.terminated)
    np.testing.assert_array_equal(np.asarray(s_new.rewards), np.zeros(2, np.float32))


def test_19x19_smoke():
    """A short deterministic 19x19 game to exercise the full-size board."""
    size = 19
    ref_env, new_env = _envs(size)
    ref_step = jax.jit(ref_env.step)
    new_step = jax.jit(new_env.step)
    key = jax.random.PRNGKey(42)
    s_ref = _ref_init(ref_env, key)
    s_new = new_env.init(key)
    rng = np.random.default_rng(42)
    for t in range(60):
        _assert_states_equal(s_ref, s_new, size, t)
        if bool(s_ref.terminated):
            break
        legal = np.flatnonzero(np.asarray(s_ref.legal_action_mask)[:-1])  # avoid passes: play stones
        action = int(rng.choice(legal))
        s_ref = ref_step(s_ref, action)
        s_new = new_step(s_new, action)


@pytest.mark.parametrize("size", [3, 4, 5])
def test_capture_and_ko(size):
    """Force captures by biasing black to fill; verifies num_captured/ko paths."""
    ref_env, new_env = _envs(size)
    ref_step = jax.jit(ref_env.step)
    new_step = jax.jit(new_env.step)
    for seed in range(20):
        key = jax.random.PRNGKey(seed)
        s_ref = _ref_init(ref_env, key)
        s_new = new_env.init(key)
        rng = np.random.default_rng(seed)
        saw_capture = False
        for t in range(2 * size * size + 4):
            _assert_states_equal(s_ref, s_new, size, t)
            saw_capture |= int(np.asarray(s_ref._x.num_captured).sum()) > 0
            if bool(s_ref.terminated):
                break
            mask = np.asarray(s_ref.legal_action_mask)
            legal_moves = np.flatnonzero(mask[:-1])
            # prefer stone placements to generate captures/kos
            if len(legal_moves) > 0 and rng.random() > 0.02:
                action = int(rng.choice(legal_moves))
            else:
                action = size * size
            s_ref = ref_step(s_ref, action)
            s_new = new_step(s_new, action)
