"""Vmapped equivalence stress: installed `pgx` Go vs `pgx1` Go on the current backend.

Mirrors how training actually calls the env — jit(vmap(step)) — and bit-compares
every field of every lane at every step of full random games, including
observations and rewards. Run on TPU to also exercise the integer-matmul
lowering paths (`uv run --extra tpu python scripts/stress_go_vmap.py`).

To compare against a specific pgx rev (e.g. the rev an experiment pins),
point PYTHONPATH at a worktree of that rev:

    git -C ~/pgx worktree add /tmp/pgx-REV REV
    PYTHONPATH=/tmp/pgx-REV uv run --extra tpu python scripts/stress_go_vmap.py 4 9 19
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

import pgx
import pgx.go as ref_go
from pgx1.go import Go

KOMI = {3: 8.5, 4: 1.5, 5: 24.5, 6: 3.5, 7: 8.5, 8: 9.5}
BATCH = 256


def run(size):
    komi = KOMI.get(size, 7.5)
    ref_env = ref_go.Go(size=size, komi=komi)
    new_env = Go(size=size, komi=komi)
    vinit_r = jax.jit(jax.vmap(ref_env.init))
    vinit_n = jax.jit(jax.vmap(new_env.init))
    vstep_r = jax.jit(jax.vmap(ref_env.step))
    vstep_n = jax.jit(jax.vmap(new_env.step))
    vobs_r = jax.jit(jax.vmap(ref_env.observe))
    vobs_n = jax.jit(jax.vmap(new_env.observe))

    keys = jax.random.split(jax.random.PRNGKey(7), BATCH)
    s_ref = vinit_r(keys)
    s_new = vinit_n(keys)
    rng = np.random.default_rng(7)

    for t in range(2 * size * size + 4):
        for name in ("current_player", "terminated", "rewards", "legal_action_mask"):
            a, b = np.asarray(getattr(s_ref, name)), np.asarray(getattr(s_new, name))
            if not np.array_equal(a, b):
                lanes = np.flatnonzero((a != b).reshape(BATCH, -1).any(axis=-1) if a.ndim > 1 else a != b)
                return f"t={t} field={name} lanes={lanes[:5]}"
        for name in ("board", "ko", "num_captured", "is_psk", "hash_history"):
            a = np.asarray(getattr(s_ref._x, name))
            b = np.asarray(getattr(s_new._x, name))
            if not np.array_equal(a, b):
                lanes = np.flatnonzero((a != b).reshape(BATCH, -1).any(axis=-1))
                return f"t={t} _x.{name} lanes={lanes[:5]}"
        a = np.asarray(vobs_r(s_ref, s_ref.current_player))
        b = np.asarray(vobs_n(s_new, s_new.current_player))
        if not np.array_equal(a, b):
            return f"t={t} vmapped obs"
        if bool(np.asarray(s_ref.terminated).all()):
            return None
        mask = np.asarray(s_new.legal_action_mask)
        u = rng.random(mask.shape)  # per-lane random legal action
        actions = jnp.int32(np.argmax(u * mask, axis=-1))
        s_ref = vstep_r(s_ref, actions)
        s_new = vstep_n(s_new, actions)
    return None


if __name__ == "__main__":
    print("pgx from:", pgx.__file__)
    print("backend:", jax.default_backend())
    for size in [int(s) for s in sys.argv[1:]] or [4, 9, 19]:
        msg = run(size)
        print(f"size {size}: {'OK' if msg is None else 'FAIL ' + msg}", flush=True)
        assert msg is None, msg
