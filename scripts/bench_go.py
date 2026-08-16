"""Throughput benchmark: reference pgx.go vs pgx1.go.

Runs a jitted scan of vmapped env steps with random legal actions and
reports steps/sec. Run on the target accelerator, e.g.:

    uv run python scripts/bench_go.py --sizes 9 19 --batch 1024 --steps 128
"""

import argparse
import time

import jax
import jax.numpy as jnp
from jax import lax


def make_runner(env, batch, steps):
    def random_action(key, mask):
        logits = jnp.where(mask, 0.0, -1e9)
        return jax.random.categorical(key, logits)

    @jax.jit
    def run(state, key):
        def body(carry, _):
            state, key = carry
            key, sub = jax.random.split(key)
            keys = jax.random.split(sub, batch)
            actions = jax.vmap(random_action)(keys, state.legal_action_mask)
            state = jax.vmap(env.step)(state, actions)
            return (state, key), None

        (state, key), _ = lax.scan(body, (state, key), None, length=steps)
        return state

    return run


def bench(name, env, batch, steps, repeats=3):
    keys = jax.random.split(jax.random.PRNGKey(0), batch)
    state = jax.jit(jax.vmap(env.init))(keys)
    run = make_runner(env, batch, steps)

    # compile + warmup
    jax.block_until_ready(run(state, jax.random.PRNGKey(1)))

    best = float("inf")
    for i in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(run(state, jax.random.PRNGKey(2 + i)))
        best = min(best, time.perf_counter() - t0)
    sps = batch * steps / best
    print(f"{name:30s}  {best * 1e3:9.1f} ms  {sps / 1e3:10.1f}k env-steps/s")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[9, 19])
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=128)
    args = parser.parse_args()

    import pgx.go as ref_go

    import pgx1.go as new_go

    print(f"backend={jax.default_backend()}  batch={args.batch}  steps={args.steps}")
    for size in args.sizes:
        ref = bench(f"pgx.go       {size}x{size}", ref_go.Go(size=size), args.batch, args.steps)
        new = bench(f"pgx1.go      {size}x{size}", new_go.Go(size=size, komi=7.5), args.batch, args.steps)
        print(f"{'speedup':30s}  {ref / new:9.2f}x")


if __name__ == "__main__":
    main()
