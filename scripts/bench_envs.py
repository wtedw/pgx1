"""pgx1 vs upstream sotetsuk/pgx step benchmark.

Times one compiled, DP-sharded `env.step` call (fixed legal actions, averaged
over `--trials`) for each env. Both implementations are driven through plain
`jax.vmap(env.step)`. pgx1 chess materializes
`State.observation` every step by default, matching upstream's behavior;
pass `--chess-no-obs` to skip it.

Upstream pgx is an ordinary dev dependency of this repo (the PyPI `pgx`
package), so one invocation benchmarks both implementations back to back,
appends JSON rows to `--out`, and prints the env x implementation pivot.
Restrict with `--impls pgx1` / `--impls upstream`; `--summarize` re-prints
the pivot from `--out` without benchmarking.

Caveats for reading results:
- Upstream `Env.step` computes `State.observation` every step; pgx1 skips it
  for everything but chess (observations are computed on demand via
  `observe`). That difference is part of the measured speedup, and it is the
  biggest term for Go's 17-plane observation.
- Upstream has no `hexnoswap`; its rows fall back to `hex_*` (with swap, one
  extra action) as the closest proxy, labeled in the output.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_ENVS = ["tic_tac_toe", "connect_four", "hexnoswap_11x11", "go_9x9", "go_19x19", "chess"]


# --- DP sharding helpers ---


def make_mesh():
    devices = np.asarray(jax.devices())
    return jax.sharding.Mesh(devices, ("x",))


def make_shardings(mesh: jax.sharding.Mesh):
    data_parallel = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x"))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    return data_parallel, replicated


def _batched_leaf_sharding(shape_struct, data_parallel, replicated):
    if getattr(shape_struct, "ndim", 0) > 0:
        return data_parallel
    return replicated


def sharding_tree_for_shape(shape_tree, data_parallel, replicated):
    return jax.tree_util.tree_map(
        lambda x: _batched_leaf_sharding(x, data_parallel, replicated),
        shape_tree,
    )


def create_sharded_env_state(
    env_init_fn: Callable,
    batch_size: int,
    mesh: jax.sharding.Mesh,
    data_parallel: jax.sharding.NamedSharding,
    replicated: jax.sharding.NamedSharding,
    seed: int = 0,
):
    """Initialize batched env state directly with leading-axis DP sharding."""
    keys = jax.random.split(jax.random.PRNGKey(seed), batch_size)
    state_shape_tree = jax.eval_shape(env_init_fn, keys)
    out_sharding_tree = sharding_tree_for_shape(state_shape_tree, data_parallel, replicated)
    sharded_init_fn = jax.jit(
        env_init_fn,
        in_shardings=data_parallel,
        out_shardings=out_sharding_tree,
    )
    with jax.set_mesh(mesh):
        sharded_keys = jax.device_put(keys, data_parallel)
        sharded_env_state = sharded_init_fn(sharded_keys)
    return sharded_env_state, out_sharding_tree


def random_legal_actions(state, rng: np.random.Generator) -> jnp.ndarray:
    masks = np.asarray(state.legal_action_mask)
    actions = np.zeros(masks.shape[0], np.int32)
    for i in range(masks.shape[0]):
        legal = np.flatnonzero(masks[i])
        actions[i] = int(rng.choice(legal)) if legal.size else 0
    return jnp.asarray(actions)


def benchmark(
    fn,
    state,
    state_sharding,
    actions,
    mesh: jax.sharding.Mesh,
    data_parallel,
    replicated,
    trials: int,
) -> float:
    out_shape = jax.eval_shape(fn, state, actions)
    out_sharding = sharding_tree_for_shape(out_shape, data_parallel, replicated)
    with jax.set_mesh(mesh):
        sharded_actions = jax.device_put(actions, data_parallel)
        compiled = jax.jit(
            fn,
            in_shardings=(state_sharding, data_parallel),
            out_shardings=out_sharding,
        ).lower(state, sharded_actions).compile()
        jax.block_until_ready(compiled(state, sharded_actions))
        start = time.perf_counter()
        for _ in range(trials):
            result = compiled(state, sharded_actions)
        jax.block_until_ready(result)
    return (time.perf_counter() - start) / trials


# --- env construction per implementation ---


def make_env_pgx1(env_id: str, chess_no_obs: bool = False):
    if env_id == "chess":
        from pgx1.chess import Chess

        return Chess(return_observation=not chess_no_obs), env_id
    if env_id.startswith("go_"):
        from pgx1.go import Go

        return Go(size=int(env_id.split("_")[1].split("x")[0])), env_id
    if env_id.startswith("hexnoswap_"):
        from pgx1.hexnoswap import Hexnoswap

        return Hexnoswap(size=int(env_id.split("_")[1].split("x")[0])), env_id
    if env_id == "connect_four":
        from pgx1.connect_four import ConnectFour

        return ConnectFour(), env_id
    if env_id == "tic_tac_toe":
        from pgx1.tic_tac_toe import TicTacToe

        return TicTacToe(), env_id
    raise ValueError(f"no pgx1 env for '{env_id}'")


def make_env_upstream(env_id: str):
    import pgx

    try:
        return pgx.make(env_id), env_id
    except ValueError:
        # Upstream sotetsuk/pgx has no hexnoswap (and registers no sized hex
        # ids); fall back to hex (with swap, one extra action) as the closest
        # available proxy, constructing it directly for the requested size.
        if env_id.startswith("hexnoswap_"):
            from pgx.hex import Hex

            size = int(env_id.split("_")[1].split("x")[0])
            proxy = f"hex_{size}x{size}"
            print(f"[bench_envs] '{env_id}' unavailable in upstream pgx; using '{proxy}' as proxy")
            return Hex(size=size), proxy
        raise


# --- driver ---


def run_bench(args) -> None:
    mesh = make_mesh()
    data_parallel, replicated = make_shardings(mesh)
    n_dev = len(mesh.devices.ravel())
    assert args.batch % n_dev == 0, f"--batch {args.batch} not divisible by {n_dev} devices"
    n_rows = 0
    for impl in args.impls:
        for env_id in args.envs:
            if impl == "pgx1":
                env, actual_id = make_env_pgx1(env_id, chess_no_obs=args.chess_no_obs)
            else:
                env, actual_id = make_env_upstream(env_id)
            init_fn = jax.vmap(env.init)
            state, state_sharding = create_sharded_env_state(
                init_fn, args.batch, mesh, data_parallel, replicated, seed=args.seed
            )
            actions = random_legal_actions(state, np.random.default_rng(args.seed))
            step_fn = jax.vmap(env.step)
            sec = benchmark(
                step_fn, state, state_sharding, actions, mesh, data_parallel, replicated, args.trials
            )
            row = {
                "label": impl,
                "impl": impl,
                "env_id": env_id,
                "actual_env_id": actual_id,
                "batch": args.batch,
                "trials": args.trials,
                "devices": n_dev,
                "sec_per_step": sec,
                "steps_per_sec": args.batch / sec,
            }
            n_rows += 1
            print(
                f"[{impl}] {env_id:>16}: {sec * 1e3:8.3f} ms/step  "
                f"{row['steps_per_sec'] / 1e6:8.2f} M steps/s  (batch={args.batch}, {n_dev} devices)"
            )
            # Append per env so a crash on a later env doesn't lose finished rows.
            if args.out:
                os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
                with open(args.out, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
    if args.out:
        print(f"[bench_envs] appended {n_rows} rows to {args.out}")
        print()
        summarize(args.out, args.baseline)


def summarize(path: str, baseline: str) -> None:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    # keep the latest row per (env, label)
    latest = {}
    for row in rows:
        latest[(row["env_id"], row["label"])] = row
    labels = list(dict.fromkeys(row["label"] for row in rows))
    envs = list(dict.fromkeys(row["env_id"] for row in rows))
    if baseline not in labels:
        baseline = labels[0]
    header = f"{'env':>16} | " + " | ".join(f"{lb:>21}" for lb in labels) + " | speedup"
    print(header)
    print("-" * len(header))
    for env_id in envs:
        cells = []
        for lb in labels:
            row = latest.get((env_id, lb))
            cells.append(
                f"{row['sec_per_step'] * 1e3:9.3f}ms {row['steps_per_sec'] / 1e6:7.2f}M"
                if row
                else f"{'-':>21}"
            )
        base = latest.get((env_id, baseline))
        best = latest.get((env_id, "pgx1")) or latest.get((env_id, labels[-1]))
        speedup = (
            f"{base['sec_per_step'] / best['sec_per_step']:6.1f}x vs {baseline}"
            if base and best and base is not best
            else ""
        )
        print(f"{env_id:>16} | " + " | ".join(cells) + " | " + speedup)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--impls",
        nargs="+",
        choices=["pgx1", "upstream"],
        default=["upstream", "pgx1"],
        help="implementations to benchmark (upstream = the installed sotetsuk/pgx package)",
    )
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="/tmp/pgx1/bench_envs.jsonl")
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="print the env x label pivot of --out instead of benchmarking",
    )
    parser.add_argument("--baseline", default="upstream", help="label speedups are computed against")
    parser.add_argument(
        "--chess-no-obs",
        action="store_true",
        help="pgx1 chess only: skip materializing State.observation each step",
    )
    args = parser.parse_args()
    if args.summarize:
        summarize(args.out, args.baseline)
        return
    run_bench(args)


if __name__ == "__main__":
    main()
