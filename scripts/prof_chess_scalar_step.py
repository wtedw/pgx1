"""Trace-profile `jax.vmap(Chess.step)`.

Warm up once, then trace N repeated steps of the compiled step. Writes to
/tmp/pgx1/scalar by default; view with `tensorboard --logdir /tmp/pgx1` or
analyze with `scripts/analyze_chess_trace.py /tmp/pgx1/scalar`.
"""

from __future__ import annotations

import argparse
import subprocess
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from pgx1.chess import Chess
from pgx1.chess import unpack_bitmask


def make_mesh():
    return jax.sharding.Mesh(np.asarray(jax.devices()), ("x",))


def make_shardings(mesh: jax.sharding.Mesh):
    data_parallel = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x"))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    return data_parallel, replicated


def sharding_tree_for_shape(shape_tree, data_parallel, replicated):
    return jax.tree_util.tree_map(
        lambda x: data_parallel if getattr(x, "ndim", 0) > 0 else replicated,
        shape_tree,
    )


def create_sharded_env_state(
    env_init_fn: Callable,
    batch_size: int,
    mesh: jax.sharding.Mesh,
    data_parallel: jax.sharding.NamedSharding,
    replicated: jax.sharding.NamedSharding,
):
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    state_shape_tree = jax.eval_shape(env_init_fn, keys)
    out_sharding_tree = sharding_tree_for_shape(state_shape_tree, data_parallel, replicated)
    sharded_init_fn = jax.jit(
        env_init_fn,
        in_shardings=data_parallel,
        out_shardings=out_sharding_tree,
    )
    with jax.set_mesh(mesh):
        return sharded_init_fn(jax.device_put(keys, data_parallel)), out_sharding_tree


def compile_sharded_step(fn, state, state_sharding, actions, action_sharding, mesh, data_parallel, replicated):
    out_shape = jax.eval_shape(fn, state, actions)
    out_sharding = sharding_tree_for_shape(out_shape, data_parallel, replicated)
    with jax.set_mesh(mesh):
        sharded_actions = jax.device_put(actions, action_sharding)
        compiled = jax.jit(
            fn,
            in_shardings=(state_sharding, action_sharding),
            out_shardings=out_sharding,
        ).lower(state, sharded_actions).compile()
    return compiled, sharded_actions


def legal_action_mask_from_state(state):
    if state.legal_action_mask is not None:
        return state.legal_action_mask
    return unpack_bitmask(state.legal_action_bitmask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        choices=["pgx1", "pgx", "chess0"],
        default="pgx1",
        help="pgx1: local original v1 chess; pgx: upstream pgx.chess; chess0: upstream pgx.chess0",
    )
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--trace-dir", type=str, default="/tmp/pgx1/scalar")
    parser.add_argument("--no-tensorboard", action="store_true", help="only write the trace, don't launch TensorBoard")
    parser.add_argument(
        "--dp-sharding",
        dest="dp_sharding",
        action="store_true",
        default=True,
        help="shard the leading batch axis of state/actions across all local devices",
    )
    parser.add_argument(
        "--no-dp-sharding",
        dest="dp_sharding",
        action="store_false",
        help="run the old unsharded jax.jit profile",
    )
    args = parser.parse_args()

    if args.env == "pgx":
        import pgx.chess as upstream_chess

        env = upstream_chess.Chess()
    elif args.env == "chess0":
        import pgx.chess0 as upstream_chess0

        env = upstream_chess0.Chess0()
    else:
        env = Chess()

    mesh = make_mesh()
    data_parallel, replicated = make_shardings(mesh)
    if args.dp_sharding and args.batch % len(jax.devices()):
        raise ValueError("--batch must be divisible by the number of local devices for DP sharding")

    if args.dp_sharding:
        state, state_sharding = create_sharded_env_state(
            jax.vmap(env.init),
            args.batch,
            mesh,
            data_parallel,
            replicated,
        )
    else:
        keys = jax.random.split(jax.random.PRNGKey(0), args.batch)
        state = jax.jit(jax.vmap(env.init))(keys)
        state_sharding = None
    # NOTE: replayed every traced step (see prof_chess_step_batch.py's note):
    # kernel/op timings stay representative, game-phase op mixes do not.
    legal = jnp.argmax(legal_action_mask_from_state(state), axis=-1)

    step_fn = jax.vmap(env.step)
    if args.dp_sharding:
        step_fn, legal = compile_sharded_step(
            step_fn,
            state,
            state_sharding,
            legal,
            data_parallel,
            mesh,
            data_parallel,
            replicated,
        )
    else:
        step_fn = jax.jit(step_fn)
    state = step_fn(state, legal)
    jax.block_until_ready(state)

    jax.profiler.start_trace(args.trace_dir)
    for _ in range(args.steps):
        state = step_fn(state, legal)
    jax.block_until_ready(state)
    jax.profiler.stop_trace()

    print(
        f"trace written to {args.trace_dir} "
        f"(env={args.env}, dp_sharding={args.dp_sharding}, local_devices={len(jax.devices())})"
    )
    if not args.no_tensorboard:
        subprocess.run(["tensorboard", "--logdir", args.trace_dir], check=True)


if __name__ == "__main__":
    main()
