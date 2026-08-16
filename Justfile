# Run `just --list` to see all recipes.

default:
    @just --list

# Run the (CPU, interpret-mode) test suites WITHOUT locking the TPU: plain
# `pytest` initializes and holds the TPU for the whole run even though the
# test modules force the cpu platform, blocking every other TPU user.
test *args="tests/":
    JAX_PLATFORMS=cpu uv run pytest {{args}}

# Replay the 128 vendored professional games (tests/data/pro_games_128.json;
# --data tests/data/pro_games_1024.json for the full set) through the
# compiled vmap(Chess.step) ON DEVICE: every move must be legal in the
# device mask, and final boards must match the python-chess-derived
# positions.
replay-pro-games *flags="":
    uv run python scripts/replay_pro_games_tpu.py {{flags}}

# Benchmark one DP-sharded vmap(env.step) across all TPU devices for every
# env (tic_tac_toe, connect_four, hexnoswap_11x11, go_9x9, go_19x19, chess),
# upstream sotetsuk/pgx (a dev dependency) vs pgx1, appending rows to
# /tmp/pgx1/bench_envs.jsonl and printing the pivot.
bench-envs batch="4096" trials="20" *flags="":
    uv run python scripts/bench_envs.py --batch {{batch}} --trials {{trials}} {{flags}}

# Trace-profile jax.vmap(Chess.step) to /tmp/pgx1/scalar. Doesn't launch
# TensorBoard -- run `just tb` separately to view it. Requires
# `uv sync --extra prof`.
prof-scalar batch="4096" steps="20":
    uv run --extra prof python scripts/prof_chess_scalar_step.py --batch {{batch}} --steps {{steps}} --no-tensorboard

# Trace ONE vmapped Chess.step call, then print an op-level timing breakdown
# parsed straight from the trace file -- no TensorBoard needed. The go-to
# loop for optimization work. Requires `uv sync --extra prof`.
prof-scalar-analyze *args="":
    #!/usr/bin/env bash
    set -euo pipefail
    args=( {{args}} )
    if [[ ${#args[@]} -gt 0 && "${args[0]}" == "--" ]]; then
        args=("${args[@]:1}")
    fi
    uv run --extra prof python scripts/prof_chess_scalar_step.py --batch 4096 --steps 1 --no-tensorboard "${args[@]}"
    uv run --extra prof python scripts/analyze_chess_trace.py /tmp/pgx1/scalar

# Print the op-level timing breakdown of an already-written trace (newest run
# under trace_dir) without re-profiling. Pass --steps N if the trace covered
# N steps (e.g. `just analyze /tmp/pgx1/scalar --steps 20` after a default
# `just prof-scalar`) so times are reported per step.
analyze trace_dir="/tmp/pgx1/scalar" *flags="":
    uv run --extra prof python scripts/analyze_chess_trace.py {{trace_dir}} {{flags}}

# Launch TensorBoard on already-written traces from prof-scalar. Requires
# `uv sync --extra prof`.
tb logdir="/tmp/pgx1" port="6006" host="0.0.0.0":
    uv run --extra prof tensorboard --logdir {{logdir}} --port {{port}} --host {{host}}
