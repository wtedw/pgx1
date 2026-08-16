# pgx1

self-contained, TPU optimized implementations of the envs found in [pgx](https://github.com/sotetsuk/pgx/).

## Benchmarks

| env              | pgx      | pgx1     | pgx1 vs pgx |
| ---------------- | -------- | -------- | ----------- |
| tic_tac_toe      | 0.379 ms | 0.356 ms | 1.1x        |
| connect_four     | 2.678 ms | 0.305 ms | 8.8x        |
| hexnoswap_11x11* | 0.864 ms | 0.298 ms | 2.9x        |
| go_9x9           | 80.5 ms  | 0.535 ms | 150x        |
| go_19x19         | 656 ms   | 1.595 ms | 411x        |
| chess            | 904 ms   | 0.832 ms | 1087x       |

## Usage

```python
import jax
from pgx1.go import Go

env = Go(size=9)  # komi defaults per size, matching pgx.make
init = jax.jit(jax.vmap(env.init))
step = jax.jit(jax.vmap(env.step))

states = init(jax.random.split(jax.random.PRNGKey(0), 1024))
states = step(states, actions)
```

## Development

```sh
uv sync            # installs dev deps incl. upstream sotetsuk/pgx (comparison baseline)
# On a Cloud TPU VM, use `uv sync --extra tpu` before the benchmarks.
uv run pytest      # incl. move-for-move equivalence vs upstream pgx
uv run python scripts/bench_go.py --sizes 9 19 --batch 1024 --steps 128
```

A [`Justfile`](Justfile) wraps the common workflows:

```sh
just test               # CPU test suite (without holding the TPU)
just bench-envs         # DP-sharded vmap(step) benchmark, pgx1 vs upstream, all envs
just replay-pro-games   # replay vendored pro games through vmap(Chess.step) on device
```

### Profiling `Chess.step`

```sh
uv sync --extra tpu --extra prof   # both extras in the SAME invocation --
                                    # `uv sync --extra X` reconciles the venv to
                                    # exactly the extras listed, so a separate
                                    # `uv sync --extra prof` call afterward would
                                    # uninstall tpu's packages (and vice versa)
just prof-scalar                   # trace jax.vmap(Chess.step) to /tmp/pgx1/scalar
just tb                            # launch TensorBoard on the trace (logdir /tmp/pgx1)
just prof-scalar-analyze           # trace ONE step call, then print an op-level
                                    # timing breakdown in the terminal (no TensorBoard)
just analyze                       # same breakdown on an existing trace, without re-profiling
```

If you're on a remote machine (e.g. a Cloud TPU VM) and `tensorboard`'s
`http://0.0.0.0:6006` isn't reachable from your local browser, either open an SSH
tunnel (`ssh -L 6006:localhost:6006 <your-vm>`, then browse to
`http://localhost:6006` locally) or open port 6006 on the VM's firewall.

You don't need TensorBoard's UI at all to read the trace, though: `jax.profiler`
writes a plain gzipped Chrome Trace Format JSON under
`<trace-dir>/plugins/profile/<run>/*.trace.json.gz`, which TensorBoard merely
renders. [`scripts/analyze_chess_trace.py`](scripts/analyze_chess_trace.py) parses
it directly and prints accumulated device time per pipeline part plus the top
individual XLA ops — `just prof-scalar-analyze` traces one `Chess.step` call and
runs it in one go, and `just analyze [trace_dir] [--steps N]` runs it alone on an
existing trace (pass `--steps` matching what the trace covered so times come out
per step).
