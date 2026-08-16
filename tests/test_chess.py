"""Tests for the single-source ``pgx1.chess`` environment."""

import ast
import hashlib
import inspect

import jax
import jax.numpy as jnp
import numpy as np

import pgx1.chess as chess


_GEOMETRY_SHA256 = {
    "BETWEEN_BY_SOURCE": "94f594401da405e222ed795af788368327be24a676eaa6b3d3aaaf32cda60fc9",
    "SLIDER_GEOMETRY": "4c66f67f291d0c8e0a7f9766814e56d8c749a6c04cfc083752d19a36d84856b3",
    "KNIGHT_DEST": "d4a7d96dbf96b5aca260047822dbd5b7434ad31c9d2702cc408aebb8bebb517a",
    "KING_DEST": "600978d64a3560c5685bc878eef3e34c7d21cbd2e92cfe2e914908d9a6716f99",
    "ENEMY_PAWN_ATTACK_DEST": "46b69932d69edc6b8d68fac2423bed4aafa49658953822c8bca944cb8afb9f3b",
    "FRIEND_PAWN_CAPTURE_DEST": "17c7d6bc31446f7f21a4f16c45bd3a5c49577be3c808451d2cc831cc31bb9b9f",
    "FRIEND_PAWN_PUSH1_DEST": "16232dd0c04ddc6942a26b6e4780d48b84b6548dcd64c827b0caa74bbdf353f7",
    "MOVE_GEOMETRY": "ebb245b2a833f899fe74b23561da13b525a7111ed823fcf732aa369f5e630149",
    "PIN_MASK_BY_SOURCE": "b26bc73f273ec3aca38793ed0174a75dabfa6aecb42a6cb583b68ce8b8fb1f06",
    "PLANE_PERM": "8e2e0e31b2b1fdc3f96314427d8d35ebe3c427e3cb02c0ec556084cc2153eab0",
    "PLANE_INDEX": "7f291217d951f1d66188bd6d2902185d35e3e9302beee18f16833a84a1aecc13",
    "UNDERPROMOTION_ORDINARY_PLANES": "91bac13cd0422475bf76a88178b7cbe7566a1b53ff22e903c2551b45f726440e",
}


def _assert_core_equal(left, right):
    for field in (
        "current_player",
        "rewards",
        "terminated",
        "truncated",
        "legal_action_mask",
        "legal_action_bitmask",
        "observation",
        "_step_count",
        "_player_order",
    ):
        left_value = getattr(left, field)
        right_value = getattr(right, field)
        if left_value is None or right_value is None:
            assert left_value is right_value
        else:
            np.testing.assert_array_equal(left_value, right_value, err_msg=field)
    for field in left._x._fields:
        np.testing.assert_array_equal(
            getattr(left._x, field), getattr(right._x, field), err_msg=f"_x.{field}"
        )


def test_chess_geometry_tables_are_stable():
    for name, expected in _GEOMETRY_SHA256.items():
        table = np.asarray(getattr(chess, name))
        assert hashlib.sha256(table.tobytes()).hexdigest() == expected, name


def test_chess_is_one_clean_implementation_file():
    source = inspect.getsource(chess)
    imports = [
        node
        for node in ast.parse(source).body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported_modules = {
        alias.name
        for node in imports
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in imports
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(module.startswith("pgx1") for module in imported_modules)
    assert "custom_vmap" not in source
    assert "custom_batching" not in source
    assert "_xla" not in source
    assert chess.__all__ == [
        "Action",
        "Chess",
        "Game",
        "GameState",
        "State",
        "INITIAL_LEGAL_ACTION_BITMASK",
        "INITIAL_LEGAL_ACTION_MASK",
        "legal_action_mask",
        "legal_action_mask_batch",
        "legal_action_bitmask",
        "legal_action_bitmask_batch",
        "pack_mask",
        "unpack_bitmask",
        "bitmask_has_action",
    ]


def test_chess_native_vmap_keeps_between_table_shared():
    batch_size = 2
    boards = jnp.broadcast_to(chess.INIT_BOARD.astype(jnp.int32), (batch_size, 64))
    en_passant = jnp.full((batch_size,), -1, jnp.int32)
    rights = jnp.ones((batch_size, 2, 2), jnp.bool_)

    closed = jax.make_jaxpr(jax.vmap(chess.legal_action_mask))(
        boards, en_passant, rights
    )
    operand_shapes = {
        variable.aval.shape
        for variable in (*closed.jaxpr.constvars, *closed.jaxpr.invars)
        if hasattr(variable.aval, "shape")
    }
    assert (64, 128, 128) in operand_shapes
    assert (batch_size, 64, 128, 128) not in operand_shapes
    assert closed.out_avals[0].shape == (batch_size, chess.NUM_ACTIONS)


def test_chess_legal_metadata_distinguishes_checkmate_and_stalemate():
    cases = (
        ({0: chess.KING, 7: -chess.ROOK}, True, False),
        ({0: chess.KING, 63: -chess.BISHOP}, True, False),
        ({0: chess.KING, 10: -chess.KNIGHT}, True, False),
        ({0: chess.KING, 9: -chess.PAWN}, True, False),
        ({0: chess.KING, 9: -chess.KING}, True, False),
        ({0: chess.KING, 1: chess.PAWN, 7: -chess.ROOK}, False, False),
        ({0: chess.KING, 9: -chess.QUEEN, 18: -chess.KING}, True, True),
        ({0: chess.KING, 10: -chess.QUEEN, 17: -chess.KING}, False, True),
    )
    rights = jnp.zeros((2, 2), dtype=jnp.bool_)
    game = chess.Game()

    for pieces, expected_check, no_legal_moves in cases:
        board = jnp.zeros(64, dtype=jnp.int32)
        for square, piece in pieces.items():
            board = board.at[square].set(piece)
        legal = chess._legal_action_result(board, jnp.int32(-1), rights)
        state = chess.GameState(
            board=board.astype(jnp.int8),
            castling_rights=rights,
        )
        assert bool(legal.in_check) is expected_check
        assert bool(chess._is_in_check(board)) is expected_check
        assert bool(~legal.mask.any()) is no_legal_moves
        rewards = game.rewards(state, legal.mask)
        if no_legal_moves and expected_check:
            np.testing.assert_array_equal(rewards, jnp.float32([-1.0, 1.0]))
        else:
            np.testing.assert_array_equal(rewards, jnp.zeros(2, jnp.float32))


def test_chess_rewards_fallback_uses_targeted_check_query():
    game = chess.Game()
    state = chess.GameState()
    mask = chess.INITIAL_LEGAL_ACTION_MASK

    closed = jax.make_jaxpr(game.rewards)(state, mask)
    operand_shapes = {
        variable.aval.shape
        for variable in (*closed.jaxpr.constvars, *closed.jaxpr.invars)
        if hasattr(variable.aval, "shape")
    }

    assert (
        64,
        chess.PADDED_SQUARES,
        chess.PADDED_SQUARES,
    ) not in operand_shapes
    np.testing.assert_array_equal(game.rewards(state, mask), jnp.zeros(2, jnp.float32))


def test_chess_combined_outcome_covers_all_termination_rules():
    game = chess.Game()
    full_mask = jnp.ones(chess.NUM_ACTIONS, dtype=jnp.bool_)
    base = chess.GameState()

    assert not bool(game.is_terminal(base, full_mask))

    fifty_move = base._replace(halfmove_count=jnp.int16(100))
    assert bool(game.is_terminal(fifty_move, full_mask))

    bare_kings = jnp.zeros(64, dtype=jnp.int8)
    bare_kings = bare_kings.at[0].set(chess.KING)
    bare_kings = bare_kings.at[63].set(-chess.KING)
    insufficient = base._replace(board=bare_kings)
    assert bool(game.is_terminal(insufficient, full_mask))

    position_hash = chess._zobrist_hash(base)
    repeated_history = base.hash_history.at[:3].set(position_hash)
    repeated = base._replace(hash_history=repeated_history)
    assert bool(game.is_terminal(repeated, full_mask))

    max_steps = base._replace(step_count=jnp.int32(chess.MAX_TERMINATION_STEPS))
    assert bool(game.is_terminal(max_steps, full_mask))
    assert not bool(chess.Game(auto_terminate=False).is_terminal(max_steps, full_mask))

    internal = game._outcome(base, full_mask.any(), jnp.bool_(False))
    np.testing.assert_array_equal(
        internal.terminated,
        game.is_terminal(base, full_mask),
    )
    np.testing.assert_array_equal(
        internal.rewards,
        game.rewards(base, full_mask, jnp.bool_(False)),
    )


def test_chess_dense_and_packed_batch_wrappers_match():
    board = jnp.broadcast_to(chess.INIT_BOARD.astype(jnp.int32), (2, 64))
    en_passant = jnp.full((2,), -1, jnp.int32)
    rights = jnp.ones((2, 2, 2), jnp.bool_)
    dense = chess.legal_action_mask_batch(board, en_passant, rights)
    packed = chess.legal_action_bitmask_batch(board, en_passant, rights)
    np.testing.assert_array_equal(dense, chess.unpack_bitmask(packed))


def test_chess_replay_preserves_terminal_mask_semantics():
    env = chess.Chess(use_bitmask=True, return_observation=False)
    num_games, plies = 2, 4
    keys = jax.random.split(jax.random.PRNGKey(52), num_games)
    initial = jax.vmap(env.init)(keys)
    state = initial
    step = jax.jit(env.step_batch)
    actions = []

    for _ in range(plies):
        masks = np.asarray(chess.unpack_bitmask(state.legal_action_bitmask))
        action = jnp.asarray(
            [np.flatnonzero(mask)[0] for mask in masks],
            dtype=jnp.int32,
        )
        actions.append(action)
        state = step(state, action)

    actions = jnp.stack(actions, axis=1)
    replayed = jax.jit(env.replay_batch)(
        initial,
        actions[:, :-1],
        actions[:, -1],
    )
    _assert_core_equal(replayed, state)

    replayed_proven = jax.jit(
        lambda s, a, f: env.replay_batch(
            s,
            a,
            f,
            final_action_is_legal=True,
        )
    )(initial, actions[:, :-1], actions[:, -1])
    _assert_core_equal(replayed_proven, state)

    padded = jnp.concatenate(
        [
            actions[:, :-1],
            jnp.full((num_games, 2), -1, dtype=jnp.int32),
        ],
        axis=1,
    )
    replayed_padded = jax.jit(env.replay_batch)(
        initial,
        padded,
        actions[:, -1],
    )
    _assert_core_equal(replayed_padded, state)

    fixed_capacity = jnp.concatenate(
        [
            actions[:, :-1],
            jnp.zeros((num_games, 2), dtype=jnp.int32),
        ],
        axis=1,
    )
    replayed_at_depth = jax.jit(
        lambda s, a, f, depth: env.replay_batch(
            s,
            a,
            f,
            replay_depth=depth,
        )
    )(
        initial,
        fixed_capacity,
        actions[:, -1],
        jnp.asarray(plies - 1, dtype=jnp.int32),
    )
    _assert_core_equal(replayed_at_depth, state)

    terminal = state.replace(
        terminated=jnp.ones(num_games, dtype=jnp.bool_),
        rewards=jnp.tile(jnp.float32([[1.0, -1.0]]), (num_games, 1)),
        legal_action_bitmask=jnp.tile(
            chess.FULL_LEGAL_ACTION_BITMASK[None],
            (num_games, 1),
        ),
    )
    final_action = jnp.zeros(num_games, dtype=jnp.int32)
    expected_noop = step(terminal, final_action)
    already_terminal = jnp.ones(num_games, dtype=jnp.bool_)

    replayed_terminal = jax.jit(env.replay_batch)(
        initial,
        actions,
        final_action,
        already_terminal,
    )
    _assert_core_equal(replayed_terminal, expected_noop)

    replayed_terminal_proven = jax.jit(
        lambda s, a, f, terminal: env.replay_batch(
            s,
            a,
            f,
            terminal,
            final_action_is_legal=True,
        )
    )(initial, actions, final_action, already_terminal)
    _assert_core_equal(replayed_terminal_proven, expected_noop)
