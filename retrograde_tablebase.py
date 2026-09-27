"""Retrograde, layer by layer, tablebase construction.

Every turn adds exactly one piece to the board: a transfer preserves the number
of occupied cells, the mandatory add increases it by one, and the rotation only
permutes cells. Starting from the empty board that means the number of occupied
cells is also the number of plies played, so the game graph is strictly layered
by occupancy and cannot contain a cycle.

``create_game_tablebase`` already walks those layers from the full board
downwards, but it solves every position from scratch, so each board re-searches
the whole subtree below it. The same waste happens inside a single search: the
transposition table collapses repeated positions but every distinct position is
still expanded once per search. This module instead runs one retrograde sweep
per rule context. The reachable boards are enumerated layer by layer from the
full board downwards and the value of a position is derived from the values of
its children, which the previous layer has already determined. Every
(board, side to move) pair is therefore evaluated exactly once for the whole
context, no matter how many positions refer to it.

Because the ply index is the occupancy, the negamax mate score
``WIN_SCORE - plies_to_terminal`` is a property of the position alone, so no
depth bookkeeping is needed and the distance to the terminal node comes out of
the sweep for free. A position is a draw exactly when none of its children
loses, which the sweep answers without any unknown-value bookkeeping.

Stored format
-------------
One file per (grid size, rotation direction, transfer rule)::

    <base_dir>/<n>x<n>/<rotation>/<transfer rule>/retrograde.npz

It holds two dense arrays indexed by the base-3 code of the row-major board:
``score`` (int16) is the negamax value from the point of view of the side to
move, or ``UNSOLVED`` when the position is not stored, and ``dtx`` (uint8) is
the number of plies to the terminal node. A lookup is a single array index
instead of a scan over a JSON array. The JSON records of a 4x4 grid already
occupy about 4 GB, while the value tables of all six rule contexts together take
roughly 60 MB.

Only positions of games where player 1 made the first move are stored. A game
where player 2 started is the same game with the two colours exchanged, so
:func:`board_index` answers those queries from the colour-swapped board and no
second table is needed. The swap is not unconditional: at an odd ply the
swapped board is not itself a player-1-started position, so the lookup routes
on the ply parity and the side to move rather than on the player alone. The
score needs no negation, because exchanging the colours is an isomorphism that
preserves the value of the player to move.

Positions are enumerated with the children of terminal positions pruned, so the
table only contains boards that can really occur in a game played from the
empty board. Without that pruning the enumeration also produces boards where a
player keeps placing pieces after the opponent has already completed a line,
which the rules score as a draw and which no real game can reach.
"""

import argparse
import ctypes
import json
import multiprocessing as mp
import os
import random
import sys

import numpy as np

import create_game_tablebase as tablebase
import orbital_logic_game_functions as olgf
import solve_game


RETROGRADE_FILENAME = "retrograde.npz"
FORMAT_VERSION = 1
UNSOLVED = np.int16(32767)
WIN_SCORE = int(solve_game.WIN_SCORE)
_NO_DISTANCE = 256
_SWAPPED_CELL = (0, 2, 1)
_ALL_ROTATIONS = tablebase.ROTATION_DIRECTIONS
_TABLE_CACHE_LIMIT = 4
_table_cache = {}


# ---------------------------------------------------------------------------
# Board coding
# ---------------------------------------------------------------------------

class _Codec:
    """Base-3 board codes plus the weights needed to update them incrementally.

    A board is a tuple of ``grid_size ** 2`` cells in row-major order and its
    code is ``sum(cell * 3 ** (cell_count - 1 - index))``. Because a move only
    ever rewrites three cells, the code of a child follows from the code of its
    parent with a handful of additions, which matters because the sweep
    evaluates tens of millions of edges.
    """

    def __init__(self, grid_size, rotate_direction, transfer_allowed):
        grid_size = int(grid_size)
        if grid_size < 1:
            raise ValueError("grid_size must be positive")
        self.grid_size = grid_size
        self.cell_count = grid_size * grid_size
        if self.cell_count > 255:
            raise ValueError(
                "a board may hold at most 255 cells, because the distance to "
                "the terminal node is stored in a single byte"
            )
        params = solve_game._engine_params(grid_size, rotate_direction, transfer_allowed)
        self.lines_first_rest = params[3]
        self.neighbours = params[5]
        self.perm = params[6]
        self.transfer_allowed = bool(transfer_allowed)
        self.size = 3 ** self.cell_count
        self.weights = [3 ** (self.cell_count - 1 - index) for index in range(self.cell_count)]
        if self.perm is None:
            self.destination_weights = self.weights
        else:
            inverse = [0] * self.cell_count
            for index, source in enumerate(self.perm):
                inverse[source] = index
            self.destination_weights = [
                self.weights[inverse[cell]] for cell in range(self.cell_count)
            ]

    def encode(self, board):
        code = 0
        for cell, weight in zip(board, self.weights):
            code += cell * weight
        return code

    def decode(self, code):
        cells = [0] * self.cell_count
        for index in range(self.cell_count - 1, -1, -1):
            code, cells[index] = divmod(code, 3)
        return tuple(cells)

    def swap(self, board):
        return tuple(_SWAPPED_CELL[cell] for cell in board)

    def base_and_destination_weights(self, board):
        """Return the code contribution shared by every child of ``board``.

        A rotation permutes the finished board, so the part of the code that
        comes from untouched cells is the same for all children of a position
        and is computed once here.
        """
        if self.perm is None:
            return self.encode(board), self.destination_weights
        base = 0
        for index, source in enumerate(self.perm):
            base += board[source] * self.weights[index]
        return base, self.destination_weights

    def child_code(self, base, destination_weights, move, player):
        transfer_source, transfer_direction, add_cell = move
        opponent = 3 - player
        code = base + player * destination_weights[add_cell]
        if transfer_source is not None:
            target = self.neighbours[transfer_source][transfer_direction]
            code += opponent * (
                destination_weights[target] - destination_weights[transfer_source]
            )
        return code


def _player_at(ply):
    """Player to move at a given ply when player 1 started from the empty board."""
    return 1 if ply % 2 == 0 else 2


def board_index(codec, board, player_turn):
    """Return the table index that holds the value of a position, or None.

    The table stores games where player 1 started. A game where player 2 started
    is the same game with the colours exchanged, so it is answered from the
    colour-swapped board, where the side to move is exchanged as well. At an odd
    ply the swapped board is not a player-1-started board, so the decision has
    to take the ply parity into account: a position is stored exactly when the
    split of pieces between the two players matches the number of plies played
    and the side to move is the one the parity calls for.
    """
    cells = tuple(int(cell) for cell in board)
    occupied = 0
    ones = 0
    for cell in cells:
        if cell:
            occupied += 1
            if cell == 1:
                ones += 1
    natural = _player_at(occupied)
    player_turn = int(player_turn)
    if ones == (occupied + 1) // 2 and player_turn == natural:
        return codec.encode(cells)
    if ones == occupied // 2 and player_turn == 3 - natural:
        return codec.encode(codec.swap(cells))
    return None


def _queryable_players(board, ply):
    """Return the sides to move a game can actually reach for this board.

    The tables hold games that player 1 started, so the number of pieces and the
    side to move are both fixed by the ply: the side to move is the one the
    parity calls for, and the split has to agree with it. The other side would
    have to be answered from a colour-swapped board, which is a different
    position and is often not one a game reaches.
    """
    ones = int(np.count_nonzero(np.asarray(board) == 1))
    natural = _player_at(ply)
    if ones == (ply + 1) // 2:
        return [natural]
    return []


# ---------------------------------------------------------------------------
# Forward enumeration of the reachable positions
# ---------------------------------------------------------------------------

_enumeration_state = {}


def _init_enumeration_worker(codec):
    _enumeration_state["codec"] = codec


def _expand_codes(task):
    """Return the codes of the children of every position in ``task``."""
    codes, player = task
    codec = _enumeration_state["codec"]
    lines_first_rest = codec.lines_first_rest
    neighbours = codec.neighbours
    transfer_allowed = codec.transfer_allowed
    cell_count = codec.cell_count
    gen_moves = solve_game._gen_moves
    winner_of = solve_game._winner
    child_code = codec.child_code
    decode = codec.decode
    weights_of = codec.base_and_destination_weights
    result = []
    append = result.append
    for code in codes:
        board = decode(int(code))
        if winner_of(board, lines_first_rest) is not None:
            continue
        base, destination_weights = weights_of(board)
        for move in gen_moves(board, player, transfer_allowed, neighbours, cell_count):
            append(child_code(base, destination_weights, move, player))
    return np.asarray(result, dtype=np.int64)


def _pool_context(workers):
    """Return a forking multiprocessing context, or None to stay single process."""
    if not workers or int(workers) < 2:
        return None
    try:
        methods = mp.get_all_start_methods()
    except (AttributeError, ValueError):
        return None
    return mp.get_context("fork") if "fork" in methods else None


def _chunk_size(total, workers):
    if not workers or int(workers) < 2 or total == 0:
        return max(1, total)
    return max(1, min(4096, -(-total // (int(workers) * 8))))


def _chunked(codes, size):
    for start in range(0, len(codes), size):
        yield codes[start:start + size]


def _map_chunks(function, tasks, workers, pool_context, initializer, initargs):
    """Map ``function`` over ``tasks``, serially or on a pool of processes."""
    if pool_context is None:
        initializer(*initargs)
        for task in tasks:
            yield function(task)
        return
    with pool_context.Pool(workers, initializer=initializer, initargs=initargs) as pool:
        for result in pool.imap(function, tasks, chunksize=1):
            yield result


def enumerate_reachable(
    grid_size,
    rotation,
    transfer_allowed,
    workers=None,
    show_progress=True,
):
    """Enumerate the reachable board codes, grouped by number of occupied cells.

    Returns the codec for the context and a list with one sorted array per ply,
    starting with the empty board at index 0. Children of positions where the
    game is already over are pruned, so only boards that can occur in a real
    game are returned.
    """
    codec = _Codec(grid_size, rotation, transfer_allowed)
    pool_context = _pool_context(workers)
    progress = tablebase._make_progress(
        codec.cell_count + 1,
        f"{grid_size}x{grid_size} {rotation} enumeration",
        show_progress,
    )
    layers = [np.zeros(1, dtype=np.int64)]
    try:
        for ply in range(codec.cell_count):
            codes = layers[ply]
            if len(codes) == 0:
                layers.append(np.zeros(0, dtype=np.int64))
                continue
            player = _player_at(ply)
            tasks = [
                (chunk, player)
                for chunk in _chunked(codes, _chunk_size(len(codes), workers))
            ]
            pieces = _map_chunks(
                _expand_codes, tasks, workers, pool_context,
                _init_enumeration_worker, (codec,),
            )
            collected = [piece for piece in pieces if len(piece)]
            if collected:
                layers.append(np.unique(np.concatenate(collected)))
            else:
                layers.append(np.zeros(0, dtype=np.int64))
            if progress is not None:
                progress.update()
    finally:
        if progress is not None:
            progress.close()
    return codec, layers


# ---------------------------------------------------------------------------
# Retrograde sweep
# ---------------------------------------------------------------------------

_sweep_state = {}


def _as_view(source, dtype):
    """Return a numpy view of ``source``, which may already be an array."""
    if isinstance(source, np.ndarray):
        return source
    return np.frombuffer(source, dtype=dtype)


def _init_sweep_worker(codec, score_source, dtx_source):
    _sweep_state["codec"] = codec
    _sweep_state["score"] = _as_view(score_source, np.int16)
    _sweep_state["dtx"] = _as_view(dtx_source, np.uint8)


def _combine_children(children):
    """Fold the values of the children of a position into the value of the position.

    ``children`` yields ``(child_score, child_dtx)`` pairs expressed from the
    point of view of the player to move at the child. The result
    ``(score, dtx, index)`` is expressed from the point of view of the player to
    move at this position, where ``index`` selects the child the rule prefers and
    is -1 when the position has no child at all.

    The mate score of ``solve_game`` counts the plies from the search root, so it
    cannot simply be negated from one ply to the next. Folding the distance to
    the terminal node instead makes the value a property of the position alone,
    which is what allows the sweep to store it once and reuse it everywhere. A
    player takes a win over a draw over a loss, a win ends the game as soon as
    possible, a loss is delayed as long as possible, and the children that are
    not the best are simply ignored.
    """
    win = (_NO_DISTANCE, -1)
    lose = (-1, -1)
    draw = (_NO_DISTANCE, -1)
    for index, (child_score, child_dtx) in enumerate(children):
        if child_score == int(UNSOLVED):
            raise RuntimeError(
                "the retrograde sweep reached a position whose value was not "
                "known yet, so the layers were not swept in a consistent order"
            )
        if child_score < 0:
            if child_dtx < win[0]:
                win = (child_dtx, index)
        elif child_score > 0:
            if child_dtx > lose[0]:
                lose = (child_dtx, index)
        elif child_dtx < draw[0]:
            draw = (child_dtx, index)
    if win[1] >= 0:
        distance = win[0] + 1
        return WIN_SCORE - distance, distance, win[1]
    if draw[1] >= 0:
        return 0, draw[0] + 1, draw[1]
    if lose[1] >= 0:
        distance = lose[0] + 1
        return distance - WIN_SCORE, distance, lose[1]
    return None


def _child_value(base, destination_weights, move, player, score, dtx, codec):
    child = codec.child_code(base, destination_weights, move, player)
    return int(score[child]), int(dtx[child])


def _evaluate_position(code, player, score, dtx, codec):
    """Return the ``(score, dtx)`` of one position from the values of its children."""
    board = codec.decode(code)
    winner = solve_game._winner(board, codec.lines_first_rest)
    if winner is not None:
        if winner == 0:
            return 0, 0
        return (WIN_SCORE, 0) if winner == player else (-WIN_SCORE, 0)

    moves = solve_game._gen_moves(
        board, player, codec.transfer_allowed, codec.neighbours, codec.cell_count,
    )
    if not moves:
        return 0, 0

    base, destination_weights = codec.base_and_destination_weights(board)
    combined = _combine_children(
        _child_value(base, destination_weights, move, player, score, dtx, codec)
        for move in moves
    )
    return combined[0], combined[1]


def _sweep_codes(task):
    """Solve every position in ``task`` and return its ``(score, dtx)`` pair."""
    codes, player = task
    codec = _sweep_state["codec"]
    score = _sweep_state["score"]
    dtx = _sweep_state["dtx"]
    evaluate = _evaluate_position
    scores = np.empty(len(codes), dtype=np.int16)
    distances = np.empty(len(codes), dtype=np.uint8)
    for index in range(len(codes)):
        value, distance = evaluate(int(codes[index]), player, score, dtx, codec)
        scores[index] = value
        distances[index] = distance
    return scores, distances


def sweep_layers(codec, layers, workers=None, show_progress=True):
    """Fill dense value tables for every reachable position, deepest layer first.

    Positions of one layer only refer to the layer below, which is already
    complete, so the layers are independent of each other and the positions
    inside a layer are independent of each other as well. That is what makes
    the sweep both linear in the size of the reachable set and easy to spread
    over several processes.
    """
    pool_context = _pool_context(workers)
    if pool_context is not None:
        score_source = pool_context.Array(ctypes.c_int16, codec.size, lock=False)
        dtx_source = pool_context.Array(ctypes.c_uint8, codec.size, lock=False)
        score = _as_view(score_source, np.int16)
        dtx = _as_view(dtx_source, np.uint8)
        score[:] = UNSOLVED
    else:
        score = np.full(codec.size, UNSOLVED, dtype=np.int16)
        dtx = np.zeros(codec.size, dtype=np.uint8)
        score_source, dtx_source = score, dtx

    progress = tablebase._make_progress(
        len(layers), f"{codec.grid_size}x{codec.grid_size} sweep", show_progress,
    )
    try:
        for ply in range(len(layers) - 1, -1, -1):
            codes = layers[ply]
            if len(codes) == 0:
                continue
            player = _player_at(ply)
            tasks = [
                (chunk, player)
                for chunk in _chunked(codes, _chunk_size(len(codes), workers))
            ]
            results = list(_map_chunks(
                _sweep_codes, tasks, workers, pool_context,
                _init_sweep_worker, (codec, score_source, dtx_source),
            ))
            score[codes] = np.concatenate([item[0] for item in results])
            dtx[codes] = np.concatenate([item[1] for item in results])
            if progress is not None:
                progress.update()
    finally:
        if progress is not None:
            progress.close()
    expected = int(sum(len(codes) for codes in layers))
    solved = int(np.count_nonzero(score != UNSOLVED))
    if solved != expected:
        raise RuntimeError(
            f"the sweep filled {solved} of the {expected} enumerated positions, "
            f"so the stored table cannot be trusted"
        )
    return score, dtx


# ---------------------------------------------------------------------------
# Building and storing
# ---------------------------------------------------------------------------

def context_directory(base_dir, grid_size, rotation, transfer_allowed):
    return os.path.join(
        str(base_dir),
        f"{int(grid_size)}x{int(grid_size)}",
        tablebase._canonical_rotation(rotation),
        tablebase._transfer_name(transfer_allowed),
    )


def context_file(base_dir, grid_size, rotation, transfer_allowed):
    return os.path.join(
        context_directory(base_dir, grid_size, rotation, transfer_allowed),
        RETROGRADE_FILENAME,
    )


def build_context_tablebase(
    base_dir,
    grid_size,
    rotation,
    transfer_allowed,
    workers=None,
    show_progress=True,
):
    """Build and store the value table for a single rule context."""
    rotation = tablebase._canonical_rotation(rotation)
    transfer_allowed = bool(transfer_allowed)
    codec, layers = enumerate_reachable(
        grid_size, rotation, transfer_allowed,
        workers=workers, show_progress=show_progress,
    )
    score, dtx = sweep_layers(
        codec, layers, workers=workers, show_progress=show_progress,
    )
    positions = int(sum(len(codes) for codes in layers))
    metadata = {
        "format_version": FORMAT_VERSION,
        "grid_size": int(grid_size),
        "rotation": rotation,
        "transfer_allowed": transfer_allowed,
        "starting_player": 1,
        "score_convention": "negamax value from the point of view of the side to move",
        "positions": positions,
        "layer_sizes": [int(len(codes)) for codes in layers],
        "table_entries": int(codec.size),
    }
    file_path = context_file(base_dir, grid_size, rotation, transfer_allowed)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savez_compressed(
        file_path,
        score=score,
        dtx=dtx,
        metadata=np.array(json.dumps(metadata)),
    )
    return {
        "file": file_path,
        "grid_size": int(grid_size),
        "rotation": rotation,
        "transfer_allowed": transfer_allowed,
        "positions": positions,
        "layer_sizes": metadata["layer_sizes"],
    }


def build_tablebase(
    base_dir,
    grid_size,
    rotations=_ALL_ROTATIONS,
    transfers=tablebase.TRANSFER_RULES,
    workers=None,
    show_progress=True,
    tablebase_type="compressed",
):
    """Build the value table for every requested rule context of a grid size."""
    grid_size = int(grid_size)
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    if base_dir is None:
        base_dir = tablebase.default_base_dir(tablebase_type)
    contexts = []
    for rotation in _normalise_rotations(rotations):
        for transfer_allowed in _normalise_transfers(transfers):
            contexts.append(build_context_tablebase(
                base_dir,
                grid_size,
                rotation,
                transfer_allowed,
                workers=workers,
                show_progress=show_progress,
            ))
    return {
        "grid_size": grid_size,
        "base_dir": str(base_dir),
        "contexts": contexts,
        "positions": sum(item["positions"] for item in contexts),
        "files": len(contexts),
    }


def _normalise_rotations(rotations):
    if isinstance(rotations, str):
        if rotations.strip().lower() == "all":
            return list(_ALL_ROTATIONS)
        rotations = rotations.split(",")
    result = []
    for rotation in rotations:
        name = tablebase._canonical_rotation(rotation)
        if name not in result:
            result.append(name)
    if not result:
        raise ValueError("at least one rotation direction is required")
    return result


def _normalise_transfers(transfers):
    if isinstance(transfers, str):
        name = transfers.strip().lower()
        if name == "all":
            return list(tablebase.TRANSFER_RULES)
        if name in ("allowed", "true", "yes"):
            return [True]
        if name in ("not_allowed", "not-allowed", "false", "no"):
            return [False]
        raise ValueError(f"unknown transfer selection: {transfers}")
    result = []
    for transfer in transfers:
        value = bool(transfer)
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError("at least one transfer rule is required")
    return result


# ---------------------------------------------------------------------------
# Reading the stored tables
# ---------------------------------------------------------------------------

class ValueTable:
    """A loaded value table for one (grid size, rotation, transfer rule) context."""

    def __init__(self, codec, score, dtx, metadata, file_path):
        self.codec = codec
        self.score = score
        self.dtx = dtx
        self.metadata = metadata
        self.file_path = file_path

    @property
    def grid_size(self):
        return self.codec.grid_size

    def index_of(self, state, player_turn):
        """Return the stored array index for a position, or None."""
        return board_index(
            self.codec,
            tuple(int(cell) for cell in np.asarray(state).ravel()),
            player_turn,
        )

    def entry(self, state, player_turn):
        """Return ``(score, dtx)`` for a position, or None when it is not stored.

        ``score`` is the negamax value from the point of view of ``player_turn``,
        which is the same convention as ``solve_game.solve_game``. ``dtx`` is the
        number of plies between the position and the terminal node.
        """
        code = self.index_of(state, player_turn)
        if code is None:
            return None
        score = int(self.score[code])
        if score == int(UNSOLVED):
            return None
        return score, int(self.dtx[code])


def load_table(base_dir, grid_size, rotation, transfer_allowed):
    """Load a stored value table, or return None when no file is available."""
    rotation = tablebase._canonical_rotation(rotation)
    transfer_allowed = bool(transfer_allowed)
    file_path = context_file(base_dir, grid_size, rotation, transfer_allowed)
    try:
        stat = os.stat(file_path)
    except OSError:
        return None
    key = (file_path, stat.st_mtime_ns, stat.st_size)
    table = _table_cache.get(key)
    if table is not None:
        return table
    try:
        with np.load(file_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            score = np.ascontiguousarray(data["score"])
            dtx = np.ascontiguousarray(data["dtx"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    if metadata.get("format_version") != FORMAT_VERSION:
        return None
    codec = _Codec(grid_size, rotation, transfer_allowed)
    if score.size != codec.size or dtx.size != codec.size:
        return None
    while len(_table_cache) >= _TABLE_CACHE_LIMIT:
        _table_cache.pop(next(iter(_table_cache)))
    _table_cache[key] = table = ValueTable(codec, score, dtx, metadata, file_path)
    return table


def available_contexts(base_dir, grid_size):
    """Return the ``(rotation, transfer_allowed)`` pairs stored for a grid size."""
    grid_folder = os.path.join(str(base_dir), f"{int(grid_size)}x{int(grid_size)}")
    found = []
    try:
        rotations = sorted(os.listdir(grid_folder))
    except OSError:
        return found
    for rotation in rotations:
        rotation_folder = os.path.join(grid_folder, rotation)
        if not os.path.isdir(rotation_folder):
            continue
        try:
            transfers = sorted(os.listdir(rotation_folder))
        except OSError:
            continue
        for transfer in transfers:
            if not os.path.exists(os.path.join(rotation_folder, transfer, RETROGRADE_FILENAME)):
                continue
            found.append((rotation, transfer == tablebase._transfer_name(True)))
    return found


def principal_variation(table, state, player_turn, rotation):
    """Rebuild the perfect-play line of a position from a stored value table.

    The tables only hold values, so the line is recovered by walking down the
    layers and letting :func:`_combine_children` pick the child at every ply.
    That is the same rule the sweep uses, so the reconstructed line is the line
    the stored values were built from.
    """
    codec = table.codec
    grid_size = codec.grid_size
    cells = tuple(int(cell) for cell in np.asarray(state).ravel())
    player = int(player_turn)
    states_sequence = [np.asarray(state, dtype=int).reshape((grid_size, grid_size))]
    moves_sequence = []
    score_of = table.score
    dtx_of = table.dtx
    for _ in range(codec.cell_count):
        if solve_game._winner(cells, codec.lines_first_rest) is not None:
            break
        children = []
        for move in solve_game._gen_moves(
            cells, player, codec.transfer_allowed, codec.neighbours, codec.cell_count,
        ):
            child = board_index(
                codec,
                solve_game._apply_move(cells, move, player, codec.perm, codec.neighbours),
                3 - player,
            )
            if child is None:
                continue
            child_score = int(score_of[child])
            if child_score == int(UNSOLVED):
                continue
            children.append((child_score, int(dtx_of[child]), move))
        if not children:
            break
        combined = _combine_children(
            (child_score, child_dtx) for child_score, child_dtx, _ in children
        )
        if combined is None:
            break
        best_move = children[combined[2]][2]
        moves_sequence.append(solve_game._move_to_dict(best_move, player, rotation, grid_size))
        cells = solve_game._apply_move(
            cells, best_move, player, codec.perm, codec.neighbours,
        )
        states_sequence.append(np.asarray(cells, dtype=int).reshape((grid_size, grid_size)))
        player = 3 - player
    return moves_sequence, states_sequence


def build_record(base_dir, state, player_turn, rotate_direction, transfer_allowed):
    """Return a tablebase record shaped like the JSON ones, or None.

    ``get_solution`` consumes records with a fixed layout, so the value tables
    are exposed through the same structure. That keeps the JSON files as a
    fallback rather than turning them into a second, divergent format.
    """
    state = tablebase._as_state(state)
    grid_size = state.shape[0]
    rotation = tablebase._canonical_rotation(rotate_direction)
    table = load_table(base_dir, grid_size, rotation, transfer_allowed)
    if table is None:
        return None
    entry = table.entry(state, player_turn)
    if entry is None:
        return None
    score = entry[0]
    if score > 0:
        game_result = f"player{int(player_turn)}_wins"
    elif score < 0:
        game_result = f"player{3 - int(player_turn)}_wins"
    else:
        game_result = "draw"
    moves_sequence, states_sequence = principal_variation(
        table, state, player_turn, rotation,
    )
    return {
        "id": tablebase.get_position_id(state, rotation, transfer_allowed, player_turn),
        "position": state.tolist(),
        "solution": {
            "score": score,
            "game_result": game_result,
            "best_move": moves_sequence[0] if moves_sequence else None,
            "moves_sequence": moves_sequence,
            "states_sequence": [item.tolist() for item in states_sequence],
        },
        "source": RETROGRADE_FILENAME,
    }


# ---------------------------------------------------------------------------
# Verification against the independent search
# ---------------------------------------------------------------------------

def _random_position(codec, ply, rng):
    """Return a random board with exactly ``ply`` occupied cells."""
    board = [0] * codec.cell_count
    for cell in rng.sample(range(codec.cell_count), ply):
        board[cell] = rng.choice((1, 2))
    return np.asarray(board, dtype=int).reshape((codec.grid_size, codec.grid_size))


def _sample_stored_positions(codec, score, ply, rng, wanted):
    """Return up to ``wanted`` random positions of ``ply`` that the table stores.

    The samples are drawn from the table itself rather than generated, because a
    board with a legal piece split can still be unreachable in the rule context,
    and such a board is correctly absent. Only positions the table stores are
    therefore the ones whose value can be cross-checked.
    """
    found = []
    attempts = 0
    budget = max(512, 512 * wanted)
    while len(found) < wanted and attempts < budget:
        attempts += 1
        code = int(rng.randrange(codec.size))
        if int(score[code]) == int(UNSOLVED):
            continue
        cells = codec.decode(code)
        if sum(1 for cell in cells if cell) != ply:
            continue
        found.append(
            np.asarray(cells, dtype=int).reshape((codec.grid_size, codec.grid_size))
        )
    return found


def verify_context(
    base_dir,
    grid_size,
    rotation,
    transfer_allowed,
    samples_per_layer=4,
    seed=0,
    show_progress=True,
):
    """Cross-check the stored values of one context against ``solve_game``.

    The samples are drawn from the stored positions themselves, so they are all
    positions the context really covers, and each one is solved again from
    scratch with the search engine and compared with the stored value. Boards no
    game can produce are generated as well and must be absent from the table.
    Each reconstructed principal variation is also replayed through
    ``play_turn`` to confirm that the line is legal, that it has the length the
    stored distance claims, and that it ends the game the way the score predicts.
    """
    rotation = tablebase._canonical_rotation(rotation)
    transfer_allowed = bool(transfer_allowed)
    table = load_table(base_dir, grid_size, rotation, transfer_allowed)
    if table is None:
        raise FileNotFoundError(
            f"no stored table for {grid_size}x{grid_size} {rotation} "
            f"transfer={transfer_allowed} in {base_dir}"
        )
    rng = random.Random(seed)
    codec = table.codec
    lines_first_rest = codec.lines_first_rest
    progress = tablebase._make_progress(
        codec.cell_count + 1,
        f"verifying {rotation} transfer={transfer_allowed}",
        show_progress,
    )
    checked = 0
    problems = []
    try:
        for ply in range(codec.cell_count + 1):
            for board in _sample_stored_positions(
                codec, table.score, ply, rng, samples_per_layer,
            ):
                for player in _queryable_players(board, ply):
                    entry = table.entry(board, player)
                    if entry is None:
                        problems.append(
                            f"ply {ply}: a stored position is not reachable through the "
                            f"lookup index for player {player}: {board.tolist()}"
                        )
                        continue
                    score, dtx = entry
                    result = solve_game.solve_game(board, rotation, transfer_allowed, player)
                    if int(result["score"]) != score:
                        problems.append(
                            f"ply {ply}: search says {result['score']}, table says "
                            f"{score} for player {player} on {board.tolist()}"
                        )
                        continue
                    moves, _states = principal_variation(table, board, player, rotation)
                    if len(moves) != dtx:
                        problems.append(
                            f"ply {ply}: the reconstructed line has {len(moves)} plies "
                            f"but the stored distance is {dtx} for {board.tolist()}"
                        )
                        continue
                    problem = _replay(board, moves, score, player, ply)
                    if problem is not None:
                        problems.append(problem)
                    checked += 1
            for _ in range(samples_per_layer):
                board = _random_position(codec, ply, rng)
                cells = tuple(int(cell) for cell in board.ravel())
                ones = sum(1 for cell in cells if cell == 1)
                if ones == (ply + 1) // 2:
                    continue
                if int(table.score[codec.encode(cells)]) != int(UNSOLVED):
                    problems.append(
                        f"ply {ply}: a board no game can reach is stored: {board.tolist()}"
                    )
            if progress is not None:
                progress.update()
    finally:
        if progress is not None:
            progress.close()
    return {
        "grid_size": int(grid_size),
        "rotation": rotation,
        "transfer_allowed": transfer_allowed,
        "checked": checked,
        "problems": problems,
    }


def _replay(board, moves, score, player_turn, ply):
    """Play a reconstructed line through the rules and check that it fits."""
    current = np.asarray(board, dtype=int).copy()
    for number, move in enumerate(moves):
        before = current.copy()
        try:
            current = np.asarray(olgf.play_turn(current, move), dtype=int)
        except (KeyError, TypeError, ValueError, IndexError) as error:
            return f"ply {ply}: the reconstructed line was rejected on move {number + 1}: {error}"
        if current.shape != before.shape:
            return f"ply {ply}: the reconstructed line changed the board shape"
    finished = olgf.evaluate_game_state(current)
    won = score > 0 and finished == player_turn
    lost = score < 0 and finished == 3 - player_turn
    drew = score == 0 and finished in (None, 0)
    if won or lost or drew:
        return None
    return (
        f"ply {ply}: the line ends with {finished}, which does not match the "
        f"score {score} for player {player_turn} on {board.tolist()}"
    )


def verify_tablebase(
    base_dir,
    grid_size,
    rotations=_ALL_ROTATIONS,
    transfers=tablebase.TRANSFER_RULES,
    samples_per_layer=4,
    seed=0,
    show_progress=True,
):
    """Run :func:`verify_context` for every requested rule context."""
    results = []
    for rotation in _normalise_rotations(rotations):
        for transfer_allowed in _normalise_transfers(transfers):
            results.append(verify_context(
                base_dir,
                grid_size,
                rotation,
                transfer_allowed,
                samples_per_layer=samples_per_layer,
                seed=seed,
                show_progress=show_progress,
            ))
    return {
        "grid_size": int(grid_size),
        "checked": sum(item["checked"] for item in results),
        "problems": [
            problem
            for item in results
            for problem in item["problems"]
        ],
        "contexts": results,
    }


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a retrograde tablebase for the orbital logic game."
    )
    parser.add_argument(
        "-n", "--grid-size", type=int, default=2,
        help="size of the square grid (default: 2)",
    )
    parser.add_argument(
        "--rotations", default="all",
        help="comma separated rotation directions, or 'all' (default: all)",
    )
    parser.add_argument(
        "--transfers", default="all",
        help="'all', 'allowed' or 'not_allowed' (default: all)",
    )
    parser.add_argument(
        "--base-dir", default=None,
        help="directory in which to save the tables (default: the selected tablebase directory)",
    )
    parser.add_argument(
        "--tablebase", "--tablebase-type", "--tablebase_type",
        dest="tablebase_type", choices=tablebase.TABLEBASE_TYPES, default="compressed",
        help="only used to pick the default directory (default: compressed)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="number of worker processes used for each layer",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="disable terminal progress output",
    )
    parser.add_argument(
        "--verify", type=int, default=0, metavar="SAMPLES",
        help="cross-check the stored values against solve_game with SAMPLES "
             "random positions per layer and context",
    )
    args = parser.parse_args(argv)
    if args.grid_size < 1:
        parser.error("grid size must be positive")

    base_dir = args.base_dir
    if base_dir is None:
        base_dir = tablebase.default_base_dir(args.tablebase_type)
    result = build_tablebase(
        base_dir,
        args.grid_size,
        rotations=args.rotations,
        transfers=args.transfers,
        workers=args.workers,
        show_progress=not args.no_progress,
        tablebase_type=args.tablebase_type,
    )
    for context in result["contexts"]:
        print(
            f"{context['rotation']} transfer={context['transfer_allowed']}: "
            f"{context['positions']} reachable positions in {context['file']}"
        )
    print(
        f"Saved {result['positions']} positions to {result['files']} files in the "
        f"{result['grid_size']}x{result['grid_size']} retrograde tablebase."
    )

    if args.verify:
        report = verify_tablebase(
            base_dir,
            args.grid_size,
            rotations=args.rotations,
            transfers=args.transfers,
            samples_per_layer=args.verify,
            show_progress=not args.no_progress,
        )
        for problem in report["problems"]:
            print(f"MISMATCH: {problem}", file=sys.stderr)
        print(f"Verified {report['checked']} positions against solve_game.")
        if report["problems"]:
            return 1
    return 0


if __name__ == "__main__":
    main()
