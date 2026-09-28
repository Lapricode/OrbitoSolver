import argparse
import contextlib
import io
import json
import math
import os

import numpy as np

import create_game_tablebase as tablebase
import orbital_logic_game_functions as olgf
import retrograde_tablebase
import solve_game


POSITION_NOT_FOUND = "Position is not in the tablebase."
_DEFAULT_PLAYER_SYMBOLS = {0: "_", 1: "x", 2: "o"}
_records_cache = {}   # (path, mtime, size) -> list of records
_score_index = {}     # (path, mtime, size) -> {position id: score}
MAX_CACHED_SCORES = 64   # a score map is tiny, so many files may stay around
_ROTATION_ALIASES = {
    "clockwise": "clockwise",
    "cw": "clockwise",
    "-": "clockwise",
    "-1": "clockwise",
    "counterclockwise": "counterclockwise",
    "ccw": "counterclockwise",
    "+": "counterclockwise",
    "1": "counterclockwise",
    "still": "still",
    "s": "still",
    "x": "still",
    "0": "still",
}


def _state_from_input(state, grid_size=None):
    if isinstance(state, str):
        text = state.strip()
        if not text:
            raise ValueError("state must not be empty")
        try:
            values = [int(cell) for cell in text]
        except ValueError as error:
            raise ValueError("state must contain only the digits 0, 1, and 2") from error
    else:
        try:
            values = [int(cell) for cell in np.asarray(state).ravel()]
        except (TypeError, ValueError) as error:
            raise ValueError("state must contain only the values 0, 1, and 2") from error

    if not values:
        raise ValueError("state must not be empty")
    if any(value not in (0, 1, 2) for value in values):
        raise ValueError("state must contain only the values 0, 1, and 2")

    if grid_size is None:
        grid_size = math.isqrt(len(values))
    else:
        try:
            grid_size = int(grid_size)
        except (TypeError, ValueError) as error:
            raise ValueError("grid_size must be an integer") from error

    if grid_size < 1 or grid_size * grid_size != len(values):
        raise ValueError("state length must be a perfect square matching grid_size")

    return np.asarray(values, dtype=int).reshape((grid_size, grid_size))


def _normalise_rotation(rotate_direction):
    rotation = str(rotate_direction).lower()
    if rotation not in _ROTATION_ALIASES:
        choices = ", ".join(tablebase.ROTATION_DIRECTIONS)
        raise ValueError(f"rotation must be one of: {choices}")
    return _ROTATION_ALIASES[rotation]


def _normalise_player(player_turn):
    try:
        player = int(player_turn)
    except (TypeError, ValueError) as error:
        raise ValueError("player_turn must be 1 or 2") from error
    if player not in (1, 2):
        raise ValueError("player_turn must be 1 or 2")
    return player


def _normalise_transfer_allowed(transfer_allowed):
    if isinstance(transfer_allowed, str):
        value = transfer_allowed.strip().lower()
        if value in {"true", "1", "yes", "allowed"}:
            return True
        if value in {"false", "0", "no", "not_allowed"}:
            return False
        raise ValueError("transfer_allowed must be true or false")
    return bool(transfer_allowed)


def _record_path(base_dir, state, rotate_direction, transfer_allowed, player_turn):
    grid_size = state.shape[0]
    transfer_folder = "transfer_allowed" if transfer_allowed else "transfer_not_allowed"
    player_folder = "player1" if player_turn == 1 else "player2"
    completion = int(np.count_nonzero(state))
    return os.path.join(
        str(base_dir),
        f"{grid_size}x{grid_size}",
        rotate_direction,
        transfer_folder,
        player_folder,
        f"completion_{completion}.json",
    )


def _cache_key(file_path):
    """The cache key of a tablebase file, or None when it cannot be read.

    The key includes the file modification time and size, so a regenerated
    tablebase is picked up without restarting the program.
    """
    try:
        stat = os.stat(file_path)
    except OSError:
        return None
    return file_path, stat.st_mtime, stat.st_size


def _read_records(file_path):
    """Read the JSON array stored in ``file_path``, or None."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            records = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    return records if isinstance(records, list) else None


def _load_records(file_path):
    """Read (and cache) the JSON array stored in ``file_path``, with its index.

    Returns the ``(records, by_id)`` pair, where ``by_id`` maps every position
    id to its record, so looking a position up does not have to walk the whole
    array. The array itself is large, so only the file read last is kept.
    """
    key = _cache_key(file_path)
    if key is None:
        return None
    loaded = _records_cache.get(key)
    if loaded is None:
        records = _read_records(file_path)
        if records is None:
            return None
        by_id = {
            record["id"]: record for record in records
            if isinstance(record, dict) and isinstance(record.get("id"), str)
        }
        _records_cache.clear()
        _records_cache[key] = (records, by_id)
    return _records_cache[key]


def _scores_by_id(file_path, key):
    """The ``{position id: score}`` map of a tablebase file, cached.

    A stored score is all :func:`lookup_score` needs, and it is small enough to
    keep for a good number of files, which matters because one evaluation walks
    through every completion level of the game tree.
    """
    scores = _score_index.get(key)
    if scores is None:
        records = _read_records(file_path)
        if records is None:
            return None
        scores = {
            record["id"]: record["solution"].get("score") for record in records
            if isinstance(record, dict)
            and isinstance(record.get("id"), str)
            and isinstance(record.get("solution"), dict)
            and record["solution"].get("score") is not None
        }
        _score_index[key] = scores
        while len(_score_index) > MAX_CACHED_SCORES:
            _score_index.pop(next(iter(_score_index)))
    return scores


def _find_record(base_dir, state, rotate_direction, transfer_allowed, player_turn):
    """Return the JSON tablebase record for a position, or None.

    Only reached for positions the retrograde value tables do not cover, either
    because no table is stored for the rule context or because the position is
    not one a game played from the empty board can produce.
    """
    file_path = _record_path(
        base_dir,
        state,
        rotate_direction,
        transfer_allowed,
        player_turn,
    )
    loaded = _load_records(file_path)
    if loaded is None:
        return None
    records, by_id = loaded

    position_id = tablebase.get_position_id(
        state,
        rotate_direction,
        transfer_allowed,
        player_turn,
    )
    record = by_id.get(position_id)
    if record is not None:
        return record

    for record in records:
        if not isinstance(record, dict) or "position" not in record:
            continue
        try:
            record_state = np.asarray(record["position"], dtype=int)
        except (TypeError, ValueError):
            continue
        if record_state.shape == state.shape and np.array_equal(record_state, state):
            return record
    return None


def _format_state(state, players_symbols):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        olgf.print_game_state(np.asarray(state, dtype=int), players_symbols)
    return output.getvalue().rstrip()


def _move_for_engine(move):
    engine_move = dict(move)
    if engine_move.get("add") is not None:
        engine_move["add"] = tuple(engine_move["add"])
    if engine_move.get("transfer") is not None:
        engine_move["transfer"] = [tuple(engine_move["transfer"][0]), engine_move["transfer"][1]]
    return engine_move


def _move_for_print(move):
    printable = {}
    for key, value in move.items():
        if key == "add" and value is not None:
            value = tuple(value)
        elif key == "transfer" and value is not None:
            value = [tuple(value[0]), value[1]]
        printable[key] = value
    return printable


def _state_sequence(record, state, moves_sequence):
    stored_states = record.get("solution", {}).get("states_sequence")
    if stored_states:
        return [np.asarray(item, dtype=int) for item in stored_states]

    states = [state.copy()]
    current_state = state.copy()
    for move in moves_sequence:
        try:
            current_state = olgf.play_turn(current_state, _move_for_engine(move))
        except (KeyError, TypeError, ValueError, IndexError):
            break
        states.append(np.asarray(current_state, dtype=int))
    return states


def _swap_game_result(game_result):
    if not isinstance(game_result, str):
        return game_result
    return {
        "player1_wins": "player2_wins",
        "player2_wins": "player1_wins",
    }.get(game_result, game_result)


def _map_compressed_record(record, representation, state, rotate_direction):
    solution = record.get("solution", {})
    stored_moves = solution.get("moves_sequence")
    if stored_moves is None:
        best_move = solution.get("best_move")
        stored_moves = [] if best_move is None else [best_move]

    symmetry = representation["symmetry"]
    swapped = representation["swapped"]
    grid_size = state.shape[0]
    mapped_solution = dict(solution)
    mapped_solution["moves_sequence"] = [
        tablebase.map_compressed_move(
            move,
            symmetry,
            grid_size,
            rotate_direction,
            swapped,
        )
        for move in stored_moves
    ]
    if solution.get("states_sequence"):
        mapped_solution["states_sequence"] = [
            tablebase.map_compressed_state(item, symmetry, swapped)
            for item in solution["states_sequence"]
        ]
    elif "states_sequence" in solution:
        mapped_solution["states_sequence"] = []
    if solution.get("best_move") is not None:
        mapped_solution["best_move"] = tablebase.map_compressed_move(
            solution["best_move"],
            symmetry,
            grid_size,
            rotate_direction,
            swapped,
        )
    if swapped:
        mapped_solution["game_result"] = _swap_game_result(solution.get("game_result"))

    mapped_record = dict(record)
    mapped_record["position"] = np.asarray(state, dtype=int).tolist()
    mapped_record["solution"] = mapped_solution
    return mapped_record


def _format_solution(
    record,
    state,
    player_turn,
    rotate_direction,
    transfer_allowed,
    players_symbols,
):
    solution = record.get("solution", {})
    moves_sequence = solution.get("moves_sequence")
    if moves_sequence is None:
        best_move = solution.get("best_move")
        moves_sequence = [] if best_move is None else [best_move]
    states_sequence = _state_sequence(record, state, moves_sequence)

    lines = [
        "Game rules:",
        f"  - Board size:          {state.shape[0]}x{state.shape[0]}",
        f"  - Players:             {players_symbols[1]} and {players_symbols[2]}",
        f"  - Rotation direction:  {rotate_direction}",
        f"  - Transfers allowed:   {transfer_allowed}",
    ]
    identifier = record.get("id")
    if identifier:
        # the ID is what the tablebase files are keyed by, so the report can be
        # traced back to the entry it came from
        lines.append(f"  - Position ID:         {identifier}")
    lines.extend([
        "",
        "Initial game state:",
        "",
        _format_state(state, players_symbols),
        "",
        f"Start player: {players_symbols[player_turn]}",
        "",
    ])

    score = solution.get("score")
    if score is None:
        result_message = solution.get("game_result", "Solution result unavailable")
    else:
        result_message = solve_game.estimated_game_result(
            score,
            players_symbols,
            player_turn,
        )
    lines.extend([result_message, "", "Perfect game evolution:", ""])

    for move_number, current_state in enumerate(states_sequence):
        lines.append(f"State after move {move_number}:")
        lines.append(_format_state(current_state, players_symbols))
        if move_number < len(moves_sequence):
            move = _move_for_print(moves_sequence[move_number])
            move_player = moves_sequence[move_number].get("player")
            if move_player not in (1, 2):
                move_player = player_turn if move_number == 0 else 3 - player_turn
            lines.extend([
                "",
                f"Move {move_number + 1} ({players_symbols[move_player]}) : {move}",
            ])

    return "\n".join(lines).rstrip()


def _normalise_lookup(
    state,
    player_turn,
    rotate_direction,
    transfer_allowed,
    base_dir,
    grid_size,
    players_symbols,
    tablebase_type,
):
    """Validate the lookup parameters shared by the public entry points."""
    state_array = _state_from_input(state, grid_size)
    player = _normalise_player(player_turn)
    rotation = _normalise_rotation(rotate_direction)
    transfer = _normalise_transfer_allowed(transfer_allowed)
    tablebase_type = tablebase.normalise_tablebase_type(tablebase_type)
    base_dir = tablebase.default_base_dir(tablebase_type) if base_dir is None else base_dir
    symbols = dict(_DEFAULT_PLAYER_SYMBOLS if players_symbols is None else players_symbols)
    if any(symbol not in symbols for symbol in (0, 1, 2)):
        raise ValueError("players_symbols must define symbols for 0, 1, and 2")
    return state_array, player, rotation, transfer, base_dir, symbols


def lookup_score(
    state,
    player_turn,
    rotate_direction="clockwise",
    transfer_allowed=True,
    base_dir=None,
    grid_size=None,
    tablebase_type="compressed",
):
    """Return the perfect score stored for a position, or None when absent.

    :func:`lookup_solution` also gives the best move and the full perfect game,
    which is a lot of work to do for a single number. Evaluating every move of a
    position, and every reply to it, needs the score of hundreds of positions,
    so this reads the score straight out of a small ``{position id: score}`` map
    of the tablebase file and never formats a report.
    """
    state_array, player, rotation, transfer, base_dir, _symbols = _normalise_lookup(
        state, player_turn, rotate_direction, transfer_allowed, base_dir, grid_size,
        None, tablebase_type,
    )
    if tablebase_type == "compressed":
        # the compressed tablebase stores one colour per position, so the
        # position is turned into the one that is stored before it is looked up
        representation = tablebase.compressed_representation(
            state_array, player, rotation)
        lookup_state = representation["state"]
        lookup_rotation = representation["stored_rotation"]
        lookup_player = 1
    else:
        lookup_state = state_array
        lookup_rotation = rotation
        lookup_player = player

    file_path = _record_path(base_dir, lookup_state, lookup_rotation, transfer, lookup_player)
    key = _cache_key(file_path)
    if key is None:
        return None
    scores = _scores_by_id(file_path, key)
    if scores is None:
        return None
    # the score is stored from the point of view of the player to move, which
    # swapping the colours of the position does not change
    position_id = tablebase.get_position_id(
        lookup_state, lookup_rotation, transfer, lookup_player)
    if position_id in scores:
        return scores[position_id]
    # a record without a usable id is only found by reading the file
    record = _find_record(base_dir, lookup_state, lookup_rotation, transfer, lookup_player)
    if record is None:
        return None
    return record.get("solution", {}).get("score")


def lookup_solution(
    state,
    player_turn,
    rotate_direction="clockwise",
    transfer_allowed=True,
    base_dir=None,
    grid_size=None,
    players_symbols=None,
    tablebase_type="compressed",
):
    """Return a structured tablebase entry for a position, or None.

    The returned dictionary always contains:

    - ``state``: the queried position as a numpy array,
    - ``player_turn``, ``rotation``, ``transfer_allowed``: the rule context,
    - ``best_move``, ``moves_sequence``, ``states_sequence``, ``score`` and
      ``game_result``: the stored solution, already mapped back to the
      orientation and colours of the queried position,
    - ``text``: the same formatted report returned by :func:`get_solution`.

    ``score`` is always expressed from the point of view of ``player_turn``
    (the player to move), and ``best_move`` is a move dictionary ready to be
    passed to ``orbital_logic_game_functions.play_turn``. ``None`` is returned
    when the position is not part of the selected tablebase. A rotating position
    is answered from the other rotation as well, because the two are the same
    game seen in a mirror, so a base directory that stored only the clockwise
    tables still answers counterclockwise queries.
    """
    (
        state_array,
        player,
        rotation,
        transfer,
        base_dir,
        symbols,
    ) = _normalise_lookup(
        state,
        player_turn,
        rotate_direction,
        transfer_allowed,
        base_dir,
        grid_size,
        players_symbols,
        tablebase_type,
    )
    tablebase_type = tablebase.normalise_tablebase_type(tablebase_type)

    record = retrograde_tablebase.build_record(
        base_dir,
        state_array,
        player,
        rotation,
        transfer,
    )
    if record is not None:
        return _build_entry(
            record, state_array, player, rotation, transfer,
            base_dir, tablebase_type, symbols,
        )

    if tablebase_type == "compressed":
        representation = tablebase.compressed_representation(
            state_array,
            player,
            rotation,
        )
        lookup_state = representation["state"]
        lookup_rotation = representation["stored_rotation"]
        lookup_player = 1
    else:
        representation = None
        lookup_state = state_array
        lookup_rotation = rotation
        lookup_player = player

    record = _find_record(
        base_dir,
        lookup_state,
        lookup_rotation,
        transfer,
        lookup_player,
    )
    if record is None:
        return None
    if representation is not None:
        record = _map_compressed_record(
            record,
            representation,
            state_array,
            rotation,
        )
    return _build_entry(
        record, state_array, player, rotation, transfer,
        base_dir, tablebase_type, symbols,
    )


def _build_entry(
    record,
    state_array,
    player,
    rotation,
    transfer,
    base_dir,
    tablebase_type,
    symbols,
):
    """Assemble the public lookup result from a tablebase record."""
    solution = record.get("solution", {})
    moves_sequence = solution.get("moves_sequence")
    if moves_sequence is None:
        best_move = solution.get("best_move")
        moves_sequence = [] if best_move is None else [best_move]
    return {
        "state": state_array,
        "player_turn": player,
        "rotation": rotation,
        "transfer_allowed": transfer,
        "base_dir": base_dir,
        "tablebase_type": tablebase_type,
        "id": record.get("id"),
        "best_move": _move_for_engine(moves_sequence[0]) if moves_sequence else None,
        "moves_sequence": [_move_for_engine(move) for move in moves_sequence],
        "states_sequence": _state_sequence(record, state_array, moves_sequence),
        "score": solution.get("score"),
        "game_result": solution.get("game_result"),
        "text": _format_solution(
            record,
            state_array,
            player,
            rotation,
            transfer,
            symbols,
        ),
    }


def available_grid_sizes(base_dir=None, tablebase_type="compressed"):
    """Return the sorted grid sizes that are available in a tablebase.

    Used by graphical front-ends to tell the user up-front which board sizes
    can be answered instantly from the tablebase. An empty list means that no
    tablebase has been generated yet.
    """
    tablebase_type = tablebase.normalise_tablebase_type(tablebase_type)
    base_dir = tablebase.default_base_dir(tablebase_type) if base_dir is None else base_dir
    grid_sizes = []
    try:
        entries = os.listdir(base_dir)
    except OSError:
        return grid_sizes
    for entry in entries:
        size = entry.partition("x")[0]
        if "x" not in entry or not size.isdigit() or int(size) < 1:
            continue
        if int(size) not in grid_sizes:
            grid_sizes.append(int(size))
    grid_sizes.sort()
    return grid_sizes


def get_solution(
    state,
    player_turn,
    rotate_direction="clockwise",
    transfer_allowed=True,
    base_dir=None,
    grid_size=None,
    players_symbols=None,
    tablebase_type="compressed",
):
    """Return a formatted solution or POSITION_NOT_FOUND.

    ``state`` is a row-major string such as ``"0120"`` or an array-like
    object. When ``grid_size`` is omitted, it is inferred from the state
    length. The compressed tablebase is used by default. The returned text
    follows the game, rules, and perfect-play evolution format used by
    ``solve_game.py``.
    """
    entry = lookup_solution(
        state,
        player_turn,
        rotate_direction=rotate_direction,
        transfer_allowed=transfer_allowed,
        base_dir=base_dir,
        grid_size=grid_size,
        players_symbols=players_symbols,
        tablebase_type=tablebase_type,
    )
    if entry is None:
        return POSITION_NOT_FOUND
    return entry["text"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Look up a saved tablebase solution.")
    parser.add_argument("state", nargs="?", help="row-major state string, such as 0120")
    parser.add_argument("--state", dest="state_option", help="row-major state string")
    parser.add_argument(
        "-p",
        "--player",
        "--player-turn",
        dest="player_turn",
        type=int,
        required=True,
        help="current player (1 or 2)",
    )
    parser.add_argument(
        "-r",
        "--rotation",
        "--rotate-direction",
        dest="rotate_direction",
        default="clockwise",
        help="rotation direction (default: clockwise)",
    )
    transfer_group = parser.add_mutually_exclusive_group()
    transfer_group.add_argument(
        "--transfer-allowed",
        "--transfer",
        dest="transfer_allowed",
        action="store_true",
        help="allow transfer moves",
    )
    transfer_group.add_argument(
        "--transfer-not-allowed",
        dest="transfer_allowed",
        action="store_false",
        help="disable transfer moves",
    )
    parser.set_defaults(transfer_allowed=True)
    parser.add_argument(
        "-n",
        "--grid-size",
        type=int,
        default=None,
        help="grid size; inferred from the state when omitted",
    )
    parser.add_argument(
        "--tablebase",
        "--tablebase-type",
        "--tablebase_type",
        dest="tablebase_type",
        choices=tablebase.TABLEBASE_TYPES,
        default="compressed",
        help="tablebase to query (default: compressed)",
    )
    parser.add_argument(
        "-b",
        "--base-dir",
        default=None,
        help="tablebase directory (default: selected tablebase directory)",
    )
    args = parser.parse_args(argv)
    state = args.state_option if args.state_option is not None else args.state
    if state is None:
        parser.error("a state is required")

    try:
        result = get_solution(
            state=state,
            player_turn=args.player_turn,
            rotate_direction=args.rotate_direction,
            transfer_allowed=args.transfer_allowed,
            base_dir=args.base_dir,
            grid_size=args.grid_size,
            tablebase_type=args.tablebase_type,
        )
    except ValueError as error:
        parser.error(str(error))
    print(result)
    return result


if __name__ == "__main__":
    main()
