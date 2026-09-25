import argparse
import contextlib
import io
import json
import math
import os

import numpy as np

import create_game_tablebase as tablebase
import orbital_logic_game_functions as olgf
import solve_game


POSITION_NOT_FOUND = "Position is not in the tablebase."
_DEFAULT_PLAYER_SYMBOLS = {0: "_", 1: "x", 2: "o"}
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


def _find_record(base_dir, state, rotate_direction, transfer_allowed, player_turn):
    file_path = _record_path(
        base_dir,
        state,
        rotate_direction,
        transfer_allowed,
        player_turn,
    )
    if not os.path.isfile(file_path):
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            records = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(records, list):
        return None

    position_id = tablebase.get_position_id(
        state,
        rotate_direction,
        transfer_allowed,
        player_turn,
    )
    for record in records:
        if isinstance(record, dict) and record.get("id") == position_id:
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
        "",
        "Initial game state:",
        "",
        _format_state(state, players_symbols),
        "",
        f"Start player: {players_symbols[player_turn]}",
        "",
    ]

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


def get_solution(
    state,
    player_turn,
    rotate_direction="clockwise",
    transfer_allowed=True,
    base_dir="game_tablebase",
    grid_size=None,
    players_symbols=None,
):
    """Return a formatted solution or POSITION_NOT_FOUND.

    ``state`` is a row-major string such as ``"0120"`` or an array-like
    object. When ``grid_size`` is omitted, it is inferred from the state
    length. The returned text follows the game, rules, and perfect-play
    evolution format used by ``solve_game.py``.
    """
    state_array = _state_from_input(state, grid_size)
    player = _normalise_player(player_turn)
    rotation = _normalise_rotation(rotate_direction)
    transfer = _normalise_transfer_allowed(transfer_allowed)
    symbols = dict(_DEFAULT_PLAYER_SYMBOLS if players_symbols is None else players_symbols)
    if any(symbol not in symbols for symbol in (0, 1, 2)):
        raise ValueError("players_symbols must define symbols for 0, 1, and 2")

    record = _find_record(
        base_dir,
        state_array,
        rotation,
        transfer,
        player,
    )
    if record is None:
        return POSITION_NOT_FOUND
    return _format_solution(
        record,
        state_array,
        player,
        rotation,
        transfer,
        symbols,
    )


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
        "-b",
        "--base-dir",
        default="game_tablebase",
        help="tablebase directory (default: game_tablebase)",
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
        )
    except ValueError as error:
        parser.error(str(error))
    print(result)
    return result


if __name__ == "__main__":
    main()
