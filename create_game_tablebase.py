import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys

import numpy as np

import orbital_logic_game_functions as olgf
import solve_game

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


ROTATION_DIRECTIONS = ("still", "clockwise", "counterclockwise")
COMPRESSED_ROTATION_DIRECTIONS = ("still", "clockwise")
TRANSFER_RULES = (False, True)
EXTRA_ROTATION_RULES = (False, True)
PLAYER_TURNS = (1, 2)
COMPRESSED_PLAYER_TURNS = (1,)
COMPRESSED_TABLEBASE_DIR = "compressed_game_tablebase"
_ROTATION_ALIASES = {
    "clockwise": "clockwise",
    "cw": "clockwise",
    "counterclockwise": "counterclockwise",
    "ccw": "counterclockwise",
    "still": "still",
    "s": "still",
}
_DIRECTION_DELTAS = {
    "u": (-1, 0),
    "d": (1, 0),
    "l": (0, -1),
    "r": (0, 1),
}


class _FallbackProgress:
    def __init__(self, total, description):
        self.total = total
        self.description = description
        self.current = 0
        self.last_percent = -1
        self.closed = False
        self._print(0)

    def _print(self, percent):
        print(
            f"\r{self.description}: {percent:3d}% ({self.current}/{self.total})",
            end="",
            file=sys.stderr,
            flush=True,
        )

    def update(self, amount=1):
        self.current += amount
        if self.total:
            percent = int(self.current * 100 / self.total)
            if percent != self.last_percent:
                self.last_percent = percent
                self._print(percent)

    def close(self):
        if not self.closed:
            print(file=sys.stderr)
            self.closed = True


def _make_progress(total, description, enabled):
    if not enabled:
        return None
    if tqdm is not None:
        return tqdm(total=total, desc=description, unit="position", dynamic_ncols=True)
    return _FallbackProgress(total, description)


def _as_state(state):
    state = np.asarray(state)
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("state must be a square two-dimensional array")
    if not np.all(np.isin(state, (0, 1, 2))):
        raise ValueError("state cells must contain only 0, 1, or 2")
    return state.astype(int, copy=False)


def _rotation_name(rotation):
    return str(rotation).lower()


def _transfer_name(transfer_allowed):
    return "transfer_allowed" if transfer_allowed else "transfer_not_allowed"


def _extra_rotation_name(extra_rotation_allowed):
    return "extra_rotation_allowed" if extra_rotation_allowed else "extra_rotation_not_allowed"


def _player_name(player_turn):
    return "player1" if player_turn == 1 else "player2"


def default_base_dir():
    """Directory of the compressed tablebase, the only one that is built here."""
    return COMPRESSED_TABLEBASE_DIR


def _canonical_rotation(rotation):
    name = str(rotation).strip().lower()
    if name not in _ROTATION_ALIASES:
        choices = ", ".join(ROTATION_DIRECTIONS)
        raise ValueError(f"rotation must be one of: {choices}")
    return _ROTATION_ALIASES[name]


def _swap_player_values(state):
    state = np.asarray(state, dtype=int)
    return np.where(state == 1, 2, np.where(state == 2, 1, state)).astype(int)


def _apply_symmetry(state, symmetry):
    reflection, turns = symmetry
    result = np.asarray(state, dtype=int)
    if reflection:
        result = result[:, ::-1]
    if turns:
        result = np.rot90(result, -turns)
    return np.ascontiguousarray(result, dtype=int).copy()


def _invert_symmetry(symmetry):
    reflection, turns = symmetry
    if reflection:
        return symmetry
    return reflection, (-turns) % 4


def _transform_position(symmetry, position, grid_size):
    row, column = int(position[0]), int(position[1])
    if symmetry[0]:
        column = grid_size - 1 - column
    for _ in range(symmetry[1]):
        row, column = column, grid_size - 1 - row
    return row, column


def _transform_direction(symmetry, source, direction, grid_size):
    delta_row, delta_column = _DIRECTION_DELTAS[str(direction).lower()]
    target = (int(source[0]) + delta_row, int(source[1]) + delta_column)
    source = _transform_position(symmetry, source, grid_size)
    target = _transform_position(symmetry, target, grid_size)
    mapped_delta = (target[0] - source[0], target[1] - source[1])
    for name, delta in _DIRECTION_DELTAS.items():
        if delta == mapped_delta:
            return name
    raise ValueError(f"unsupported transfer direction: {direction}")


def _symmetry_transformations(rotate_direction):
    rotation = _canonical_rotation(rotate_direction)
    if rotation == "still":
        return tuple((reflection, turns) for reflection in (False, True) for turns in range(4))
    if rotation == "clockwise":
        return tuple((False, turns) for turns in range(4))
    return tuple((True, turns) for turns in range(4))


def _turning_changes_the_result(state, rotation):
    """Tell whether turning a full board can change what the endgame presses decide.

    A full board without a line is decided by the forced presses, and a press is
    nothing but a rotation, so a board and its rotation can well belong to
    different games. Storing a rotated representative under the name of the
    board it came from would then answer for the wrong game, which is why such a
    board is not canonicalized by rotation. A "still" rotation turns nothing at
    all, so the presses can only end in a draw there and rotating is harmless.
    """
    if _canonical_rotation(rotation) != "clockwise":
        return False
    state = _as_state(state)
    if np.any(state == 0):
        return False
    return olgf.evaluate_game_state(state) is None


def _symmetry_transformations_for_state(state, rotation):
    """The symmetries that may be used to store this board, rotations excluded
    when turning it would change the answer."""
    transformations = _symmetry_transformations(rotation)
    if not _turning_changes_the_result(state, rotation):
        return transformations
    reflections = tuple(symmetry for symmetry in transformations if symmetry[0])
    return reflections if reflections else ((False, 0),)


def canonicalize_state(state, rotate_direction):
    state = _as_state(state)
    best_key = None
    best_state = None
    best_symmetry = None
    for symmetry in _symmetry_transformations_for_state(state, rotate_direction):
        candidate = _apply_symmetry(state, symmetry)
        key = tuple(int(cell) for cell in candidate.ravel())
        if best_key is None or key < best_key:
            best_key = key
            best_state = candidate
            best_symmetry = symmetry
    rotation = _canonical_rotation(rotate_direction)
    stored_rotation = "clockwise" if rotation == "counterclockwise" else rotation
    return best_state, best_symmetry, stored_rotation


def compressed_representation(state, player_turn, rotate_direction):
    state = _as_state(state)
    player_turn = int(player_turn)
    if player_turn not in (1, 2):
        raise ValueError("player_turn must be 1 or 2")
    swapped = player_turn == 2
    oriented_state = _swap_player_values(state) if swapped else state
    canonical, symmetry, stored_rotation = canonicalize_state(oriented_state, rotate_direction)
    return {
        "state": canonical,
        "symmetry": symmetry,
        "stored_rotation": stored_rotation,
        "swapped": swapped,
    }


def map_compressed_state(state, symmetry, swap_players=False):
    mapped = _apply_symmetry(state, _invert_symmetry(symmetry))
    return _swap_player_values(mapped) if swap_players else mapped


def map_compressed_move(move, symmetry, grid_size, rotate_direction, swap_players=False):
    inverse = _invert_symmetry(symmetry)
    mapped = dict(move)
    if mapped.get("add") is not None:
        mapped["add"] = _transform_position(inverse, mapped["add"], grid_size)
    if mapped.get("transfer") is not None:
        source = mapped["transfer"][0]
        direction = _transform_direction(
            inverse,
            source,
            direction=mapped["transfer"][1],
            grid_size=grid_size,
        )
        mapped["transfer"] = [_transform_position(inverse, source, grid_size), direction]
    if swap_players:
        mapped["player"] = 3 - int(mapped["player"])
    mapped["rotate"] = _canonical_rotation(rotate_direction)
    return mapped


def get_file_path(base_dir, state, rotation, transfer_allowed, extra_rotation_allowed, player_turn):
    """Build the existing tablebase path with a JSON completion file."""
    state = _as_state(state)
    grid_size = state.shape[0]
    grid_folder = f"{grid_size}x{grid_size}"
    rotation_folder = _rotation_name(rotation)
    transfer_folder = _transfer_name(transfer_allowed)
    extra_folder = _extra_rotation_name(extra_rotation_allowed)
    player_folder = _player_name(player_turn)
    pieces_count = int(np.count_nonzero(state))
    completion_file = f"completion_{pieces_count}.json"
    folder_path = os.path.join(base_dir, grid_folder, rotation_folder, transfer_folder, extra_folder, player_folder)
    os.makedirs(folder_path, exist_ok=True)
    return os.path.join(folder_path, completion_file)


def position_state_id(state):
    """Return the ID of a position: the state alone, as base-3 digits.

    The cells are written row by row, one digit per cell, so a 4x4 position
    reads as sixteen digits such as ``0000121102102211``. Nothing else is part
    of the ID: the grid size is given by the number of digits, and the rules and
    the side to move are part of the tablebase context, not of the position.
    """
    state = _as_state(state)
    return "".join(str(int(cell)) for cell in state.ravel())


def _json_compatible(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _write_records(file_path, records):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(_json_compatible(records), file, indent=2, ensure_ascii=False)
        file.write("\n")


def _read_records(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {file_path}")
    return records


def _solve_position(state, rotation, transfer_allowed, extra_rotation_allowed, player_turn):
    result = solve_game.solve_game(state, rotation, transfer_allowed, player_turn,
                                   extra_rotation_allowed=extra_rotation_allowed)
    score = int(result["score"])
    if score > 0:
        game_result = f"player{player_turn}_wins"
    elif score < 0:
        game_result = f"player{3 - player_turn}_wins"
    else:
        game_result = "draw"
    moves_sequence = _json_compatible(result["moves_sequence"])
    return {
        "score": score,
        "game_result": game_result,
        "best_move": moves_sequence[0] if moves_sequence else None,
        "moves_sequence": moves_sequence,
        "states_sequence": _json_compatible(result["states_sequence"]),
    }


def _make_record(state, solution, rotation, transfer_allowed, extra_rotation_allowed, player_turn, position_id=None):
    return {
        "id": position_state_id(state) if position_id is None else position_id,
        "position": state.tolist(),
        "solution": _json_compatible(solution),
    }


def save_game_record(
    base_dir,
    state,
    best_move,
    game_result,
    rotation,
    transfer_allowed,
    extra_rotation_allowed,
    player_turn,
    position_id=None,
):
    """Save one record in the JSON array for its completion file.

    An existing record with the same ID is replaced, which makes repeated
    calls idempotent and keeps every generated file valid JSON.
    """
    state = _as_state(state)
    file_path = get_file_path(base_dir, state, rotation, transfer_allowed, extra_rotation_allowed, player_turn)
    record = _make_record(
        state,
        {"best_move": best_move, "game_result": game_result},
        rotation,
        transfer_allowed,
        extra_rotation_allowed,
        player_turn,
        position_id,
    )
    records = _read_records(file_path)
    for index, existing_record in enumerate(records):
        if existing_record.get("id") == record["id"]:
            records[index] = record
            break
    else:
        records.append(record)
    _write_records(file_path, records)
    print(f"Record saved to {file_path}")


def _solve_position_task(item):
    state, rotation, transfer_allowed, extra_rotation_allowed, player_turn = item
    return _solve_position(state, rotation, transfer_allowed, extra_rotation_allowed, player_turn)


def _solve_states_parallel(states, rotation, transfer_allowed, extra_rotation_allowed, player_turn, workers):
    """Solve many independent positions using a pool of worker processes."""
    tasks = [(state, rotation, transfer_allowed, extra_rotation_allowed, player_turn) for state in states]
    with mp.Pool(workers) as pool:
        return pool.map(_solve_position_task, tasks)


def _iter_states_for_completion(grid_size, completion):
    cell_count = grid_size * grid_size
    for occupied in itertools.combinations(range(cell_count), completion):
        for values in itertools.product((1, 2), repeat=completion):
            state_values = [0] * cell_count
            for cell, value in zip(occupied, values):
                state_values[cell] = value
            yield np.asarray(state_values, dtype=int).reshape((grid_size, grid_size))


def create_game_tablebase(
    base_dir=None,
    grid_size=2,
    show_progress=True,
    workers=None,
):
    """Solve every board state of the compressed tablebase for an n x n grid.

    The compressed tablebase keeps one record per class of symmetric boards, for
    both settings of the endgame presses, so the lookup can answer any of the
    eight rule contexts of a grid from a single file set.
    """
    if base_dir is None:
        base_dir = default_base_dir()
    grid_size = int(grid_size)
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    if workers is not None:
        workers = int(workers)

    rotation_directions = COMPRESSED_ROTATION_DIRECTIONS
    player_turns = COMPRESSED_PLAYER_TURNS
    state_count = 3 ** (grid_size * grid_size)
    context_count = (
        len(rotation_directions)
        * len(TRANSFER_RULES)
        * len(EXTRA_ROTATION_RULES)
        * len(player_turns)
    )
    total_positions = state_count * context_count
    progress = _make_progress(
        total_positions,
        f"{grid_size}x{grid_size} compressed tablebase",
        show_progress,
    )
    positions_written = 0
    files_written = 0

    try:
        for completion in range(grid_size * grid_size, -1, -1):
            for rotation in rotation_directions:
                for transfer_allowed in TRANSFER_RULES:
                    for extra_rotation_allowed in EXTRA_ROTATION_RULES:
                        for player_turn in player_turns:
                            all_states = list(_iter_states_for_completion(grid_size, completion))
                            states = [
                                state
                                for state in all_states
                                if np.array_equal(
                                    canonicalize_state(state, rotation)[0],
                                    state,
                                )
                            ]
                            if progress is not None:
                                progress.update(len(all_states))
                            if not states:
                                continue
                            file_path = get_file_path(
                                base_dir,
                                states[0],
                                rotation,
                                transfer_allowed,
                                extra_rotation_allowed,
                                player_turn,
                            )
                            if workers is not None and workers > 1 and len(states) > 1:
                                solutions = _solve_states_parallel(
                                    states,
                                    rotation,
                                    transfer_allowed,
                                    extra_rotation_allowed,
                                    player_turn,
                                    workers,
                                )
                            else:
                                solutions = [
                                    _solve_position(state, rotation, transfer_allowed, extra_rotation_allowed, player_turn)
                                    for state in states
                                ]
                            records = []
                            # the ID is the position alone, so the same board can
                            # be stored in every context: the check is per file
                            position_ids = set()
                            for state, solution in zip(states, solutions):
                                record = _make_record(
                                    state,
                                    solution,
                                    rotation,
                                    transfer_allowed,
                                    extra_rotation_allowed,
                                    player_turn,
                                )
                                if record["id"] in position_ids:
                                    raise RuntimeError(f"Duplicate position ID: {record['id']}")
                                position_ids.add(record["id"])
                                records.append(record)
                            _write_records(file_path, records)
                            files_written += 1
                            positions_written += len(position_ids)
    finally:
        if progress is not None:
            progress.close()

    return {
        "grid_size": grid_size,
        "positions": positions_written,
        "files": files_written,
    }


def generate_tablebase(
    base_dir=None,
    grid_size=2,
    show_progress=True,
    workers=None,
):
    """Backward-compatible alias for create_game_tablebase."""
    return create_game_tablebase(
        base_dir,
        grid_size,
        show_progress,
        workers=workers,
    )


def solve_all_2x2_positions(
    base_dir=None,
    show_progress=True,
    workers=None,
):
    """Solve the 2x2 compressed tablebase."""
    return create_game_tablebase(
        base_dir,
        grid_size=2,
        show_progress=show_progress,
        workers=workers,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create the orbital logic game compressed tablebase.")
    parser.add_argument(
        "-n",
        "--grid-size",
        type=int,
        default=2,
        help="size of the square grid (default: 2)",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help=f"directory in which to save the tablebase (default: {COMPRESSED_TABLEBASE_DIR})",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable terminal progress output",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="number of worker processes to use for solving positions in parallel",
    )
    args = parser.parse_args(argv)
    if args.grid_size < 1:
        parser.error("grid size must be positive")

    result = create_game_tablebase(
        base_dir=args.base_dir,
        grid_size=args.grid_size,
        show_progress=not args.no_progress,
        workers=args.workers,
    )
    print(
        f"Saved {result['positions']} positions to {result['files']} JSON files "
        f"in the {result['grid_size']}x{result['grid_size']} compressed tablebase."
    )
    return result


if __name__ == "__main__":
    main()
