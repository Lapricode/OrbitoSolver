import argparse
import itertools
import json
import os
import sys

import numpy as np

import solve_game

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


ROTATION_DIRECTIONS = ("still", "clockwise", "counterclockwise")
TRANSFER_RULES = (False, True)
PLAYER_TURNS = (1, 2)


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


def _player_name(player_turn):
    return "player1" if player_turn == 1 else "player2"


def get_file_path(base_dir, state, rotation, transfer_allowed, player_turn):
    """Build the existing tablebase path with a JSON completion file."""
    state = _as_state(state)
    grid_size = state.shape[0]
    grid_folder = f"{grid_size}x{grid_size}"
    rotation_folder = _rotation_name(rotation)
    transfer_folder = _transfer_name(transfer_allowed)
    player_folder = _player_name(player_turn)
    pieces_count = int(np.count_nonzero(state))
    completion_file = f"completion_{pieces_count}.json"
    folder_path = os.path.join(base_dir, grid_folder, rotation_folder, transfer_folder, player_folder)
    os.makedirs(folder_path, exist_ok=True)
    return os.path.join(folder_path, completion_file)


def get_position_id(state, rotation, transfer_allowed, player_turn):
    """Return a stable ID for a board and its complete tablebase rule context.

    The board portion is a row-major sequence of base-3 digits. The grid,
    rotation, transfer mode, and current player are included so the same board
    evaluated under different rules still receives a different ID.
    """
    state = _as_state(state)
    grid_size = state.shape[0]
    grid_folder = f"{grid_size}x{grid_size}"
    board_code = "".join(str(int(cell)) for cell in state.ravel())
    return "_".join((
        grid_folder,
        board_code,
        _rotation_name(rotation),
        _transfer_name(transfer_allowed),
        _player_name(player_turn),
    ))


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


def _solve_position(state, rotation, transfer_allowed, player_turn):
    result = solve_game.solve_game(state, rotation, transfer_allowed, player_turn)
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


def _make_record(state, solution, rotation, transfer_allowed, player_turn, position_id=None):
    return {
        "id": get_position_id(state, rotation, transfer_allowed, player_turn)
        if position_id is None else position_id,
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
    player_turn,
    position_id=None,
):
    """Save one record in the JSON array for its completion file.

    An existing record with the same ID is replaced, which makes repeated
    calls idempotent and keeps every generated file valid JSON.
    """
    state = _as_state(state)
    file_path = get_file_path(base_dir, state, rotation, transfer_allowed, player_turn)
    record = _make_record(
        state,
        {"best_move": best_move, "game_result": game_result},
        rotation,
        transfer_allowed,
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


def _iter_states_for_completion(grid_size, completion):
    cell_count = grid_size * grid_size
    for occupied in itertools.combinations(range(cell_count), completion):
        for values in itertools.product((1, 2), repeat=completion):
            state_values = [0] * cell_count
            for cell, value in zip(occupied, values):
                state_values[cell] = value
            yield np.asarray(state_values, dtype=int).reshape((grid_size, grid_size))


def create_game_tablebase(base_dir="game_tablebase", grid_size=2, show_progress=True):
    """Solve every board state for an n x n grid and all rule combinations."""
    grid_size = int(grid_size)
    if grid_size < 1:
        raise ValueError("grid_size must be positive")

    state_count = 3 ** (grid_size * grid_size)
    context_count = len(ROTATION_DIRECTIONS) * len(TRANSFER_RULES) * len(PLAYER_TURNS)
    total_positions = state_count * context_count
    progress = _make_progress(
        total_positions,
        f"{grid_size}x{grid_size} tablebase",
        show_progress,
    )
    position_ids = set()
    files_written = 0

    try:
        for completion in range(grid_size * grid_size, -1, -1):
            for rotation in ROTATION_DIRECTIONS:
                for transfer_allowed in TRANSFER_RULES:
                    for player_turn in PLAYER_TURNS:
                        records_by_file = {}
                        file_paths = {}
                        for state in _iter_states_for_completion(grid_size, completion):
                            if completion not in file_paths:
                                file_paths[completion] = get_file_path(
                                    base_dir,
                                    state,
                                    rotation,
                                    transfer_allowed,
                                    player_turn,
                                )
                            file_path = file_paths[completion]
                            solution = _solve_position(
                                state,
                                rotation,
                                transfer_allowed,
                                player_turn,
                            )
                            record = _make_record(
                                state,
                                solution,
                                rotation,
                                transfer_allowed,
                                player_turn,
                            )
                            if record["id"] in position_ids:
                                raise RuntimeError(f"Duplicate position ID: {record['id']}")
                            position_ids.add(record["id"])
                            records_by_file.setdefault(file_path, []).append(record)
                            if progress is not None:
                                progress.update()

                        for file_path in sorted(records_by_file):
                            _write_records(file_path, records_by_file[file_path])
                            files_written += 1
    finally:
        if progress is not None:
            progress.close()

    return {
        "grid_size": grid_size,
        "positions": len(position_ids),
        "files": files_written,
    }


def generate_tablebase(base_dir="game_tablebase", grid_size=2, show_progress=True):
    """Backward-compatible alias for create_game_tablebase."""
    return create_game_tablebase(base_dir, grid_size, show_progress)


def solve_all_2x2_positions(base_dir="game_tablebase", show_progress=True):
    """Solve all 81 possible 2x2 boards for all 12 rule combinations."""
    return create_game_tablebase(base_dir, grid_size=2, show_progress=show_progress)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create an orbital logic game tablebase.")
    parser.add_argument(
        "-n",
        "--grid-size",
        type=int,
        default=2,
        help="size of the square grid (default: 2)",
    )
    parser.add_argument(
        "--base-dir",
        default="game_tablebase",
        help="directory in which to save the tablebase (default: game_tablebase)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable terminal progress output",
    )
    args = parser.parse_args(argv)
    if args.grid_size < 1:
        parser.error("grid size must be positive")

    result = create_game_tablebase(
        base_dir=args.base_dir,
        grid_size=args.grid_size,
        show_progress=not args.no_progress,
    )
    print(
        f"Saved {result['positions']} positions to {result['files']} JSON files "
        f"in the {result['grid_size']}x{result['grid_size']} tablebase."
    )
    return result


if __name__ == "__main__":
    main()
