"""Computer opponent logic shared by the graphical front-end.

The engine always tries to answer from the compressed game tablebase, which
stores the perfect score and the principal variation of every position it
covers. When the position is missing, the caller decides what happens next:

* :data:`FALLBACK_RANDOM` plays a uniformly random legal move,
* :data:`FALLBACK_SEARCH` runs the iterative deepening minimax search from
  ``solve_game`` for at most ``time_limit`` seconds and at most
  :data:`MAX_SEARCH_DEPTH` moves of lookahead,
* :data:`FALLBACK_NONE` reports that no move is available.

The module has no GUI dependency, so its helpers can also be used from a
worker process (see :class:`EngineProcess`) to keep the interface responsive
while the computer thinks.
"""

import multiprocessing as mp
import os
import random
import time

import numpy as np

import get_solution
import orbital_logic_game_functions as olgf
import solve_game


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPRESSED_TABLEBASE_DIR = os.path.join(MODULE_DIR, "compressed_game_tablebase")
FULL_TABLEBASE_DIR = os.path.join(MODULE_DIR, "game_tablebase")

TABLEBASE_TYPE = "compressed"
USE_TABLEBASE = True
FALLBACK_RANDOM = "random"
FALLBACK_SEARCH = "search"
FALLBACK_NONE = "none"
FALLBACKS = (FALLBACK_RANDOM, FALLBACK_SEARCH, FALLBACK_NONE)

SOURCE_TABLEBASE = "tablebase"
SOURCE_RANDOM = "random"
SOURCE_SEARCH = "search"
SOURCE_NONE = "none"

PLAYER_SYMBOLS = {0: "_", 1: "Black", 2: "White"}
DEFAULT_TIME_LIMIT = 3.0
MAX_TIME_LIMIT = 600.0
MAX_SEARCH_DEPTH = 100


def default_tablebase_dir(tablebase_type=TABLEBASE_TYPE):
    """Absolute path of a tablebase shipped next to this module."""
    return (COMPRESSED_TABLEBASE_DIR if tablebase_type == TABLEBASE_TYPE
            else FULL_TABLEBASE_DIR)


def available_grid_sizes(tablebase_type=TABLEBASE_TYPE, base_dir=None):
    """Grid sizes covered by the tablebase, as sorted integers."""
    if base_dir is None:
        base_dir = default_tablebase_dir(tablebase_type)
    return get_solution.available_grid_sizes(base_dir, tablebase_type)


def clamp_time_limit(time_limit):
    """Keep a user supplied thinking time inside a usable range."""
    try:
        time_limit = float(time_limit)
    except (TypeError, ValueError):
        return DEFAULT_TIME_LIMIT
    if time_limit != time_limit or time_limit <= 0:  # NaN or non positive
        return DEFAULT_TIME_LIMIT
    return min(time_limit, MAX_TIME_LIMIT)


def game_result_message(score, player_turn):
    """Describe a score that is relative to the player to move."""
    if score is None:
        return "unknown"
    if score > 0:
        return f"{PLAYER_SYMBOLS[player_turn]} wins"
    if score < 0:
        return f"{PLAYER_SYMBOLS[3 - player_turn]} wins"
    return "draw"


def move_to_text(move):
    """Human readable description of a move dictionary."""
    if move is None:
        return "no move"
    parts = []
    transfer = move.get("transfer")
    if transfer is not None:
        source, direction = transfer[0], transfer[1]
        parts.append(f"transfer ({source[0]},{source[1]})->{direction}")
    add = move.get("add")
    if add is not None:
        parts.append(f"add ({add[0]},{add[1]})")
    return ", ".join(parts) if parts else "no move"


def tablebase_entry(state, player_turn, rotation, transfer_allowed, base_dir=None):
    """Return the tablebase entry of a position, or None when absent.

    Positions whose game already ended have no legal move to suggest, so they
    report None as well.
    """
    if olgf.evaluate_game_state(state) is not None:
        return None
    try:
        return get_solution.lookup_solution(
            state=state,
            player_turn=player_turn,
            rotate_direction=rotation,
            transfer_allowed=bool(transfer_allowed),
            base_dir=default_tablebase_dir() if base_dir is None else base_dir,
            tablebase_type=TABLEBASE_TYPE,
        )
    except ValueError:
        return None


def _result(source, move=None, score=None, message="", **extra):
    result = {
        "move": move,
        "score": score,
        "source": source,
        "message": message,
        "result": game_result_message(score, extra.get("player_turn", 1)),
    }
    result.update(extra)
    return result


def choose_move(
    state,
    player_turn,
    rotation="clockwise",
    transfer_allowed=True,
    use_tablebase=USE_TABLEBASE,
    fallback=FALLBACK_RANDOM,
    time_limit=DEFAULT_TIME_LIMIT,
    max_depth=MAX_SEARCH_DEPTH,
    base_dir=None,
    seed=None,
):
    """Return the computer move for ``state`` together with a report.

    The result dictionary always holds the chosen ``move`` (a dictionary ready
    for ``orbital_logic_game_functions.play_turn``, or None), the ``score`` of
    the move relative to ``player_turn`` (None when unknown), the ``source``
    of the move (one of the ``SOURCE_*`` constants), a human readable
    ``message`` and the ``elapsed`` search time in seconds. Tablebase answers
    additionally carry the full formatted perfect-play report in ``text`` and
    the whole evolution in ``moves_sequence`` / ``states_sequence``.

    ``max_depth`` caps how far the minimax search may look, in moves; an
    answer that comes from the deepest allowed search is reported as not
    guaranteed, because the moves below the horizon were never examined.
    """
    state = np.asarray(state, dtype=int)
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("state must be a square two-dimensional array")
    if player_turn not in (1, 2):
        raise ValueError("player_turn must be 1 or 2")
    if fallback not in FALLBACKS:
        raise ValueError(f"fallback must be one of: {', '.join(FALLBACKS)}")
    start = time.monotonic()

    if olgf.evaluate_game_state(state) is not None:
        return _result(
            SOURCE_NONE,
            message="The game is already over.",
            player_turn=player_turn,
            elapsed=0.0,
        )

    if use_tablebase:
        entry = tablebase_entry(state, player_turn, rotation, transfer_allowed, base_dir)
        if entry is not None and entry["best_move"] is not None:
            return _result(
                SOURCE_TABLEBASE,
                move=entry["best_move"],
                score=entry["score"],
                message=(
                    f"Tablebase (perfect play): {move_to_text(entry['best_move'])} "
                    f"-> {game_result_message(entry['score'], player_turn)}"
                ),
                player_turn=player_turn,
                elapsed=time.monotonic() - start,
                text=entry["text"],
                moves_sequence=entry["moves_sequence"],
                states_sequence=[np.asarray(item, dtype=int) for item in entry["states_sequence"]],
                position_id=entry["id"],
            )

    if fallback == FALLBACK_NONE:
        return _result(
            SOURCE_NONE,
            message="This position is not in the tablebase.",
            player_turn=player_turn,
            elapsed=time.monotonic() - start,
        )

    if fallback == FALLBACK_RANDOM:
        rng = None if seed is None else random.Random(seed)
        move, _value = solve_game.find_random_move(
            state, rotation, transfer_allowed, player_turn, rng,
        )
        if move is None:
            return _result(
                SOURCE_NONE,
                message="No legal move is available.",
                player_turn=player_turn,
                elapsed=time.monotonic() - start,
            )
        return _result(
            SOURCE_RANDOM,
            move=move,
            message=f"Random move (not in the tablebase): {move_to_text(move)}",
            player_turn=player_turn,
            elapsed=time.monotonic() - start,
        )

    limit = clamp_time_limit(time_limit)
    move, score, info = solve_game.find_best_move_within_time(
        state, rotation, transfer_allowed, player_turn, time_limit=limit,
        max_depth=max_depth,
    )
    if move is None:
        return _result(
            SOURCE_NONE,
            message="No legal move is available.",
            player_turn=player_turn,
            elapsed=time.monotonic() - start,
            search=info,
        )
    depth = info["depth"]
    message = (
        f"Minimax search (not in the tablebase): {move_to_text(move)} -> "
        f"{game_result_message(score, player_turn)}"
    )
    if depth == 0:
        message = (
            f"Minimax search could not finish the first iteration in {limit:g}s, "
            f"playing the best ordered move: {move_to_text(move)} "
            f"(max depth {max_depth}; the result is not guaranteed)"
        )
    else:
        message += (
            f" (depth {depth} of at most {max_depth}, {info['nodes']} nodes, "
            f"{info['elapsed']:.1f}s"
            + (", timed out" if info["timed_out"] else "")
            + ")"
        )
        if info["reason"] == "max_depth":
            message += (
                f"\nThe search reached the maximum depth of {max_depth} moves, so "
                f"the result is not guaranteed: the moves below that depth were "
                f"not examined."
            )
    return _result(
        SOURCE_SEARCH,
        move=move,
        score=score,
        message=message,
        player_turn=player_turn,
        elapsed=time.monotonic() - start,
        search=info,
    )


def worker(request_queue, result_queue):
    """Process entry point: answer a single request put on ``request_queue``."""
    while True:
        request = request_queue.get()
        if request is None:
            return
        try:
            result = choose_move(**request)
        except Exception as error:  # never leave the front-end waiting forever
            result = _result(
                SOURCE_NONE,
                message=f"Engine error: {type(error).__name__}: {error}",
                player_turn=int(request.get("player_turn", 1)),
                elapsed=0.0,
                error=f"{type(error).__name__}: {error}",
            )
        result_queue.put(result)


class EngineProcess:
    """Run :func:`choose_move` in a separate process.

    Searching a 4x4 board takes seconds, which would freeze the interface if
    it ran in the main thread. One long lived worker process is started on the
    first request and reused afterwards, and the result is handed back
    asynchronously through a queue.
    """

    def __init__(self):
        self._context = mp.get_context(
            "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        )
        self._request_queue = None
        self._result_queue = None
        self._process = None
        self._busy = False

    @property
    def busy(self):
        """True while a submitted request has not been collected yet."""
        return self._busy

    def _ensure_worker(self):
        if self._process is not None and self._process.is_alive():
            return
        self._request_queue = self._context.Queue()
        self._result_queue = self._context.Queue()
        self._process = self._context.Process(
            target=worker,
            args=(self._request_queue, self._result_queue),
            daemon=True,
        )
        self._process.start()

    def submit(self, request):
        """Send a request to the worker; returns False when one is pending."""
        if self._busy:
            return False
        self._ensure_worker()
        self._busy = True
        self._request_queue.put(request)
        return True

    def take_result(self):
        """Return the finished result of the pending request, or None."""
        if not self._busy:
            return None
        try:
            result = self._result_queue.get(timeout=0)
        except Exception:
            return None
        self._busy = False
        return result

    def shutdown(self):
        """Stop the worker process if one is running."""
        self._busy = False
        if self._process is None:
            return
        if self._request_queue is not None:
            try:
                self._request_queue.put(None)
            except (OSError, ValueError):
                pass
        self._process.join(timeout=1.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        self._process = None
        self._request_queue = None
        self._result_queue = None


if __name__ == "__main__":
    demo_state = np.array([[1, 2, 0], [0, 0, 0], [0, 1, 2]])
    for fallback in FALLBACKS:
        answer = choose_move(demo_state, 1, "clockwise", True, True, fallback, 1.0)
        print(f"[{fallback}] {answer['source']}: {answer['message']}")
