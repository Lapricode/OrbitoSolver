"""Computer opponent logic shared by the graphical front-end.

The engine answers from the retrograde value tables in
:data:`RETROGRADE_TABLEBASE_DIR`: they hold the perfect score of every position a
game can reach and are consulted as a single array index. Positions no stored
table covers, for instance a rule context that has not been built yet, fall back
to the compressed game tablebase, which stores the perfect score and the
principal variation of every position it covers. When neither tablebase has the
position, the caller decides what happens next:

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

import create_game_tablebase
import get_solution
import orbital_logic_game_functions as olgf
import retrograde_tablebase
import solve_game


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
RETROGRADE_TABLEBASE_DIR = os.path.join(MODULE_DIR, "retrograde_game_tablebase")
COMPRESSED_TABLEBASE_DIR = os.path.join(MODULE_DIR, "compressed_game_tablebase")

# Most preferred first: the retrograde tables are exact for every reachable
# position and answer with one array index, the JSON tablebase is the fallback.
TABLEBASE_DIRS = (RETROGRADE_TABLEBASE_DIR, COMPRESSED_TABLEBASE_DIR)

USE_TABLEBASE = True
FALLBACK_RANDOM = "random"
FALLBACK_SEARCH = "search"
FALLBACK_NONE = "none"
FALLBACKS = (FALLBACK_RANDOM, FALLBACK_SEARCH, FALLBACK_NONE)

SOURCE_TABLEBASE = "tablebase"
SOURCE_RANDOM = "random"
SOURCE_SEARCH = "search"
SOURCE_ENDGAME = "endgame"
SOURCE_NONE = "none"

PLAYER_SYMBOLS = {0: "_", 1: "White", 2: "Black"}
DEFAULT_TIME_LIMIT = 3.0
MAX_TIME_LIMIT = 600.0
MAX_SEARCH_DEPTH = 100

# how many presses the official rule gives once the board is full without a line
EXTRA_TURNS = solve_game.EXTRA_TURNS


def default_tablebase_dir():
    """Absolute path of the JSON tablebase shipped next to this module."""
    return COMPRESSED_TABLEBASE_DIR


def tablebase_dirs(base_dir=None):
    """Directories to consult, most preferred first.

    An explicit ``base_dir`` replaces the whole order, which keeps a caller that
    points the engine at one directory in full control of what it reads.
    """
    if base_dir is not None:
        return (base_dir,)
    return TABLEBASE_DIRS


def available_grid_sizes(base_dir=None):
    """Grid sizes covered by any tablebase, as sorted integers."""
    if base_dir is not None:
        return get_solution.available_grid_sizes(base_dir)
    sizes = set()
    for directory in tablebase_dirs():
        sizes.update(get_solution.available_grid_sizes(directory))
    return sorted(sizes)


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
    if olgf.is_rotation_only(move):
        return f"press Orbito ({move.get('rotate', 'still')})"
    parts = []
    transfer = move.get("transfer")
    if transfer is not None:
        source, direction = transfer[0], transfer[1]
        parts.append(f"transfer ({source[0]},{source[1]})->{direction}")
    add = move.get("add")
    if add is not None:
        parts.append(f"add ({add[0]},{add[1]})")
    return ", ".join(parts) if parts else "no move"


RESULT_WIN = "win"
RESULT_DRAW = "draw"
RESULT_LOSS = "loss"
RESULT_ORDER = {RESULT_WIN: 0, RESULT_DRAW: 1, RESULT_LOSS: 2}
WIN_SCORE = 1000


def result_of_score(score):
    """Name the game result a score means for the player it is relative to."""
    if score is None:
        return None
    if score > 0:
        return RESULT_WIN
    if score < 0:
        return RESULT_LOSS
    return RESULT_DRAW


def moves_to_result(score):
    """How many plies separate a win or a loss from the end of the game."""
    if score is None or score == 0:
        return None
    return WIN_SCORE - abs(score)


def position_score(state, player_turn, rotation, transfer_allowed, base_dir=None,
                   extra_rotation_allowed=True):
    """Perfect ``(score, plies)`` of a stored position, or None when missing.

    Every tablebase directory is asked in the order :func:`tablebase_entry` uses.
    The retrograde value tables come first, because there a single array lookup
    is the whole answer, and the JSON tablebase picks the positions up
    afterwards: it also holds the boards no game from the empty board can reach,
    such as a position drawn by hand in the editor. ``plies`` is None when only
    the score is known.
    """
    for directory in tablebase_dirs(base_dir):
        entry = retrograde_tablebase.stored_score(
            directory, state, player_turn, rotation, transfer_allowed, extra_rotation_allowed)
        if entry is not None:
            return entry
        score = get_solution.lookup_score(
            state=state,
            player_turn=player_turn,
            rotate_direction=rotation,
            transfer_allowed=transfer_allowed,
            extra_rotation_allowed=extra_rotation_allowed,
            base_dir=directory,
        )
        if score is not None:
            return score, None
    return None


def endgame_score(state, player_turn, rotation, transfer_allowed, extra_rotation_allowed,
                  extra_turns=0):
    """Perfect ``(score, plies)`` of a position that only has the presses left.

    A full board without a line has no move to look up, and the presses are
    forced, so the value comes from the search engine instead. ``plies`` counts
    the presses that are still to come, which is what the value tables store for
    such a position as well.
    """
    if not extra_rotation_allowed:
        return None
    grid_size = np.asarray(state).shape[0]
    perm = solve_game._engine_params(grid_size, rotation, transfer_allowed, True)[6]
    lines_first_rest = solve_game._build_grid(grid_size)[1]
    cells = tuple(int(cell) for cell in np.asarray(state, dtype=int).ravel())
    outcome = solve_game.endgame_outcome(
        cells, perm, lines_first_rest, extra_rotation_allowed, extra_turns)
    if outcome is None:
        return None
    winner, turns, _boards = outcome
    return solve_game.endgame_score(winner, turns, player_turn), turns


def evaluate_moves(
    state,
    player_turn,
    rotation="clockwise",
    transfer_allowed=True,
    base_dir=None,
    count_traps=True,
    extra_rotation_allowed=True,
):
    """Evaluate every legal move of a position with the stored perfect scores.

    Each move is played out and the resulting position is looked up in the
    tablebase, so every move is described as a win, a draw or a loss **for the
    player to move**, together with the number of plies the game still needs.
    A stored score is measured from the position it belongs to, so the value of
    a move is rebuilt around the plies left in the new position plus the one the
    move itself used. The moves come back ordered as the player asked for them:
    wins first, then draws, then losses, and inside each group the move that ends
    the game soonest (wins) or lasts longest (losses) comes first.

    A position in the endgame has no move but the forced Orbito press, so the
    list holds that single move, valued by replaying the presses that follow it.

    Every entry is a dictionary:

    - ``move``: the move, ready for :func:`orbital_logic_game_functions.play_turn`
    - ``state``: the position the move leads to
    - ``score``: the perfect score of the move for the player to move
    - ``result``: ``"win"``, ``"draw"`` or ``"loss"``. A move that gives both
      players a line at once is a draw, the way the game itself calls it.
    - ``moves_to_result``: plies left until the game ends, None for a draw
    - ``traps``: how many of the opponent's replies in the new position lose
      for the opponent, the number of chances the move offers to slip up
    - ``text``: one line describing the move, e.g. ``"add (1,1) -> Black wins"``

    ``count_traps=False`` skips the ``traps`` count, which is what the counting
    itself needs: it is one tablebase lookup per move of the opponent.

    ``complete`` tells whether every legal move could be evaluated; it is False
    when a resulting position is missing from the tablebase. An empty list means
    the position itself is not stored.
    """
    state = np.asarray(state, dtype=int)
    if player_turn not in (1, 2):
        raise ValueError("player_turn must be 1 or 2")
    if olgf.evaluate_game_state(state) is not None:
        return [], True
    opponent = 3 - player_turn
    evaluated = []
    complete = True
    possible = olgf.get_possible_moves(state, rotation, transfer_allowed, player_turn)
    if not possible and extra_rotation_allowed:
        # nothing can be placed or transferred: the game is down to the presses
        press = olgf.rotation_only_move(player_turn, rotation)
        following = olgf.play_turn(state, press)
        finished = olgf.evaluate_game_state(following)
        if finished is not None:
            # the first press already shows a line, which settles the game at once
            score = (WIN_SCORE - 1) if finished == player_turn else -(WIN_SCORE - 1)
            plies = 1
        else:
            entry = endgame_score(following, opponent, rotation, transfer_allowed, True, 1)
            if entry is None:
                return [], True
            child_score, child_plies = entry
            plies = child_plies + 1
            if child_score == 0:
                score = 0
            else:
                # the child value is the opponent's: a win of theirs is this
                # player's loss, and both are measured over the presses left
                distance = WIN_SCORE - plies
                score = distance if child_score < 0 else -distance
        evaluated.append({
            "move": press,
            "state": following,
            "score": score,
            "result": result_of_score(score),
            "moves_to_result": (plies if score != 0 else None),
            "traps": 0,
            "text": move_result_text(press, score, player_turn, plies),
        })
        return evaluated, True
    for move in possible:
        following = olgf.play_turn(state, move)
        finished = olgf.evaluate_game_state(following)
        if finished == 0:
            # the move ended the game, and it ended it in a draw: a rotation can
            # give both players a line at the same time, which is not a loss
            score = 0
            plies = 1
        elif finished is not None:
            # the move ended the game, its score is settled on the spot
            score = (WIN_SCORE - 1) if finished == player_turn else -(WIN_SCORE - 1)
            plies = 1
        else:
            entry = position_score(
                following, opponent, rotation, transfer_allowed, base_dir, extra_rotation_allowed)
            if entry is None:
                complete = False
                continue
            child_score, child_plies = entry
            if child_plies is None:
                # a position that is only stored as a score has no distance to
                # count on, so the move keeps the stored one and reports none
                score = -child_score
                plies = None
            else:
                # a stored score is measured from the position it belongs to:
                # the plies that are left in it. The move that led there is one
                # more, and the value has to be rebuilt around that number. A
                # plain negation of the score would be one ply too short, which
                # is what turned a reply that loses into "loses at once".
                plies = child_plies + 1
                if child_score == 0:
                    score = 0
                else:
                    distance = WIN_SCORE - plies
                    score = distance if child_score < 0 else -distance
        evaluated.append({
            "move": move,
            "state": following,
            "score": score,
            "result": result_of_score(score),
            "moves_to_result": (plies if score != 0 else None),
            "traps": (count_losing_replies(following, opponent, rotation,
                                           transfer_allowed, base_dir,
                                           extra_rotation_allowed)
                      if count_traps else 0),
            "text": move_result_text(move, score, player_turn, plies),
        })
    evaluated.sort(key=lambda item: (RESULT_ORDER[item["result"]],
                                     -item["score"], -item["traps"]))
    return evaluated, complete


def count_losing_replies(state, player_turn, rotation, transfer_allowed, base_dir,
                         extra_rotation_allowed=True):
    """How many of the moves of ``player_turn`` lose the game for them.

    A position that wins for the player to move usually offers them more than
    one way to play it, and every reply that ends up losing is a chance for the
    opponent to make a bad next move. Counting them is what turns two moves of
    the same score into a choice.
    """
    moves, _complete = evaluate_moves(
        state, player_turn, rotation, transfer_allowed, base_dir, count_traps=False,
        extra_rotation_allowed=extra_rotation_allowed)
    return sum(1 for item in moves if item["result"] == RESULT_LOSS)


def move_result_text(move, score, player_turn, plies=None):
    """One line describing a move and what it achieves for the player to move.

    The line is written from the point of view of the player to move, so
    "Black loses in 3 moves" is a move that hands the win to White, and the
    report reads the same way down the whole list.

    ``plies`` is the length of the perfect line that starts with this move, the
    move itself included, so it is one more than the value of the position the
    move leads to. It is taken from the score when it is not given.
    """
    move_text = move_to_text(move)
    if score == 0:
        return f"{move_text} -> draw"
    if plies is None:
        plies = moves_to_result(score)
    outcome = "wins" if score > 0 else "loses"
    if plies == 1:
        return f"{move_text} -> {PLAYER_SYMBOLS[player_turn]} {outcome} at once"
    return f"{move_text} -> {PLAYER_SYMBOLS[player_turn]} {outcome} in {plies} moves"


def best_evaluated_move(evaluated):
    """Pick the move of an evaluation list to actually play.

    The perfect score decides, and moves of the same score are separated by the
    number of losing replies they leave to the opponent, so the computer takes
    the move that gives the opponent the most chances to slip up. That only
    breaks ties, so the result of the game is the one the tablebase proves.
    """
    if not evaluated:
        return None
    return max(evaluated, key=lambda item: (item["score"], item["traps"]))["move"]


def tablebase_entry(state, player_turn, rotation, transfer_allowed, base_dir=None,
                    extra_rotation_allowed=True):
    """Return the tablebase entry of a position, or None when absent.

    The retrograde value tables are consulted first and the compressed JSON
    tablebase afterwards, so the computer follows the stored perfect play of the
    value table whenever one exists for the rule context. A counterclockwise
    position is answered from the clockwise table by a reflection of the board,
    which is why a build only needs the ``still`` and ``clockwise`` contexts.
    Positions whose game already ended have no legal move to suggest, so they
    report None as well. A position in the endgame is answered from the forced
    press, which the tables store as the value of that very position.
    """
    if olgf.evaluate_game_state(state) is not None:
        return None
    for directory in tablebase_dirs(base_dir):
        try:
            entry = get_solution.lookup_solution(
                state=state,
                player_turn=player_turn,
                rotate_direction=rotation,
                transfer_allowed=bool(transfer_allowed),
                extra_rotation_allowed=extra_rotation_allowed,
                base_dir=directory,
            )
        except ValueError:
            return None
        if entry is not None:
            return entry
    return None


def position_id(state, player_turn, rotation="clockwise", transfer_allowed=True,
                extra_rotation_allowed=True):
    """The ID of a position: its state alone, as base-3 digits.

    The cells are written row by row, so a 4x4 position reads as sixteen digits
    such as ``0000121102102211``. The rule context and the side to move are not
    part of it, which is why the arguments are accepted but not used: they say
    how the position would be played, not which position it is. The result does
    not depend on the position being stored anywhere.
    """
    return create_game_tablebase.position_state_id(state)


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
    extra_rotation_allowed=True,
    extra_turns=0,
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

    A tablebase answer also carries ``moves``: every legal move of the position
    with its perfect score, ordered wins, draws, losses (see
    :func:`evaluate_moves`). ``move`` is then the best of them, ties broken by
    the number of losing replies left to the opponent.

    ``extra_turns`` counts the endgame presses that have already been played.
    They are not part of the board, so the value tables only know the presses
    that start at one; a position that is already into them is answered by
    replaying the rest.

    Every answer carries ``position_id``, the tablebase ID of the position (see
    :func:`position_id`), whether or not a tablebase knows it.
    """
    state = np.asarray(state, dtype=int)
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("state must be a square two-dimensional array")
    if player_turn not in (1, 2):
        raise ValueError("player_turn must be 1 or 2")
    if fallback not in FALLBACKS:
        raise ValueError(f"fallback must be one of: {', '.join(FALLBACKS)}")
    start = time.monotonic()
    # every answer names the position it is about, stored or not, so a report
    # can always be traced back to a tablebase entry
    identifier = position_id(state, player_turn, rotation, transfer_allowed, extra_rotation_allowed)

    if olgf.evaluate_game_state(state) is not None:
        return _result(
            SOURCE_NONE,
            message="The game is already over.",
            player_turn=player_turn,
            elapsed=0.0,
            position_id=identifier,
        )

    if use_tablebase and extra_turns <= 0:
        entry = tablebase_entry(
            state, player_turn, rotation, transfer_allowed, base_dir, extra_rotation_allowed)
        if entry is not None and entry["best_move"] is None:
            # the position is stored but has no move left to make: a full board
            # without a line, which is the draw the endgame presses used to be
            return _result(
                SOURCE_TABLEBASE,
                move=None,
                score=entry["score"],
                message=(
                    f"Tablebase (perfect play): the board is full and nobody has a "
                    f"line, so it is a {game_result_message(entry['score'], player_turn)}"
                ),
                player_turn=player_turn,
                elapsed=time.monotonic() - start,
                text=entry["text"],
                moves_sequence=entry["moves_sequence"],
                states_sequence=[np.asarray(item, dtype=int) for item in entry["states_sequence"]],
                position_id=entry["id"],
                moves=[],
                moves_complete=True,
            )
        if entry is not None and entry["best_move"] is not None:
            evaluated, complete = evaluate_moves(
                state, player_turn, rotation, transfer_allowed, base_dir,
                extra_rotation_allowed=extra_rotation_allowed)
            # the tablebase decides the result, the traps only pick between the
            # moves that reach it
            best = best_evaluated_move(evaluated) or entry["best_move"]
            counted = len(evaluated)
            note = (f" of {counted} legal move{'' if counted == 1 else 's'}"
                    if complete else
                    f" of {counted} evaluated move{'' if counted == 1 else 's'}")
            return _result(
                SOURCE_TABLEBASE,
                move=best,
                score=entry["score"],
                message=(
                    f"Tablebase (perfect play, best{note}): "
                    f"{move_to_text(best)} -> "
                    f"{game_result_message(entry['score'], player_turn)}"
                ),
                player_turn=player_turn,
                elapsed=time.monotonic() - start,
                text=entry["text"],
                moves_sequence=entry["moves_sequence"],
                states_sequence=[np.asarray(item, dtype=int) for item in entry["states_sequence"]],
                position_id=entry["id"],
                moves=evaluated,
                moves_complete=complete,
            )

    if extra_turns > 0 and extra_rotation_allowed:
        # part way through the presses: the board is settled, the presses are not
        entry = endgame_score(
            state, player_turn, rotation, transfer_allowed, extra_rotation_allowed, extra_turns)
        if entry is not None:
            value, presses_left = entry
            if presses_left <= 0:
                # every press is spent and nobody ever held a line: the game is
                # over, there is no further press to offer
                return _result(
                    SOURCE_ENDGAME,
                    move=None,
                    score=value,
                    message=(
                        f"All {int(solve_game.EXTRA_TURNS)} endgame presses are spent "
                        f"without a line: {game_result_message(value, player_turn)}"
                    ),
                    player_turn=player_turn,
                    elapsed=time.monotonic() - start,
                    position_id=identifier,
                    presses_remaining=0,
                )
            press = olgf.rotation_only_move(player_turn, rotation)
            message = (
                f"Endgame press {extra_turns + 1} of {int(solve_game.EXTRA_TURNS)} "
                f"({presses_left} to go): {game_result_message(value, player_turn)}"
            )
            return _result(
                SOURCE_ENDGAME,
                move=press,
                score=value,
                message=message,
                player_turn=player_turn,
                elapsed=time.monotonic() - start,
                position_id=identifier,
                presses_remaining=presses_left,
            )

    if fallback == FALLBACK_NONE:
        return _result(
            SOURCE_NONE,
            message="This position is not in the tablebase.",
            player_turn=player_turn,
            elapsed=time.monotonic() - start,
            position_id=identifier,
        )

    if fallback == FALLBACK_RANDOM:
        rng = None if seed is None else random.Random(seed)
        move, _value = solve_game.find_random_move(
            state, rotation, transfer_allowed, player_turn, rng,
            extra_rotation_allowed=extra_rotation_allowed, extra_turns=extra_turns,
        )
        if move is None:
            return _result(
                SOURCE_NONE,
                message="No legal move is available.",
                player_turn=player_turn,
                elapsed=time.monotonic() - start,
                position_id=identifier,
            )
        return _result(
            SOURCE_RANDOM,
            move=move,
            message=f"Random move (not in the tablebase): {move_to_text(move)}",
            player_turn=player_turn,
            elapsed=time.monotonic() - start,
            position_id=identifier,
        )

    limit = clamp_time_limit(time_limit)
    move, score, info = solve_game.find_best_move_within_time(
        state, rotation, transfer_allowed, player_turn, time_limit=limit,
        max_depth=max_depth,
        extra_rotation_allowed=extra_rotation_allowed, extra_turns=extra_turns,
    )
    if move is None:
        return _result(
            SOURCE_NONE,
            message="No legal move is available.",
            player_turn=player_turn,
            elapsed=time.monotonic() - start,
            search=info,
            position_id=identifier,
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
        position_id=identifier,
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
            try:
                identifier = position_id(
                    request["state"],
                    int(request.get("player_turn", 1)),
                    request.get("rotation", "clockwise"),
                    request.get("transfer_allowed", True),
                    request.get("extra_rotation_allowed", True),
                )
            except Exception:
                identifier = None
            result = _result(
                SOURCE_NONE,
                message=f"Engine error: {type(error).__name__}: {error}",
                player_turn=int(request.get("player_turn", 1)),
                elapsed=0.0,
                position_id=identifier,
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
