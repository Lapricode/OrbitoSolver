import math
import multiprocessing as mp
import os
import random
import numpy as np
import time

import orbital_logic_game_functions as olgf

# ---------------------------------------------------------------------------
# Fast game engine
#
# Boards are represented internally as flat tuples of cells (0 = empty,
# 1 = player 1, 2 = player 2). All per-boardsize data (winning lines,
# neighbor tables, rotation permutations) is precomputed once and cached.
# The search uses a negamax formulation with alpha/beta pruning, a
# transposition table (depth-independent scores) and a cheap static move
# ordering heuristic. Only the public API below (minimax, find_best_move,
# solve_game, simulate_principal_variation, estimated_game_result) converts
# between numpy boards and the fast internal representation.
#
# The endgame presses
# -------------------
# When the last cell is filled and nobody holds a line, the board has no
# empty cell left, so no piece can be added and nothing can be transferred.
# The rules then hand the game to the Orbito button: the board is pressed
# ``EXTRA_TURNS`` more times, and the first press that shows a line of n
# decides the game; if no press shows one, the game is a draw. Because
# nothing is placed, a press is nothing but a rotation, and the presses are
# forced, so a position in the endgame has exactly one continuation. The
# rule is optional (``extra_rotation_allowed``): switched off, a full board
# without a line is simply the draw the endgame presses used to be.
# ---------------------------------------------------------------------------

WIN_SCORE = 1000          # same scale used by the original minimax
INF = 10**9
EXACT, LOWER, UPPER = 0, 1, 2
TT_MAX = 4_000_000        # hard cap on transposition table entries
EXTRA_TURNS = 5           # Orbito presses a full board without a line gets

_grid_cache = {}          # n -> (lines, line_first_rest, cell_lines, neighbours, perm_cw, perm_ccw)


def _build_grid(n):
    data = _grid_cache.get(n)
    if data is not None:
        return data
    nn = n * n
    lines_full = []
    for r in range(n):
        lines_full.append(tuple(r * n + c for c in range(n)))
    for c in range(n):
        lines_full.append(tuple(r * n + c for r in range(n)))
    lines_full.append(tuple(i * n + i for i in range(n)))
    lines_full.append(tuple(i * n + (n - 1 - i) for i in range(n)))
    lines_first_rest = [(line[0], line[1:]) for line in lines_full]
    cell_lines = [[] for _ in range(nn)]
    for line in lines_full:
        for idx in line:
            cell_lines[idx].append(line)
    neighbours = []
    for r in range(n):
        for c in range(n):
            i = r * n + c
            neighbours.append((i - n if r > 0 else None,
                               i + n if r < n - 1 else None,
                               i - 1 if c > 0 else None,
                               i + 1 if c < n - 1 else None))
    perm_cw = _build_rotation_perm(n, True)
    perm_ccw = _build_rotation_perm(n, False)
    data = (lines_full, lines_first_rest, cell_lines, neighbours, perm_cw, perm_ccw)
    _grid_cache[n] = data
    return data


def _build_rotation_perm(n, clockwise):
    nn = n * n
    perm = list(range(nn))
    assigned = set()
    for layer in range(n // 2):
        elements = ([(layer, j) for j in range(layer, n - layer)] +
                    [(i, n - layer - 1) for i in range(layer + 1, n - layer - 1)] +
                    [(n - layer - 1, j) for j in range(n - layer - 1, layer - 1, -1)] +
                    [(i, layer) for i in range(n - layer - 2, layer, -1)])
        flat = [r * n + c for r, c in elements]
        L = len(flat)
        if clockwise:
            # destination elements[k] receives source elements[k - 1]
            for k in range(L):
                perm[flat[k]] = flat[k - 1]
                assigned.add(flat[k])
        else:
            for k in range(L):
                perm[flat[k]] = flat[(k + 1) % L]
                assigned.add(flat[k])
    for i in range(nn):
        if i not in assigned:
            perm[i] = i  # odd-board center cell keeps its value
    return tuple(perm)


def _engine_params(n, rotate_direction, transfer_allowed, extra_rotation_allowed = True):
    """Bundle all per-search data into one object handed to the search."""
    lines_full, lines_first_rest, cell_lines, neighbours, perm_cw, perm_ccw = _build_grid(n)
    d = rotate_direction.lower() if isinstance(rotate_direction, str) else rotate_direction
    if d in olgf.clockwise_rotation_keywords:
        perm = perm_cw
    elif d in olgf.counterclockwise_rotation_keywords:
        perm = perm_ccw
    else:
        perm = None  # "still" (or invalid -> no rotation)
    return (n, n * n, lines_full, lines_first_rest, cell_lines, neighbours, perm,
            transfer_allowed, bool(extra_rotation_allowed))


def _winner(b, lines_first_rest):
    """Return 1 or 2 if that player has won, 0 if BOTH players have a full line
    (a draw), or None if the game is not over yet."""
    win = 0
    for first, rest in lines_first_rest:
        v = b[first]
        if v:
            ok = True
            for idx in rest:
                if b[idx] != v:
                    ok = False
                    break
            if ok:
                if win == 0:
                    win = v
                elif win != v:
                    return 0  # both players reached a line at once -> draw
    return win or None


def _apply_move(b, move, player, perm, neighbours):
    """Apply a move (transfer_source, transfer_dir, add_cell) and rotate."""
    tsrc, tdir, add = move
    lst = list(b)
    if tsrc is not None:
        tgt = neighbours[tsrc][tdir]
        lst[tgt] = b[tsrc]
        lst[tsrc] = 0
    lst[add] = player
    if perm is None:
        return tuple(lst)
    return tuple(lst[i] for i in perm)


def endgame_score(winner, turns, player):
    """Negamax value of an endgame press sequence for the side to move.

    ``winner`` is the player whose line the first press that shows one
    completes, or 0 when the presses show no line at all or show one for both
    players at once. ``turns`` is the number of presses the game took, counted
    from the position the press sequence starts at, and ``player`` is the side
    to move there. A draw is worth 0, everything else is ``WIN_SCORE`` less the
    number of presses, so a faster win scores higher, exactly as in the search.
    """
    if winner == 0:
        return 0
    return WIN_SCORE - turns if winner == player else turns - WIN_SCORE


def _endgame_presses(board, perm, lines_first_rest, presses):
    """Turn a full board up to ``presses`` times and report what the presses show.

    Returns ``(winner, turns, boards)``: the player whose line the first press
    that shows one completes (0 when no press shows one, or one is shown for
    both players at once), how many presses were needed, and the boards those
    presses led to. A still rotation leaves the board as it is, so a game that
    rotates by ``still`` never sees a line appear and always ends in a draw.
    """
    if presses <= 0:
        return 0, 0, []
    cells = board
    boards = []
    for turn in range(1, presses + 1):
        cells = cells if perm is None else tuple(cells[i] for i in perm)
        boards.append(cells)
        winner = _winner(cells, lines_first_rest)
        if winner is not None:
            return winner, turn, boards
    return 0, presses, boards


def endgame_outcome(board, perm, lines_first_rest, extra_rotation_allowed, extra_turns = 0):
    """The endgame presses of a position, or None when the position has none.

    A position has endgame presses when the rule is switched on, the board is
    full, and nobody holds a line yet. ``extra_turns`` counts how many of the
    ``EXTRA_TURNS`` presses are already spent, which is not part of the board:
    only the caller knows it, so the presses are replayed rather than read from
    a table.

    Returns the ``(winner, turns, boards)`` triple of :func:`_endgame_presses`,
    with the number of presses left taken into account.
    """
    if not extra_rotation_allowed:
        return None
    if any(cell == 0 for cell in board):
        return None
    if _winner(board, lines_first_rest) is not None:
        return None
    spent = max(0, min(EXTRA_TURNS, int(extra_turns)))
    return _endgame_presses(board, perm, lines_first_rest, EXTRA_TURNS - spent)


def _gen_moves(b, player, transfer_allowed, neighbours, nn):
    """Generate all legal moves for `player`, in the exact same order as
    get_possible_moves(). A move is the tuple (transfer_source, transfer_dir,
    add_cell), where transfer_source is a flat cell index (or None) and
    transfer_dir is 0=up,1=down,2=left,3=right."""
    moves = []
    append = moves.append
    opponent = 3 - player
    for i in range(nn):
        if b[i] == 0:
            append((None, None, i))
    if transfer_allowed:
        for i in range(nn):
            if b[i] == opponent:
                for d in range(4):
                    tgt = neighbours[i][d]
                    if tgt is not None and b[tgt] == 0:
                        for e in range(nn):
                            if (b[e] == 0 and e != tgt) or e == i:
                                append((i, d, e))
    return moves


def _cheap_score(move, b, player, n, cell_lines):
    """Static move-ordering score based on the lines that pass through the
    add cell. Prefers moves that complete (or approach) a line."""
    add = move[2]
    opponent = 3 - player
    total = 0
    for line in cell_lines[add]:
        own = 0
        blocked = False
        for idx in line:
            if idx == add:
                continue  # this cell is about to become the player's piece
            v = b[idx]
            if v == player:
                own += 1
            elif v == opponent:
                blocked = True
                break
        if not blocked:
            if own == n - 1:
                return 10**6  # immediate win
            total += own * own
    return total


def _order_moves(moves, b, player, P, bm_hint):
    """Order moves: transposition-table move first (if legal), then the
    remaining moves sorted by the cheap heuristic (best first)."""
    if bm_hint is not None:
        try:
            idx = moves.index(bm_hint)
        except ValueError:
            idx = -1
        if idx >= 0:
            rest = moves[:idx] + moves[idx + 1:]
            rest.sort(key=lambda m: _cheap_score(m, b, player, P[0], P[4]), reverse=True)
            return [bm_hint] + rest
    moves.sort(key=lambda m: _cheap_score(m, b, player, P[0], P[4]), reverse=True)
    return moves


def _tt_store(tt, key, depth, value, bound, move):
    if len(tt) >= TT_MAX:
        tt.clear()
    tt[key] = (depth, value, bound, move)


# ---------------------------------------------------------------------------
# Search clock
#
# A deadline can be installed around a search so that it gives up instead of
# running forever. Only the root loop of an iterative deepening search polls
# the clock itself, so the check inside the hot recursion is reduced to a
# single global lookup and a counter increment (the actual clock is read once
# every _NODE_CHECK_MASK nodes).
# ---------------------------------------------------------------------------

_SEARCH_DEADLINE = None
_NODE_COUNT = 0
_NODE_CHECK_MASK = 2047


class _SearchTimeout(Exception):
    """Raised inside the search when the installed deadline has passed."""


def _start_search(deadline):
    global _SEARCH_DEADLINE, _NODE_COUNT
    _SEARCH_DEADLINE = deadline
    _NODE_COUNT = 0


def _stop_search():
    """Uninstall the deadline and return the number of nodes visited."""
    global _SEARCH_DEADLINE, _NODE_COUNT
    _SEARCH_DEADLINE = None
    nodes = _NODE_COUNT
    _NODE_COUNT = 0
    return nodes


def _tick_clock():
    global _NODE_COUNT
    _NODE_COUNT += 1
    if not (_NODE_COUNT & _NODE_CHECK_MASK) and time.monotonic() >= _SEARCH_DEADLINE:
        raise _SearchTimeout


def _endgame_value(b, player, perm, lines_first_rest, extra_rotation_allowed, depth = 0):
    """Value of a position without a single legal move.

    With the endgame presses switched off a full board without a line is the
    draw the game has always been. With them switched on the presses are
    forced, so the value follows from the board alone: it is the outcome of the
    first press that shows a line, ``depth`` presses further down the search.
    """
    outcome = endgame_outcome(b, perm, lines_first_rest, extra_rotation_allowed)
    if outcome is None:
        return 0
    winner, turns, _boards = outcome
    return endgame_score(winner, turns + depth, player)


def _negamax(b, player, depth, alpha, beta, tt, P, exact_only=False):
    """Negamax with alpha/beta pruning and a transposition table.

    Scores are relative to the side to move and shift linearly with the
    absolute search depth (faster wins are preferred). The transposition
    table therefore stores the depth at which a value was computed so it can
    be reused at a different absolute depth, and each entry also keeps a
    bound flag (EXACT/LOWER/UPPER). When exact_only is True, only EXACT
    bounds are trusted (used while reconstructing a principal variation)."""
    if _SEARCH_DEADLINE is not None:
        _tick_clock()
    n, nn, lines_full, lines_first_rest, cell_lines, neighbours, perm, transfer_allowed, extra_rotation_allowed = P

    winner = _winner(b, lines_first_rest)
    if winner is not None:
        if winner == player:
            return WIN_SCORE - depth
        if winner == 0:
            return 0
        return depth - WIN_SCORE

    key = (b, player)
    entry = tt.get(key)
    bm_hint = None
    if entry is not None:
        e_depth, e_val, e_bound, e_move = entry
        if not exact_only:
            adj = e_val + (e_depth - depth)  # shift value to the current depth
            if e_bound == EXACT:
                return adj
            if e_bound == LOWER:
                if adj >= beta:
                    return adj
                if adj > alpha:
                    alpha = adj
            else:  # UPPER
                if adj <= alpha:
                    return adj
                if adj < beta:
                    beta = adj
        bm_hint = e_move

    moves = _gen_moves(b, player, transfer_allowed, neighbours, nn)
    if not moves:
        return _endgame_value(b, player, perm, lines_first_rest, extra_rotation_allowed, depth)
    moves = _order_moves(moves, b, player, P, bm_hint)

    alpha_orig = alpha
    best = -INF
    best_move = moves[0]
    for move in moves:
        child = _apply_move(b, move, player, perm, neighbours)
        v = -_negamax(child, 3 - player, depth + 1, -beta, -alpha, tt, P, exact_only)
        if v > best:
            best = v
            best_move = move
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break

    if best >= beta:
        bound = LOWER
    elif best <= alpha_orig:
        bound = UPPER
    else:
        bound = EXACT
    if not exact_only:
        _tt_store(tt, key, depth, best, bound, best_move)
    return best


def _best_move_at(b, player, depth, tt, P):
    """Exact search of the subtree rooted at (b, player) that returns only
    the very best move (used to reconstruct a principal variation without
    trusting the possibly-pruned move stored inside a bounded TT entry)."""
    n, nn, lines_full, lines_first_rest, cell_lines, neighbours, perm, transfer_allowed, extra_rotation_allowed = P
    moves = _gen_moves(b, player, transfer_allowed, neighbours, nn)
    if not moves:
        return None
    moves = _order_moves(moves, b, player, P, None)
    best = -INF
    best_move = None
    for move in moves:
        child = _apply_move(b, move, player, perm, neighbours)
        v = -_negamax(child, 3 - player, depth + 1, -INF, INF, tt, P, exact_only=True)
        if v > best:
            best = v
            best_move = move
    return best_move


_DIGITS = ("u", "d", "l", "r")
TOP_MOVES = 5          # how many scored moves the time limited search remembers


def _scored_value(item):
    """Sort key for a (value, move) pair; moves are never compared to each other."""
    return item[0]


def _best_scored(scored, count):
    """Return the ``count`` best (value, move) pairs, best first."""
    return sorted(scored, key=_scored_value, reverse=True)[:count]


# ---------------------------------------------------------------------------
# Optional parallel root search
# The top-level (root) move evaluations are completely independent subtrees,
# so they can be dispatched to a pool of worker processes. Each worker keeps
# its own transposition table; the tables are merged afterwards so the
# principal-variation reconstruction still hits exact entries. With workers
# left as None the original single-process search is used, which is exact and
# bit-for-bit identical to before.
# ---------------------------------------------------------------------------

_WORKER_P = None


def _worker_init_parallel(P):
    global _WORKER_P
    _WORKER_P = P


def _worker_root_eval(task):
    """Evaluate the subtree after one root move for the given player.

    Returns (value, move, tt) where the value is already negated so it uses
    the same sign convention as the serial root loop, and tt is the worker's
    local transposition table (freed from pruning bounds by searching each
    root child fully).
    """
    b, move, player, depth = task
    P = _WORKER_P
    child = _apply_move(b, move, player, P[6], P[5])
    tt = {}
    v = -_negamax(child, 3 - player, depth + 1, -INF, INF, tt, P)
    return v, move, tt


def _solve_root_moves_parallel(b, player, depth, moves, P, workers):
    """Values for every root move, computed on a pool of worker processes."""
    tasks = [(b, move, player, depth) for move in moves]
    with mp.Pool(workers, initializer=_worker_init_parallel, initargs=(P,)) as pool:
        results = pool.map(_worker_root_eval, tasks)
    merged_tt = {}
    for _v, _m, sub_tt in results:
        merged_tt.update(sub_tt)
    return results, merged_tt


def _endgame_answer(b, player, P, rotate_direction, n, extra_turns = 0):
    """The single move of a position in the endgame, and its value.

    The presses are forced, so a position that has them has exactly one
    continuation: an Orbito press that only turns the board. ``(None, None)`` is
    returned when the position has no legal move at all, which is what a full
    board without a line is when the rule is switched off. Once every press is
    spent the position is over, and the score is returned on its own.
    """
    outcome = endgame_outcome(b, P[6], P[3], P[8], extra_turns)
    if outcome is None:
        return None, None
    winner, turns, _boards = outcome
    score = endgame_score(winner, turns, player)
    if turns <= 0:
        return None, score
    return olgf.rotation_only_move(player, rotate_direction), score


def _endgame_line(b, player, P, rotate_direction, n, extra_turns = 0):
    """The presses of the endgame as a move list, and the value they end on.

    Returns ``(moves, states, score)``, or ``([], [], None)`` when the position
    has no endgame presses. The moves are the forced presses, so a line that
    runs into the endgame is finished by them instead of stopping short.
    """
    outcome = endgame_outcome(b, P[6], P[3], P[8], extra_turns)
    if outcome is None:
        return [], [], None
    winner, turns, boards = outcome
    moves = []
    states = []
    mover = player
    for cells in boards:
        moves.append(olgf.rotation_only_move(mover, rotate_direction))
        mover = 3 - mover
        states.append(np.array(cells, dtype = int).reshape(n, n))
    return moves, states, endgame_score(winner, turns, player)


def _move_to_dict(move, player, rotate_direction, n):
    """Convert a compact (transfer_source, transfer_dir, add_cell) move back
    into the dictionary format used by the public API."""
    tsrc, tdir, add = move
    if tsrc is None:
        transfer = None
    else:
        transfer = [(tsrc // n, tsrc % n), _DIGITS[tdir]]
    add_pos = (add // n, add % n)
    return {"player": player, "transfer": transfer, "add": add_pos, "rotate": rotate_direction}


# ---------------------------------------------------------------------------
# Public API (same signatures and semantics as the original implementation)
# ---------------------------------------------------------------------------

def minimax(state, rotate_direction = "clockwise", transfer_allowed = True, player = 1, maximizing_player = None, depth = 0, alpha = -math.inf, beta = math.inf, extra_rotation_allowed = True):
    """
    Recursively evaluates the game tree using minimax with alpha-beta pruning.
    Parameters:
        state: the current game state (a numpy array)
        rotate_direction: the direction in which the board is rotated (clockwise, counterclockwise, or still)
        transfer_allowed: whether transfers are allowed between adjacent cells
        player: the player whose turn it is (1 or 2)
        maximizing_player: the player we are trying to maximize (the one for whom we want the best move)
        depth: current depth of recursion (used to favor faster wins/longer delays of losses)
        alpha: the best already explored option along the path to the maximizer
        beta: the best already explored option along the path to the minimizer
        extra_rotation_allowed: whether a full board without a line is decided by the EXTRA_TURNS Orbito presses
    Returns:
        An evaluation score: high positive if maximizing_player wins, high negative if loses, or 0 for a draw.
    """
    if maximizing_player is None:
        maximizing_player = player
    P = _engine_params(state.shape[0], rotate_direction, transfer_allowed, extra_rotation_allowed)
    b = tuple(state.ravel().tolist())
    value = _negamax(b, player, depth, -INF if alpha == -math.inf else alpha,
                     INF if beta == math.inf else beta, {}, P)
    if player != maximizing_player:
        value = -value
    return value


def find_best_move(state, rotate_direction = "clockwise", transfer_allowed = True, player = 1, workers = None, extra_rotation_allowed = True, extra_turns = 0):
    """
    Determines the best move for the given player from the current state.
    Parameters:
        state: the current game state (a numpy array)
        player: the player whose move is to be determined (1 or 2)
        workers: optional number of worker processes used to evaluate the
            root move subtrees in parallel (None / 1 = single process)
        extra_rotation_allowed: whether a full board without a line is decided by the EXTRA_TURNS Orbito presses
        extra_turns: how many of those presses have already been played
    Returns:
        A tuple (best_move, best_value) where best_move is the move (a dictionary)
        and best_value is its minimax evaluation. A position in the endgame has a
        single forced continuation, an Orbito press that only turns the board.
    """
    n = state.shape[0]
    P = _engine_params(n, rotate_direction, transfer_allowed, extra_rotation_allowed)
    b = tuple(state.ravel().tolist())
    moves = _gen_moves(b, player, P[7], P[5], P[1])
    if not moves:
        return _endgame_answer(b, player, P, rotate_direction, n, extra_turns)
    moves = _order_moves(moves, b, player, P, None)
    best = -INF
    best_move = None
    if workers is not None and int(workers) > 1 and len(moves) > 1:
        try:
            results, _tt = _solve_root_moves_parallel(b, player, 0, moves, P, int(workers))
        except Exception:
            results = None
        if results is not None:
            for v, move, _sub in results:
                if v > best:
                    best = v
                    best_move = move
            return _move_to_dict(best_move, player, rotate_direction, n), best
    tt = {}
    for move in moves:
        child = _apply_move(b, move, player, P[6], P[5])
        v = -_negamax(child, 3 - player, 1, -INF, INF, tt, P)
        if v > best:
            best = v
            best_move = move
    return _move_to_dict(best_move, player, rotate_direction, n), best


def find_random_move(state, rotate_direction = "clockwise", transfer_allowed = True, player = 1, rng = None, extra_rotation_allowed = True, extra_turns = 0):
    """
    Picks one legal move at random, which is useful for quick games and as a
    fallback for a computer opponent that should not think too hard.
    Parameters:
        state: the current game state (a numpy array)
        rotate_direction: the direction in which the board is rotated
        transfer_allowed: whether transfers are allowed between adjacent cells
        player: the player whose move is chosen (1 or 2)
        rng: an optional random.Random instance (a fresh one is used when omitted)
        extra_rotation_allowed: whether a full board without a line is decided by the EXTRA_TURNS Orbito presses
        extra_turns: how many of those presses have already been played
    Returns:
        A tuple (move, value) where move is a random legal move (a dictionary) and
        value is always None, or (None, None) when no move is available. A position
        in the endgame has only the forced press, so that is the move returned.
    """
    n = state.shape[0]
    P = _engine_params(n, rotate_direction, transfer_allowed, extra_rotation_allowed)
    b = tuple(state.ravel().tolist())
    moves = _gen_moves(b, player, P[7], P[5], P[1])
    if not moves:
        move, _value = _endgame_answer(b, player, P, rotate_direction, n, extra_turns)
        return move, None
    generator = random if rng is None else rng
    move = generator.choice(moves)
    return _move_to_dict(move, player, rotate_direction, n), None


def _search_with_deadline(b, player, P, deadline, max_depth):
    """Iterative deepening root search that stops as soon as the deadline hits.

    Each iteration searches every root move to the same depth, so the move
    values of a single iteration are directly comparable; the values of the
    last *completed* iteration are the ones returned. Returns
    (best_move, best_value, depth_reached, top_moves, timed_out, reason) with
    ``reason`` telling why the deepening stopped: "max_depth" when the cap was
    reached, "timeout" when the deadline cut an iteration short, "proven" when
    the horizon already decided the result and "no_moves" when the root has
    nothing left to try."""
    tt = {}
    ordered = _order_moves(_gen_moves(b, player, P[7], P[5], P[1]), b, player, P, None)
    best_move, best_value, best_depth, top_moves = ordered[0], None, 0, []
    timed_out = False
    reason = "max_depth"
    depth = 1
    while max_depth is None or depth <= max_depth:
        scored = []
        partial_best, partial_value, partial_timed_out = None, None, False
        for move in ordered:
            if time.monotonic() >= deadline:
                partial_timed_out = True
                break
            child = _apply_move(b, move, player, P[6], P[5])
            try:
                value = -_negamax(child, 3 - player, depth, -INF, INF, tt, P)
            except _SearchTimeout:
                partial_timed_out = True
                break
            scored.append((value, move))
            if partial_value is None or value > partial_value:
                partial_value, partial_best = value, move
        if partial_timed_out:
            timed_out = True
            reason = "timeout"
            if best_depth == 0 and partial_best is not None:
                # the very first iteration was cut short: keep the best move
                # that was fully evaluated (its score is not reported though)
                best_move = partial_best
                top_moves = _best_scored(scored, TOP_MOVES)
            break
        if not scored:
            reason = "no_moves"
            break
        best_move = max(scored, key=_scored_value)[1]
        best_value, best_depth = max(scored, key=_scored_value)[0], depth
        top_moves = _best_scored(scored, TOP_MOVES)
        if abs(best_value) >= WIN_SCORE - depth:
            reason = "proven"
            break  # the horizon already proves the result
        depth += 1
    return best_move, best_value, best_depth, top_moves, timed_out, reason


def find_best_move_within_time(state, rotate_direction = "clockwise", transfer_allowed = True, player = 1, time_limit = 1.0, max_depth = None, top_moves_count = 5, extra_rotation_allowed = True, extra_turns = 0):
    """
    Determines the best move the search can find within a limited amount of
    thinking time, using iterative deepening. Deeper iterations only start
    once the previous one finished, so the returned move always comes from a
    fully completed search.
    Parameters:
        state: the current game state (a numpy array)
        rotate_direction: the direction in which the board is rotated
        transfer_allowed: whether transfers are allowed between adjacent cells
        player: the player whose move is chosen (1 or 2)
        time_limit: maximum thinking time in seconds (must be positive)
        max_depth: optional cap on the searched depth in moves; ``None`` means
            the search is only bounded by ``time_limit``
        top_moves_count: how many alternatives to report in the info dictionary
        extra_rotation_allowed: whether a full board without a line is decided by the EXTRA_TURNS Orbito presses
        extra_turns: how many of those presses have already been played
    Returns:
        A tuple (best_move, best_value, info) where best_move is a move
        (a dictionary), best_value is its minimax evaluation (None when even the
        first iteration could not finish) and info is a dictionary with the
        "depth", "max_depth", "nodes", "elapsed", "timed_out", "reason" and
        "top_moves" details. ``reason`` tells why the deepening stopped, so the
        caller can warn when the answer comes from the deepest allowed search.
        A position in the endgame needs no search: it returns the forced press
        and the value of the presses that follow it, with reason "endgame".
    """
    n = state.shape[0]
    time_limit = float(time_limit)
    if not time_limit > 0:
        raise ValueError("time_limit must be positive")
    P = _engine_params(n, rotate_direction, transfer_allowed, extra_rotation_allowed)
    b = tuple(state.ravel().tolist())
    if not _gen_moves(b, player, P[7], P[5], P[1]):
        info = {"depth": 0, "max_depth": max_depth, "nodes": 0,
                "elapsed": 0.0, "timed_out": False,
                "reason": "no_moves", "top_moves": []}
        move, value = _endgame_answer(b, player, P, rotate_direction, n, extra_turns)
        if move is None:
            return None, None, info
        info["reason"] = "endgame"
        return move, value, info

    start = time.monotonic()
    _start_search(start + time_limit)
    try:
        best_move, best_value, depth, top_moves, timed_out, reason = _search_with_deadline(
            b, player, P, start + time_limit, max_depth,
        )
    finally:
        nodes = _stop_search()
    info = {
        "depth": depth,
        "max_depth": max_depth,
        "nodes": nodes,
        "elapsed": time.monotonic() - start,
        "timed_out": timed_out,
        "reason": reason,
        "top_moves": [
            (_move_to_dict(move, player, rotate_direction, n), value)
            for value, move in top_moves[:top_moves_count]
        ],
    }
    return _move_to_dict(best_move, player, rotate_direction, n), best_value, info


def simulate_principal_variation(state, rotate_direction = "clockwise", transfer_allowed = True, player = 1, max_steps = 50, extra_rotation_allowed = True):
    """
    Simulate a possible evolution of the game (the principal variation) by alternately choosing best moves.
    Parameters:
        state: the starting game state (a numpy array)
        player: the player whose turn it is (1 or 2)
        max_steps: maximum number of moves to simulate to avoid infinite loops
        extra_rotation_allowed: whether a full board without a line is decided by the EXTRA_TURNS Orbito presses
    Returns:
        A list of game states representing the evolution from the current state.
        The forced presses of the endgame are part of it, so a line that reaches a
        full board is followed through to the end of the game.
    """
    evolution = [{"move": None, "state": state}]
    current_state = state
    current_player = player
    steps = 0
    while steps < max_steps:
        if olgf.evaluate_game_state(current_state) is not None:
            break
        best_move, score = find_best_move(current_state, rotate_direction, transfer_allowed, current_player,
                                          extra_rotation_allowed = extra_rotation_allowed)
        if best_move is None:
            break
        current_state = olgf.play_turn(current_state, best_move)
        evolution.append({"move": best_move, "state": current_state})
        current_player = 3 - current_player
        steps += 1
    return evolution


def estimated_game_result(score, players_symbols = {1: "x", 2: "o", 0: "_"}, start_player = 1, maximizing_player = None):
    """
    Returns a message describing the estimated result of the game based on the minimax score.
    Parameters:
        score: the minimax evaluation score
        players_symbols: a dictionary mapping player values to symbols (default: {1: "x", 2: "o",  0: "_"})
        start_player: the player who started the game (1 or 1)
        maximizing_player: the player for whom we are optimizing (1 or 2)
    Returns:
        A message describing the estimated result of the game.
    """
    if maximizing_player is None:
        maximizing_player = start_player
    if score > 0:
        return f"Player {players_symbols[[3 - maximizing_player, maximizing_player][start_player == maximizing_player]]} wins!"
    elif score < 0:
        return f"Player {players_symbols[[maximizing_player, 3 - maximizing_player][start_player == maximizing_player]]} wins!"
    elif score == 0:
        return "It's a draw!"


def solve_game(state, rotate_direction = "clockwise", transfer_allowed = True, player = 1, maximizing_player = None, depth = 0, alpha = -math.inf, beta = math.inf, workers = None, extra_rotation_allowed = True):
    """
    Recursively solves the game from the given state, returning a dictionary with:
        - "score": the minimax evaluation score,
        - "states_sequence": a list of game states representing the evolution from the current state to a terminal state,
        - "moves_sequence": a list of moves (dictionaries) that lead from one state to the next.

    Parameters:
        state: the current game state (a numpy array)
        rotate_direction: the direction in which the board is rotated (clockwise, counterclockwise, or still)
        transfer_allowed: whether transfers are allowed between adjacent cells
        player: the player whose turn it is (1 or 2)
        maximizing_player: the player for whom we are optimizing (defaults to the initial player)
        depth: current depth of recursion (used to favor faster wins/longer losses)
        alpha: best value found so far for the maximizer
        beta: best value found so far for the minimizer
        extra_rotation_allowed: whether a full board without a line is decided by the EXTRA_TURNS Orbito presses

    Returns:
        A dictionary with keys "score", "states_sequence", and "moves_sequence".
        A line that ends in the endgame is followed through the forced presses, so
        the returned sequence reaches the actual end of the game.
    """
    if maximizing_player is None:
        maximizing_player = player

    n = state.shape[0]
    P = _engine_params(n, rotate_direction, transfer_allowed, extra_rotation_allowed)
    b = tuple(state.ravel().tolist())

    tt = {}
    if workers is not None and int(workers) > 1 and alpha == -math.inf and beta == math.inf:
        root_winner = _winner(b, P[3])
        if root_winner is not None:
            if root_winner == player:
                value = WIN_SCORE - depth
            elif root_winner == 0:
                value = 0
            else:
                value = depth - WIN_SCORE
        else:
            root_moves = _gen_moves(b, player, P[7], P[5], P[1])
            if not root_moves:
                value = _endgame_value(b, player, P[6], P[3], extra_rotation_allowed, depth)
            else:
                root_moves = _order_moves(root_moves, b, player, P, None)
                try:
                    results, tt = _solve_root_moves_parallel(b, player, depth, root_moves, P, int(workers))
                    value = max(v for v, _m, _t in results)
                except Exception:
                    value = _negamax(b, player, depth, -INF if alpha == -math.inf else alpha,
                                     INF if beta == math.inf else beta, tt, P)
    else:
        value = _negamax(b, player, depth, -INF if alpha == -math.inf else alpha,
                         INF if beta == math.inf else beta, tt, P)

    # Reconstruct the principal variation by following the exact best moves,
    # falling back to a clean exact re-search whenever a TT entry is bounded.
    states_seq = [state]
    moves_seq = []
    cur = b
    p = player
    d = depth
    while True:
        if _winner(cur, P[3]) is not None:
            break
        entry = tt.get((cur, p))
        if entry is not None and entry[2] == EXACT and entry[3] is not None:
            move = entry[3]
        else:
            move = _best_move_at(cur, p, d, tt, P)
        if move is None:
            # No move left: with the rule on, the forced presses finish the game.
            press_moves, press_states, _score = _endgame_line(cur, p, P, rotate_direction, n)
            moves_seq.extend(press_moves)
            states_seq.extend(press_states)
            break
        moves_seq.append(_move_to_dict(move, p, rotate_direction, n))
        cur = _apply_move(cur, move, p, P[6], P[5])
        d += 1
        p = 3 - p
        states_seq.append(np.array(cur, dtype = state.dtype).reshape(n, n))

    if player != maximizing_player:
        value = -value
    return {"score": value, "states_sequence": states_seq, "moves_sequence": moves_seq}


if __name__ == "__main__":
    print("Welcome to the \"Orbital Logic Game\" solver!")
    # initial_state = np.zeros((2, 2))
    # initial_state = np.array([[0, 0, 0], \
    #                             [0, 0, 0], \
    #                             [0, 0, 0]])
    # initial_state = np.array([[1, 2, 0, 1], \
    #                             [0, 0, 2, 0], \
    #                             [0, 0, 0, 1], \
    #                             [2, 0, 0, 0]])
    n = 4; initial_state = np.zeros((n, n))
    board_size = initial_state.shape[0]
    players_symbols = {1: "x", 2: "o", 0: "_"}
    start_player = 1
    rotate_direction = "still"
    transfer_allowed = True

    olgf.print_game_statistics(board_size)
    print("\nGame rules:")
    print(f"  - Board size:          {board_size}x{board_size}")
    print(f"  - Players:             {players_symbols[1]} and {players_symbols[2]}")
    print(f"  - Rotation direction:  {rotate_direction}")
    print(f"  - Transfers allowed:   {transfer_allowed}")
    print("\n\nInitial game state:\n")
    olgf.print_game_state(initial_state, players_symbols)
    print(f"\n\nStart player: {players_symbols[start_player]}")

    # # Find the best move and its evaluation score.
    # start_time = time.time()
    # best_move, score = find_best_move(initial_state, rotate_direction, transfer_allowed, start_player)
    # print(f"\n\nBest move found. Time taken: {time.time() - start_time:.3f} sec")
    # print(best_move)
    # print(f"{estimated_game_result(score, players_symbols, start_player)} Minimax evaluation score: {score}")

    # Solve the game from the current state.
    start_time = time.time()
    solution = solve_game(initial_state, rotate_direction, transfer_allowed, start_player)
    print(f"\n\nTime taken to solve the game: {time.time() - start_time:.3f} sec")
    score = solution["score"]
    perfect_states_seq = solution["states_sequence"]
    perfect_moves_seq = solution["moves_sequence"]

    # Print the final evaluation.
    result_message = estimated_game_result(score, players_symbols, start_player)
    print("\n" + result_message)

    # Print the perfect game evolution.
    print("\nPerfect game evolution:\n")
    player = start_player
    for move_number, s in enumerate(perfect_states_seq):
        print(f"State after move {move_number}:")
        olgf.print_game_state(s, players_symbols)
        if move_number < len(perfect_moves_seq):
            print(f"\n\nMove {move_number + 1} ({players_symbols[player]}) : {perfect_moves_seq[move_number]}")
            player = 3 - player
    print("\n")

    # # Simulate and print a possible evolution of the game.
    # print("\nPossible evolution of the game (principal variation):")
    # evolution = simulate_principal_variation(initial_state, rotate_direction, start_player)
    # for move_number, evolution_moment in enumerate(evolution):
    #     print(f"\n\n\nMove {move_number + 1}: \t {evolution_moment['move']}\n")
    #     olgf.print_game_state(evolution_moment["state"], players_symbols)
    # print("\n")
