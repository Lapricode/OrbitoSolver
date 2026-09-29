import numpy as np
import random
import time
import copy
import itertools
import orbital_logic_game_functions as olgf
import solve_game


# a full board without a line is finished by this many Orbito presses when the
# rule is on; with the rule off such a board is a draw straight away
EXTRA_TURNS = solve_game.EXTRA_TURNS


def _sample_move_encoded(row, player, transfer_allowed, n, rng):
    """Pick one random legal move for a single board, returned encoded as
    (transfer_source_flat, transfer_dir, add_flat) with -1 meaning "none".

    The selection is uniform over get_possible_moves' enumeration order:
    first the add-only moves (row-major over empty cells), then the transfer
    moves (row-major over opponent pieces, then u/d/l/r, then the add cells).
    """
    L = n * n
    empties = [k for k in range(L) if row[k] == 0]
    e = len(empties)
    transfers = []
    if transfer_allowed:
        opponent = 3 - player
        for src in range(L):
            if row[src] == opponent:
                r, c = divmod(src, n)
                if r > 0 and row[src - n] == 0:
                    transfers.append((src, 0, src - n))
                if r < n - 1 and row[src + n] == 0:
                    transfers.append((src, 1, src + n))
                if c > 0 and row[src - 1] == 0:
                    transfers.append((src, 2, src - 1))
                if c < n - 1 and row[src + 1] == 0:
                    transfers.append((src, 3, src + 1))
    t = len(transfers)
    total = e + t * e  # every transfer produces exactly e add options
    pick = int(rng.integers(0, total))
    if pick < e:
        return -1, -1, empties[pick]
    v = pick - e
    q, r = divmod(v, e)
    src, direction, target = transfers[q]
    position_in_add_order = 0
    for k in range(L):
        if (row[k] == 0 and k != target) or k == src:
            if position_in_add_order == r:
                return src, direction, k
            position_in_add_order += 1
    return src, direction, src


def _play_random_games_batch(
    board_size,
    games_played,
    rotate_direction,
    transfer_allowed,
    batch_size=None,
    seed=None,
    extra_rotation_allowed=True,
):
    """CPU batched rewrite of the per-game random play loop.

    All running games advance in lockstep: one move is sampled per active game,
    then every move is applied together via vectorized numpy operations and the
    whole batch is evaluated at once. Returns the same ``counters`` dictionary
    produced by the scalar loop, so the aggregate statistics are identical.
    """
    n = board_size
    L = n * n
    rotate_keyword = str(rotate_direction).lower()
    if rotate_keyword in olgf.clockwise_rotation_keywords:
        perm = olgf._rotation_perm(n, True)
    elif rotate_keyword in olgf.counterclockwise_rotation_keywords:
        perm = olgf._rotation_perm(n, False)
    else:
        perm = np.arange(L, dtype=np.intp)
    neighbor_delta = np.asarray((-n, n, -1, 1), dtype=np.intp)

    counters = {
        "1_start_1_win": 0, "1_start_2_win": 0,
        "2_start_1_win": 0, "2_start_2_win": 0,
        "1_start_draw": 0, "2_start_draw": 0,
    }
    rng = np.random.default_rng(seed)
    batch_size = batch_size if batch_size and batch_size > 0 else max(64, min(1024, games_played))
    games_done = 0
    ar = np.arange(n)

    def results_of(rows):
        """The winner of every board in the batch: 1, 2, 0 for a shared line, -1."""
        w = np.where(rows == 2, -1, rows)
        row_sums = w.sum(axis=2)
        col_sums = w.sum(axis=1)
        d1 = w[:, ar, ar].sum(axis=1)
        d2 = w[:, ar, n - 1 - ar].sum(axis=1)
        p1 = (row_sums == n).any(axis=1) | (col_sums == n).any(axis=1) | (d1 == n) | (d2 == n)
        p2 = (row_sums == -n).any(axis=1) | (col_sums == -n).any(axis=1) | (d1 == -n) | (d2 == -n)
        return np.where(p1 & p2, 0, np.where(p1, 1, np.where(p2, 2, -1)))

    def book(rows_results, mask):
        """Count the outcomes of the masked games, keyed by who started them."""
        for i in np.flatnonzero(mask):
            key = "1_start" if start_players[i] == 1 else "2_start"
            if rows_results[i] == 1:
                counters[f"{key}_1_win"] += 1
            elif rows_results[i] == 2:
                counters[f"{key}_2_win"] += 1
            else:
                counters[f"{key}_draw"] += 1

    while games_done < games_played:
        m = min(batch_size, games_played - games_done)
        boards = np.zeros((m, L), dtype=np.int64)
        start_players = np.where(((np.arange(m) + games_done) % 2) == 0, 2, 1).astype(np.int64)
        active = np.ones(m, dtype=bool)
        for step in range(L):
            if not active.any():
                break
            players = np.where((step % 2) == 0, start_players, 3 - start_players)
            transfer_src = np.full(m, -1, dtype=np.int64)
            transfer_dir = np.full(m, -1, dtype=np.int64)
            add_src = np.full(m, -1, dtype=np.int64)
            for i in np.flatnonzero(active):
                tsrc, tdir, add = _sample_move_encoded(
                    boards[i], int(players[i]), transfer_allowed, n, rng
                )
                transfer_src[i] = tsrc
                transfer_dir[i] = tdir
                add_src[i] = add
            m_transfer = transfer_src >= 0
            if m_transfer.any():
                rows = np.flatnonzero(m_transfer)
                s = transfer_src[m_transfer]
                trg = s + neighbor_delta[transfer_dir[m_transfer]]
                src_vals = boards[rows, s].copy()
                boards[rows, trg] = src_vals
                boards[rows, s] = 0
            m_add = add_src >= 0
            if m_add.any():
                rows = np.flatnonzero(m_add)
                boards[rows, add_src[m_add]] = players[m_add]
            boards = boards[:, perm]
            results = results_of(boards.reshape(m, n, n))
            # a board with a line is finished, and a shared line is a draw
            newly_finished = (results != -1) & active
            if newly_finished.any():
                book(results, newly_finished)
                active &= ~newly_finished
        # the games still running hold a full board without a line: the official
        # rule finishes them with forced presses, which add nothing and only turn
        # the rings, so they are applied to the whole batch at once
        for press in range(EXTRA_TURNS if extra_rotation_allowed else 0):
            if not active.any():
                break
            boards = boards[:, perm]
            results = results_of(boards.reshape(m, n, n))
            pressed = (results != -1) & active
            if pressed.any():
                book(results, pressed)
                active &= ~pressed
        if active.any():
            book(np.zeros(m, dtype=np.int64), active)
        games_done += m

    return counters


def _print_probabilities(counters, games_played, start_time):
    print()
    print(f"Total games played:                {games_played:10d}")
    print(f"Player 1 starts:                   {counters['1_start_1_win'] + counters['1_start_2_win'] + counters['1_start_draw']:10d}")
    print(f"Player 2 starts:                   {counters['2_start_1_win'] + counters['2_start_2_win'] + counters['2_start_draw']:10d}")
    print()
    print(f"Player 1 wins:                     {counters['1_start_1_win'] + counters['2_start_1_win']:10d} \t|\t {(counters['1_start_1_win'] + counters['2_start_1_win'])/games_played*100:.2f}%")
    print(f"   - Player 1 starts and wins:     {counters['1_start_1_win']:10d} \t|\t {counters['1_start_1_win']/games_played*100:.2f}%")
    print(f"   - Player 2 starts and loses:    {counters['2_start_1_win']:10d} \t|\t {counters['2_start_1_win']/games_played*100:.2f}%")
    print()
    print(f"Player 2 wins:                     {counters['2_start_2_win'] + counters['1_start_2_win']:10d} \t|\t {(counters['2_start_2_win'] + counters['1_start_2_win'])/games_played*100:.2f}%")
    print(f"   - Player 1 starts and loses:    {counters['1_start_2_win']:10d} \t|\t {counters['1_start_2_win']/games_played*100:.2f}%")
    print(f"   - Player 2 starts and wins:     {counters['2_start_2_win']:10d} \t|\t {counters['2_start_2_win']/games_played*100:.2f}%")
    print()
    print(f"Draws:                             {counters['1_start_draw'] + counters['2_start_draw']:10d} \t|\t {(counters['1_start_draw'] + counters['2_start_draw'])/games_played*100:.2f}%")
    print(f"   - Player 1 starts and draws:    {counters['1_start_draw']:10d} \t|\t {counters['1_start_draw']/games_played*100:.2f}%")

    print(f"\nTime taken: {time.time() - start_time:.2f} seconds")


def check_game_play_probabilities(board_size = 3, games_played = 10000, rotate_direction = "1", transfer_allowed = True, players_symbols = {1: "x", 2: "o", 0: "_"}, verbose_print = False, batch_size = None, seed = None, extra_rotation_allowed = True):
    start_time = time.time()
    olgf.print_game_statistics(board_size)
    counters = {"1_start_1_win": 0, "1_start_2_win": 0, "2_start_1_win": 0, "2_start_2_win": 0, "1_start_draw": 0, "2_start_draw": 0}
    if not verbose_print:
        counters = _play_random_games_batch(
            board_size, games_played, rotate_direction, transfer_allowed,
            batch_size=batch_size, seed=seed,
            extra_rotation_allowed=extra_rotation_allowed,
        )
        _print_probabilities(counters, games_played, start_time)
        return counters
    start_player = 1
    for i in range(games_played):
        state = np.zeros((board_size, board_size))
        start_player = 3 - start_player
        player = start_player
        if verbose_print: print(f"\nGame {i+1}\n"); olgf.print_game_state(state, players_symbols, [1, "  ", ""]); print(2*"\n")
        for j in range(board_size ** 2):
            possible_moves = olgf.get_possible_moves(state, rotate_direction, transfer_allowed, player)
            next_move = random.choice(possible_moves)
            new_state = olgf.play_turn(state, next_move)
            if verbose_print: print(next_move); olgf.print_game_state(new_state, players_symbols, [1, "  ", ""]); print(2*"\n")
            player = 3 - player
            state = new_state
            result = olgf.evaluate_game_state(state)
            if result in (1, 2):
                if verbose_print: print(f"Player {result} wins\n")
                break
        else:
            # the board filled up without a line, so the official rule presses on
            for _ in range(EXTRA_TURNS if extra_rotation_allowed else 0):
                press = olgf.rotation_only_move(player, rotate_direction)
                state = olgf.play_turn(state, press)
                if verbose_print: print(press); olgf.print_game_state(state, players_symbols, [1, "  ", ""]); print(2*"\n")
                player = 3 - player
                result = olgf.evaluate_game_state(state)
                if result in (1, 2):
                    if verbose_print: print(f"Player {result} wins\n")
                    break
            else:
                result = 0
                if verbose_print: print("It's a draw\n")
        prefix = "1_start" if start_player == 1 else "2_start"
        if result == 1:
            counters[f"{prefix}_1_win"] += 1
        elif result == 2:
            counters[f"{prefix}_2_win"] += 1
        else:
            counters[f"{prefix}_draw"] += 1
    _print_probabilities(counters, games_played, start_time)
    return counters


if __name__ == "__main__":
    # board is a square grid
    # 1 for 1st player and 2 for 2nd player
    board_size = 3
    players_symbols = {1: "x", 2: "o", 0: "_"}
    rotate_direction = "1"
    transfer_allowed = True
    state = np.zeros((board_size, board_size))
    # state = np.array([[0, 0, 0], \
    #                   [0, 1, 0], \
    #                   [0, 0, 0]])
    # state = np.array([[1, 2, 2, 2], \
    #                   [0, 1, 2, 2], \
    #                   [0, 1, 1, 2], \
    #                   [1, 1, 0, 2]])
    # olgf.print_game_statistics(board_size)

    # # test 1:
    # check_game_play_probabilities(board_size, 10000, rotate_direction, transfer_allowed, players_symbols, True)
    
    # test 2:
    board_size = 4
    olgf.print_game_statistics(board_size)
    # all_strings = []
    # for pair in itertools.product(list(range(board_size**2)), repeat = 2):
    #     if abs(pair[0] - pair[1]) <= 1 and pair[0] + pair[1] <= board_size**2:
    #         all_strings += olgf.generate_state_strings(players_symbols, board_size**2, pair[0], pair[1])
    # print(f"\nTotal strings:\t {len(all_strings)}")
    # all_strings, all_strings_numbers = olgf.sort_state_strings(all_strings, players_symbols)
    # unique_state_strings, unique_state_numbers = olgf.find_unique_rotationally_symmetric_states(all_strings, players_symbols)
    # print(f"\nUnique strings:\t {len(unique_state_strings)}")
    # for k in range(len(all_strings)):
    #     print((" ").join(all_strings[k]) + 2*"\t" + str(all_strings_numbers[k]), end = "\n")
    # for k in range(len(unique_state_strings)):
    #     print((" ").join(unique_state_strings[k]) + 2*"\t" + str(unique_state_numbers[k]), end = "\n")
    # states = olgf.convert_strings_to_states(unique_state_strings, players_symbols)
    # for state in states:
    #     olgf.print_game_state(state, players_symbols); print("\n\n")
    # print(olgf.stringify_states_numbers(unique_state_numbers, board_size**2, "10", players_symbols))
    
    # # test 3:
    # max = 0
    # s_max = ""
    # for s in all_strings:
    #     s2 = s.replace(players_symbols[0], "0").replace(players_symbols[1], "1").replace(players_symbols[2], "2")
    #     if int(s2, 3) > max:
    #         max = int(s2, 3)
    #         s_max = s
    # print(max, s_max)


# Initial game state:

# x  o  _  x  
      
# x  o  o  _  
      
# _  o  x  x  
      
# o  _  _  _  

# Current player: 1


# Time taken: 5.8157689571380615 sec


# Estimated draw

# Perfect game evolution (states with perfect play):

# State after move 0:
# x  o  _  x  
      
# x  o  o  _  
      
# _  o  x  x  
      
# o  _  _  _  

# Move 1: {'player': 1, 'transfer': [(1, 2), 'r'], 'add': (2, 0), 'rotate': '-1'}
# State after move 1:
# x  x  o  _  
      
# x  o  o  x  
      
# o  x  _  o  
      
# _  _  _  x  

# Move 2: {'player': 2, 'transfer': None, 'add': (3, 0), 'rotate': '-1'}
# State after move 2:
# x  x  x  o  
      
# o  x  o  _  
      
# o  _  o  x  
      
# _  _  x  o  

# Move 3: {'player': 1, 'transfer': [(1, 2), 'r'], 'add': (1, 2), 'rotate': '-1'}
# State after move 3:
# o  x  x  x  
      
# o  _  x  o  
      
# _  o  x  o  
      
# _  x  o  x  

# Move 4: {'player': 2, 'transfer': [(1, 2), 'l'], 'add': (1, 2), 'rotate': '-1'}
# State after move 4:
# o  o  x  x  
      
# _  o  x  x  
      
# _  x  o  o  
      
# x  o  x  o


# Welcome to the game solver!

# --- Game statistics ---
# board size (# of side cells):   	4
# # of total board cells:         	16
# # of possible board states:     	10165778


# Initial game state:

# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  

# Current player: 1


# Time taken: 53970.53447508812 sec


# Estimated draw

# Perfect game evolution (states with perfect play):

# State after move 0:
# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  

# Move 1: {'player': 1, 'transfer': None, 'add': (0, 0), 'rotate': '0'}
# State after move 1:
# x  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  

# Move 2: {'player': 2, 'transfer': None, 'add': (0, 1), 'rotate': '0'}
# State after move 2:
# x  o  _  _  
      
# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  

# Move 3: {'player': 1, 'transfer': None, 'add': (0, 2), 'rotate': '0'}
# State after move 3:
# x  o  x  _  
      
# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  

# Move 4: {'player': 2, 'transfer': None, 'add': (0, 3), 'rotate': '0'}
# State after move 4:
# x  o  x  o  
      
# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  

# Move 5: {'player': 1, 'transfer': None, 'add': (1, 0), 'rotate': '0'}
# State after move 5:
# x  o  x  o  
      
# x  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  

# Move 6: {'player': 2, 'transfer': None, 'add': (1, 1), 'rotate': '0'}
# State after move 6:
# x  o  x  o  
      
# x  o  _  _  
      
# _  _  _  _  
      
# _  _  _  _  

# Move 7: {'player': 1, 'transfer': None, 'add': (1, 2), 'rotate': '0'}
# State after move 7:
# x  o  x  o  
      
# x  o  x  _  
      
# _  _  _  _  
      
# _  _  _  _  

# Move 8: {'player': 2, 'transfer': None, 'add': (1, 3), 'rotate': '0'}
# State after move 8:
# x  o  x  o  
      
# x  o  x  o  
      
# _  _  _  _  
      
# _  _  _  _  

# Move 9: {'player': 1, 'transfer': None, 'add': (2, 0), 'rotate': '0'}
# State after move 9:
# x  o  x  o  
      
# x  o  x  o  
      
# x  _  _  _  
      
# _  _  _  _  

# Move 10: {'player': 2, 'transfer': None, 'add': (3, 0), 'rotate': '0'}
# State after move 10:
# x  o  x  o  
      
# x  o  x  o  
      
# x  _  _  _  
      
# o  _  _  _  

# Move 11: {'player': 1, 'transfer': None, 'add': (2, 1), 'rotate': '0'}
# State after move 11:
# x  o  x  o  
      
# x  o  x  o  
      
# x  x  _  _  
      
# o  _  _  _  

# Move 12: {'player': 2, 'transfer': None, 'add': (2, 2), 'rotate': '0'}
# State after move 12:
# x  o  x  o  
      
# x  o  x  o  
      
# x  x  o  _  
      
# o  _  _  _  

# Move 13: {'player': 1, 'transfer': None, 'add': (2, 3), 'rotate': '0'}
# State after move 13:
# x  o  x  o  
      
# x  o  x  o  
      
# x  x  o  x  
      
# o  _  _  _  

# Move 14: {'player': 2, 'transfer': None, 'add': (3, 1), 'rotate': '0'}
# State after move 14:
# x  o  x  o  
      
# x  o  x  o  
      
# x  x  o  x  
      
# o  o  _  _  

# Move 15: {'player': 1, 'transfer': None, 'add': (3, 2), 'rotate': '0'}
# State after move 15:
# x  o  x  o  
      
# x  o  x  o  
      
# x  x  o  x  
      
# o  o  x  _  

# Move 16: {'player': 2, 'transfer': None, 'add': (3, 3), 'rotate': '0'}
# State after move 16:
# x  o  x  o  
      
# x  o  x  o  
      
# x  x  o  x  
      
# o  o  x  o


# Welcome to the game solver!

# --- Game statistics ---
# board size (# of side cells):   	4
# # of total board cells:         	16
# # of possible board states:     	10165778


# Initial game state:

# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  

# Current player: 1

 
# Time taken: 171310.278 sec

# Player x wins!

# Perfect game evolution (states with perfect play):

# State after move 0:
# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  

# Move 1 (x) : {'player': 1, 'transfer': None, 'add': (1, 1), 'rotate': '+'}
# State after move 1:
# _  _  _  _  
      
# _  _  _  _  
      
# _  x  _  _  
      
# _  _  _  _  

# Move 2 (o) : {'player': 2, 'transfer': None, 'add': (0, 0), 'rotate': '+'}
# State after move 2:
# _  _  _  _  
      
# o  _  _  _  
      
# _  _  x  _  
      
# _  _  _  _  

# Move 3 (x) : {'player': 1, 'transfer': None, 'add': (0, 0), 'rotate': '+'}
# State after move 3:
# _  _  _  _  
      
# x  _  x  _  
      
# o  _  _  _  
      
# _  _  _  _  

# Move 4 (o) : {'player': 2, 'transfer': None, 'add': (2, 1), 'rotate': '+'}
# State after move 4:
# _  _  _  _  
      
# _  x  _  _  
      
# x  _  o  _  
      
# o  _  _  _  

# Move 5 (x) : {'player': 1, 'transfer': None, 'add': (1, 2), 'rotate': '+'}
# State after move 5:
# _  _  _  _  
      
# _  x  o  _  
      
# _  x  _  _  
      
# x  o  _  _  

# Move 6 (o) : {'player': 2, 'transfer': None, 'add': (1, 3), 'rotate': '+'}
# State after move 6:
# _  _  _  o  
      
# _  o  _  _  
      
# _  x  x  _  
      
# _  x  o  _  

# Move 7 (x) : {'player': 1, 'transfer': None, 'add': (1, 0), 'rotate': '+'}
# State after move 7:
# _  _  o  _  
      
# _  _  x  _  
      
# x  o  x  _  
      
# _  _  x  o  

# Move 8 (o) : {'player': 2, 'transfer': None, 'add': (0, 3), 'rotate': '+'}
# State after move 8:
# _  o  o  _  
      
# _  x  x  _  
      
# _  _  o  o  
      
# x  _  _  x  

# Move 9 (x) : {'player': 1, 'transfer': None, 'add': (0, 3), 'rotate': '+'}
# State after move 9:
# o  o  x  _  
      
# _  x  o  o  
      
# _  x  _  x  
      
# _  x  _  _  

# Move 10 (o) : {'player': 2, 'transfer': None, 'add': (2, 2), 'rotate': '+'}
# State after move 10:
# o  x  _  o  
      
# o  o  o  x  
      
# _  x  x  _  
      
# _  _  x  _  

# Move 11 (x) : {'player': 1, 'transfer': None, 'add': (3, 3), 'rotate': '+'}
# State after move 11:
# x  _  o  x  
      
# o  o  x  _  
      
# o  o  x  x  
      
# _  _  _  x  

# Move 12 (o) : {'player': 2, 'transfer': None, 'add': (0, 1), 'rotate': '+'}
# State after move 12:
# o  o  x  _  
      
# x  x  x  x  
      
# o  o  o  x  
      
# o  _  _  _
