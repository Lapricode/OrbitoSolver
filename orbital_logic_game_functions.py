import numpy as np
import math as m
import itertools
import sys


# important global variables for the functions
clockwise_rotation_keywords = ["clockwise", "cw", "-", "-1"]
counterclockwise_rotation_keywords = ["counterclockwise", "ccw", "+", "1"]
stand_still_keywords = ["still", "s", "x", "0"]
players_symbols_default = {1: "x", 2: "o", 0: "_"}

# Key that marks a move as a press of the Orbito button that only turns the
# board. Those are the moves of the endgame: the board is full, so there is
# nothing to add and no empty cell to transfer a piece into, and the press
# only rotates. See :func:`rotation_only_move` and the endgame rule in
# ``solve_game``.
rotation_only_key = "rotation_only"

# caches used by the optimized rotation / encoding helpers
_rotation_perms = {}     # (n, is_clockwise) -> numpy permutation for flat arrays
_ring_index_cache = {}   # n -> list of ring index arrays (flat indices, in ring order)
_rotation_perm_cache = {}  # (n, kind) -> list of source flat indices for string rotations, kind in {"cw","ccw","half"}


def _build_rings(n):
    rings = _ring_index_cache.get(n)
    if rings is not None:
        return rings
    rings = []
    for layer in range(n // 2):
        elements = ([(layer, j) for j in range(layer, n - layer)] +
                    [(i, n - layer - 1) for i in range(layer + 1, n - layer - 1)] +
                    [(n - layer - 1, j) for j in range(n - layer - 1, layer - 1, -1)] +
                    [(i, layer) for i in range(n - layer - 2, layer, -1)])
        rings.append(np.array([r * n + c for r, c in elements], dtype=np.intp))
    _ring_index_cache[n] = rings
    return rings


def _rotation_perm(n, clockwise=True):
    key = (n, clockwise)
    perm = _rotation_perms.get(key)
    if perm is None:
        perm = np.arange(n * n, dtype=np.intp)
        shift = 1 if clockwise else -1
        for ring in _build_rings(n):
            perm[ring] = np.roll(ring, shift)
        perm.flags.writeable = False
        _rotation_perms[key] = perm
    return perm


def _string_rotation_perm(n, kind):
    """Return perm such that result[perm[dest]] = source[dest] mapping old cell k -> new position."""
    key = (n, kind)
    perm = _rotation_perm_cache.get(key)
    if perm is not None:
        return perm
    perm = [0] * (n * n)
    for k in range(n * n):
        i, j = divmod(k, n)
        if kind == "cw":
            ni, nj = j, n - 1 - i
        elif kind == "ccw":
            ni, nj = n - 1 - j, i
        elif kind == "half":
            ni, nj = n - 1 - i, n - 1 - j
        perm[n * ni + nj] = k
    perm = tuple(perm)
    _rotation_perm_cache[key] = perm
    return perm


def _rotation_apply(state_string, perm):
    """Return state_string rotated by the given permutation (fast path)."""
    return "".join(state_string[perm[k]] for k in range(len(perm)))

# evaluate and score the game state
def evaluate_game_state(state):
    '''
    check if the game state is a win for a player
    return 1 if player 1 wins, 2 if player 2 wins, and 0 if it is a draw
    '''
    n = state.shape[0]
    rows = state.tolist()
    row_sums = [0] * n
    col_sums = [0] * n
    diag_1 = 0
    diag_2 = 0
    for i in range(n):
        row = rows[i]
        s = 0
        for j in range(n):
            value = row[j]
            w = value if value != 2 else -1
            s += w
            col_sums[j] += w
            if i == j:
                diag_1 += w
            if i == n - 1 - j:
                diag_2 += w
        row_sums[i] = s
    n_pos = n
    n_neg = -n
    p1 = any(x == n_pos for x in row_sums) or any(x == n_pos for x in col_sums) or diag_1 == n_pos or diag_2 == n_pos
    p2 = any(x == n_neg for x in row_sums) or any(x == n_neg for x in col_sums) or diag_1 == n_neg or diag_2 == n_neg
    if p1 and p2:
        return 0
    elif p1:
        return 1
    elif p2:
        return 2
    else:
        return None

# rotate the game board along a certain direction
def rotate_board(state, rotate_direction = "clockwise", out = None):
    '''
    rotate the game board along a certain direction
    rotate_direction can be clockwise, counterclockwise, or still
    if out is given, the rotated result is written into out (and returned), reusing its buffer
    '''
    n = state.shape[0]
    rotate_direction = rotate_direction.lower() if isinstance(rotate_direction, str) else rotate_direction
    if rotate_direction in stand_still_keywords:
        if out is not None:
            out[:] = state
            return out
        return state.copy()
    elif rotate_direction in clockwise_rotation_keywords:
        perm = _rotation_perm(n, True)
    elif rotate_direction in counterclockwise_rotation_keywords:
        perm = _rotation_perm(n, False)
    else:
        print("Direction must be \"clockwise\", \"counterclockwise\", or \"still\".")
        return state.copy()
    rotated = state.ravel()[perm].reshape(n, n)
    if out is not None:
        out[:] = rotated
        return out
    return rotated

def board_is_full(state):
    '''
    tell whether the board has no empty cell left, so that no piece can be added
    '''
    return not bool(np.any(np.asarray(state) == 0))


def is_rotation_only(next_move):
    '''
    tell whether a move is an Orbito press that only turns the board
    next_move is a move dictionary, as built by :func:`rotation_only_move`
    '''
    return bool(next_move) and bool(next_move.get(rotation_only_key))


def rotation_only_move(player = 1, rotate_direction = "clockwise"):
    '''
    build the move of an Orbito press that only turns the board
    it is the move of the endgame, played once the last cell is filled: with
    every cell occupied there is no transfer and no add, only the rotation
    '''
    return {"player": player, "transfer": None, "add": None,
            "rotate": rotate_direction, rotation_only_key: True}


# make a player's turn
def play_turn(state, next_move):
    '''
    make a player's turn
    next_move is a dictionary with the following keys:
    "player" for the player making the move (1 for player 1, 2 for player 2),
    "transfer" for the transfer move of an opponent's piece to an adjacent cell (optional),
    "add" for the add move of a player's piece to an empty cell (optional),
    "rotate" for the direction of the board rotation
    a move carrying the "rotation_only" key is an Orbito press of the endgame:
    every cell is occupied, so only the rotation is played
    '''
    state_copy = state.copy()
    n = state.shape[0]
    player = next_move["player"]  # 1 for player 1, 2 for player 2
    transfer = next_move["transfer"]  # [position, direction] of the piece to be transferred
    add = next_move["add"]  # position of the piece to be added
    rotate = next_move["rotate"]  # direction of the board rotation
    opponent = 3 - player
    if is_rotation_only(next_move):
        return rotate_board(state_copy, str(rotate), out = state_copy)  # a press only turns the board
    # transfer move of an opponent's piece to an adjacent cell
    # 4 possible directions: "u" for up, "d" for down, "l" for left, "r" for right
    if transfer is not None:
        from_pos = transfer[0]
        transfer_direction = transfer[1].lower()
        # determine target position based on transfer direction
        if transfer_direction in ["up", "u"]:
            target = (from_pos[0] - 1, from_pos[1])
        elif transfer_direction in ["down", "d"]:
            target = (from_pos[0] + 1, from_pos[1])
        elif transfer_direction in ["left", "l"]:
            target = (from_pos[0], from_pos[1] - 1)
        elif transfer_direction in ["right", "r"]:
            target = (from_pos[0], from_pos[1] + 1)
        else: print("Invalid transfer direction!"); return state  # check if the transfer direction is valid
        if not (0 <= target[0] < n and 0 <= target[1] < n): print("Transfer move out of board bounds!"); return state  # check board boundaries
        if state_copy[from_pos] != opponent: print("Transfer move invalid: selected piece is not the opponent's!"); return state  # check if the selected piece is the opponent's
        if state_copy[target] != 0: print("Transfer move invalid: target cell is not empty!"); return state  # check if the target cell is empty
        # perform the transfer moving the opponent's piece
        state_copy[target] = state_copy[from_pos]
        state_copy[from_pos] = 0
    # add move of a player's piece to an empty cell
    if add is not None:
        if state_copy[add] == 0:
            # place the player's piece in the target cell
            state_copy[add] = player
        else: print("Add move invalid: target cell is not empty!"); return state  # check if the target cell is empty, to place the piece
    else: print("No add move!"); return state  # check if there is an add move, else the turn is invalid
    return rotate_board(state_copy, str(rotate), out = state_copy)  # rotate the board in place

# return a list of all possible moves for a player, from the current game state
def get_possible_moves(state, rotate_direction = "clockwise", transfer_allowed = True, player = 1):
    '''
    return a list of all possible moves for a player, from the current game state
    player is 1 for player 1 and 2 for player 2
    rotate_direction is clockwise, counterclockwise, or still
    a full board has no empty cell, so the list is empty there: the Orbito
    presses of the endgame are not moves, they are built by
    :func:`rotation_only_move`
    '''
    n = state.shape[0]
    rows = state.tolist()
    opponent = 3 - player
    possible_moves = []
    append = possible_moves.append
    for row in range(n):
        row_cells = rows[row]
        for column in range(n):
            if row_cells[column] == 0:
                append({"player": player, "transfer": None, "add": (row, column), "rotate": rotate_direction})
    # create possible moves with both transfer and add
    if transfer_allowed:
        for row in range(n):
            row_cells = rows[row]
            for column in range(n):
                if row_cells[column] == opponent:
                    source_cell = (row, column)
                    for transfer_direction in ("u", "d", "l", "r"):
                        # transfer part
                        target_cell = None
                        if transfer_direction == "u":
                            target = row - 1
                            if target >= 0 and rows[target][column] == 0:
                                target_cell = (target, column)
                        elif transfer_direction == "d":
                            target = row + 1
                            if target < n and rows[target][column] == 0:
                                target_cell = (target, column)
                        elif transfer_direction == "l":
                            target = column - 1
                            if target >= 0 and row_cells[target] == 0:
                                target_cell = (row, target)
                        else:
                            target = column + 1
                            if target < n and row_cells[target] == 0:
                                target_cell = (row, target)
                        # add part
                        if target_cell is not None:
                            for i in range(n):
                                add_row = rows[i]
                                for j in range(n):
                                    if (add_row[j] == 0 and (i, j) != target_cell) or (i, j) == source_cell:
                                        append({"player": player, "transfer": [source_cell, transfer_direction], "add": (i, j), "rotate": rotate_direction})
    return possible_moves

# rotate the board 90 degrees, clockwise or counterclockwise, where the board state is given as a string
def rotate_90_degrees(state_string, rotate_direction = "clockwise", rotate_times = 1):
    '''
    return the new state (as a string), after the "rotate_direction" given (as 90 degrees steps) is applied "rotate_times" times on the "state_string" old state
    the board size is n
              state             string
    old cell: (i, j)            k = ni+j
    new cell: (j, n-1-i)        nj+n-1-i = nk+n-1-(n^2+1)(k//n)             clockwise rotation
              (n-1-j, i)        n(n-1-j)+i = n(n-1-k)+(n^2+1)(k//n)         counterclockwise rotation
              (n-1-i, n-1-j)    n(n-1-i)+(n-1-j) = n^2-1-k                  double clockwise/counterclockwise rotation
    '''
    n = int(len(state_string)**0.5)
    rotate_times = rotate_times % 4
    if rotate_times in (1, 3):
        if rotate_direction in clockwise_rotation_keywords and rotate_times == 1 or rotate_direction in counterclockwise_rotation_keywords and rotate_times == 3:
            perm = _string_rotation_perm(n, "cw")
        elif rotate_direction in clockwise_rotation_keywords and rotate_times == 3 or rotate_direction in counterclockwise_rotation_keywords and rotate_times == 1:
            perm = _string_rotation_perm(n, "ccw")
        else:
            return state_string
        return _rotation_apply(state_string, perm)
    elif rotate_times == 2 and rotate_direction not in stand_still_keywords:
        perm = _string_rotation_perm(n, "half")
        return _rotation_apply(state_string, perm)
    else:
        return state_string

# generate all possible strings of a certain length n with k and l occurrences of two symbols
def generate_state_strings(players_symbols = players_symbols_default, n = 4, k = 1, l = 1):
    '''
    players_symbols is a dictionary mapping player symbols to values (for example: {1: "x", 2: "o", 0: "_"})
    n is the length of the string, k is the number of times the first symbol appears (player 1 /1), and l is the number of times the second symbol appears (player 2 /2)
    '''
    if k + l > n: print("The sum of k and l must not exceed n."); return None
    positions = list(range(n))  # positions in the string
    all_strings = []  # list to store the generated strings
    append = all_strings.append
    sym0 = str(players_symbols[0])
    sym1 = str(players_symbols[1])
    sym2 = str(players_symbols[2])
    for x_positions in itertools.combinations(positions, k):  # choose k positions for one symbol in the string
        o_positions_iter = itertools.combinations(sorted(set(positions) - set(x_positions)), l)  # choose l positions for the other symbol in the string
        for o_positions in o_positions_iter:
            s = [sym0] * n  # initialize the string with default symbols
            for pos in x_positions:  # place the first symbol in the chosen related positions
                s[pos] = sym1
            for pos in o_positions:  # place the second symbol in the chosen related positions
                s[pos] = sym2
            append(''.join(s))  # add the string to the list
    return all_strings

# convert game states to their respective strings
def convert_states_to_strings(states, players_symbols = players_symbols_default):
    '''
    convert game states to their respective strings
    players_symbols is a dictionary mapping player symbols to values (for example: {1: "x", 2: "o", 0: "_"})
    '''
    symbols = players_symbols
    if all(len(symbols[k]) == 1 for k in (0, 1, 2)):
        trans = bytes.maketrans(bytes([0, 1, 2]), (symbols[0] + symbols[1] + symbols[2]).encode("ascii"))
        return [state.astype(np.uint8).ravel().tobytes().translate(trans).decode("ascii") for state in states]
    return ["".join(symbols[cell] for row in state for cell in row) for state in states]

# convert strings of the game states to their processable form
def convert_strings_to_states(strings, players_symbols = players_symbols_default):
    '''
    convert strings of the game states to their processable form
    players_symbols is a dictionary mapping player symbols to values (for example: {1: "x", 2: "o", 0: "_"})
    '''
    inverse_mapping = {v: k for k, v in players_symbols.items()}
    symbols = players_symbols
    states = []
    append = states.append
    if all(len(symbols[k]) == 1 for k in (0, 1, 2)):
        trans = bytes.maketrans((symbols[0] + symbols[1] + symbols[2]).encode("ascii"), b"012")
        for s in strings:
            n = int(len(s) ** 0.5)
            if n * n != len(s):
                raise ValueError(f"String '{s}' does not represent a square board.")
            arr = (np.frombuffer(s.encode("ascii").translate(trans), dtype = np.uint8).astype(np.int16) - 48)
            append(arr.reshape(n, n))
    else:
        for s in strings:
            n = int(len(s) ** 0.5)
            if n * n != len(s):
                raise ValueError(f"String '{s}' does not represent a square board.")
            state = [[inverse_mapping[s[r * n + c]] for c in range(n)] for r in range(n)]
            append(np.array(state))
    return states

# convert state strings to numbers, with a 1-1 matching, for easier and overall better identification and classification
def numberify_state_strings(state_strings, players_symbols = players_symbols_default):
    '''
    numberify game states for easier and overall better identification, using the base-3 arithmetic system
    state_strings are the strings representations of states
    players_symbols is a dictionary mapping player symbols to values (for example: {1: "x", 2: "o", 0: "_"})
    '''
    symbols = players_symbols
    values = []
    append = values.append
    if all(len(symbols[k]) == 1 for k in (0, 1, 2)):
        trans = str.maketrans({symbols[0]: "0", symbols[1]: "1", symbols[2]: "2"})
        for s in state_strings:
            append(int(s.translate(trans), 3))
    else:
        for s in state_strings:
            append(int(s.replace(symbols[0], "0").replace(symbols[1], "1").replace(symbols[2], "2"), 3))
    return values

# convert state numbers to strings, with a 1-1 matching, for easier and better identification and classification
def stringify_states_numbers(state_numbers, state_length, numbers_base = "3", players_symbols = players_symbols_default):
    '''
    stringify game states for easier and overall better identification, using the base-3 arithmetic system
    state_numbers are the numbers representations of states
    numbers_base is the arithmetic base of the numbers in the state_numbers list, it can be "3" (base 3) or "10" (base 10)
    players_symbols is a dictionary mapping player symbols to values (for example: {1: "x", 2: "o", 0: "_"})
    '''
    symbols = players_symbols
    digits_strings = []
    append = digits_strings.append
    if numbers_base == "10":
        for num in state_numbers:
            append(np.base_repr(int(num), base = 3))
    else:
        for num in state_numbers:
            append(str(num))
    state_strings = []
    s_append = state_strings.append
    if all(len(symbols[k]) == 1 for k in (0, 1, 2)):
        trans = str.maketrans({"0": symbols[0], "1": symbols[1], "2": symbols[2]})
        for s in digits_strings:
            s_append(s.translate(trans).rjust(state_length, symbols[0]))
    else:
        for s in digits_strings:
            s_append(s.replace("0", symbols[0]).replace("1", symbols[1]).replace("2", symbols[2]).rjust(state_length, symbols[0]))
    return state_strings

# sort the given state strings based on their corresponding numbers
def sort_state_strings(state_strings, players_symbols = players_symbols_default):
    '''
    state_strings are the strings representations of states
    players_symbols is a dictionary mapping player symbols to values (for example: {1: "x", 2: "o", 0: "_"})
    '''
    state_numbers = numberify_state_strings(state_strings, players_symbols) 
    zipped_data = zip(state_strings, state_numbers)
    sorted_pairs = sorted(zipped_data, key = lambda x: x[1])
    try:
        sorted_strings, sorted_numbers = zip(*sorted_pairs)
    except:
        sorted_strings, sorted_numbers = [], []
    return list(sorted_strings), list(sorted_numbers)

# given the set of state strings, find the smallest subset of them that fully describes it, up to rotational transformations
def find_unique_rotationally_symmetric_states(state_strings, players_symbols = players_symbols_default):
    '''
    find the unique strings in the list state strings, up to all possible 90 degrees rotations 
    return the unique state strings (sorted to their corresponding numbers), along to their numbers
    '''
    # state_strings_numbers = numberify_state_strings(state_strings, players_symbols)
    # zipped_data = zip(state_strings, state_strings_numbers)
    # sorted_pairs = sorted(zipped_data, key = lambda x: x[1])
    # state_strings, state_strings_numbers = zip(*sorted_pairs)
    # unique_state_strings = list(state_strings)
    # counter = 0
    # while counter < len(unique_state_strings):
    #     state = unique_state_strings[counter]
    #     rotated_states = [state]
    #     for k in range(1, 4):
    #         rotation = rotate_90_degrees(state, "cw", k)
    #         if rotation not in rotated_states:
    #             rotated_states.append(rotation)
    #     rotated_states_numbers = numberify_state_strings(rotated_states, players_symbols)
    #     zipped_data = zip(rotated_states, rotated_states_numbers)
    #     sorted_pairs = sorted(zipped_data, key = lambda x: x[1])
    #     try:
    #         rotated_states, _ = zip(*sorted_pairs)
    #         for string in rotated_states[1:]:
    #             unique_state_strings.remove(string)
    #     except: unique_state_strings, unique_state_numbers = [], []
    #     counter += 1
    # unique_state_numbers = numberify_state_strings(unique_state_strings, players_symbols)
    unique_strings = []
    unique_numbers = []
    seen_unique_numbers = set()  # all rotation-related numbers of the kept (unique) states
    append_string = unique_strings.append
    append_number = unique_numbers.append
    for k in range(len(state_strings)):
        state = state_strings[k]
        rotated_states = [state, rotate_90_degrees(state, "cw", 1), rotate_90_degrees(state, "cw", 2), rotate_90_degrees(state, "cw", 3)]
        rotated_states_numbers = numberify_state_strings(rotated_states, players_symbols)
        is_unique = not any(num in seen_unique_numbers for num in rotated_states_numbers)
        if is_unique:
            min_index = rotated_states_numbers.index(min(rotated_states_numbers))
            seen_unique_numbers.update(rotated_states_numbers)
            append_number(rotated_states_numbers[min_index])
            append_string(rotated_states[min_index])
    zipped_data = zip(unique_strings, unique_numbers)
    sorted_pairs = sorted(zipped_data, key = lambda x: x[1])
    try:
        unique_state_strings, unique_state_numbers = zip(*sorted_pairs)
    except:
        unique_state_strings, unique_state_numbers = [], []
    return list(unique_state_strings), list(unique_state_numbers)

# print the game state
def print_game_state(state, players_symbols = players_symbols_default, print_gap_info = [1, "  ", ""]):
    '''
    print the game state
    players_symbols is a dictionary mapping player values to symbols (for example: {1: "x", 2: "o", 0: "_"})
    print_gap_info is a list with the following elements:
        the number of repeated columns, the gap between columns, and the gap between rows
    '''
    n = state.shape[0]
    state = state.tolist()
    columns_gap = print_gap_info[0] * print_gap_info[1]
    rows_gap = print_gap_info[2]
    rows_gap_total = print_gap_info[0] * ("\n" + (n - 1) * (rows_gap + columns_gap) + rows_gap)
    columns_margin = (len(rows_gap) - 1) * " "
    for row in range(n):
        for column in range(n):
            if state[row][column] == 1:
                print(columns_margin + players_symbols[1], end = columns_gap)
            elif state[row][column] == 2:
                print(columns_margin + players_symbols[2], end = columns_gap)
            else:
                if len(players_symbols) > 2:
                    print(columns_margin + players_symbols[0], end = columns_gap)
                else:
                    print(columns_margin + "_", end = columns_gap)
        if row < n - 1:
            print(rows_gap_total)

# print some game statistics
def multinomial(n, k, r):  # n! / (k! * r! * (n - k - r)!)
    ''' 
    calculate the multinomial coefficient
    n is the total number of elements, k is the number of the first type of elements, and r is the number of the second type of elements
    '''
    return m.comb(n, k) * m.comb(n - k, r)

# print some game statistics
def print_game_statistics(board_size):
    '''
    print some game statistics
    '''
    board_cells = board_size ** 2
    # all possible states, not accounting for rotational (90 degrees step) symmetries
    N_all_symmetries = 1  # the case of a blank grid
    for k in range(1, board_cells + 1):
        N_all_symmetries += 2**(k%2) * multinomial(board_cells, int(np.floor(k / 2)), int(np.ceil(k / 2)))
    # accounting for rotational (90 degrees step) symmetries
    # - at least single (180 degrees step) 2-way symmetry
    N_2 = 0
    cells_fill_2 = int(np.floor(board_size**2 / 2))
    color_lim_2 = int(np.floor(board_size**2 / 4))
    for w in range(0, color_lim_2 + 1):
        for b in range(w, min(w + board_size % 2, color_lim_2) + 1):
            N_2 += multinomial(cells_fill_2, w, b) * (3 - int(np.sign(abs(w - b))))**(board_size % 2)
    # - only double (90 degrees step) 4-way symmetry
    N_4 = 0
    cells_fill_4 = int(np.floor(board_size**2 / 4))
    color_lim_4 = int(np.floor(board_size**2 / 8))
    for k in range(0, color_lim_4 + 1):
        N_4 += multinomial(cells_fill_4, k, k) * 3**(board_size % 2)
    # - all possible states accounting for every rotational symmetry
    N_all_unique = int((N_all_symmetries - N_2) / 4 + (N_2 + N_4) / 2)
    
    print("\n--- Game statistics ---")
    print("# board dimension (board size):                                  " + f"{board_size}")
    print("# of total board cells:                                          " + f"{board_cells}")
    print("# of possible board states (including the blank grid case):")
    # print(" - wrt 1 player:                                                 " + f"{int((N_all_symmetries - 1) / 2) + 1}")
    print("    - with rotationally symmetric cases:                         " + f"{N_all_symmetries}")
    print("    - unique / no rotationally symmetric cases:                  " + f"{N_all_unique}")
    print("         - assymetric:                                           " + f"{int((N_all_symmetries - N_2) / 4)}")
    print("         - 2-way symmetry (180 degrees step):                    " + f"{int((N_2 - N_4) / 2)}")
    print("         - 4-way symmetry (90 degrees step):                     " + f"{N_4}")


# generated_string = generate_states_strings(players_symbols = {1: "x", 2: "o", 0: "_"}, n = 9, k = 3, l = 2)[:10]
# recovered_state = convert_strings_to_states(generated_string)
# converted_string = convert_states_to_strings(recovered_state)
# for string in generated_string: print(string)
# print("\nRecovered states:")
# for state in recovered_state: print(state)
# print("\nConverted states to strings:", converted_string)
