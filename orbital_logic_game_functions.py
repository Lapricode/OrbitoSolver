import numpy as np
import math as m
from itertools import combinations


# important global variables for the functions
clockwise_rotation_keywords = ["clockwise", "cw", "-", "-1"]
counterclockwise_rotation_keywords = ["counterclockwise", "ccw", "+", "1"]
stand_still_keywords = ["still", "s", "x", "0"]
players_symbols_default = {1: "x", 2: "o", 0: "_"}

# evaluate and score the game state
def evaluate_game_state(state):
    '''
    check if the game state is a win for a player
    return 1 if player 1 wins, 2 if player 2 wins, and 0 if it is a draw
    '''
    state_copy = np.copy(state)
    state_copy[state_copy == 2] = -1
    players_check = [False, False]
    for k in range(2):
        board_size = state_copy.shape[0]
        state_columns_check = np.prod(np.ones((1, board_size)) @ (state_copy - (-1)**k * board_size * np.eye(board_size))) == 0
        state_rows_check = np.prod(np.ones((1, board_size)) @ (state_copy.T - (-1)**k * board_size * np.eye(board_size))) == 0
        state_diagonals_check = np.sum(np.diag(state_copy)) == (-1)**k * board_size or np.sum(np.diag(np.fliplr(state_copy))) == (-1)**k * board_size
        if state_columns_check or state_rows_check or state_diagonals_check:
            players_check[k] = True
    if players_check.count(True) == 2:
        return 0
    elif players_check.count(True) == 1:
        return 1 + players_check.index(True)
    else:
        return None

# rotate the game board along a certain direction
def rotate_board(state, rotate_direction = "clockwise"):
    '''
    rotate the game board along a certain direction
    rotate_direction can be clockwise, counterclockwise, or still
    '''
    n = state.shape[0]
    rotated_state = state.copy()
    for layer in range(n // 2):
        # in order: top row (left to right), right column (excluding corners), bottom row (right to left), left column (excluding corners)
        elements = ([(layer, j) for j in range(layer, n - layer)] + \
                    [(i, n - layer - 1) for i in range(layer + 1, n - layer - 1)] + \
                    [(n - layer - 1, j) for j in range(n - layer - 1, layer - 1, -1)] + \
                    [(i, layer) for i in range(n - layer - 2, layer, -1)])
        if rotate_direction.lower() in clockwise_rotation_keywords:
            for k in range(len(elements)):
                rotated_state[elements[k]] = state[elements[k - 1]]
        elif rotate_direction.lower() in counterclockwise_rotation_keywords:
            for k in range(len(elements)):
                rotated_state[elements[k]] = state[elements[(k + 1) % len(elements)]]
        elif rotate_direction.lower() in stand_still_keywords:
            continue
        else:
            print("Direction must be \"clockwise\", \"counterclockwise\", or \"still\".")
    return rotated_state

# make a player's turn
def play_turn(state, next_move):
    '''
    make a player's turn
    next_move is a dictionary with the following keys:
    "player" for the player making the move (1 for player 1, 2 for player 2),
    "transfer" for the transfer move of an opponent's piece to an adjacent cell (optional),
    "add" for the add move of a player's piece to an empty cell (optional),
    "rotate" for the direction of the board rotation
    '''
    state_copy = state.copy()
    n = state.shape[0]
    player = next_move["player"]  # 1 for player 1, 2 for player 2
    transfer = next_move["transfer"]  # [position, direction] of the piece to be transferred
    add = next_move["add"]  # position of the piece to be added
    rotate = next_move["rotate"]  # direction of the board rotation
    opponent = 3 - player
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
    final_state = rotate_board(state_copy, str(rotate))  # rotate the board
    return final_state

# return a list of all possible moves for a player, from the current game state
def get_possible_moves(state, rotate_direction = "clockwise", transfer_allowed = True, player = 1):
    '''
    return a list of all possible moves for a player, from the current game state
    player is 1 for player 1 and 2 for player 2
    rotate_direction is clockwise, counterclockwise, or still
    '''
    n = state.shape[0]
    opponent = 3 - player
    possible_moves = []
    # create possible moves with add only
    for row in range(n):
        for column in range(n):
            if state[row, column] == 0:
                possible_moves.append({"player": player, "transfer": None, "add": (row, column), "rotate": rotate_direction})
    # create possible moves with both transfer and add
    if transfer_allowed:
        for row in range(n):
            for column in range(n):
                if state[row, column] == opponent:
                    for transfer_direction in ["u", "d", "l", "r"]:
                        # transfer part
                        source_cell = (row, column)
                        target_cell = None
                        if transfer_direction in ["u", "d"]:
                            target = row + (-1) ** (transfer_direction == "u")
                            if 0 <= target < n and state[target, column] == 0:
                                target_cell = (target, column)
                        elif transfer_direction in ["l", "r"]:
                            target = column + (-1) ** (transfer_direction == "l")
                            if 0 <= target < n and state[row, target] == 0:
                                target_cell = (row, target) 
                        # add part
                        if target_cell is not None:
                            for i in range(n):
                                for j in range(n):
                                    if (state[i, j] == 0 and (i, j) != target_cell) or (i, j) == source_cell:
                                        possible_moves.append({"player": player, "transfer": [(row, column), transfer_direction], "add": (i, j), "rotate": rotate_direction})
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
    new_state_string = len(state_string) * ["_"]
    if rotate_direction in clockwise_rotation_keywords and rotate_times % 4 == 1 or rotate_direction in counterclockwise_rotation_keywords and rotate_times % 4 == 3:
        for k in range(len(state_string)): new_state_string[int(n*k+n-1-(n**2+1)*(k//n))] = state_string[k]
        return "".join(new_state_string)
    elif rotate_direction in clockwise_rotation_keywords and rotate_times % 4 == 3 or rotate_direction in counterclockwise_rotation_keywords and rotate_times % 4 == 1:
        for k in range(len(state_string)): new_state_string[int(n*(n-1-k)+(n**2+1)*(k//n))] = state_string[k]
        return "".join(new_state_string)
    elif rotate_times % 4 == 2 and rotate_direction not in stand_still_keywords:
        for k in range(len(state_string)): new_state_string[n**2-1-k] = state_string[k]
        return "".join(new_state_string)
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
    for x_positions in combinations(positions, k):  # choose k positions for one symbol in the string
        remaining_positions = set(positions) - set(x_positions)  # remaining positions for the other symbols in the string
        for o_positions in combinations(sorted(remaining_positions), l):  # choose l positions for the other symbol in the string
            s = [str(players_symbols[0])] * n  # initialize the string with default symbols
            for pos in x_positions:  # place the first symbol in the chosen related positions
                s[pos] = str(players_symbols[1])
            for pos in o_positions:  # place the second symbol in the chosen related positions
                s[pos] = str(players_symbols[2])
            all_strings.append(''.join(s))  # add the string to the list
    return all_strings

# convert game states to their respective strings
def convert_states_to_strings(states, players_symbols = players_symbols_default):
    '''
    convert game states to their respective strings
    players_symbols is a dictionary mapping player symbols to values (for example: {1: "x", 2: "o", 0: "_"})
    '''
    strings = ["".join(players_symbols[cell] for row in state for cell in row) for state in states]
    return strings

# convert strings of the game states to their processable form
def convert_strings_to_states(strings, players_symbols = players_symbols_default):
    '''
    convert strings of the game states to their processable form
    players_symbols is a dictionary mapping player symbols to values (for example: {1: "x", 2: "o", 0: "_"})
    '''
    inverse_mapping = {v: k for k, v in players_symbols.items()}
    states = []
    for s in strings:
        n = int(len(s) ** 0.5)
        if n * n != len(s):
            raise ValueError(f"String '{s}' does not represent a square board.")
        state = [[inverse_mapping[s[r * n + c]] for c in range(n)] for r in range(n)]
        states.append(np.array(state))
    return states

# convert state strings to numbers, with a 1-1 matching, for easier and overall better identification and classification
def numberify_state_strings(state_strings, players_symbols = players_symbols_default):
    '''
    numberify game states for easier and overall better identification, using the base-3 arithmetic system
    players_symbols is a dictionary mapping player symbols to values (for example: {1: "x", 2: "o", 0: "_"})
    '''
    values = []
    for s in state_strings:
        s2 = s.replace(players_symbols[0], "0").replace(players_symbols[1], "1").replace(players_symbols[2], "2")
        values.append(int(s2, 3))
    return values

# convert state numbers to strings, with a 1-1 matching, for easier and better identification and classification
def stringify_states_numbers(state_numbers, state_length, numbers_base = "3", players_symbols = players_symbols_default):
    '''
    stringify game states for easier and overall better identification, using the base-3 arithmetic system
    numbers_base is the arithmetic base of the numbers in the state_numbers list, it can be "3" (base 3) or "10" (base 10)
    players_symbols is a dictionary mapping player symbols to values (for example: {1: "x", 2: "o", 0: "_"})
    '''
    state_strings = []
    if numbers_base == "10":
        for k in range(len(state_numbers)):
            num = int(state_numbers[k])
            if num != 0:
                digits = []
                while num:
                    digits.append(str(num % 3))
                    num //= 3
                state_numbers[k] = "".join(reversed(digits))
            else:
                state_numbers[k] = str(num)
    for k in range(len(state_numbers)):
        state_strings.append(str(state_numbers[k]).replace("0", players_symbols[0]).replace("1", players_symbols[1]).replace("2", players_symbols[2]).rjust(state_length, players_symbols[0]))
    return state_strings

#
def find_unique_states(state_strings, players_symbols = players_symbols_default):
    pass

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
    possible_board_states = 1  # the case of a blank grid
    # calculate the number of possible board states starting with the same player (else it is double that number)
    # not accounting for rotational symmetries
    for k in range(1, board_cells + 1):
        # print(multinomial(board_cells, int(np.floor(k / 2)), int(np.ceil(k / 2))))
        possible_board_states += 2**(k%2) * multinomial(board_cells, int(np.floor(k / 2)), int(np.ceil(k / 2)))
    # accounting for rotational symmetries
    
    print("\n--- Game statistics ---")
    print("# board dimension (board size):     " + f"{board_size}")
    print("# of total board cells:             " + f"{board_cells}")
    print("# of possible board states:         " + f"{possible_board_states}" + "\t(including the blank grid case)")
    print("# of possible states wrt 1 player:  " + f"{int((possible_board_states - 1) / 2)}" + "\t(excluding the blank grid case)")


# generated_string = generate_states_strings(players_symbols = {1: "x", 2: "o", 0: "_"}, n = 9, k = 3, l = 2)[:10]
# recovered_state = convert_strings_to_states(generated_string)
# converted_string = convert_states_to_strings(recovered_state)
# for string in generated_string: print(string)
# print("\nRecovered states:")
# for state in recovered_state: print(state)
# print("\nConverted states to strings:", converted_string)
