import numpy as np
import math as m
from itertools import combinations


# evaluate and score the game state
def evaluate_game_state(state):
    ''' check if the game state is a win for a player
    return 1 if player 1 wins, -1 if player 2 wins, and 0 if it is a draw '''
    players_check = [False, False]
    for k in range(2):
        board_size = state.shape[0]
        state_columns_check = np.prod(np.ones((1, board_size)) @ (state - (-1)**k * board_size * np.eye(board_size))) == 0
        state_rows_check = np.prod(np.ones((1, board_size)) @ (state.T - (-1)**k * board_size * np.eye(board_size))) == 0
        state_diagonals_check = np.sum(np.diag(state)) == (-1)**k * board_size or np.sum(np.diag(np.fliplr(state))) == (-1)**k * board_size
        if state_columns_check or state_rows_check or state_diagonals_check:
            players_check[k] = True
    if players_check.count(True) == 2:
        return 0
    elif players_check.count(True) == 1:
        return (-1)**players_check.index(True)
    else:
        return None

# rotate the game board along a certain direction
def rotate_board(state, rotate_direction = "clockwise"):
    ''' rotate the game board along a certain direction
    rotate_direction can be clockwise or "counterclockwise '''
    n = state.shape[0]
    rotated_state = state.copy()
    for layer in range(n // 2):
        # in order: top row (left to right), right column (excluding corners), bottom row (right to left), left column (excluding corners)
        elements = ([(layer, j) for j in range(layer, n - layer)] + \
                    [(i, n - layer - 1) for i in range(layer + 1, n - layer - 1)] + \
                    [(n - layer - 1, j) for j in range(n - layer - 1, layer - 1, -1)] + \
                    [(i, layer) for i in range(n - layer - 2, layer, -1)])
        if rotate_direction.lower() in ["clockwise", "cw", "-", "-1"]:
            for k in range(len(elements)):
                rotated_state[elements[k]] = state[elements[k - 1]]
        elif rotate_direction.lower() in ["counterclockwise", "ccw", "+", "1"]:
            for k in range(len(elements)):
                rotated_state[elements[k]] = state[elements[(k + 1) % len(elements)]]
        else:
            print("Direction must be \"clockwise\" or \"counterclockwise\".")
    return rotated_state

# make a player's turn
def play_turn(state, next_move):
    ''' make a player's turn
    next_move is a dictionary with the following keys:
    "player" for the player making the move (1 for player 1, -1 for player 2),
    "transfer" for the transfer move of an opponent's piece to an adjacent cell (optional),
    "add" for the add move of a player's piece to an empty cell (optional),
    "rotate" for the direction of the board rotation '''
    state_copy = state.copy()
    n = state.shape[0]
    player = next_move["player"]  # 1 for player 1, -1 for player 2
    transfer = next_move["transfer"]  # [position, direction] of the piece to be transferred
    add = next_move["add"]  # position of the piece to be added
    rotate = next_move["rotate"]  # direction of the board rotation
    opponent = -player
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
def get_possible_moves(state, player = 1, rotate_direction = "clockwise"):
    ''' return a list of all possible moves for a player, from the current game state
    player is 1 for player 1 and -1 for player 2
    rotate_direction is clockwise or counterclockwise '''
    n = state.shape[0]
    opponent = -player
    possible_moves = []
    # create possible moves with both transfer and add
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
    # create possible moves with add only
    for row in range(n):
        for column in range(n):
            if state[row, column] == 0:
                possible_moves.append({"player": player, "transfer": None, "add": (row, column), "rotate": rotate_direction})
    return possible_moves

# generate all possible strings of a certain length n with k and l occurrences of two symbols
def generate_states_strings(n, k, l):
    ''' n is the length of the string, k is the number of the first symbol, and l is the number of the second symbol in the string '''
    symbols = ["1", "2", "0"]
    if k + l > n: print("The sum of k and l must not exceed n."); return None
    positions = list(range(n))  # positions in the string
    all_strings = []  # list to store the generated strings
    for x_positions in combinations(positions, k):  # choose k positions for one symbol in the string
        remaining_positions = set(positions) - set(x_positions)  # remaining positions for the other symbols in the string
        for o_positions in combinations(sorted(remaining_positions), l):  # choose l positions for the other symbol in the string
            s = [str(symbols[2])] * n  # initialize the string with default symbols
            for pos in x_positions:  # place the first symbol in the chosen related positions
                s[pos] = str(symbols[0])
            for pos in o_positions:  # place the second symbol in the chosen related positions
                s[pos] = str(symbols[1])
            all_strings.append(''.join(s))  # add the string to the list
    return all_strings

# convert strings of the game states to their processable form
def convert_strings_to_states(states_strings, old_symbols = ["1", "2", "0"], new_symbols = [1, -1, 0]):
    ''' convert strings of the game states to their processable form
    old_symbols is a list of the symbols in the strings
    new_symbols is a list of the symbols in the processable form '''
    converted_states = []
    n = np.sqrt(len(states_strings[0])).astype(int)
    for state_string in states_strings:
        converted_state = np.zeros((n, n))
        for k in range(len(old_symbols)):
            converted_state[state_string == old_symbols[k]] = new_symbols[k]
        converted_states.append(converted_state)
    return converted_states

# convert the game state from its readable form to its processable form
def convert_game_state(state, old_symbols = [1, 2, 0], new_symbols = [1, -1, 0]):
    ''' convert the game state from its readable form to its processable form
    old_symbols is a list of the symbols in the readable form
    new_symbols is a list of the symbols in the processable form '''
    converted_state = np.copy(state)
    for k in range(len(old_symbols)):
        converted_state[state == old_symbols[k]] = new_symbols[k]
    return converted_state

# print the game state
def print_game_state(state, players = ["x", "o"], print_gap_info = [1, "  ", ""]):
    ''' print the game state
    players is a list of the symbols for the players
    print_gap_info is a list with the following elements:
    the number of repeated columns, the gap between columns, and the gap between rows '''
    n = state.shape[0]
    state = state.tolist()
    columns_gap = print_gap_info[0] * print_gap_info[1]
    rows_gap = print_gap_info[2]
    rows_gap_total = print_gap_info[0] * ("\n" + (n - 1) * (rows_gap + columns_gap) + rows_gap)
    columns_margin = (len(rows_gap) - 1) * " "
    for row in range(n):
        for column in range(n):
            if state[row][column] == 1:
                print(columns_margin + players[0], end = columns_gap)
            elif state[row][column] == -1:
                print(columns_margin + players[1], end = columns_gap)
            else:
                if len(players) > 2:
                    print(columns_margin + players[2], end = columns_gap)
                else:
                    print(columns_margin + "_", end = columns_gap)
        if row < n - 1:
            print(rows_gap_total)

# print some game statistics
def multinomial(n, k, r):  # n! / (k! * r! * (n - k - r)!)
    ''' calculate the multinomial coefficient
    n is the total number of elements, k is the number of the first type of elements, and r is the number of the second type of elements '''
    return m.comb(n, k) * m.comb(n - k, r)

def print_game_statistics(board_size):
    ''' print some game statistics '''
    board_cells = board_size ** 2
    possible_board_states = 0
    # calculate the number of possible board states starting with the same player (else it is double that number)
    # not accounting for rotational symmetries
    for k in range(1, board_cells + 1):
        # print(multinomial(board_cells, int(np.floor(k / 2)), int(np.ceil(k / 2))))
        possible_board_states += multinomial(board_cells, int(np.floor(k / 2)), int(np.ceil(k / 2)))
    # accounting for rotational symmetries
    
    print("\n--- Game statistics ---")
    align_gap = 1
    print("board size (# of side cells):   " + "\t" + f"{board_size}")
    print("# of total board cells:         " + "\t" + f"{board_cells}")
    print("# of possible board states:     " + "\t" + f"{possible_board_states}")
