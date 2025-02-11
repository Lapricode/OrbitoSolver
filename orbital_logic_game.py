import numpy as np


# evaluate and score the game state
def evaluate_game_state(state):
    state = convert_game_state(state)
    players_check = [False, False]
    for k in range(2):
        tablo_size = state.shape[0]
        state_columns_check = np.prod(np.ones((1, tablo_size)) @ (state - (-1)**k * tablo_size * np.eye(tablo_size))) == 0
        state_rows_check = np.prod(np.ones((1, tablo_size)) @ (state.T - (-1)**k * tablo_size * np.eye(tablo_size))) == 0
        state_diagonals_check = np.sum(np.diag(state)) == (-1)**k * tablo_size or np.sum(np.diag(np.fliplr(state))) == (-1)**k * tablo_size
        if state_columns_check or state_rows_check or state_diagonals_check:
            players_check[k] = True
    print(players_check)
    if players_check.count(True) == 2:
        return 0
    elif players_check.count(True) == 1:
        return (-1)**players_check.index(True)
    else:
        return None

# rotate the game tablo along a certain direction
def rotate_tablo(state, rotate_direction = "clockwise"):
    n = state.shape[0]
    rotated_state = state.copy()
    for layer in range(n // 2):
        elements = (
            [(layer, j) for j in range(layer, n - layer)] +  # Top row (left to right)
            [(i, n - layer - 1) for i in range(layer + 1, n - layer - 1)] +  # Right column (excluding corners)
            [(n - layer - 1, j) for j in range(n - layer - 1, layer - 1, -1)] +  # Bottom row (right to left)
            [(i, layer) for i in range(n - layer - 2, layer, -1)]  # Left column (excluding corners)
        )
        if rotate_direction.lower() in ["clockwise", "cw", "+", "1"]:
            for k in range(len(elements)):
                rotated_state[elements[k]] = state[elements[k - 1]]
        elif rotate_direction.lower() in ["counterclockwise", "ccw", "-", "-1"]:
            for k in range(len(elements)):
                rotated_state[elements[k]] = state[elements[(k + 1) % len(elements)]]
        else:
            print("Direction must be \"clockwise\" or \"counterclockwise\".")
    return rotated_state

# make a given player's turn
def play_turn(state, player, add_position, remove_position, rotate_direction):
    converted_state = convert_game_state(state)
    if add_position != None:
        converted_state[add_position] = player
    if remove_position != None and state[remove_position] != player:
        converted_state[remove_position] = 0
    final_state = rotate_tablo(converted_state, str(rotate_direction))
    return final_state

# convert the game state from its readable form to its processable form
def convert_game_state(state):
    converted_state = np.copy(state)
    converted_state[state == 1] = 1
    converted_state[state == 2] = -1
    return converted_state

# print the game state
def print_game_state(state):
    state[state == 1] = 1
    state[state == -1] = 2
    print(state)


if __name__ == "__main__":
    # tablo is a square grid
    # 1 for 1st player and -1 for 2nd player
    tablo_size = 4
    state = np.zeros((4, 4))
    state = np.array([[1, 2, 2, 2], \
                      [0, 1, 2, 2], \
                      [0, 1, 1, 2], \
                      [1, 1, 0, 2]])
    state_evaluation = evaluate_game_state(state)
    new_state = play_turn(state, 2, (0, 0), (0, 1), "+")
    rotated_state_cw = rotate_tablo(state, "+")
    rotated_state_ccw = rotate_tablo(state, "-")
    print(state_evaluation)
    print_game_state(state)
    print_game_state(new_state)
    print_game_state(rotated_state_cw)
    print_game_state(rotated_state_ccw)
    
