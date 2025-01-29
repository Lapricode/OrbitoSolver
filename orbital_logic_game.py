import numpy as np


def convert_game_state(state):
    converted_state = np.copy(state)
    converted_state[state == 1] = 1
    converted_state[state == 2] = -1
    return converted_state

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

def rotate_tablo(state, direction):
    return np.copy(state)

def play_turn(state, player, add_position, remove_position, rotate_direction):
    converted_state = convert_game_state(state)
    if add_position != None:
        converted_state[add_position] = player
    if remove_position != None and state[remove_position] != player:
        converted_state[remove_position] = 0
    final_state = rotate_tablo(converted_state, rotate_direction)
    return final_state

def print_game_sate(state):
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
    new_state = play_turn(state, 2, (0, 0), (0, 1), 1)
    print(state_evaluation)
    print_game_sate(state)
    print_game_sate(new_state)