import numpy as np


def evaluate_game_state(state):
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
        return 2
    
def print_game_sate(state):
    print(state)

def make_turn(state, player, move):
    pass

def rotate_tablo(state, direction):
    pass


if __name__ == "__main__":
    # tablo is a square grid
    # 1 for 1st player and -1 for 2nd player
    tablo_size = 4
    state = np.zeros((4, 4))
    # state = np.array([[1, -1, -1, -1], \
    #                   [0, 1, -1, -1], \
    #                   [0, 1, 1, 1], \
    #                   [1, 1, 0, -1]])
    evaluation = evaluate_game_state(state)
    print(evaluation)
    print_game_sate(state)