import numpy as np
import orbital_logic_game_functions as olgf


if __name__ == "__main__":
    # board is a square grid
    # 1 for 1st player and -1 for 2nd player
    players = ["x", "o"]
    board_size = 3
    olgf.print_game_statistics(board_size)
    state = np.zeros((board_size, board_size))
    # state = np.array([[1, 0, 0], \
    #                   [0, 0, 0], \
    #                   [0, 0, 0]])
    # state = np.array([[1, 2, 2, 2], \
    #                   [0, 1, 2, 2], \
    #                   [0, 1, 1, 2], \
    #                   [1, 1, 0, 2]])
    state = olgf.convert_game_state(state, old_symbols = [1, 2, 0], new_symbols = [1, -1, 0])
    state_evaluation = olgf.evaluate_game_state(state)
    new_state = olgf.play_turn(state, 1, [(0, 0), "d"], (2, 0), "+")
    print(2*"\n")
    olgf.print_game_state(state, players, [2, "  ", ""])
    print(3*"\n")
    olgf.print_game_state(new_state, players, [2, "  ", ""])
