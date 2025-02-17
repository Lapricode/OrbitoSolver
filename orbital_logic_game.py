import numpy as np
import orbital_logic_game_functions as olgf


if __name__ == "__main__":
    # board is a square grid
    # 1 for 1st player and -1 for 2nd player
    players = ["x", "o"]
    board_size = 3
    rotate_direction = "+"
    olgf.print_game_statistics(board_size)
    state = np.zeros((board_size, board_size))
    # state = np.array([[0, 0, 0], \
    #                   [0, 1, 0], \
    #                   [0, 0, 0]])
    # state = np.array([[1, 2, 2, 2], \
    #                   [0, 1, 2, 2], \
    #                   [0, 1, 1, 2], \
    #                   [1, 1, 0, 2]])
    state = olgf.convert_game_state(state, old_symbols = [1, 2, 0], new_symbols = [1, -1, 0])
    state_evaluation = olgf.evaluate_game_state(state)
    next_move = {"player": -1, "transfer": [(0, 0), "d"], "addition": (2, 0), "rotation": "+"}
    new_state = olgf.play_turn(state, next_move)
    
    print(2*"\n")
    olgf.print_game_state(state, players, [2, "  ", ""])
    
    # possible_moves = olgf.get_possible_moves(state, 1, rotate_direction)
    # print(3*"\n")
    # for k in range(len(possible_moves)):
    #     print(possible_moves[k])
    # print("\n")
    # possible_moves = olgf.get_possible_moves(state, -1, rotate_direction)
    # for k in range(len(possible_moves)):
    #     print(possible_moves[k])
    
    print(3*"\n")
    olgf.print_game_state(new_state, players, [2, "  ", ""])

    print(3*"\n")

    # Example usage:
    n = 4
    k = 1
    l = 1

    all_strings = olgf.generate_strings(n, k, l)
    print(f"Total strings: {len(all_strings)}")
    # Print a few example strings:
    for s in all_strings[:]:
        print(s)
