import numpy as np
import orbital_logic_game_functions as olgf


if __name__ == "__main__":
    # board is a square grid
    # 1 for 1st player and -1 for 2nd player
    players = ["x", "o"]
    transfer_directions = ["u", "d", "l", "r"]
    rotate_directions = ["-", "+"]
    board_size = 4
    state = np.zeros((board_size, board_size))
    # state = np.array([[0, 0, 0], \
    #                   [0, 1, 0], \
    #                   [0, 0, 0]])
    # state = np.array([[1, 2, 2, 2], \
    #                   [0, 1, 2, 2], \
    #                   [0, 1, 1, 2], \
    #                   [1, 1, 0, 2]])
    state = olgf.convert_game_state(state, old_symbols = [1, 2, 0], new_symbols = [1, -1, 0])
    # olgf.print_game_statistics(board_size)

    
    verbose_print = True
    games_played = 1000
    board_size = 4
    state = np.zeros((board_size, board_size))
    print(2*"\n")
    olgf.print_game_state(state, players, [1, "  ", ""])
    print(3*"\n")
    counters = [0, 0, 0]
    for i in range(games_played):
        state = np.zeros((board_size, board_size))
        player = (-1) ** i
        for j in range(board_size ** 2):
            possible_moves = olgf.get_possible_moves(state, player, "+")
            next_move = np.random.choice(possible_moves)
            new_state = olgf.play_turn(state, next_move)
            if verbose_print: print(next_move); olgf.print_game_state(new_state, players, [1, "  ", ""]); print(2*"\n")
            player *= -1
            state = np.copy(new_state)
            if olgf.evaluate_game_state(state) == 1:
                if verbose_print: print(f"Player 1 wins\n")
                counters[0] += 1
                break
            if olgf.evaluate_game_state(state) == -1:
                if verbose_print: print(f"Player 2 wins\n")
                counters[1] += 1
                break
            if j == board_size ** 2 - 1:
                if verbose_print: print("It's a draw\n")
                counters[2] += 1
                break
    print(f"Player 1 wins {counters[0]} times ({counters[0] / games_played * 100}%)")
    print(f"Player 2 wins {counters[1]} times ({counters[1] / games_played * 100}%)")
    print(f"It's a draw {counters[2]} times ({counters[2] / games_played * 100}%)")


    # # Example usage:
    # n = 4
    # k = 1
    # l = 1

    # all_strings = olgf.generate_strings(n, k, l)
    # print(f"Total strings: {len(all_strings)}")
    # # Print a few example strings:
    # for s in all_strings[:]:
    #     print(s)
