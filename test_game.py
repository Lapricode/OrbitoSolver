import numpy as np
import time
import orbital_logic_game_functions as olgf


if __name__ == "__main__":
    # board is a square grid
    # 1 for 1st player and -1 for 2nd player
    processable_symbols = [1, -1, 0]
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
    state = olgf.convert_game_state(state, old_symbols = [1, 2, 0], new_symbols = processable_symbols)
    # olgf.print_game_statistics(board_size)


    start_time = time.time()
    
    players = ["x", "o"]
    verbose_print = False
    games_played = 10000
    board_size = 3
    olgf.print_game_statistics(board_size)
    counters = {"1_start_1_win": 0, "1_start_2_win": 0, "2_start_1_win": 0, "2_start_2_win": 0, "1_start_draw": 0, "2_start_draw": 0}
    start_player = 1
    for i in range(games_played):
        state = np.zeros((board_size, board_size))
        start_player *= -1
        player = start_player
        if verbose_print: print(f"\nGame {i+1}\n"); olgf.print_game_state(state, players, [1, "  ", ""]); print(2*"\n")
        for j in range(board_size ** 2):
            possible_moves = olgf.get_possible_moves(state, player, "+")
            next_move = np.random.choice(possible_moves)
            new_state = olgf.play_turn(state, next_move)
            if verbose_print: print(next_move); olgf.print_game_state(new_state, players, [1, "  ", ""]); print(2*"\n")
            player *= -1
            state = np.copy(new_state)
            if olgf.evaluate_game_state(state) == 1:
                if start_player == 1:
                    counters["1_start_1_win"] += 1
                elif start_player == -1:
                    counters["2_start_1_win"] += 1
                if verbose_print: print(f"Player 1 wins\n")
                break
            if olgf.evaluate_game_state(state) == -1:
                if start_player == 1:
                    counters["1_start_2_win"] += 1
                elif start_player == -1:
                    counters["2_start_2_win"] += 1
                if verbose_print: print(f"Player 2 wins\n")
                break
            if j == board_size ** 2 - 1:
                if start_player == 1:
                    counters["1_start_draw"] += 1
                elif start_player == -1:
                    counters["2_start_draw"] += 1
                if verbose_print: print("It's a draw\n")
                break
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
    print(f"   - Player 2 starts and draws:    {counters['2_start_draw']:10d} \t|\t {counters['2_start_draw']/games_played*100:.2f}%")

    print(f"\nTime taken: {time.time() - start_time:.2f} seconds")


    # # Example usage:
    # n = 4
    # k = 1
    # l = 1

    # all_strings = olgf.generate_states_strings(n, k, l)
    # print(f"Total strings: {len(all_strings)}")
    # # Print a few example strings:
    # for s in all_strings[:]:
    #     print(s)
