import numpy as np
import time
import orbital_logic_game_functions as olgf


if __name__ == "__main__":
    # board is a square grid
    # 1 for 1st player and -1 for 2nd player
    processable_symbols = [1, -1, 0]
    transfer_directions = ["u", "d", "l", "r"]
    rotate_directions = ["-", "+", "0"]
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
    
    verbose_print = False
    games_played = 10000
    board_size = 3
    olgf.print_game_statistics(board_size)
    counters = {"1_start_1_win": 0, "1_start_2_win": 0, "2_start_1_win": 0, "2_start_2_win": 0, "1_start_draw": 0, "2_start_draw": 0}
    start_player = 1
    players_symbols = {1: "x", -1: "o", 0: "_"}
    rotate_direction = "0"
    transfer_allowed = True
    for i in range(games_played):
        state = np.zeros((board_size, board_size))
        start_player *= -1
        player = start_player
        if verbose_print: print(f"\nGame {i+1}\n"); olgf.print_game_state(state, players, [1, "  ", ""]); print(2*"\n")
        for j in range(board_size ** 2):
            possible_moves = olgf.get_possible_moves(state, rotate_direction, transfer_allowed, player)
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


# Initial game state:

# x  o  _  x  
      
# x  o  o  _  
      
# _  o  x  x  
      
# o  _  _  _  

# Current player: 1


# Time taken: 5.8157689571380615 sec


# Estimated draw

# Perfect game evolution (states with perfect play):

# State after move 0:
# x  o  _  x  
      
# x  o  o  _  
      
# _  o  x  x  
      
# o  _  _  _  

# Move 1: {'player': 1, 'transfer': [(1, 2), 'r'], 'add': (2, 0), 'rotate': '-1'}
# State after move 1:
# x  x  o  _  
      
# x  o  o  x  
      
# o  x  _  o  
      
# _  _  _  x  

# Move 2: {'player': -1, 'transfer': None, 'add': (3, 0), 'rotate': '-1'}
# State after move 2:
# x  x  x  o  
      
# o  x  o  _  
      
# o  _  o  x  
      
# _  _  x  o  

# Move 3: {'player': 1, 'transfer': [(1, 2), 'r'], 'add': (1, 2), 'rotate': '-1'}
# State after move 3:
# o  x  x  x  
      
# o  _  x  o  
      
# _  o  x  o  
      
# _  x  o  x  

# Move 4: {'player': -1, 'transfer': [(1, 2), 'l'], 'add': (1, 2), 'rotate': '-1'}
# State after move 4:
# o  o  x  x  
      
# _  o  x  x  
      
# _  x  o  o  
      
# x  o  x  o


# Welcome to the game solver!

# --- Game statistics ---
# board size (# of side cells):   	4
# # of total board cells:         	16
# # of possible board states:     	10165778


# Initial game state:

# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  

# Current player: 1


# Time taken: 53970.53447508812 sec


# Estimated draw

# Perfect game evolution (states with perfect play):

# State after move 0:
# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  

# Move 1: {'player': 1, 'transfer': None, 'add': (0, 0), 'rotate': '0'}
# State after move 1:
# x  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  

# Move 2: {'player': -1, 'transfer': None, 'add': (0, 1), 'rotate': '0'}
# State after move 2:
# x  o  _  _  
      
# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  

# Move 3: {'player': 1, 'transfer': None, 'add': (0, 2), 'rotate': '0'}
# State after move 3:
# x  o  x  _  
      
# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  

# Move 4: {'player': -1, 'transfer': None, 'add': (0, 3), 'rotate': '0'}
# State after move 4:
# x  o  x  o  
      
# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  

# Move 5: {'player': 1, 'transfer': None, 'add': (1, 0), 'rotate': '0'}
# State after move 5:
# x  o  x  o  
      
# x  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  

# Move 6: {'player': -1, 'transfer': None, 'add': (1, 1), 'rotate': '0'}
# State after move 6:
# x  o  x  o  
      
# x  o  _  _  
      
# _  _  _  _  
      
# _  _  _  _  

# Move 7: {'player': 1, 'transfer': None, 'add': (1, 2), 'rotate': '0'}
# State after move 7:
# x  o  x  o  
      
# x  o  x  _  
      
# _  _  _  _  
      
# _  _  _  _  

# Move 8: {'player': -1, 'transfer': None, 'add': (1, 3), 'rotate': '0'}
# State after move 8:
# x  o  x  o  
      
# x  o  x  o  
      
# _  _  _  _  
      
# _  _  _  _  

# Move 9: {'player': 1, 'transfer': None, 'add': (2, 0), 'rotate': '0'}
# State after move 9:
# x  o  x  o  
      
# x  o  x  o  
      
# x  _  _  _  
      
# _  _  _  _  

# Move 10: {'player': -1, 'transfer': None, 'add': (3, 0), 'rotate': '0'}
# State after move 10:
# x  o  x  o  
      
# x  o  x  o  
      
# x  _  _  _  
      
# o  _  _  _  

# Move 11: {'player': 1, 'transfer': None, 'add': (2, 1), 'rotate': '0'}
# State after move 11:
# x  o  x  o  
      
# x  o  x  o  
      
# x  x  _  _  
      
# o  _  _  _  

# Move 12: {'player': -1, 'transfer': None, 'add': (2, 2), 'rotate': '0'}
# State after move 12:
# x  o  x  o  
      
# x  o  x  o  
      
# x  x  o  _  
      
# o  _  _  _  

# Move 13: {'player': 1, 'transfer': None, 'add': (2, 3), 'rotate': '0'}
# State after move 13:
# x  o  x  o  
      
# x  o  x  o  
      
# x  x  o  x  
      
# o  _  _  _  

# Move 14: {'player': -1, 'transfer': None, 'add': (3, 1), 'rotate': '0'}
# State after move 14:
# x  o  x  o  
      
# x  o  x  o  
      
# x  x  o  x  
      
# o  o  _  _  

# Move 15: {'player': 1, 'transfer': None, 'add': (3, 2), 'rotate': '0'}
# State after move 15:
# x  o  x  o  
      
# x  o  x  o  
      
# x  x  o  x  
      
# o  o  x  _  

# Move 16: {'player': -1, 'transfer': None, 'add': (3, 3), 'rotate': '0'}
# State after move 16:
# x  o  x  o  
      
# x  o  x  o  
      
# x  x  o  x  
      
# o  o  x  o


# Welcome to the game solver!

# --- Game statistics ---
# board size (# of side cells):   	4
# # of total board cells:         	16
# # of possible board states:     	10165778


# Initial game state:

# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  

# Current player: 1

 
# Time taken: 171310.278 sec

# Player x wins!

# Perfect game evolution (states with perfect play):

# State after move 0:
# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  
      
# _  _  _  _  

# Move 1 (x) : {'player': 1, 'transfer': None, 'add': (1, 1), 'rotate': '+'}
# State after move 1:
# _  _  _  _  
      
# _  _  _  _  
      
# _  x  _  _  
      
# _  _  _  _  

# Move 2 (o) : {'player': -1, 'transfer': None, 'add': (0, 0), 'rotate': '+'}
# State after move 2:
# _  _  _  _  
      
# o  _  _  _  
      
# _  _  x  _  
      
# _  _  _  _  

# Move 3 (x) : {'player': 1, 'transfer': None, 'add': (0, 0), 'rotate': '+'}
# State after move 3:
# _  _  _  _  
      
# x  _  x  _  
      
# o  _  _  _  
      
# _  _  _  _  

# Move 4 (o) : {'player': -1, 'transfer': None, 'add': (2, 1), 'rotate': '+'}
# State after move 4:
# _  _  _  _  
      
# _  x  _  _  
      
# x  _  o  _  
      
# o  _  _  _  

# Move 5 (x) : {'player': 1, 'transfer': None, 'add': (1, 2), 'rotate': '+'}
# State after move 5:
# _  _  _  _  
      
# _  x  o  _  
      
# _  x  _  _  
      
# x  o  _  _  

# Move 6 (o) : {'player': -1, 'transfer': None, 'add': (1, 3), 'rotate': '+'}
# State after move 6:
# _  _  _  o  
      
# _  o  _  _  
      
# _  x  x  _  
      
# _  x  o  _  

# Move 7 (x) : {'player': 1, 'transfer': None, 'add': (1, 0), 'rotate': '+'}
# State after move 7:
# _  _  o  _  
      
# _  _  x  _  
      
# x  o  x  _  
      
# _  _  x  o  

# Move 8 (o) : {'player': -1, 'transfer': None, 'add': (0, 3), 'rotate': '+'}
# State after move 8:
# _  o  o  _  
      
# _  x  x  _  
      
# _  _  o  o  
      
# x  _  _  x  

# Move 9 (x) : {'player': 1, 'transfer': None, 'add': (0, 3), 'rotate': '+'}
# State after move 9:
# o  o  x  _  
      
# _  x  o  o  
      
# _  x  _  x  
      
# _  x  _  _  

# Move 10 (o) : {'player': -1, 'transfer': None, 'add': (2, 2), 'rotate': '+'}
# State after move 10:
# o  x  _  o  
      
# o  o  o  x  
      
# _  x  x  _  
      
# _  _  x  _  

# Move 11 (x) : {'player': 1, 'transfer': None, 'add': (3, 3), 'rotate': '+'}
# State after move 11:
# x  _  o  x  
      
# o  o  x  _  
      
# o  o  x  x  
      
# _  _  _  x  

# Move 12 (o) : {'player': -1, 'transfer': None, 'add': (0, 1), 'rotate': '+'}
# State after move 12:
# o  o  x  _  
      
# x  x  x  x  
      
# o  o  o  x  
      
# o  _  _  _
