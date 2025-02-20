import math
import numpy as np
import time
from orbital_logic_game_functions import evaluate_game_state, get_possible_moves, play_turn, print_game_state, print_game_statistics


def minimax(state, rotate_direction = "clockwise", transfer_allowed = True, player = 1, maximizing_player = None, depth = 0, alpha = -math.inf, beta = math.inf):
    """
    Recursively evaluates the game tree using minimax with alpha-beta pruning.
    Parameters:
        state: the current game state (a numpy array)
        rotate_direction: the direction in which the board is rotated (clockwise, counterclockwise, or still)
        transfer_allowed: whether transfers are allowed between adjacent cells
        player: the player whose turn it is (1 or -1)
        maximizing_player: the player we are trying to maximize (the one for whom we want the best move)
        depth: current depth of recursion (used to favor faster wins/longer delays of losses)
        alpha: the best already explored option along the path to the maximizer
        beta: the best already explored option along the path to the minimizer
    Returns:
        An evaluation score: high positive if maximizing_player wins, high negative if loses, or 0 for a draw.
    """
    result = evaluate_game_state(state)
    if result is not None:
        # Terminal state reached
        if result == 0:
            return 0
        # Reward quicker wins and delay losses
        if result == maximizing_player:
            return 1000 - depth  # faster win is better
        else:
            return -1000 + depth  # slower loss is slightly better

    # Get all possible moves for the current player.
    moves = get_possible_moves(state, rotate_direction, transfer_allowed, player)
    if not moves:
        return 0  # No moves available means a draw (should not normally happen)

    if player == maximizing_player:
        max_eval = -math.inf
        for move in moves:
            new_state = play_turn(state, move)
            eval = minimax(new_state, rotate_direction, transfer_allowed, -player, maximizing_player, depth + 1, alpha, beta)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # beta cut-off
        return max_eval
    else:
        min_eval = math.inf
        for move in moves:
            new_state = play_turn(state, move)
            eval = minimax(new_state, rotate_direction, transfer_allowed, -player, maximizing_player, depth + 1, alpha, beta)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # alpha cut-off
        return min_eval

def find_best_move(state, rotate_direction = "clockwise", transfer_allowed = True, player = 1):
    """
    Determines the best move for the given player from the current state.
    Parameters:
        state: the current game state (a numpy array)
        player: the player whose move is to be determined (1 or -1)
    Returns:
        A tuple (best_move, best_value) where best_move is the move (a dictionary)
        and best_value is its minimax evaluation.
    """
    best_move = None
    best_value = -math.inf
    moves = get_possible_moves(state, rotate_direction, transfer_allowed, player)

    for move in moves:
        new_state = play_turn(state, move)
        move_value = minimax(new_state, rotate_direction, transfer_allowed, -player, player, depth = 1, alpha = -math.inf, beta = math.inf)
        if move_value > best_value:
            best_value = move_value
            best_move = move
    return best_move, best_value

def simulate_principal_variation(state, rotate_direction = "clockwise", player = 1, max_steps = 50):
    """
    Simulate a possible evolution of the game (the principal variation) by alternately choosing best moves.
    Parameters:
        state: the starting game state (a numpy array)
        player: the player whose turn it is (1 or -1)
        max_steps: maximum number of moves to simulate to avoid infinite loops
    Returns:
        A list of game states representing the evolution from the current state.
    """
    evolution = [{"move": None, "state": state}]
    current_state = state
    current_player = player
    steps = 0
    while steps < max_steps:
        if evaluate_game_state(current_state) is not None:
            break
        best_move, score = find_best_move(current_state, rotate_direction, transfer_allowed, current_player)
        if best_move is None:
            break
        current_state = play_turn(current_state, best_move)
        evolution.append({"move": best_move, "state": current_state})
        current_player = -current_player
        steps += 1
    return evolution

def estimated_game_result(score, players = ["x", "o"], start_player = 1, maximizing_player = None):
    """
    Returns a message describing the estimated result of the game based on the minimax score.
    Parameters:
        score: the minimax evaluation score
        players: a list of player symbols (e.g., ["x", "o"])
        start_player: the player who started the game (1 or -1)
        maximizing_player: the player for whom we are optimizing (1 or -1)
    Returns:
        A message describing the estimated result of the game.
    """
    if maximizing_player is None:
        maximizing_player = start_player
    if score > 0:
        return f"Player {players[[maximizing_player == 1, maximizing_player == -1][start_player == maximizing_player]]} wins!"
    elif score < 0:
        return f"Player {players[[maximizing_player == -1, maximizing_player == 1][start_player == maximizing_player]]} wins!"
    elif score == 0:
        return "It's a draw!"

def solve_game(state, rotate_direction = "clockwise", transfer_allowed = True, player = 1, maximizing_player = None, depth = 0, alpha = -math.inf, beta = math.inf):
    """
    Recursively solves the game from the given state, returning a dictionary with:
        - "score": the minimax evaluation score,
        - "states_sequence": a list of game states representing the evolution from the current state to a terminal state,
        - "moves_sequence": a list of moves (dictionaries) that lead from one state to the next.
    
    Parameters:
        state: the current game state (a numpy array)
        rotate_direction: the direction in which the board is rotated (clockwise, counterclockwise, or still)
        transfer_allowed: whether transfers are allowed between adjacent cells
        player: the player whose turn it is (1 or -1)
        maximizing_player: the player for whom we are optimizing (defaults to the initial player)
        depth: current recursion depth (used to favor faster wins/longer losses)
        alpha: best value found so far for the maximizer
        beta: best value found so far for the minimizer
      
    Returns:
        A dictionary with keys "score", "states_sequence", and "moves_sequence".
    """
    if maximizing_player is None:
        maximizing_player = player

    # Check if the current state is terminal.
    result = evaluate_game_state(state)
    if result is not None:
        if result == 0:
            return {"score": 0, "states_sequence": [state], "moves_sequence": []}
        elif result == maximizing_player:
            return {"score": 1000 - depth, "states_sequence": [state], "moves_sequence": []}
        else:
            return {"score": -1000 + depth, "states_sequence": [state], "moves_sequence": []}

    moves = get_possible_moves(state, rotate_direction, transfer_allowed, player)
    if not moves:
        return {"score": 0, "states_sequence": [state], "moves_sequence": []}

    # For the maximizing player, choose the move with the highest score.
    if player == maximizing_player:
        best_eval = -math.inf
        best_state_seq = None
        best_moves_seq = None
        for move in moves:
            new_state = play_turn(state, move)
            child = solve_game(new_state, rotate_direction, transfer_allowed, -player, maximizing_player, depth + 1, alpha, beta)
            child_eval = child["score"]
            if child_eval > best_eval:
                best_eval = child_eval
                best_state_seq = [state] + child["states_sequence"]
                best_moves_seq = [move] + child["moves_sequence"]
            alpha = max(alpha, child_eval)
            if beta <= alpha:
                break  # beta cutoff
        return {"score": best_eval, "states_sequence": best_state_seq, "moves_sequence": best_moves_seq}

    # For the minimizing player, choose the move with the lowest score.
    else:
        best_eval = math.inf
        best_state_seq = None
        best_moves_seq = None
        for move in moves:
            new_state = play_turn(state, move)
            child = solve_game(new_state, rotate_direction, transfer_allowed, -player, maximizing_player, depth + 1, alpha, beta)
            child_eval = child["score"]
            if child_eval < best_eval:
                best_eval = child_eval
                best_state_seq = [state] + child["states_sequence"]
                best_moves_seq = [move] + child["moves_sequence"]
            beta = min(beta, child_eval)
            if beta <= alpha:
                break  # alpha cutoff
        return {"score": best_eval, "states_sequence": best_state_seq, "moves_sequence": best_moves_seq}


if __name__ == "__main__":
    print("Welcome to the game solver!")
    initial_state = np.array([[0, 0, 0], \
                                [0, 0, 0], \
                                [0, 0, 0]])
    # initial_state = np.array([[1, -1, 0, 1], \
    #                             [0, 0, -1, 0], \
    #                             [0, 0, 0, 1], \
    #                             [-1, 0, 0, 0]])
    initial_state = np.zeros((3, 3))
    board_size = initial_state.shape[0]
    print_game_statistics(board_size)
    players = ["x", "o"]
    start_player = -1
    rotate_direction = "0"
    transfer_allowed = True
    
    print("\n\nInitial game state:\n")
    print_game_state(initial_state)
    print(f"\n\nCurrent player: {start_player}")

    # Solve the game from the current state.
    start_time = time.time()
    solution = solve_game(initial_state, rotate_direction, transfer_allowed, start_player)
    print(f"\n\nTime taken: {time.time() - start_time} sec")
    score = solution["score"]
    perfect_states_seq = solution["states_sequence"]
    perfect_moves_seq = solution["moves_sequence"]
    
    # Print the final evaluation.
    result_message = estimated_game_result(score, players, start_player)
    print("\n" + result_message)
    
    # Print the perfect game evolution.
    print("\nPerfect game evolution (states with perfect play):\n")
    player = start_player
    for move_number, s in enumerate(perfect_states_seq):
        print(f"State after move {move_number}:")
        print_game_state(s, players)
        if move_number < len(perfect_moves_seq):
            print(f"\n\nMove {move_number + 1} ({players[[1, 0][player == 1]]}) : {perfect_moves_seq[move_number]}")
            player *= -1
    print("\n")

    # # Find the best move and its evaluation score.
    # start_time = time.time()
    # best_move, score = find_best_move(initial_state, rotate_direction, transfer_allowed, start_player)
    # print("\n\n\nBest move found:")
    # print(best_move)
    # print(f"\nMinimax evaluation score: {score}")
    # print(f"\n\nTime taken: {time.time() - start_time} sec")

    # # Apply the best move to get the new state.
    # new_state = play_turn(initial_state, best_move)
    # print("\n\nGame state after best move:\n")
    # print_game_state(new_state)

    # # Print the estimated result of the game.
    # result_message = estimated_game_result(score, players, start_player)
    # print("\n" + result_message)

    # # Simulate and print a possible evolution of the game.
    # print("\nPossible evolution of the game (principal variation):")
    # evolution = simulate_principal_variation(initial_state, rotate_direction, start_player)
    # for move_number, evolution_moment in enumerate(evolution):
    #     print(f"\n\n\nMove {move_number + 1}: \t {evolution_moment['move']}\n")
    #     print_game_state(evolution_moment["state"])
    # print("\n")
