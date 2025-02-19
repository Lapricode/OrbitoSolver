import math
import numpy as np
from orbital_logic_game_functions import evaluate_game_state, get_possible_moves, play_turn, print_game_statistics


def minimax(state, player, maximizing_player, depth, alpha, beta):
    """
    Recursively evaluates the game tree using minimax with alpha-beta pruning.
    Parameters:
        state: the current game state (a numpy array)
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
    moves = get_possible_moves(state, player, rotate_direction="clockwise")
    if not moves:
        return 0  # No moves available means a draw (should not normally happen)

    if player == maximizing_player:
        max_eval = -math.inf
        for move in moves:
            new_state = play_turn(state, move)
            eval = minimax(new_state, -player, maximizing_player, depth + 1, alpha, beta)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # beta cut-off
        return max_eval
    else:
        min_eval = math.inf
        for move in moves:
            new_state = play_turn(state, move)
            eval = minimax(new_state, -player, maximizing_player, depth + 1, alpha, beta)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # alpha cut-off
        return min_eval

def find_best_move(state, player):
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
    moves = get_possible_moves(state, player, rotate_direction="clockwise")

    for move in moves:
        new_state = play_turn(state, move)
        move_value = minimax(new_state, -player, player, depth = 1, alpha = -math.inf, beta = math.inf)
        if move_value > best_value:
            best_value = move_value
            best_move = move
    return best_move, best_value


if __name__ == "__main__":
    print("Welcome to the game solver!")
    # Set up an empty board for a 3x3 game (change board_size as desired)
    board_size = 3
    state = np.zeros((board_size, board_size))
    current_player = 1  # Let's say player 1 is to move
    best_move, score = find_best_move(state, current_player)
    print("Best move found:")
    print(best_move)
    print("Minimax evaluation score:", score)
