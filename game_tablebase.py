import os
import numpy as np


def get_file_path(base_dir, state, rotation, transfer_allowed, player_turn):
    """
    Build the file path according to:
      1) Grid size (e.g., "2x2", "3x3", etc.)
      2) Rotation mode folder (e.g., "clockwise", "counterclockwise", "still")
      3) Transfer rule (e.g., "transfer_allowed" or "transfer_not_allowed")
      4) Player's turn ("player1" or "player2")
      5) Completion level of the grid (number of non-zero cells, e.g., "completion_3.txt")
    """
    # 1. Grid size folder:
    grid_size = state.shape[0]  # assuming square grid
    grid_folder = f"{grid_size}x{grid_size}"
    
    # 2. Rotation folder:
    rotation_folder = rotation.lower()
    
    # 3. Transfer allowed folder:
    transfer_folder = "transfer_allowed" if transfer_allowed else "transfer_not_allowed"
    
    # 4. Player turn folder:
    player_folder = "player1" if player_turn == 1 else "player2"
    
    # 5. Completion: count how many cells are filled (non-zero)
    pieces_count = np.count_nonzero(state)
    # Here we assume that the "completion" is determined solely by the number of filled cells.
    completion_file = f"completion_{pieces_count}.txt"
    
    # Build the folder tree:
    folder_path = os.path.join(base_dir, grid_folder, rotation_folder, transfer_folder, player_folder)
    # Ensure the directory exists:
    os.makedirs(folder_path, exist_ok=True)
    
    # Final file path:
    file_path = os.path.join(folder_path, completion_file)
    return file_path

def save_game_record(base_dir, state, best_move, game_result, rotation, transfer_allowed, player_turn):
    """
    Saves one line of data for a given game state, its best move, and its perfect play result.
    Each line in the file will have the state, move, and result separated by a delimiter.
    """
    file_path = get_file_path(base_dir, state, rotation, transfer_allowed, player_turn)
    
    # Convert the state to a string; here we use a flattened, comma-separated representation.
    state_str = np.array2string(state, separator=',')
    # Convert best move and game result to strings (customize formatting as needed)
    best_move_str = str(best_move)
    game_result_str = str(game_result)
    
    # Create the record line; you can choose a delimiter that suits your needs.
    record_line = f"{state_str} | {best_move_str} | {game_result_str}\n"
    
    # Append the record to the file.
    with open(file_path, "a") as f:
        f.write(record_line)
    
    print(f"Record saved to {file_path}")

# Example usage:
if __name__ == "__main__":
    # Define a base directory for the tablebase
    base_dir = "game_tablebase"
    
    # Example game state (2x2 board)
    state = np.array([[1, 0],
                      [0, -1]])
    
    # Example best move (could be any representation; here a dictionary)
    best_move = {"player": 1, "move": (0, 1)}
    
    # Example game result (e.g., from perfect play analysis)
    game_result = "Player X wins"
    
    # Parameters for the folder hierarchy:
    rotation = "clockwise"       # Options might be "clockwise", "counterclockwise", "still"
    transfer_allowed = True      # or False
    player_turn = 1              # 1 for player1 or -1 for player2
    
    # Save the game record:
    save_game_record(base_dir, state, best_move, game_result, rotation, transfer_allowed, player_turn)
