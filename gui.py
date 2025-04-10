import sys
import pygame
import numpy as np
import time
from orbital_logic_game_functions import evaluate_game_state, play_turn
from solve_game import find_best_move

pygame.init()

# ---------------------
# UI Element Classes
# ---------------------

class InputBox:
    def __init__(self, x, y, w, h, font_size, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.color_inactive = pygame.Color('lightskyblue3')
        self.color_active = pygame.Color('dodgerblue2')
        self.color = self.color_inactive
        self.text = text
        self.font_size = font_size
        self.font = pygame.font.Font(None, self.font_size)
        self.txt_surface = self.font.render(text, True, pygame.Color('black'))
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True
            else:
                self.active = False
            self.color = self.color_active if self.active else self.color_inactive
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
                self.color = self.color_inactive
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                if event.unicode.isdigit():
                    self.text += event.unicode
            self.txt_surface = self.font.render(self.text, True, pygame.Color('black'))

    def draw(self, screen):
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        pygame.draw.rect(screen, self.color, self.rect, 2)

class Button:
    def __init__(self, x, y, w, h, font_size, text, callback=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.normal_color = pygame.Color('gray')
        self.hover_color = pygame.Color('lightgray')
        self.pressed_color = pygame.Color('darkgray')
        self.text = text
        self.font_size = font_size
        self.font = pygame.font.Font(None, self.font_size)
        self.txt_surface = self.font.render(self.text, True, pygame.Color('black'))
        self.callback = callback
        self.is_pressed = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.is_pressed = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.is_pressed and self.rect.collidepoint(event.pos):
                if self.callback:
                    self.callback()
            self.is_pressed = False

    def draw(self, screen):
        mouse_pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(mouse_pos):
            if self.is_pressed:
                color = self.pressed_color
            else:
                color = self.hover_color
        else:
            color = self.normal_color
        pygame.draw.rect(screen, color, self.rect)
        text_rect = self.txt_surface.get_rect(center=self.rect.center)
        screen.blit(self.txt_surface, text_rect)

class Checkbox:
    def __init__(self, x, y, size, font_size, text, checked=False):
        self.rect = pygame.Rect(x, y, size, size)
        self.checked = checked
        self.text = text
        self.font_size = font_size
        self.font = pygame.font.Font(None, self.font_size)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.checked = not self.checked

    def draw(self, screen):
        pygame.draw.rect(screen, pygame.Color('white'), self.rect)
        pygame.draw.rect(screen, pygame.Color('black'), self.rect, 2)
        if self.checked:
            pygame.draw.line(screen, pygame.Color('black'), self.rect.topleft, self.rect.bottomright, 2)
            pygame.draw.line(screen, pygame.Color('black'), self.rect.topright, self.rect.bottomleft, 2)
        txt_surface = self.font.render(self.text, True, pygame.Color('black'))
        screen.blit(txt_surface, (self.rect.right + 5, self.rect.y))

class RadioButton:
    def __init__(self, x, y, radius, font_size, text, selected=False):
        self.x = x
        self.y = y
        self.radius = radius
        self.text = text
        self.font_size = font_size
        self.selected = selected
        self.font = pygame.font.Font(None, self.font_size)
        self.circle_rect = pygame.Rect(x - radius, y - radius, 2*radius, 2*radius)

    def handle_event(self, event, group):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.circle_rect.collidepoint(event.pos):
                for btn in group:
                    btn.selected = False
                self.selected = True

    def draw(self, screen):
        pygame.draw.circle(screen, pygame.Color('black'), (self.x, self.y), self.radius, 2)
        if self.selected:
            pygame.draw.circle(screen, pygame.Color('black'), (self.x, self.y), self.radius - 4)
        txt_surface = self.font.render(self.text, True, pygame.Color('black'))
        screen.blit(txt_surface, (self.x + self.radius + 10, self.y - self.radius))

# ---------------------
# Drawing functions
# ---------------------

def draw_board(screen, board_state, board_rect, hover_cell, selections, font):
    board_color = (150, 75, 0)  # brown background
    grid_color = (0, 0, 0)
    pygame.draw.rect(screen, board_color, board_rect)
    rows, cols = board_state.shape
    cell_width = board_rect.width // cols
    cell_height = board_rect.height // rows

    # Draw grid
    for i in range(rows):
        for j in range(cols):
            cell_rect = pygame.Rect(board_rect.x + j*cell_width, board_rect.y + i*cell_height, cell_width, cell_height)
            pygame.draw.rect(screen, grid_color, cell_rect, 1)

    # Hover highlight (blue)
    if hover_cell is not None:
        hi, hj = hover_cell
        cell_rect = pygame.Rect(board_rect.x + hj*cell_width, board_rect.y + hi*cell_height, cell_width, cell_height)
        pygame.draw.rect(screen, (0, 0, 255), cell_rect, 3)

    # Highlight selections
    if selections.get("transfer_source") is not None:
        i, j = selections["transfer_source"]
        cell_rect = pygame.Rect(board_rect.x + j*cell_width, board_rect.y + i*cell_height, cell_width, cell_height)
        pygame.draw.rect(screen, (255, 0, 0), cell_rect, 3)
    if selections.get("transfer_target") is not None:
        i, j = selections["transfer_target"]
        cell_rect = pygame.Rect(board_rect.x + j*cell_width, board_rect.y + i*cell_height, cell_width, cell_height)
        pygame.draw.rect(screen, (255, 0, 0), cell_rect, 3)
    if selections.get("add") is not None:
        i, j = selections["add"]
        cell_rect = pygame.Rect(board_rect.x + j*cell_width, board_rect.y + i*cell_height, cell_width, cell_height)
        pygame.draw.rect(screen, (0, 255, 0), cell_rect, 3)

    # Draw pieces.
    for i in range(rows):
        for j in range(cols):
            center = (board_rect.x + j*cell_width + cell_width//2, board_rect.y + i*cell_height + cell_height//2)
            radius = min(cell_width, cell_height) // 3
            if board_state[i, j] == 1:
                pygame.draw.circle(screen, (0, 0, 0), center, radius)
            elif board_state[i, j] == -1:
                pygame.draw.circle(screen, (255, 255, 255), center, radius)

# ---------------------
# Main Program
# ---------------------

def main():
    # Window dimensions.
    BOARD_WIDTH = 600
    MENU_WIDTH = 300
    BOARD_HEIGHT = BOARD_WIDTH
    WINDOW_WIDTH, WINDOW_HEIGHT = BOARD_WIDTH + MENU_WIDTH, BOARD_WIDTH
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Orbital Logic Game")
    clock = pygame.time.Clock()

    # ---------------- State Variables ----------------
    mode_state = "menu"  # "menu" or "playing"
    game_over = False
    game_message = ""

    # Game settings (set in menu).
    grid_size = 3
    transfer_allowed = True
    rotation_allowed = True
    move_completion_mode = "auto"  # "auto" or "manual"
    game_mode_choice = "2 Players"   # or "Vs Computer"
    rotate_direction = "clockwise" if rotation_allowed else "still"

    # Game variables (set when game starts).
    board_state = None
    current_player = 1  # 1 = Black, -1 = White
    current_move = {"transfer": {"source": None, "target": None}, "add": None}
    move_phase = "none"  # "none", "transfer_target", "add_move"
    selections = {"transfer_source": None, "transfer_target": None, "add": None}

    # Confirmation dialog state.
    confirm_box_active = False
    confirm_action = None  # "restart" or "new_game"
    # Define a confirmation dialog rectangle.
    confirm_box_rect = pygame.Rect(WINDOW_WIDTH//2 - 150, WINDOW_HEIGHT//2 - 100, 300, 200)
    
    # ---------------- Menu Layout for Settings ----------------
    menu_area = pygame.Rect(BOARD_WIDTH, 0, MENU_WIDTH, WINDOW_HEIGHT)
    num_items = 11
    margin_top = 20
    margin_bottom = 20
    available_height = menu_area.height - margin_top - margin_bottom
    spacing = available_height / (num_items)

    title_y           = margin_top
    grid_label_y      = title_y + spacing
    input_y           = grid_label_y
    transfer_cb_y     = grid_label_y + spacing
    rotation_cb_y     = transfer_cb_y + spacing
    game_mode_label_y = rotation_cb_y + spacing
    rb_game1_y        = game_mode_label_y + spacing
    rb_game2_y        = rb_game1_y + spacing
    move_mode_label_y = rb_game2_y + spacing
    rb_move1_y        = move_mode_label_y + spacing
    rb_move2_y        = rb_move1_y + spacing
    start_button_y    = rb_move2_y + spacing

    menu_start_x = BOARD_WIDTH + 20

    menu_title_font   = pygame.font.Font(None, 44)
    menu_options_font = pygame.font.Font(None, 30)
    game_mode_font    = pygame.font.Font(None, 40)
    InputBox_font    = 28
    Button_font      = 28
    Checkbox_font    = 28
    RadioButton_font = 28
    
    grid_size_box = InputBox(menu_start_x + 130, input_y, 60, 30, InputBox_font, text="4")
    transfer_checkbox = Checkbox(menu_start_x, transfer_cb_y, 20, Checkbox_font, "Allow Transfer Moves", checked=True)
    rotation_checkbox = Checkbox(menu_start_x, rotation_cb_y, 20, Checkbox_font, "Allow Rotation", checked=True)
    rb_game1 = RadioButton(menu_start_x, rb_game1_y, 10, RadioButton_font, "2 Players", selected=True)
    rb_game2 = RadioButton(menu_start_x, rb_game2_y, 10, RadioButton_font, "Vs Computer", selected=False)
    rb_move1 = RadioButton(menu_start_x, rb_move1_y, 10, RadioButton_font, "Auto Move Completion", selected=True)
    rb_move2 = RadioButton(menu_start_x, rb_move2_y, 10, RadioButton_font, "Manual Move Completion", selected=False)

    def start_game():
        nonlocal mode_state, board_state, current_player, game_over, game_message
        nonlocal grid_size, transfer_allowed, rotation_allowed, move_completion_mode, game_mode_choice, rotate_direction
        try:
            grid_size = int(grid_size_box.text)
            if grid_size < 2:
                grid_size = 2
        except:
            grid_size = 3
        board_state = np.zeros((grid_size, grid_size), dtype=int)
        current_player = 1
        game_over = False
        game_message = ""
        transfer_allowed = transfer_checkbox.checked
        rotation_allowed = rotation_checkbox.checked
        rotate_direction = "clockwise" if rotation_allowed else "still"
        game_mode_choice = "Vs Computer" if rb_game2.selected else "2 Players"
        move_completion_mode = "manual" if rb_move2.selected else "auto"
        reset_turn()
        mode_state = "playing"
    start_button = Button(menu_start_x, start_button_y, 260, 60, Button_font, "Start Game", start_game)

    # ---------------- In-Game UI Elements (Fixed Positions) ----------------
    # These four buttons are always shown in game mode.
    game_menu_x = BOARD_WIDTH + 20
    game_mode_buttons_length = 260
    game_mode_buttons_height = 60
    game_mode_buttons_spacing = 80
    restart_button_y = 150
    new_game_button_y = restart_button_y + game_mode_buttons_spacing
    take_back_button_y = new_game_button_y + game_mode_buttons_spacing
    complete_move_button_y = take_back_button_y + game_mode_buttons_spacing
    restart_button = Button(game_menu_x, restart_button_y, game_mode_buttons_length, game_mode_buttons_height, Button_font, "Restart")
    new_game_button = Button(game_menu_x, new_game_button_y, game_mode_buttons_length, game_mode_buttons_height, Button_font, "New Game")
    take_back_button = Button(game_menu_x, take_back_button_y, game_mode_buttons_length, game_mode_buttons_height, Button_font, "Take Back")
    complete_move_button = Button(game_menu_x, complete_move_button_y, game_mode_buttons_length, game_mode_buttons_height, Button_font, "Complete Move")

    # Set up confirmation callbacks for Restart and New Game.
    def on_restart_button():
        nonlocal confirm_box_active, confirm_action
        confirm_box_active = True
        confirm_action = "restart"

    def on_new_game_button():
        nonlocal confirm_box_active, confirm_action
        confirm_box_active = True
        confirm_action = "new_game"

    restart_button.callback = on_restart_button
    new_game_button.callback = on_new_game_button

    def take_back():
        nonlocal move_phase, current_move, selections
        if current_move["add"] is not None:
            current_move["add"] = None
            selections["add"] = None
            if transfer_allowed and current_move["transfer"]["target"] is not None:
                move_phase = "transfer_target"
            else:
                move_phase = "none"
        elif current_move["transfer"]["target"] is not None:
            current_move["transfer"]["target"] = None
            selections["transfer_target"] = None
            move_phase = "transfer_target"
        elif current_move["transfer"]["source"] is not None:
            current_move["transfer"]["source"] = None
            selections["transfer_source"] = None
            move_phase = "none"
    take_back_button.callback = take_back

    def complete_move():
        nonlocal board_state, current_player, current_move, move_phase, game_over, game_message
        if move_phase == "add_move" and current_move["add"] is not None:
            move = {"player": current_player, "transfer": None, "add": current_move["add"], "rotate": rotate_direction}
            if transfer_allowed and current_move["transfer"]["source"] is not None and current_move["transfer"]["target"] is not None:
                src = current_move["transfer"]["source"]
                tgt = current_move["transfer"]["target"]
                dr = tgt[0] - src[0]
                dc = tgt[1] - src[1]
                if dr == -1 and dc == 0:
                    direction = "u"
                elif dr == 1 and dc == 0:
                    direction = "d"
                elif dc == -1 and dr == 0:
                    direction = "l"
                elif dc == 1 and dr == 0:
                    direction = "r"
                else:
                    direction = None
                move["transfer"] = [src, direction]
            new_state = play_turn(board_state, move)
            if np.array_equal(new_state, board_state):
                print("Invalid move! Try again.")
                return
            else:
                board_state = new_state
                result = evaluate_game_state(board_state)
                if result is None and not np.any(board_state == 0):
                    result = 0
                if result is not None:
                    game_over = True
                    if result == 0:
                        game_message = "It's a draw!"
                    elif result == 1:
                        game_message = "Black wins!"
                    elif result == -1:
                        game_message = "White wins!"
                else:
                    current_player *= -1
                reset_turn()
    complete_move_button.callback = complete_move

    def restart_game():
        nonlocal board_state, current_player, game_over, game_message
        board_state = np.zeros(board_state.shape, dtype=int)
        current_player = 1
        game_over = False
        game_message = ""
        reset_turn()

    def switch_to_menu():
        nonlocal mode_state
        mode_state = "menu"

    def reset_turn():
        nonlocal current_move, move_phase, selections
        current_move = {"transfer": {"source": None, "target": None}, "add": None}
        move_phase = "none"
        selections = {"transfer_source": None, "transfer_target": None, "add": None}

    # Confirmation dialog callbacks.
    def confirm_yes():
        nonlocal confirm_box_active, confirm_action
        if confirm_action == "restart":
            restart_game()
        elif confirm_action == "new_game":
            switch_to_menu()
        confirm_box_active = False
        confirm_action = None

    def confirm_no():
        nonlocal confirm_box_active, confirm_action
        confirm_box_active = False
        confirm_action = None

    def handle_resize(new_width, new_height):
        pass

    # Create confirmation dialog buttons.
    confirm_yes_button = Button(confirm_box_rect.x + 30, confirm_box_rect.y + 140, 100, 40, Button_font, "Yes", confirm_yes)
    confirm_no_button = Button(confirm_box_rect.x + 170, confirm_box_rect.y + 140, 100, 40, Button_font, "No", confirm_no)

    # ---------------- Main Loop ----------------
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        hover_cell = None
        board_rect = pygame.Rect(0, 0, BOARD_WIDTH, BOARD_HEIGHT)
        if board_state is not None and board_rect.collidepoint(mouse_pos):
            rows, cols = board_state.shape
            cell_w = board_rect.width // cols
            cell_h = board_rect.height // rows
            col = (mouse_pos[0] - board_rect.x) // cell_w
            row = (mouse_pos[1] - board_rect.y) // cell_h
            hover_cell = (row, col)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                handle_resize(event.w, event.h)

            # If confirmation dialog is active, only process its events.
            if confirm_box_active:
                confirm_yes_button.handle_event(event)
                confirm_no_button.handle_event(event)
                continue

            if mode_state == "menu":
                grid_size_box.handle_event(event)
                transfer_checkbox.handle_event(event)
                rotation_checkbox.handle_event(event)
                rb_game1.handle_event(event, [rb_game1, rb_game2])
                rb_game2.handle_event(event, [rb_game1, rb_game2])
                rb_move1.handle_event(event, [rb_move1, rb_move2])
                rb_move2.handle_event(event, [rb_move1, rb_move2])
                start_button.handle_event(event)
            elif mode_state == "playing":
                # Only process board moves if the game is not over.
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if not game_over and board_rect.collidepoint(event.pos):
                        cell_w = board_rect.width // board_state.shape[1]
                        cell_h = board_rect.height // board_state.shape[0]
                        col = (event.pos[0] - board_rect.x) // cell_w
                        row = (event.pos[1] - board_rect.y) // cell_h
                        if (game_mode_choice == "2 Players") or (game_mode_choice == "Vs Computer" and current_player == 1):
                            if transfer_allowed and move_phase == "none" and board_state[row, col] == -current_player:
                                current_move["transfer"]["source"] = (row, col)
                                selections["transfer_source"] = (row, col)
                                move_phase = "transfer_target"
                            elif move_phase == "transfer_target":
                                src = current_move["transfer"]["source"]
                                if (abs(row - src[0]) == 1 and col == src[1]) or (abs(col - src[1]) == 1 and row == src[0]):
                                    if board_state[row, col] == 0:
                                        current_move["transfer"]["target"] = (row, col)
                                        selections["transfer_target"] = (row, col)
                                        move_phase = "add_move"
                            elif move_phase in ["none", "add_move"]:
                                # Allow add move even if clicking on the transfer source cell.
                                if board_state[row, col] == 0 or (transfer_allowed and current_move["transfer"]["source"] is not None and (row, col) == current_move["transfer"]["source"]):
                                    current_move["add"] = (row, col)
                                    selections["add"] = (row, col)
                                    move_phase = "add_move"
                                    if move_completion_mode == "auto":
                                        complete_move()
                # Always handle the in-game buttons.
                take_back_button.handle_event(event)
                complete_move_button.handle_event(event)
                restart_button.handle_event(event)
                new_game_button.handle_event(event)

        if mode_state == "playing" and not game_over and game_mode_choice == "Vs Computer" and current_player == -1:
            best_move, score = find_best_move(board_state, rotate_direction, transfer_allowed, current_player)
            if best_move is not None:
                board_state = play_turn(board_state, best_move)
                result = evaluate_game_state(board_state)
                if result is None and not np.any(board_state == 0):
                    result = 0
                if result is not None:
                    game_over = True
                    if result == 0:
                        game_message = "It's a draw!"
                    elif result == 1:
                        game_message = "Black wins!"
                    elif result == -1:
                        game_message = "White wins!"
                else:
                    current_player *= -1
                reset_turn()
                pygame.time.delay(300)

        # --------------------- Drawing ---------------------
        screen.fill((200, 200, 200))
        if mode_state == "menu":
            pygame.draw.rect(screen, (220, 220, 220), (BOARD_WIDTH, 0, MENU_WIDTH, WINDOW_HEIGHT))
            title_surface = menu_title_font.render("Game Settings", True, pygame.Color('black'))
            screen.blit(title_surface, (menu_start_x, title_y))
            grid_label = menu_options_font.render("Grid Size:", True, pygame.Color('black'))
            screen.blit(grid_label, (menu_start_x, grid_label_y))
            grid_size_box.draw(screen)
            transfer_checkbox.draw(screen)
            rotation_checkbox.draw(screen)
            game_mode_label = menu_options_font.render("Game Mode:", True, pygame.Color('black'))
            screen.blit(game_mode_label, (menu_start_x, game_mode_label_y))
            rb_game1.draw(screen)
            rb_game2.draw(screen)
            move_mode_label = menu_options_font.render("Move Completion:", True, pygame.Color('black'))
            screen.blit(move_mode_label, (menu_start_x, move_mode_label_y))
            rb_move1.draw(screen)
            rb_move2.draw(screen)
            start_button.draw(screen)
        elif mode_state == "playing":
            draw_board(screen, board_state, board_rect, hover_cell, selections, menu_options_font)
            pygame.draw.rect(screen, (220, 220, 220), (BOARD_WIDTH, 0, MENU_WIDTH, WINDOW_HEIGHT))
            if game_over:
                info_text = game_message
                info_surface = game_mode_font.render(info_text, True, pygame.Color('black'))
                screen.blit(info_surface, (menu_start_x, BOARD_HEIGHT - 40))
            else:
                info_text = "Turn: Black" if current_player == 1 else "Turn: White"
                info_surface = game_mode_font.render(info_text, True, pygame.Color('black'))
                screen.blit(info_surface, (menu_start_x, 20))
            # Always show Restart, New Game, Take Back, and Complete Move buttons.
            restart_button.draw(screen)
            new_game_button.draw(screen)
            take_back_button.draw(screen)
            complete_move_button.draw(screen)
        
        # Draw confirmation dialog if active.
        if confirm_box_active:
            pygame.draw.rect(screen, (180, 180, 180), confirm_box_rect)
            pygame.draw.rect(screen, (0, 0, 0), confirm_box_rect, 2)
            confirm_text = ""
            if confirm_action == "restart":
                confirm_text = "Restart game?"
            elif confirm_action == "new_game":
                confirm_text = "Return to main menu?"
            confirm_surface = menu_options_font.render(confirm_text, True, pygame.Color('black'))
            screen.blit(confirm_surface, (confirm_box_rect.x + 20, confirm_box_rect.y + 40))
            confirm_yes_button.draw(screen)
            confirm_no_button.draw(screen)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
