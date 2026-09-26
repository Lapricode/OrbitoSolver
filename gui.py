import sys
import time

import numpy as np
import pygame

import computer_engine as engine
import orbital_logic_game_functions as olgf


# ---------------------
# Layout and colours
# ---------------------

BOARD_SIZE = 600
MENU_WIDTH = 380
STATUS_HEIGHT = 140
WINDOW_WIDTH = BOARD_SIZE + MENU_WIDTH
WINDOW_HEIGHT = BOARD_SIZE + STATUS_HEIGHT

PANEL_COLOR = (226, 226, 230)
INPUT_BACKGROUND = (250, 250, 250)
BOARD_LIGHT = (176, 122, 70)
BOARD_DARK = (152, 101, 55)
GRID_COLOR = (45, 26, 10)
HOVER_COLOR = (0, 90, 255)
SOURCE_COLOR = (220, 30, 30)
TARGET_COLOR = (255, 140, 0)
ADD_COLOR = (0, 190, 0)
HINT_COLOR = (215, 0, 215)
WIN_COLOR = (0, 150, 255)

PLAYER_FILL = {1: (15, 15, 15), 2: (250, 250, 250)}
PLAYER_OUTLINE = {1: (250, 250, 250), 2: (15, 15, 15)}
PLAYER_NAMES = {0: "Empty", 1: "Black", 2: "White"}
PLAYER_SYMBOLS = {1: "x", 2: "o"}

MIN_GRID_SIZE = 1
MAX_GRID_SIZE = 8
MIN_THINK_TIME = 0.35

FILL = "fill"  # panel row width meaning "use the whole panel width"

PIECE_VALUES = {"Empty": 0, "Black": 1, "White": 2}

ENGINE_OPTIONS = {
    # radio button label: (look in the tablebase first, what to do otherwise)
    "Tablebase, else random move": (True, engine.FALLBACK_RANDOM),
    "Tablebase, else search": (True, engine.FALLBACK_SEARCH),
    "Random move only": (False, engine.FALLBACK_RANDOM),
    "Search only": (False, engine.FALLBACK_SEARCH),
}

EDITOR_HELP = (
    "Left click places the selected piece, right click erases it and middle "
    "click places the other player. The rules below are the ones the engine "
    "uses, and they are shared with the game settings."
)


def dimmed_surface(surface, factor=0.6):
    """Return a darker copy of a text surface (works on pygame and pygame-ce)."""
    try:
        return pygame.transform.multiply_alpha(surface, factor)
    except AttributeError:
        level = int(255 * factor)
        dimmed = surface.copy()
        dimmed.fill((level, level, level, 255), special_flags=pygame.BLEND_RGBA_MULT)
        return dimmed


# ---------------------
# UI Element Classes
# ---------------------

class Label:
    """A piece of static text."""

    def __init__(self, text, font, color=(20, 20, 20)):
        self.font = font
        self.color = color
        self.text = text
        self.surface = self.font.render(text, True, color)
        self.rect = pygame.Rect(0, 0, *self.surface.get_size())

    def set_text(self, text):
        if text != self.text:
            self.text = text
            self.surface = self.font.render(text, True, self.color)
            self.rect.size = self.surface.get_size()

    def handle_event(self, event):
        return False

    def draw(self, screen):
        screen.blit(self.surface, self.rect)


class InputBox:
    """A single line numeric text field."""

    def __init__(self, w, h, font_size, text='', decimal=False, max_len=5, on_enter=None):
        self.rect = pygame.Rect(0, 0, w, h)
        self.color_inactive = pygame.Color('lightskyblue3')
        self.color_active = pygame.Color('dodgerblue2')
        self.color = self.color_inactive
        self.text = text
        self.decimal = decimal
        self.max_len = max_len
        self.on_enter = on_enter
        self.font = pygame.font.Font(None, font_size)
        self.txt_surface = self.font.render(text, True, pygame.Color('black'))
        self.active = False

    @property
    def number(self):
        """The field content as a float, or None when it is not a number."""
        try:
            return float(self.text)
        except ValueError:
            return None

    def set_text(self, text):
        self.text = text
        self.txt_surface = self.font.render(self.text, True, pygame.Color('black'))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            inside = self.rect.collidepoint(event.pos)
            self.active = inside
            self.color = self.color_active if inside else self.color_inactive
            return inside
        if event.type == pygame.KEYDOWN and self.active:
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.active = False
                self.color = self.color_inactive
                if self.on_enter is not None:
                    self.on_enter()
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_DELETE:
                self.text = ''
            elif len(self.text) < self.max_len and (
                    event.unicode.isdigit() or
                    (self.decimal and event.unicode == '.' and '.' not in self.text)):
                self.text += event.unicode
            else:
                return False
            self.txt_surface = self.font.render(self.text, True, pygame.Color('black'))
            return True
        return False

    def draw(self, screen):
        pygame.draw.rect(screen, INPUT_BACKGROUND, self.rect)
        pygame.draw.rect(screen, self.color, self.rect, 2)
        screen.blit(self.txt_surface, (self.rect.x + 5, self.rect.y + 5))


class Button:
    """A clickable button with an optional callback."""

    def __init__(self, w, h, font_size, text, callback=None, enabled=True):
        self.rect = pygame.Rect(0, 0, w, h)
        self.normal_color = pygame.Color('gray')
        self.hover_color = pygame.Color('lightgray')
        self.pressed_color = pygame.Color('darkgray')
        self.disabled_color = pygame.Color(206, 206, 210)
        self.text = text
        self.font = pygame.font.Font(None, font_size)
        self.txt_surface = self.font.render(self.text, True, pygame.Color('black'))
        self.faded_surface = dimmed_surface(self.txt_surface)
        self.callback = callback
        self.enabled = enabled
        self.is_pressed = False

    def set_text(self, text):
        if text != self.text:
            self.text = text
            self.txt_surface = self.font.render(self.text, True, pygame.Color('black'))
            self.faded_surface = dimmed_surface(self.txt_surface)

    def handle_event(self, event):
        if not self.enabled:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.is_pressed = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.is_pressed:
                pressed_inside = self.rect.collidepoint(event.pos)
                self.is_pressed = False
                if pressed_inside:
                    if self.callback:
                        self.callback()
                    return True
        return False

    def draw(self, screen):
        if not self.enabled:
            color = self.disabled_color
        else:
            mouse_pos = pygame.mouse.get_pos()
            if self.rect.collidepoint(mouse_pos):
                color = self.pressed_color if self.is_pressed else self.hover_color
            else:
                color = self.normal_color
        pygame.draw.rect(screen, color, self.rect)
        text_rect = self.txt_surface.get_rect(center=self.rect.center)
        screen.blit(self.faded_surface if not self.enabled else self.txt_surface, text_rect)


class Checkbox:
    """A labelled on/off box."""

    def __init__(self, size, font_size, text, checked=False):
        self.rect = pygame.Rect(0, 0, size, size)
        self.checked = checked
        self.text = text
        self.font = pygame.font.Font(None, font_size)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.checked = not self.checked
            return True
        return False

    def draw(self, screen):
        pygame.draw.rect(screen, pygame.Color('white'), self.rect)
        pygame.draw.rect(screen, pygame.Color('black'), self.rect, 2)
        if self.checked:
            pygame.draw.line(screen, pygame.Color('black'), self.rect.topleft, self.rect.bottomright, 2)
            pygame.draw.line(screen, pygame.Color('black'), self.rect.topright, self.rect.bottomleft, 2)
        txt_surface = self.font.render(self.text, True, pygame.Color('black'))
        screen.blit(txt_surface, (self.rect.right + 8, self.rect.y))


class RadioButton:
    """One option of a group of mutually exclusive radio buttons.

    Passing a ``group`` list registers the button in it, so the group can be
    built while the buttons are created and a single click always leaves
    exactly one button of the group selected.
    """

    def __init__(self, radius, font_size, text, group=None, selected=False):
        self.rect = pygame.Rect(0, 0, 2 * radius, 2 * radius)
        self.radius = radius
        self.text = text
        self.font = pygame.font.Font(None, font_size)
        self.selected = selected
        self.group = [] if group is None else group
        self.group.append(self)
        if selected:
            for button in self.group:
                button.selected = button is self

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            for button in self.group:
                button.selected = button is self
            return True
        return False

    def draw(self, screen):
        center = self.rect.center
        pygame.draw.circle(screen, pygame.Color('black'), center, self.radius, 2)
        if self.selected:
            pygame.draw.circle(screen, pygame.Color('black'), center, self.radius - 4)
        txt_surface = self.font.render(self.text, True, pygame.Color('black'))
        screen.blit(txt_surface, (self.rect.right + 10, self.rect.centery - txt_surface.get_height() // 2))


def radio_group(font_size, options, default=0):
    """Build a list of radio buttons that know each other.

    ``options`` holds the labels and the one at ``default`` starts selected.
    A click on any of them leaves exactly one selected.
    """
    group = []
    for index, text in enumerate(options):
        RadioButton(9, font_size, text, group, selected=index == default)
    return group


class TextPanel:
    """A word wrapped read-only text area, optionally scrollable."""

    WHEEL_STEP = 3

    def __init__(self, w, h, font, title=None, background=(248, 248, 248),
                 border=(110, 110, 110), padding=8, scrollable=True):
        self.rect = pygame.Rect(0, 0, w, h)
        self.font = font
        self.title = title
        self.title_surface = font.render(title, True, (20, 20, 20)) if title else None
        self.background = background
        self.border = border
        self.padding = padding
        self.scrollable = scrollable
        self.scroll = 0
        self.lines = [""]
        self.line_height = font.get_linesize() + 2
        self._raw = None
        self._wrap_width = None

    @property
    def text(self):
        return self._raw or ""

    @property
    def text_width(self):
        return max(20, self.rect.width - 2 * self.padding)

    def set_text(self, text):
        if text == self._raw:
            return
        self._raw = text
        self._rewrap()

    def clear(self):
        self.set_text("")
        self.scroll = 0

    def _rewrap(self):
        width = self.text_width
        lines = []
        for raw_line in self.text.split("\n"):
            if not raw_line:
                lines.append("")
                continue
            current = ""
            for word in raw_line.split(" "):
                candidate = word if not current else current + " " + word
                if not current or self.font.size(candidate)[0] <= width:
                    current = candidate
                else:
                    lines.append(current)
                    current = word
            lines.append(current)
        self.lines = lines or [""]
        if width != self._wrap_width:
            self.scroll = min(self.scroll, self.max_scroll)
            self._wrap_width = width

    @property
    def header_height(self):
        return self.line_height + 4 if self.title_surface is not None else 0

    @property
    def body_height(self):
        return max(self.line_height, self.rect.height - 2 * self.padding - self.header_height)

    @property
    def visible_lines(self):
        return max(1, self.body_height // self.line_height)

    @property
    def max_scroll(self):
        return max(0, len(self.lines) - self.visible_lines)

    def scroll_to_end(self):
        self.scroll = self.max_scroll

    def handle_event(self, event):
        if (self.scrollable and event.type == pygame.MOUSEWHEEL
                and self.rect.collidepoint(pygame.mouse.get_pos())):
            self.scroll = max(0, min(self.max_scroll, self.scroll - event.y * self.WHEEL_STEP))
            return True
        return False

    def draw(self, screen):
        self._rewrap()
        pygame.draw.rect(screen, self.background, self.rect)
        top = self.rect.y + self.padding
        if self.title_surface is not None:
            screen.blit(self.title_surface, (self.rect.x + self.padding, top))
            top += self.header_height
        previous_clip = screen.get_clip()
        screen.set_clip(pygame.Rect(self.rect.x, top, self.rect.width,
                                    max(0, self.rect.bottom - top - self.padding)))
        for index in range(int(self.scroll), len(self.lines)):
            y = top + (index - self.scroll) * self.line_height
            surface = self.font.render(self.lines[index], True, (25, 25, 25))
            screen.blit(surface, (self.rect.x + self.padding, y))
        screen.set_clip(previous_clip)
        pygame.draw.rect(screen, self.border, self.rect, 2)
        if self.scrollable and self.max_scroll > 0:
            track = pygame.Rect(self.rect.right - 8, top, 5, max(1, self.rect.bottom - top - self.padding))
            pygame.draw.rect(screen, (200, 200, 200), track)
            knob_height = max(16, int(track.height * self.visible_lines / len(self.lines)))
            knob_y = track.y + int((track.height - knob_height) * self.scroll / self.max_scroll)
            pygame.draw.rect(screen, (110, 110, 110),
                             pygame.Rect(track.x, knob_y, track.width, knob_height))


class Panel:
    """A scrollable column of widgets drawn on a coloured background."""

    WHEEL_STEP = 26

    def __init__(self, rect, background=PANEL_COLOR, padding=14, gap=4, spacing=8):
        self.rect = pygame.Rect(rect)
        self.background = background
        self.padding = padding
        self.gap = gap
        self.spacing = spacing
        self.scroll = 0
        self.rows = []

    def add_row(self, height, widgets, fill=False):
        """Add a row of ``(widget, x, width, height)`` items.

        ``x`` may be None to place the widget right after the previous one,
        ``width`` and ``height`` may be None to keep the natural size of the
        widget (centered inside the row), and the width :data:`FILL` uses the
        rest of the panel. A row added with ``fill`` takes the vertical space
        left over by the other rows and stretches its widgets.
        """
        self.rows.append({"height": height, "widgets": list(widgets), "fill": fill})
        return self

    @property
    def content_width(self):
        return max(20, self.rect.width - 2 * self.padding)

    def _row_height(self, row):
        """Height a row needs without being stretched by a fill row."""
        natural = max((widget.rect.height for widget, _x, _w, _h in row["widgets"]), default=0)
        return max(row["height"], natural)

    def _fixed_height(self):
        """Height of every row that is not a fill row, gaps included."""
        rows = [row for row in self.rows if not row["fill"]]
        if not rows:
            return 0
        return sum(self._row_height(row) for row in rows) + self.gap * (len(rows) - 1)

    def _fill_height(self):
        """Height every fill row shares: what the other rows leave over."""
        flexible = [row for row in self.rows if row["fill"]]
        if not flexible:
            return 0
        leftover = self.rect.height - 2 * self.padding - self._fixed_height()
        return max(0, leftover // len(flexible))

    def content_height(self):
        """Total height the rows need when nothing is scrolled out."""
        fill_height = self._fill_height()
        heights = [fill_height if row["fill"] else self._row_height(row) for row in self.rows]
        gaps = self.gap * max(0, len(heights) - 1)
        return max(0, sum(heights) + gaps)

    @property
    def max_scroll(self):
        return max(0, self.content_height() - self.rect.height)

    def layout(self):
        """Give every widget its final on screen rectangle."""
        fill_height = self._fill_height()
        y = self.padding - int(self.scroll)
        for row in self.rows:
            height = fill_height if row["fill"] else self._row_height(row)
            cursor = 0
            for widget, x, width, widget_height in row["widgets"]:
                offset = cursor if x is None else x
                if width == FILL:
                    final_width = max(20, self.content_width - offset)
                elif width is None:
                    final_width = widget.rect.width
                else:
                    final_width = width
                if row["fill"] and not widget_height:
                    final_height = height
                elif widget_height is None:
                    final_height = widget.rect.height
                else:
                    final_height = widget_height
                widget.rect.size = (final_width, final_height)
                widget.rect.topleft = (self.rect.x + self.padding + offset,
                                       y + (height - final_height) // 2)
                cursor = offset + final_width + self.spacing
            y += height + self.gap
        self.scroll = max(0, min(self.max_scroll, self.scroll))


    def handle_event(self, event):
        self.layout()
        handled = False
        for row in self.rows:
            for widget, _x, _w, _h in row["widgets"]:
                if not self.rect.colliderect(widget.rect):
                    continue
                if widget.handle_event(event):
                    handled = True
        if (not handled and event.type == pygame.MOUSEWHEEL
                and self.rect.collidepoint(pygame.mouse.get_pos())):
            self.scroll = max(0, min(self.max_scroll, self.scroll - event.y * self.WHEEL_STEP))
            handled = True
        self.layout()
        return handled

    def draw(self, screen):
        pygame.draw.rect(screen, self.background, self.rect)
        self.layout()
        previous_clip = screen.get_clip()
        screen.set_clip(self.rect)
        for row in self.rows:
            for widget, _x, _w, _h in row["widgets"]:
                widget.draw(screen)
        screen.set_clip(previous_clip)
        if self.max_scroll > 0:
            track = pygame.Rect(self.rect.right - 8, self.rect.y + 4, 5, self.rect.height - 8)
            pygame.draw.rect(screen, (205, 205, 205), track)
            content = max(1, self.content_height())
            knob_height = max(20, int(track.height * self.rect.height / content))
            knob_y = track.y + int((track.height - knob_height) * self.scroll / self.max_scroll)
            pygame.draw.rect(screen, (120, 120, 120),
                             pygame.Rect(track.x, knob_y, track.width, knob_height))


# ---------------------
# Drawing functions
# ---------------------

def find_winning_lines(state):
    """Return every (line, player) pair that fills a row, column or diagonal."""
    n = state.shape[0]
    lines = [tuple((r, c) for c in range(n)) for r in range(n)]
    lines += [tuple((r, c) for r in range(n)) for c in range(n)]
    lines.append(tuple((i, i) for i in range(n)))
    lines.append(tuple((i, n - 1 - i) for i in range(n)))
    winners = []
    for line in lines:
        value = int(state[line[0]])
        if value and all(int(state[cell]) == value for cell in line):
            winners.append((line, value))
    return winners


def cell_at(position, board_rect, board_state):
    """Board cell under a window position, or None when outside the board."""
    if board_state is None or not board_rect.collidepoint(position):
        return None
    rows, cols = board_state.shape
    col = (position[0] - board_rect.x) // max(1, board_rect.width // cols)
    row = (position[1] - board_rect.y) // max(1, board_rect.height // rows)
    if not (0 <= row < rows and 0 <= col < cols):
        return None
    return int(row), int(col)


def transfer_target(move):
    """Board cell an already built transfer move lands on, or None."""
    transfer = move.get("transfer")
    if transfer is None:
        return None
    source, direction = transfer[0], str(transfer[1]).lower()
    row, col = int(source[0]), int(source[1])
    if direction in ("u", "up"):
        row -= 1
    elif direction in ("d", "down"):
        row += 1
    elif direction in ("l", "left"):
        col -= 1
    elif direction in ("r", "right"):
        col += 1
    else:
        return None
    return row, col


def move_highlights(move):
    """Highlight cells used by a move dictionary."""
    if move is None:
        return []
    highlights = []
    transfer = move.get("transfer")
    if transfer is not None:
        highlights.append(((int(transfer[0][0]), int(transfer[0][1])), SOURCE_COLOR, 3))
        target = transfer_target(move)
        if target is not None:
            highlights.append((target, TARGET_COLOR, 3))
    add = move.get("add")
    if add is not None:
        highlights.append(((int(add[0]), int(add[1])), ADD_COLOR, 3))
    return highlights


def selection_highlights(selections):
    """Highlight cells of the move currently being built by hand."""
    highlights = []
    for key, color in (("transfer_source", SOURCE_COLOR), ("transfer_target", TARGET_COLOR),
                       ("add", ADD_COLOR)):
        if selections.get(key) is not None:
            highlights.append((tuple(int(v) for v in selections[key]), color, 3))
    return highlights


def draw_board(screen, board_state, board_rect, hover_cell=None, highlights=(), winning_lines=()):
    """Draw the board with its grid, its pieces and the cell highlights."""
    pygame.draw.rect(screen, BOARD_DARK, board_rect)
    rows, cols = board_state.shape
    cell_width = max(1, board_rect.width // cols)
    cell_height = max(1, board_rect.height // rows)

    def cell_rect(cell):
        row, col = cell
        return pygame.Rect(board_rect.x + col * cell_width,
                           board_rect.y + row * cell_height,
                           cell_width, cell_height)

    for i in range(rows):
        for j in range(cols):
            cell = cell_rect((i, j))
            pygame.draw.rect(screen, BOARD_LIGHT if (i + j) % 2 == 0 else BOARD_DARK, cell)
            pygame.draw.rect(screen, GRID_COLOR, cell, 1)

    for cell, color, width in highlights:
        if cell is not None:
            pygame.draw.rect(screen, color, cell_rect(cell), width)

    for line, _player in winning_lines:
        points = [cell_rect(cell).center for cell in line]
        if len(points) > 1:
            pygame.draw.lines(screen, WIN_COLOR, False, points, max(3, cell_width // 12))

    if hover_cell is not None:
        pygame.draw.rect(screen, HOVER_COLOR, cell_rect(hover_cell), 3)

    for i in range(rows):
        for j in range(cols):
            value = int(board_state[i, j])
            if value == 0:
                continue
            center = cell_rect((i, j)).center
            radius = max(2, min(cell_width, cell_height) // 3)
            pygame.draw.circle(screen, PLAYER_FILL[value], center, radius)
            pygame.draw.circle(screen, PLAYER_OUTLINE[value], center, radius, 2)


# ---------------------
# Main Program
# ---------------------

def main():
    pygame.init()
    board_width = BOARD_SIZE
    menu_width = MENU_WIDTH
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Orbital Logic Game")
    clock = pygame.time.Clock()

    # ---------------- State ----------------
    mode_state = "menu"  # "menu", "playing" or "editor"

    board_rect = pygame.Rect(0, 0, board_width, BOARD_SIZE)
    status_rect = pygame.Rect(0, board_rect.bottom, board_width, STATUS_HEIGHT)
    menu_rect = pygame.Rect(board_width, 0, menu_width, WINDOW_HEIGHT)

    game = {
        "board": np.zeros((3, 3), dtype=int),
        "player": 1,        # player to move (1 = Black, 2 = White)
        "over": False,
        "message": "",
        "move": {"transfer": {"source": None, "target": None}, "add": None},
        "phase": "none",    # "none", "transfer_target" or "add_move"
        "selections": {"transfer_source": None, "transfer_target": None, "add": None},
        "log": [],
        "message_line": "",
        "hint": None,       # move suggested by the hint button
        "busy": False,      # a computer answer is on its way
        "busy_kind": "",    # "computer" or "hint"
        "busy_since": 0.0,
        "pending": None,    # (result, arrival time, kind) waiting out the think time
        "blocked": False,   # the computer cannot answer, wait for the human
        "vs_computer": False,  # the computer plays the other side
    }

    editor = {
        "board": np.zeros((3, 3), dtype=int),
        "player": 1,        # player to move in the drawn position
        "suggestion": None, # move suggested by the engine
        "busy": False,
        "busy_since": 0.0,
    }
    tablebase_sizes = engine.available_grid_sizes()

    # ---------------- Fonts ----------------
    title_font = pygame.font.Font(None, 46)
    heading_font = pygame.font.Font(None, 30)
    body_font = pygame.font.Font(None, 26)
    small_font = pygame.font.Font(None, 22)
    huge_font = pygame.font.Font(None, 42)

    # ---------------- Shared setting widgets ----------------
    grid_size_box = InputBox(62, 32, body_font.get_height(), text="3")
    time_limit_box = InputBox(72, 32, body_font.get_height(), text="2.0", decimal=True, max_len=5)
    transfer_checkbox = Checkbox(20, body_font.get_height(), "Allow Transfer Moves", checked=True)
    rotation_group = radio_group(body_font.get_height(),
                                 ["Still", "Clockwise", "Counterclockwise"], default=1)
    mode_group = radio_group(body_font.get_height(),
                             ["2 Players", "Vs Computer", "Position Editor"])
    mode_computer, mode_editor = mode_group[1], mode_group[2]
    engine_group = radio_group(body_font.get_height(),
                               ["Tablebase, else random move", "Tablebase, else search",
                                "Random move only", "Search only"])
    completion_group = radio_group(body_font.get_height(),
                                   ["Auto Move Completion", "Manual Move Completion"])

    def read_grid_size(default=3):
        try:
            size = int(grid_size_box.text)
        except ValueError:
            return default
        return max(MIN_GRID_SIZE, min(MAX_GRID_SIZE, size))

    def current_rotation():
        chosen = next(radio.text for radio in rotation_group if radio.selected)
        return {"Clockwise": "clockwise",
                "Counterclockwise": "counterclockwise"}.get(chosen, "still")

    def read_time_limit():
        value = time_limit_box.number
        return engine.DEFAULT_TIME_LIMIT if value is None else engine.clamp_time_limit(value)

    def engine_config():
        chosen = next(radio.text for radio in engine_group if radio.selected)
        use_tablebase, fallback = ENGINE_OPTIONS[chosen]
        return {"use_tablebase": use_tablebase, "fallback": fallback}

    def engine_config_text():
        config = engine_config()
        prefix = "tablebase, then " if config["use_tablebase"] else "no tablebase, "
        if config["fallback"] == engine.FALLBACK_SEARCH:
            return f"Engine: {prefix}search for {read_time_limit():g}s"
        return f"Engine: {prefix}random move"

    def vs_computer():
        return game["vs_computer"]

    def resize_board(board, size):
        resized = np.zeros((size, size), dtype=int)
        rows = min(size, board.shape[0])
        cols = min(size, board.shape[1])
        resized[:rows, :cols] = board[:rows, :cols]
        return resized

    def evaluate_position(board):
        """Return (board, game_over, message) for a board, empty or finished."""
        result = olgf.evaluate_game_state(board)
        if result is None and not np.any(board == 0):
            result = 0
        if result is None:
            return board, False, ""
        if result == 0:
            return board, True, "It's a draw!"
        if result == 1:
            return board, True, "Black wins!"
        return board, True, "White wins!"

    def reset_turn():
        game["move"] = {"transfer": {"source": None, "target": None}, "add": None}
        game["phase"] = "none"
        game["selections"] = {"transfer_source": None, "transfer_target": None, "add": None}

    def stop_search():
        game["busy"] = False
        game["busy_kind"] = ""
        game["pending"] = None
        editor["busy"] = False

    def switch_mode(new_mode):
        nonlocal mode_state
        mode_state = new_mode

    # ---------------- Menu panel ----------------
    menu_panel = Panel(menu_rect)
    menu_panel.add_row(40, [(Label("Game Settings", title_font), 0, None, None)])
    menu_panel.add_row(32, [
        (Label("Grid Size:", body_font), 0, None, None),
        (grid_size_box, None, 62, 32),
        (Label("(1-8)", small_font), None, None, None),
    ])
    menu_panel.add_row(28, [(transfer_checkbox, 0, None, None)])
    menu_panel.add_row(24, [(Label("Rotation:", body_font), 0, None, None)])
    for radio in rotation_group:
        menu_panel.add_row(22, [(radio, 0, None, None)])
    menu_panel.add_row(24, [(Label("Game Mode:", body_font), 0, None, None)])
    for radio in mode_group:
        menu_panel.add_row(22, [(radio, 0, None, None)])
    menu_panel.add_row(24, [(Label("Computer Engine:", body_font), 0, None, None)])
    for radio in engine_group:
        menu_panel.add_row(22, [(radio, 0, None, None)])
    menu_panel.add_row(32, [
        (Label("Search time (s):", body_font), 0, None, None),
        (time_limit_box, None, 72, 32),
    ])
    menu_panel.add_row(24, [(Label("Move Completion:", body_font), 0, None, None)])
    for radio in completion_group:
        menu_panel.add_row(22, [(radio, 0, None, None)])
    start_button = Button(0, 46, heading_font.get_height(), "Start")
    menu_panel.add_row(50, [(start_button, 0, FILL, 46)])

    # ---------------- Position editor panel ----------------
    editor_hint = TextPanel(0, 46, small_font, scrollable=False, background=(240, 240, 244))
    editor_piece_group = radio_group(body_font.get_height(),
                                     ["Empty", "Black", "White"], default=1)
    editor_turn_group = radio_group(body_font.get_height(), ["Black", "White"])
    editor_opponent_group = radio_group(body_font.get_height(),
                                        ["2 Players", "Vs Computer"], default=1)
    editor_opponent_computer = editor_opponent_group[1]
    editor_engine_label = Label("Engine:", small_font)
    editor_apply_button = Button(70, 30, body_font.get_height(), "Apply")
    editor_ask_button = Button(0, 46, heading_font.get_height(), "Ask Computer")
    editor_clear_button = Button(160, 32, body_font.get_height(), "Clear Board")
    editor_play_button = Button(0, 32, body_font.get_height(), "Play This Position")
    editor_report = TextPanel(0, 150, small_font, title="Report")

    editor_panel = Panel(menu_rect)
    editor_panel.add_row(38, [(Label("Position Editor", title_font), 0, None, None)])
    editor_panel.add_row(editor_hint.rect.height, [(editor_hint, 0, FILL, editor_hint.rect.height)])
    editor_panel.add_row(32, [
        (Label("Grid:", body_font), 0, None, None),
        (grid_size_box, None, 62, 32),
        (editor_apply_button, None, 70, 30),
    ])
    editor_panel.add_row(24, [(Label("Piece to place:", body_font), 0, None, None)])
    for radio in editor_piece_group:
        editor_panel.add_row(22, [(radio, 0, None, None)])
    editor_panel.add_row(24, [(Label("Player to move:", body_font), 0, None, None)])
    for radio in editor_turn_group:
        editor_panel.add_row(22, [(radio, 0, None, None)])
    editor_panel.add_row(24, [(Label("Play this position:", body_font), 0, None, None)])
    for index, radio in enumerate(editor_opponent_group):
        editor_panel.add_row(22, [(radio, 0 if index == 0 else 90, None, None)])
    editor_panel.add_row(24, [(Label("Rotation:", body_font), 0, None, None)])
    for radio in rotation_group:
        editor_panel.add_row(22, [(radio, 0, None, None)])
    editor_panel.add_row(26, [(transfer_checkbox, 0, None, None)])
    editor_panel.add_row(20, [(editor_engine_label, 0, FILL, None)])
    editor_panel.add_row(46, [(editor_ask_button, 0, FILL, 46)])
    editor_panel.add_row(32, [
        (editor_clear_button, 0, 160, 32),
        (editor_play_button, None, FILL, 32),
    ])
    editor_panel.add_row(0, [(editor_report, 0, FILL, 0)], fill=True)

    # ---------------- Playing panel ----------------
    play_status_label = Label("", huge_font)
    play_message = TextPanel(0, 74, small_font, title="Computer", scrollable=False,
                             background=(240, 240, 244))
    play_hint_button = Button(0, 40, body_font.get_height(), "Hint (ask the computer)")
    play_take_back_button = Button(0, 40, body_font.get_height(), "Take Back")
    play_complete_button = Button(0, 40, body_font.get_height(), "Complete Move")
    play_restart_button = Button(0, 40, body_font.get_height(), "Restart")
    play_new_game_button = Button(0, 40, body_font.get_height(), "New Game")
    play_log = TextPanel(0, 120, small_font, title="Moves")

    play_panel = Panel(menu_rect)
    play_panel.add_row(44, [(play_status_label, 0, FILL, None)])
    play_panel.add_row(play_message.rect.height, [(play_message, 0, FILL, play_message.rect.height)])
    play_panel.add_row(40, [(play_hint_button, 0, FILL, 40)])
    play_panel.add_row(40, [(play_take_back_button, 0, FILL, 40)])
    play_panel.add_row(40, [(play_complete_button, 0, FILL, 40)])
    play_panel.add_row(40, [(play_restart_button, 0, FILL, 40)])
    play_panel.add_row(40, [(play_new_game_button, 0, FILL, 40)])
    play_panel.add_row(0, [(play_log, 0, FILL, 0)], fill=True)

    status_panel = TextPanel(board_width, STATUS_HEIGHT, small_font,
                             background=(240, 240, 243), scrollable=False)
    status_panel.rect = pygame.Rect(status_rect)

    # ---------------- Confirmation dialog ----------------
    dialog_rect = pygame.Rect(WINDOW_WIDTH // 2 - 170, WINDOW_HEIGHT // 2 - 90, 340, 180)
    dialog_title = Label("", heading_font)
    dialog_yes_button = Button(120, 44, body_font.get_height(), "Yes")
    dialog_no_button = Button(120, 44, body_font.get_height(), "No")
    dialog = {"active": False, "action": None}

    def place_dialog_widgets():
        dialog_title.rect.topleft = (dialog_rect.x + 22, dialog_rect.y + 40)
        dialog_yes_button.rect.topleft = (dialog_rect.x + 34, dialog_rect.y + 118)
        dialog_no_button.rect.topleft = (dialog_rect.x + 186, dialog_rect.y + 118)

    place_dialog_widgets()

    # ---------------- Computer engine ----------------
    engine_process = engine.EngineProcess()

    def make_request(state, player, config):
        """Build a picklable engine request out of the current settings."""
        return dict(
            state=[[int(value) for value in row] for row in state],
            player_turn=int(player),
            rotation=current_rotation(),
            transfer_allowed=bool(transfer_checkbox.checked),
            time_limit=read_time_limit(),
            **config,
        )

    def poll_engine():
        """Collect a finished answer from the engine process and use it."""
        result = engine_process.take_result()
        if result is None:
            return
        if game["busy"]:
            kind, game["busy_kind"] = game["busy_kind"], ""
            game["busy"] = False
            if time.monotonic() - game["busy_since"] < MIN_THINK_TIME:
                # hold a fast answer back so that the computer does not look instant
                game["pending"] = (result, time.monotonic(), kind)
            elif kind == "hint":
                apply_hint(result)
            else:
                apply_computer_move(result)
        elif editor["busy"]:
            editor["busy"] = False
            apply_editor_answer(result)
        flush_pending()

    def flush_pending():
        """Use a computer answer that was held back to keep the game readable."""
        if game["pending"] is None:
            return
        result, arrived, kind = game["pending"]
        if time.monotonic() - arrived < MIN_THINK_TIME:
            return
        game["pending"] = None
        if kind == "hint":
            apply_hint(result)
        else:
            apply_computer_move(result)

    # ---------------- Game actions ----------------
    def append_log(move, player):
        game["log"].append(f"{len(game['log']) + 1}. {PLAYER_NAMES[player]}: "
                           f"{engine.move_to_text(move)}")
        play_log.set_text("\n".join(game["log"]))
        play_log.scroll_to_end()

    def play_move(move, player):
        """Play a move for a player and update the game status and log."""
        board = olgf.play_turn(game["board"], move)
        if np.array_equal(board, game["board"]):
            game["message_line"] = "That move is not legal, try again."
            return False
        game["board"] = board
        game["board"], game["over"], game["message"] = evaluate_position(board)
        game["blocked"] = False
        append_log(move, player)
        game["player"] = 3 - player
        reset_turn()
        return True

    def complete_move():
        """Play the move that was assembled with the mouse."""
        if game["phase"] != "add_move" or game["move"]["add"] is None:
            return
        move = {"player": game["player"], "transfer": None,
                "add": game["move"]["add"], "rotate": current_rotation()}
        source = game["move"]["transfer"]["source"]
        target = game["move"]["transfer"]["target"]
        if transfer_checkbox.checked and source is not None and target is not None:
            if target[0] - source[0] == -1:
                direction = "u"
            elif target[0] - source[0] == 1:
                direction = "d"
            elif target[1] - source[1] == -1:
                direction = "l"
            else:
                direction = "r"
            move["transfer"] = [source, direction]
        game["hint"] = None
        play_move(move, game["player"])

    def take_back():
        """Undo the last step of the move that is being assembled."""
        if game["move"]["add"] is not None:
            game["move"]["add"] = None
            game["selections"]["add"] = None
            if transfer_checkbox.checked and game["move"]["transfer"]["target"] is not None:
                game["phase"] = "transfer_target"
            else:
                game["phase"] = "none"
        elif game["move"]["transfer"]["target"] is not None:
            game["move"]["transfer"]["target"] = None
            game["selections"]["transfer_target"] = None
            game["phase"] = "transfer_target"
        elif game["move"]["transfer"]["source"] is not None:
            game["move"]["transfer"]["source"] = None
            game["selections"]["transfer_source"] = None
            game["phase"] = "none"

    def start_game():
        """Leave the menu: either open the editor or start a fresh game."""
        size = read_grid_size()
        stop_search()
        if mode_editor.selected:
            editor["board"] = resize_board(editor["board"], size)
            switch_mode("editor")
            return
        game["board"] = np.zeros((size, size), dtype=int)
        game["player"] = 1
        game["log"] = []
        game["message_line"] = ""
        game["hint"] = None
        game["blocked"] = False
        game["vs_computer"] = mode_computer.selected
        play_log.clear()
        reset_turn()
        game["board"], game["over"], game["message"] = evaluate_position(game["board"])
        switch_mode("playing")

    def restart_game():
        """Start the current game again from an empty board."""
        stop_search()
        size = game["board"].shape[0]
        game["board"] = np.zeros((size, size), dtype=int)
        game["player"] = 1
        game["log"] = []
        game["message_line"] = ""
        game["hint"] = None
        game["blocked"] = False
        play_log.clear()
        reset_turn()
        game["board"], game["over"], game["message"] = evaluate_position(game["board"])

    def request_computer_move():
        """Let the computer answer for the side to move."""
        if game["busy"] or game["over"] or game["blocked"] or game["pending"] is not None:
            return
        if not vs_computer() or game["player"] != 2:
            return
        if not engine_process.submit(make_request(game["board"], game["player"], engine_config())):
            return
        game["busy"] = True
        game["busy_kind"] = "computer"
        game["busy_since"] = time.monotonic()
        game["message_line"] = "White is thinking..."

    def apply_computer_move(result):
        game["message_line"] = result["message"]
        if result["move"] is None:
            game["blocked"] = True
            return
        game["hint"] = None
        play_move(result["move"], game["player"])

    def request_hint():
        """Ask the computer which move the human should play."""
        if game["busy"] or game["over"] or game["pending"] is not None:
            return
        if not engine_process.submit(make_request(
                game["board"], game["player"],
                {"use_tablebase": True, "fallback": engine.FALLBACK_SEARCH})):
            return
        game["busy"] = True
        game["busy_kind"] = "hint"
        game["busy_since"] = time.monotonic()
        game["message_line"] = f"{PLAYER_NAMES[game['player']]} is asking the computer..."

    def apply_hint(result):
        game["message_line"] = result["message"]
        game["hint"] = result["move"]

    def open_dialog(action):
        dialog["active"] = True
        dialog["action"] = action

    def confirm_yes():
        action = dialog["action"]
        dialog["active"] = False
        dialog["action"] = None
        if action == "restart":
            restart_game()
        elif action == "new_game":
            stop_search()
            switch_mode("menu")

    def confirm_no():
        dialog["active"] = False
        dialog["action"] = None

    # ---------------- Editor actions ----------------
    def reset_editor():
        editor["suggestion"] = None
        editor_report.set_text('Draw a position and press "Ask Computer".\n\n' + EDITOR_HELP)

    def selected_piece():
        return PIECE_VALUES[next(radio.text for radio in editor_piece_group if radio.selected)]

    def apply_editor_size():
        size = read_grid_size()
        editor["board"] = resize_board(editor["board"], size)
        grid_size_box.set_text(str(size))
        reset_editor()

    def clear_editor():
        editor["board"] = np.zeros(editor["board"].shape, dtype=int)
        reset_editor()

    def play_editor_position():
        """Start a game that begins from the drawn position."""
        stop_search()
        game["board"] = editor["board"].copy()
        game["player"] = editor["player"]
        game["message"] = ""
        game["log"] = []
        game["message_line"] = ""
        game["hint"] = None
        game["blocked"] = False
        game["vs_computer"] = editor_opponent_computer.selected
        play_log.clear()
        reset_turn()
        game["board"], game["over"], game["message"] = evaluate_position(game["board"])
        switch_mode("playing")

    def request_editor_answer():
        """Ask the engine for the best move in the drawn position."""
        if editor["busy"]:
            return
        if not engine_process.submit(make_request(
                editor["board"], editor["player"], engine_config())):
            return
        editor["busy"] = True
        editor["busy_since"] = time.monotonic()
        editor["suggestion"] = None
        editor_report.set_text("Asking the computer, please wait...")

    def apply_editor_answer(result):
        editor["suggestion"] = result["move"]
        report = result["message"]
        if result.get("text"):
            report += "\n\n" + result["text"]
        elif result["source"] == engine.SOURCE_SEARCH and result.get("search"):
            report += "\n\nBest moves found:\n"
            for move, value in result["search"]["top_moves"]:
                report += f"  {engine.move_to_text(move)} (score {value})\n"
        editor_report.set_text(report)
        editor_report.scroll = 0

    def editor_click(cell, button):
        if cell is None:
            return
        row, col = cell
        if button == 3:
            value = 0
        elif button == 2:
            value = 3 - selected_piece()
        else:
            value = selected_piece()
        if editor["board"][row, col] != value:
            editor["board"][row, col] = value
            reset_editor()

    def handle_board_click(event, hover_cell):
        """Build a move with the mouse on the board of a running game."""
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return
        if game["over"] or hover_cell is None:
            return
        if vs_computer() and game["player"] != 1:
            return
        row, col = hover_cell
        board = game["board"]
        if transfer_checkbox.checked and game["phase"] == "none" \
                and board[row, col] == 3 - game["player"]:
            game["move"]["transfer"]["source"] = (row, col)
            game["selections"]["transfer_source"] = (row, col)
            game["phase"] = "transfer_target"
        elif game["phase"] == "transfer_target":
            source = game["move"]["transfer"]["source"]
            if ((abs(row - source[0]) == 1 and col == source[1])
                    or (abs(col - source[1]) == 1 and row == source[0])):
                if board[row, col] == 0:
                    game["move"]["transfer"]["target"] = (row, col)
                    game["selections"]["transfer_target"] = (row, col)
                    game["phase"] = "add_move"
        elif game["phase"] in ("none", "add_move"):
            source = game["move"]["transfer"]["source"]
            if board[row, col] == 0 or (transfer_checkbox.checked
                                        and source is not None and (row, col) == source):
                game["move"]["add"] = (row, col)
                game["selections"]["add"] = (row, col)
                game["phase"] = "add_move"
                if next(radio.text for radio in completion_group
                        if radio.selected) == "Auto Move Completion":
                    complete_move()

    def handle_editor_click(event, hover_cell):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 2, 3):
            editor_click(hover_cell, event.button)

    # ---------------- Widget callbacks ----------------
    start_button.callback = start_game
    play_complete_button.callback = complete_move
    play_take_back_button.callback = take_back
    play_restart_button.callback = lambda: open_dialog("restart")
    play_new_game_button.callback = lambda: open_dialog("new_game")
    play_hint_button.callback = request_hint
    editor_apply_button.callback = apply_editor_size
    editor_ask_button.callback = request_editor_answer
    editor_clear_button.callback = clear_editor
    editor_play_button.callback = play_editor_position
    dialog_yes_button.callback = confirm_yes
    dialog_no_button.callback = confirm_no

    def handle_resize(width, height):
        """Follow the window size: the board stays square, the panel is the rest."""
        nonlocal board_rect, status_rect, menu_rect, dialog_rect
        size = max(200, min(width - menu_width, height - 120))
        board_rect = pygame.Rect(0, 0, size, size)
        status_rect = pygame.Rect(0, size, size, max(60, height - size))
        menu_rect = pygame.Rect(size, 0, max(200, width - size), height)
        menu_panel.rect = pygame.Rect(menu_rect)
        editor_panel.rect = pygame.Rect(menu_rect)
        play_panel.rect = pygame.Rect(menu_rect)
        status_panel.rect = pygame.Rect(status_rect)
        dialog_rect = pygame.Rect(width // 2 - 170, height // 2 - 90, 340, 180)
        place_dialog_widgets()
        for panel in (menu_panel, editor_panel, play_panel):
            panel.layout()

    # ---------------- Drawing ----------------
    def rules_text():
        return (f"rotation: {current_rotation()}, transfers: "
                f"{'allowed' if transfer_checkbox.checked else 'not allowed'}")

    def tablebase_text():
        if not tablebase_sizes:
            return "tablebase: not generated"
        return "tablebase: " + ", ".join(f"{size}x{size}" for size in tablebase_sizes)

    def cell_text(hover_cell):
        if hover_cell is None:
            return f"Legend: {PLAYER_SYMBOLS[1]} = Black, {PLAYER_SYMBOLS[2]} = White."
        return f"Cell: ({hover_cell[0]}, {hover_cell[1]})   " \
               f"Legend: {PLAYER_SYMBOLS[1]} = Black, {PLAYER_SYMBOLS[2]} = White."

    def draw_status(text):
        status_panel.set_text(text)

    def draw_playing(screen, hover_cell):
        highlights = selection_highlights(game["selections"])
        if game["hint"] is not None:
            highlights += move_highlights(game["hint"])
        winning = find_winning_lines(game["board"]) if game["over"] else ()
        draw_board(screen, game["board"], board_rect, hover_cell, highlights, winning)

        if game["over"]:
            play_status_label.set_text(game["message"])
        else:
            play_status_label.set_text(f"Turn: {PLAYER_NAMES[game['player']]}")
        if game["busy"]:
            play_message.set_text(
                f"{PLAYER_NAMES[game['player']]} is thinking... "
                f"{time.monotonic() - game['busy_since']:.1f}s")
        else:
            play_message.set_text(game["message_line"])
        play_panel.draw(screen)
        size = game["board"].shape[0]
        draw_status(
            f"Playing a {size}x{size} game"
            f"{' against the computer' if vs_computer() else ' with two players'}"
            f"  |  {rules_text()}\n"
            f"{tablebase_text()}\n"
            f"{cell_text(hover_cell)}"
        )

    def draw_editor(screen, hover_cell):
        highlights = move_highlights(editor["suggestion"])
        winning = find_winning_lines(editor["board"])
        draw_board(screen, editor["board"], board_rect, hover_cell, highlights, winning)
        editor_engine_label.set_text(engine_config_text())
        editor_hint.set_text(EDITOR_HELP)
        editor_panel.draw(screen)
        size = editor["board"].shape[0]
        if editor["busy"]:
            note = f"The computer is thinking... {time.monotonic() - editor['busy_since']:.1f}s"
        else:
            note = f"{size}x{size} grid, {PLAYER_NAMES[editor['player']]} to move."
        draw_status(
            f"Position editor  |  {rules_text()}\n"
            f"{note}\n"
            f"{tablebase_text()}\n"
            f"{cell_text(hover_cell)}"
        )

    def draw_menu(screen):
        pygame.draw.rect(screen, PANEL_COLOR, menu_rect)
        menu_panel.draw(screen)
        draw_status(
            "Choose the settings on the right and press Start.\n"
            "Position Editor lets you draw a position and ask the computer for "
            "the best move in it.\n"
            f"{tablebase_text()}\n"
            f"Legend: {PLAYER_SYMBOLS[1]} = Black, {PLAYER_SYMBOLS[2]} = White."
        )

    def draw_mode(screen, hover_cell):
        if mode_state == "playing":
            draw_playing(screen, hover_cell)
        elif mode_state == "editor":
            draw_editor(screen, hover_cell)
        else:
            draw_menu(screen)

    def draw_dialog():
        pygame.draw.rect(screen, (185, 185, 190), dialog_rect)
        pygame.draw.rect(screen, (0, 0, 0), dialog_rect, 2)
        if dialog["action"] == "restart":
            dialog_title.set_text("Restart game?")
        else:
            dialog_title.set_text("Return to main menu?")
        dialog_title.draw(screen)
        dialog_yes_button.draw(screen)
        dialog_no_button.draw(screen)

    # ---------------- Panels and initial texts ----------------
    for panel in (menu_panel, editor_panel, play_panel):
        panel.layout()
    reset_editor()
    play_log.clear()
    status_panel.set_text(
        "Choose the settings on the right and press Start.\n"
        f"{tablebase_text()}."
    )

    # ---------------- Main loop ----------------
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        if mode_state == "playing":
            active_board = game["board"]
        elif mode_state == "editor":
            active_board = editor["board"]
        else:
            active_board = None
        hover_cell = cell_at(mouse_pos, board_rect, active_board)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue
            if event.type == pygame.VIDEORESIZE:
                handle_resize(event.w, event.h)
                continue
            if dialog["active"]:
                dialog_yes_button.handle_event(event)
                dialog_no_button.handle_event(event)
                continue
            if mode_state == "menu":
                menu_panel.handle_event(event)
            elif mode_state == "playing":
                play_panel.handle_event(event)
                handle_board_click(event, hover_cell)
            else:
                editor_panel.handle_event(event)
                handle_editor_click(event, hover_cell)

        poll_engine()
        if mode_state == "playing" and not game["over"] and vs_computer() and game["player"] == 2:
            request_computer_move()
        flush_pending()

        # --------------------- Drawing ---------------------
        screen.fill((200, 200, 200))
        draw_mode(screen, hover_cell)
        status_panel.draw(screen)
        if dialog["active"]:
            draw_dialog()
        pygame.display.flip()
        clock.tick(60)

    engine_process.shutdown()
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
