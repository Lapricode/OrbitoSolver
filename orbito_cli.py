"""Interactive command line front-end for the ORBITO (Orbital Logic Game) solver.

The tool asks for the rules of the game once, one short question at a time, and
then answers the same two questions for as long as the user wants them:

* a position can be typed in and solved, which prints the perfect move, how many
  legal moves keep that result, and the whole perfect game that follows,
* a game can be played, either by both players or by the user against the
  computer, which follows the very same table.

Both read the retrograde value tablebase, so every answer is exact rather than
searched. Pressing ``?`` in the middle of a game asks the same question about
the position the game has reached.

Several moves often reach the same result, so the rules ask how to choose between
them. Ranked is the way the graphical interface does it: among the moves that win,
the one that ends the game soonest and leaves the opponent the most losing
replies; among the moves that lose, the one that survives longest. First takes the
first move of the stored line instead, which is one lookup rather than one per
move. Either way the result of the game is the one the tablebase proves.

Nothing has to be spelled out: every question offers a default, so Enter is
always an answer, and every prompt accepts a single letter.

Usage::

    python orbito_cli.py

Cells are named ``a1``, ``b2``, ... with the column as a letter and the row as a
number, counted from the top left. A move is the cell to add a piece on, and,
optionally before it, the transfer of an opponent's piece:

* ``c3``            add your own piece at c3,
* ``a1b1 c3``       move the piece on a1 to the neighbouring cell b1, add at c3,
* ``a1r c3``        the same move, with a direction letter instead of the target.

A position is written row by row, either with digits (``0012``) or with symbols
(``xo..``), and ``empty`` is the untouched board.
"""

import collections
import re
import sys
import textwrap

import numpy as np

import computer_engine
import get_solution
import orbital_logic_game_functions as olgf
import retrograde_tablebase
import solve_game


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TABLEBASE_DIR = computer_engine.RETROGRADE_TABLEBASE_DIR

# The board symbols. A dot is used for an empty cell because it reads better in
# a grid than the underscore of the tablebase symbols, and it is what the help
# text and the position input use as well.
SYMBOLS = {0: ".", 1: "x", 2: "o"}

# Every character a position may be written with.
STATE_SYMBOLS = {
    "0": 0, ".": 0, "_": 0, "-": 0,
    "1": 1, "x": 1,
    "2": 2, "o": 2,
}
STATE_SEPARATORS = " \t\r\n/|,"

# A cell is a column letter followed by a row number, so a move is read by
# picking the cells out of whatever the user typed and ignoring the separators.
CELL_PATTERN = re.compile(r"([a-z])\s*([0-9]+)")
MOVE_SEPARATORS = " \t,+-/;:_"
DIRECTIONS = {"u": (-1, 0), "d": (1, 0), "l": (0, -1), "r": (0, 1)}

MAX_LISTED_MOVES = 40
COMPUTER_SECONDS = 10.0

# What one answer to "what should be played here" holds: the move, the perfect
# game that follows it, the value of the position, and the evaluation of every
# legal move that the choice was made from.
Plan = collections.namedtuple(
    "Plan", "move moves states score plies evaluated complete")

MODE_X = 1
MODE_O = 2
MODE_BOTH = 3
HUMAN_PLAYER = {MODE_X: 1, MODE_O: 2, MODE_BOTH: None}
MODE_LABELS = {
    MODE_X: "you play x, the computer plays o",
    MODE_O: "you play o, the computer plays x",
    MODE_BOTH: "you play both sides and type the moves of both players",
}
REPEAT_LABELS = {
    "game": "another game",
    "solve": "another position",
}

# How the best move is picked when several of them reach the same result.
THINK_RANKED = "ranked"
THINK_FIRST = "first"
THINK_LABELS = {
    THINK_RANKED: "every legal move ranked, the way the graphical interface ranks them",
    THINK_FIRST: "the first move of the tablebase line, with no further analysis",
}

# The mirror image of a rotation answers the same game, so one of the two stored
# rotation tables is enough to serve both.
COUNTERPART_ROTATION = {
    "clockwise": "counterclockwise",
    "counterclockwise": "clockwise",
    "still": None,
}

SOURCE_FIRST = "tablebase_first"
SOURCE_NOTES = {
    "tablebase": "perfect play from the tablebase",
    SOURCE_FIRST: "the first move the tablebase gives, not ranked",
    "search": "minimax search, the tablebase has no entry",
    "endgame": "the forced Orbito press",
    "random": "a random move",
    "none": "no move to make",
}

NOT_A_MOVE = (
    "I did not read a move from that. Write the cell to add your piece on, "
    "optionally preceded by a transfer: `c3` adds at c3, `a1b1 c3` moves the "
    "piece on a1 to the neighbouring cell b1 and adds at c3, and `a1r c3` does "
    "the same with a direction letter (u, d, l, r)."
)

NOT_STORED = (
    "The tablebase has no entry for that position. The stored tables only hold "
    "the positions a game can reach from the empty board, so check the board, "
    "the side to move and the rules."
)

GAME_COMMANDS = {
    "?": "solve", "best": "solve", "hint": "solve",
    "*": "play", "play": "play",
    "m": "moves", "moves": "moves", "list": "moves",
    "u": "undo", "undo": "undo",
    "b": "board", "board": "board",
    "h": "help", "help": "help",
    "q": "left", "quit": "left", "menu": "left", "back": "left",
}

GAME_HELP = """\
Moves
  c3        add your own piece at c3
  a1b1 c3   move the opponent's piece on a1 to the neighbouring cell b1,
            then add your own piece at c3
  a1r c3    the same move, with a direction letter (u, d, l, r) for the transfer

Commands
  ?         the best move and the perfect continuation from here
  ?1  ?2    the same, seen from player 1's or player 2's side
  *         let the computer play the move it would recommend
  m         value every legal move: result, length, and losing replies
  u         undo the last move (two plies against the computer)
  b         show the board again
  h         this help
  q         leave the game
"""

MAIN_HELP = """\
Every prompt takes a single letter, and Enter repeats what you were doing:

  [Enter]  do the same thing again
  s        solve a position
  g        play a game
  r        answer the rules again and start over
  q        leave

In a game, ? analyses the position, * plays the move the tool recommends, m values
every legal move, u undoes, b prints the board and q leaves the game.

The rules are asked once: the board size, the rotation, whether transfers and the
endgame presses are on, which side you play, and how the best move is picked.
Ranked, the tool compares every legal move the way the graphical interface does:
among the moves that win it takes the one that wins soonest and leaves the most
losing replies, and among the moves that lose it takes the one that survives
longest. First, it takes the first move of the stored line and does not rank the
moves around it.
"""


# ---------------------------------------------------------------------------
# Asking questions
# ---------------------------------------------------------------------------

class Quit(Exception):
    """Raised to leave the tool, on Ctrl-D or Ctrl-C."""


def prompt(text):
    """Read one line, treating Ctrl-D and Ctrl-C as a request to leave."""
    try:
        return input(text)
    except (EOFError, KeyboardInterrupt):
        print()
        raise Quit()


def wrap(text, indent="  "):
    """Return ``text`` as one paragraph indented for the terminal."""
    return textwrap.fill(
        " ".join(str(text).split()),
        width=76,
        initial_indent=indent,
        subsequent_indent=indent,
    )


def ask_yes_no(question, default):
    """Ask a yes or no question, with ``default`` on an empty answer."""
    hint = "Y/n" if default else "y/N"
    while True:
        answer = prompt(f"{question} ({hint}) ").strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes", "true", "1", "t"):
            return True
        if answer in ("n", "no", "false", "0", "f"):
            return False
        print("Please answer y or n.")


def ask_int(question, default, allowed=None):
    """Ask for a whole number, with ``default`` on an empty answer."""
    label = f"{question} {'/'.join(str(value) for value in allowed)}" if allowed else question
    while True:
        answer = prompt(f"{label} [{default}] ").strip()
        if not answer:
            return default
        try:
            value = int(answer)
        except ValueError:
            print("That is not a whole number.")
            continue
        if allowed and value not in allowed:
            print(f"Choose one of: {', '.join(str(item) for item in allowed)}.")
            continue
        return value


def ask_rotation(default="counterclockwise"):
    """Ask how the board turns after every turn."""
    choices = {
        "c": "clockwise", "cw": "clockwise", "-": "clockwise",
        "a": "counterclockwise", "ccw": "counterclockwise", "+": "counterclockwise",
        "s": "still", "st": "still",
    }
    while True:
        answer = prompt(
            f"Rotation after each turn: (c)lockwise, (a)nticlockwise, s(t)ill [{default}] "
        ).strip().lower()
        if not answer:
            return default
        if answer in choices:
            return choices[answer]
        print("Please answer c, a or s.")


def ask_mode(default=MODE_X):
    """Ask which sides the user takes."""
    while True:
        answer = prompt("You play: (x), (o), or (b)oth sides [x] ").strip().lower()
        if not answer:
            return default
        if answer in ("x", "1"):
            return MODE_X
        if answer in ("o", "2"):
            return MODE_O
        if answer in ("b", "both", "3"):
            return MODE_BOTH
        print("Please answer x, o or b.")


def ask_next(repeat_label):
    """Ask what to do once the current step is finished.

    Returns ``"repeat"``, ``"solve"``, ``"game"``, ``"rules"`` or ``"quit"``,
    and ``None`` when the answer was not understood, so the question can simply
    be asked again.
    """
    while True:
        answer = prompt(
            f"Now: [Enter] {repeat_label}   [s]olve   [g]ame   [r] new rules   [q]uit\n> "
        ).strip().lower()
        if answer in ("", "enter", "again"):
            return "repeat"
        if answer in ("s", "solve", "position"):
            return "solve"
        if answer in ("g", "game", "play", "new"):
            return "game"
        if answer in ("r", "rules", "restart", "new rules"):
            return "rules"
        if answer in ("q", "quit", "exit"):
            return "quit"
        print("Type Enter to carry on, s to solve a position, g to play a game, "
              "r for new rules, or q to quit.")


def ask_in_game(board, player, rules, ply):
    """Read one line during a game.

    Returns ``(action, value)``: ``("move", move)`` with a move ready for
    :func:`orbital_logic_game_functions.play_turn`, ``("solve", player)`` with
    the side to analyse, or one of ``"play"``, ``"undo"``, ``"moves"``,
    ``"board"``, ``"help"`` and ``"left"`` with no value.
    """
    while True:
        answer = prompt(f"\nmove {ply} ({SYMBOLS[player]} to play) > ").strip().lower()
        if not answer:
            continue
        if answer.startswith("?"):
            side = answer[1:].strip()
            if side in ("", "?"):
                return "solve", player
            if side in ("1", "x"):
                return "solve", 1
            if side in ("2", "o"):
                return "solve", 2
            print("Type ? for the side to move, ?1 or ?2 for one side, or h for help.")
            continue
        command = GAME_COMMANDS.get(answer)
        if command == "solve":
            return "solve", player
        if command is not None:
            return command, None
        parsed = parse_move(answer)
        if parsed is None:
            print(wrap(NOT_A_MOVE))
            continue
        error = validate_move(board, player, parsed)
        if error is not None:
            print(error)
            continue
        transfer, add = parsed
        return "move", {
            "player": player,
            "transfer": None if transfer is None else [transfer[0], transfer[1]],
            "add": add,
            "rotate": rules.rotation,
        }


# ---------------------------------------------------------------------------
# The rules of a game
# ---------------------------------------------------------------------------

class Rules:
    """The rule context a game is played and solved in."""

    def __init__(self, grid_size, rotation, transfer_allowed, extra_rotation_allowed,
                 thinking=THINK_RANKED):
        self.grid_size = int(grid_size)
        self.rotation = rotation
        self.transfer_allowed = bool(transfer_allowed)
        self.extra_rotation_allowed = bool(extra_rotation_allowed)
        self.thinking = thinking

    def ranks_moves(self):
        """Tell whether the best move is ranked over every legal move."""
        return self.thinking == THINK_RANKED

    def lookup(self):
        """Return the keyword arguments that select this context in a lookup."""
        return {
            "rotate_direction": self.rotation,
            "transfer_allowed": self.transfer_allowed,
            "extra_rotation_allowed": self.extra_rotation_allowed,
            "base_dir": TABLEBASE_DIR,
        }

    def describe(self):
        """Return one line naming every rule of the context."""
        parts = [
            f"{self.grid_size}x{self.grid_size} board",
            f"rotation {self.rotation} after every turn",
            "transfers allowed" if self.transfer_allowed else "no transfers",
        ]
        if self.extra_rotation_allowed:
            parts.append(f"{solve_game.EXTRA_TURNS} endgame Orbito presses")
        else:
            parts.append("no endgame presses")
        return ", ".join(parts)

    def describe_thinking(self):
        """Return the sentence naming how the best move is picked."""
        return THINK_LABELS[self.thinking]


def ask_thinking(default=THINK_RANKED):
    """Ask how the best move should be chosen between the ones that hold."""
    choices = {
        "r": THINK_RANKED, "rank": THINK_RANKED, "ranked": THINK_RANKED,
        "deep": THINK_RANKED, "gui": THINK_RANKED,
        "f": THINK_FIRST, "first": THINK_FIRST, "faster": THINK_FIRST,
    }
    while True:
        answer = prompt(
            "Best move: (r)ank every legal move, or take the (f)irst move the "
            f"tablebase gives [{default}] "
        ).strip().lower()
        if not answer:
            return default
        if answer in choices:
            return choices[answer]
        print("Please answer r or f.")


def ask_rules():
    """Ask for the rule context of the game and who plays which side.

    Every question offers a default, so a whole context is set up by pressing
    Enter six times.
    """
    sizes = computer_engine.available_grid_sizes(TABLEBASE_DIR)
    if sizes:
        grid_size = ask_int("Board size", sizes[-1], allowed=sizes)
    else:
        print(wrap(
            f"No tablebase was found in {TABLEBASE_DIR}, so no position can be "
            "answered yet. Build one with `python retrograde_tablebase.py -n 3` "
            "and start again. Press Enter to carry on regardless."
        ))
        prompt("")
        grid_size = 4
    rotation = ask_rotation()
    transfer_allowed = ask_yes_no(
        "Transfers: an opponent's piece may move to a neighbouring empty cell", True)
    extra_rotation_allowed = ask_yes_no(
        f"Endgame: a full board without a line gets {solve_game.EXTRA_TURNS} more "
        "Orbito presses", True)
    mode = ask_mode()
    thinking = ask_thinking()
    return Rules(grid_size, rotation, transfer_allowed, extra_rotation_allowed,
                 thinking), mode


def warn_missing_context(rules):
    """Say so when the chosen rules cannot be answered from the tables."""
    stored = set(retrograde_tablebase.available_contexts(TABLEBASE_DIR, rules.grid_size))
    if not stored:
        print(f"  ! no table is stored for a {rules.grid_size}x{rules.grid_size} board")
        return
    wanted = (rules.rotation, rules.transfer_allowed, rules.extra_rotation_allowed)
    mirror = COUNTERPART_ROTATION.get(rules.rotation)
    if wanted in stored:
        return
    if mirror is not None and (mirror, wanted[1], wanted[2]) in stored:
        return
    listed = ", ".join(
        f"{rotation}/{'transfer' if transfer else 'no transfer'}"
        f"/{'presses' if extra else 'no presses'}"
        for rotation, transfer, extra in sorted(stored)
    )
    print(f"  ! no table is stored for these rules; stored here: {listed}")


# ---------------------------------------------------------------------------
# Notation
# ---------------------------------------------------------------------------

def cell_name(cell):
    """Return the name of a cell, the column letter first and the row number after."""
    row, column = cell
    return f"{chr(ord('a') + column)}{row + 1}"


def neighbour(cell, direction):
    """Return the cell one step from ``cell`` in a direction letter."""
    row, column = cell
    step_row, step_column = DIRECTIONS[direction]
    return row + step_row, column + step_column


def direction_between(source, target):
    """Return the direction letter from one cell to a neighbour, or None."""
    for letter, (step_row, step_column) in DIRECTIONS.items():
        if (source[0] + step_row, source[1] + step_column) == target:
            return letter
    return None


def _cells_and_letters(text):
    """Pull the board cells and the stray direction letters out of typed text."""
    squeezed = "".join(char for char in text.lower() if char not in MOVE_SEPARATORS)
    cells = []
    letters = []
    position = 0
    for match in CELL_PATTERN.finditer(squeezed):
        letters.extend(
            char for char in squeezed[position:match.start()] if char in DIRECTIONS
        )
        cells.append((int(match.group(2)) - 1, ord(match.group(1)) - ord("a")))
        position = match.end()
    letters.extend(char for char in squeezed[position:] if char in DIRECTIONS)
    return cells, letters


def parse_move(text):
    """Return ``((source, direction), add)``, ``(None, add)``, or None.

    ``source`` and ``direction`` are the pair
    :func:`orbital_logic_game_functions.play_turn` wants for a transfer. The
    target cell is left out on purpose: it is always the neighbour of the source
    in the given direction, so it never has to be typed twice.
    """
    cells, letters = _cells_and_letters(text)
    if letters:
        if len(letters) != 1 or len(cells) != 2:
            return None
        return (cells[0], letters[0]), cells[1]
    if len(cells) == 3:
        direction = direction_between(cells[0], cells[1])
        if direction is None:
            return None
        return (cells[0], direction), cells[2]
    if len(cells) == 1:
        return None, cells[0]
    return None


def format_move(move):
    """Return the notation of a move dictionary, in the form the tool accepts."""
    if olgf.is_rotation_only(move):
        return "press"
    parts = []
    transfer = move.get("transfer")
    if transfer:
        source = tuple(transfer[0])
        direction = str(transfer[1]).lower()
        if direction in DIRECTIONS:
            parts.append(f"{cell_name(source)}{cell_name(neighbour(source, direction))}")
        else:
            parts.append(cell_name(source))
    if move.get("add") is not None:
        parts.append(cell_name(tuple(move["add"])))
    return " ".join(parts) if parts else "press"


def parse_state(text, grid_size):
    """Return the position typed by the user, or None when it cannot be read.

    Digits and symbols are both accepted, the cells are read row by row, and
    ``empty`` is the untouched board.
    """
    text = text.strip().lower()
    if "".join(char for char in text if char.isalpha()) == "empty":
        return np.zeros((grid_size, grid_size), dtype=int)
    characters = [char for char in text if char not in STATE_SEPARATORS]
    values = [STATE_SYMBOLS.get(char) for char in characters]
    if len(values) != grid_size * grid_size or any(value is None for value in values):
        return None
    return np.asarray(values, dtype=int).reshape((grid_size, grid_size))


def on_board(cell, grid_size):
    """Tell whether a cell lies inside the board."""
    return 0 <= cell[0] < grid_size and 0 <= cell[1] < grid_size


# ---------------------------------------------------------------------------
# Showing things
# ---------------------------------------------------------------------------

def render_board(board):
    """Return the board as a labelled grid of x, o and dot."""
    grid_size = board.shape[0]
    lines = ["     " + " ".join(chr(ord("a") + column) for column in range(grid_size))]
    for row in range(grid_size):
        cells = " ".join(SYMBOLS[int(cell)] for cell in board[row])
        lines.append(f"  {row + 1}  {cells}")
    return "\n".join(lines)


def render_brief(board):
    """Return the board as a single line of symbols, to keep a long game short."""
    return " ".join(SYMBOLS[int(cell)] for cell in np.asarray(board).ravel())


def outcome(board, rules):
    """Return 1, 2 or 0 when the game is over, and None while it is not."""
    result = olgf.evaluate_game_state(board)
    if result is not None:
        return result
    if not rules.extra_rotation_allowed and olgf.board_is_full(board):
        return 0
    return None


def describe_result(result):
    """Return the sentence naming the end of a game."""
    if result == 0:
        return "draw"
    return f"{SYMBOLS[result]} wins"


def winning_line(board, player):
    """Return the name of a line of the player that completes the board, or None.

    The lines are looked for in the order the rules judge them in, so the first
    one found is the one the game ended on.
    """
    size = board.shape[0]
    for row in range(size):
        if all(board[row, col] == player for col in range(size)):
            return f"row {row + 1}"
    for col in range(size):
        if all(board[row, col] == player for row in range(size)):
            return f"column {chr(ord('a') + col)}"
    if all(board[index, index] == player for index in range(size)):
        return "main diagonal"
    if all(board[index, size - 1 - index] == player for index in range(size)):
        return "other diagonal"
    return None


def describe_winner(board, result):
    """Return the sentence naming the end of a game and the line that made it."""
    ending = describe_result(result)
    line = winning_line(board, result)
    if line is None:
        return ending
    return f"{ending} with the {line} line"


def presses_pending(board, rules):
    """Tell whether the board is down to the forced Orbito presses.

    A full board without a line is not over while presses are left: the game is
    turned until a line shows up or every press is spent.
    """
    return (rules.extra_rotation_allowed
            and olgf.board_is_full(board)
            and olgf.evaluate_game_state(board) is None)


def play_presses(board, player, rules):
    """Play the forced Orbito presses out and print each one.

    A press only turns the board, so there is nothing left to choose and nobody
    is asked: the presses go on until a line appears or all of them are spent,
    and the game is over either way. Returns the result and the boards the
    presses led to.
    """
    total = int(solve_game.EXTRA_TURNS)
    boards = []
    for number in range(1, total + 1):
        mover = player
        press = olgf.rotation_only_move(player, rules.rotation)
        board = np.asarray(olgf.play_turn(board, press), dtype=int)
        player = 3 - player
        boards.append(board)
        print(f"  {SYMBOLS[mover]} presses the ring   "
              f"(Orbito press {number} of {total})")
        print()
        print(render_board(board))
        if olgf.evaluate_game_state(board) is not None:
            return olgf.evaluate_game_state(board), boards
    return 0, boards


def infer_side_to_move(board):
    """Return the player a position belongs to, or None when it cannot be one.

    Every turn fills exactly one cell, so the pieces on the board say how many
    turns were played and which colour is to move. That is how the tablebase
    reads a position too, and answering under the wrong colour would be worse
    than refusing.
    """
    occupied = int(np.count_nonzero(board))
    ones = int(np.count_nonzero(np.asarray(board) == 1))
    mover = 1 if occupied % 2 == 0 else 2
    if ones == (occupied + 1) // 2:
        return mover
    if ones == occupied // 2:
        return 3 - mover
    return None


def print_banner():
    """Say what the tool is and what it can answer."""
    sizes = computer_engine.available_grid_sizes(TABLEBASE_DIR)
    covered = ", ".join(f"{size}x{size}" for size in sizes) or "none"
    print("ORBITO solver: best move and perfect continuation, from the retrograde tablebase")
    print(f"  tablebase:   {TABLEBASE_DIR}")
    print(f"  board sizes: {covered}")
    print(wrap(
        "The rules are asked once, then the tool waits: Enter repeats what you "
        "were doing, ? analyses the position, and q leaves."
    ))


def print_rules(rules, mode):
    """Print the rule context the next game or position will use."""
    print()
    print(f"  Rules  {rules.describe()}")
    print(f"  Play   {MODE_LABELS[mode]}")
    print(f"  Best   {rules.describe_thinking()}")
    warn_missing_context(rules)


def print_move_evaluation(board, player, rules, evaluated=None, complete=True, best=None):
    """Print every legal move of a position with what it achieves.

    Each line gives the move, whether it wins, draws or loses for the side to
    move, how many plies the game still lasts, and how many of the opponent's
    replies lose the game for the opponent. The move the tool would play is
    marked, and the list is ordered as the ranking orders it.
    """
    if evaluated is None:
        evaluated, complete = evaluate(board, player, rules, count_traps=rules.ranks_moves())
    if not evaluated:
        moves = olgf.get_possible_moves(board, rules.rotation, rules.transfer_allowed, player)
        if not moves:
            print(wrap(
                "There is no move left: the board is full and nobody has a line, so "
                "the game is decided by the Orbito presses."
            ))
            return
        print(f"  {len(moves)} legal moves for {SYMBOLS[player]}, but the tablebase "
              "stores no value for them:")
        shown = moves[:MAX_LISTED_MOVES]
        for move in shown:
            print(f"    {format_move(move)}")
        if len(moves) > len(shown):
            print(f"    ... and {len(moves) - len(shown)} more")
        return
    counted = any(item["traps"] for item in evaluated)
    print(wrap(
        "Every legal move with the result it reaches, how long the game then "
        "lasts" + (", and how many of the opponent's replies lose the game for "
                   "the opponent" if counted else "")
        + ". The marked move is the one the tool would play."
    ))
    print()
    print(f"  {len(evaluated)} legal moves for {SYMBOLS[player]}, best first:")
    shown = evaluated[:MAX_LISTED_MOVES]
    for number, item in enumerate(shown, start=1):
        mark = "->" if best is not None and same_move(item["move"], best) else "  "
        line = (f"  {number:>3}. {mark} {format_move(item['move']):<12} "
                f"{describe_move_value(item, player)}")
        if counted:
            line = f"{line:<44} {item['traps']} losing replies"
        print(line)
    if len(evaluated) > len(shown):
        print(f"    ... and {len(evaluated) - len(shown)} more")
    if not complete:
        print("    (the tablebase is missing some of the moves, so this list is short)")


def same_move(one, other):
    """Tell whether two move dictionaries name the same move."""
    if one is None or other is None:
        return False
    return format_move(one) == format_move(other)


def describe_move_value(item, player):
    """Return the sentence naming what one evaluated move achieves."""
    result = item["result"]
    plies = item["moves_to_result"]
    if result == computer_engine.RESULT_DRAW:
        return "draw with perfect play"
    if result == computer_engine.RESULT_WIN:
        if plies is None:
            return f"{SYMBOLS[player]} wins"
        return f"{SYMBOLS[player]} wins in {print_plies(plies)}"
    if plies is None:
        return f"{SYMBOLS[3 - player]} wins"
    return f"{SYMBOLS[3 - player]} wins in {print_plies(plies)}"


def print_line(moves, states):
    """Print a perfect continuation move by move, one line per ply."""
    presses = 0
    for number, move in enumerate(moves, start=1):
        if olgf.is_rotation_only(move):
            presses += 1
            text = f"press {presses}"
        else:
            text = format_move(move)
        print(f"  {number:>3}. {SYMBOLS[move['player']]}  {text:<12} {render_brief(states[number])}")


def print_plies(plies):
    """Return a number of plies with its noun, for the lines that quote it."""
    return f"{plies} ply" if plies == 1 else f"{plies} plies"


def print_verdict(score, player, plies):
    """Return the sentence describing how a position ends with perfect play."""
    if score is None:
        return f"{SYMBOLS[player]} to play; the tablebase stores no value for it"
    if score > 0:
        return f"{SYMBOLS[player]} wins with perfect play, mate in {print_plies(plies)}"
    if score < 0:
        return f"{SYMBOLS[3 - player]} wins with perfect play, mate in {print_plies(plies)}"
    return f"the game is a draw with perfect play, in {print_plies(plies)}"


def print_move_count(evaluated, score):
    """Print how many of the legal moves keep the perfect result."""
    total = len(evaluated)
    if total <= 1 or score is None:
        return
    keeping = sum(1 for item in evaluated if item["score"] == score)
    if keeping == total:
        print(f"  Every one of the {total} legal moves keeps that result.")
    else:
        verb = "keeps" if keeping == 1 else "keep"
        print(f"  {keeping} of the {total} legal moves {verb} that result.")


def show_analysis(board, player, rules, show_board=True):
    """Print the best move and the perfect continuation of a position.

    Returns the tablebase entry, or None when the position is not stored.
    """
    board = np.asarray(board, dtype=int)
    result = outcome(board, rules)
    if result is not None:
        print(f"  The game is already over here: {describe_winner(board, result)}.")
        return None
    entry = lookup(board, player, rules)
    if entry is None:
        print()
        print(wrap(NOT_STORED))
        return None
    if show_board:
        print()
        print(render_board(board))
        print(f"  {SYMBOLS[player]} to play.")
    plan = best_move_and_line(board, player, rules, entry)
    print()
    if plan.plies is None:
        print(f"  {SYMBOLS[player]} to play; the tablebase stores no distance for it")
    else:
        print(f"  {print_verdict(plan.score, player, plan.plies)}")
    if plan.move is not None:
        print(f"  Best move: {format_move(plan.move)}")
    if not plan.complete and plan.evaluated:
        print("  (the tablebase is missing some of the moves, so this is from "
              "what it does hold)")
    print_move_count(plan.evaluated, plan.score)
    if plan.moves:
        print()
        print(f"  Perfect play, {print_plies(len(plan.moves))}:")
        print_line(plan.moves, plan.states)
    return entry


# ---------------------------------------------------------------------------
# Reading and playing
# ---------------------------------------------------------------------------

def lookup(board, player, rules):
    """Return the tablebase entry of a position, or None when it is not stored."""
    try:
        return get_solution.lookup_solution(board, player, **rules.lookup())
    except ValueError:
        return None


def evaluate(board, player, rules, count_traps=False):
    """Return every legal move of a position with its perfect value.

    This is the ranking the graphical interface uses: the value of a move says
    whether it wins, draws or loses for the side to move and how many plies the
    game still needs, and ``traps`` counts the replies of the opponent that lose
    the game for them. Counting the traps asks the tablebase once per reply of
    every move, so it is only worth it when the ranking needs it.

    Returns ``(moves, complete)``. The list is empty when the position is over or
    is not stored, and ``complete`` is False when the tablebase is missing some of
    the moves that were legal.
    """
    try:
        return computer_engine.evaluate_moves(
            np.asarray(board, dtype=int),
            player,
            rules.rotation,
            rules.transfer_allowed,
            TABLEBASE_DIR,
            count_traps=count_traps,
            extra_rotation_allowed=rules.extra_rotation_allowed,
        )
    except (ValueError, IndexError, KeyError):
        return [], False


def rank_best(evaluated):
    """Pick the best move of an evaluation list the way the interface does.

    The perfect value comes first: among the moves that win, the one that ends
    the game soonest, and among the moves that lose, the one that survives
    longest. Moves of the same value are then separated by how many of the
    opponent's replies lose the game for the opponent, so the winning side gives
    away as many chances to slip up as it can.
    """
    return computer_engine.best_evaluated_move(evaluated)


def best_move_and_line(board, player, rules, entry, evaluated=None):
    """Return the best move, the perfect game that follows it, and its value.

    ``thinking`` decides how the move is picked. In the ranked mode every legal
    move is valued and :func:`rank_best` chooses one of them, and the
    continuation is then read from the position that move leads to, so the line
    always starts with the move that was reported. In the first mode the stored
    line is used as it stands, which is one lookup instead of one per move.

    ``evaluated`` is the ``(moves, complete)`` pair of an evaluation the caller
    already has, so the moves are not walked twice. The evaluation comes back in
    the result for the same reason.
    """
    ranked = rules.ranks_moves()
    if evaluated is None:
        evaluated = evaluate(board, player, rules, count_traps=ranked)
    evaluated, complete = evaluated
    stored = entry["moves_sequence"]

    def plan_with(move, moves, states, score, plies):
        return Plan(move, moves, states, score, plies, evaluated, complete)

    if not ranked:
        return plan_with(entry["best_move"], stored, entry["states_sequence"],
                         entry["score"], len(stored))
    move = rank_best(evaluated)
    if move is None:
        # nothing could be valued, so fall back on the line the tablebase stored
        return plan_with(entry["best_move"], stored, entry["states_sequence"],
                         entry["score"], len(stored))
    item = next(candidate for candidate in evaluated if candidate["move"] is move)
    if stored and same_move(stored[0], move):
        # the stored line already starts with this move, so it is the line to
        # print. This is what settles a position that is down to the presses.
        return plan_with(move, stored, entry["states_sequence"], item["score"],
                         len(stored))
    following = olgf.play_turn(board, move)
    if following is None:
        return plan_with(move, stored, entry["states_sequence"], item["score"],
                         len(stored))
    child = lookup(following, 3 - player, rules)
    if child is None:
        # the move is proved, but there is no stored line behind it to print
        return plan_with(move, [], [], item["score"], item["moves_to_result"])
    if not child["moves_sequence"]:
        # the move ended the game on the spot, so the line is the move itself
        return plan_with(move, [move],
                         [np.asarray(board, dtype=int), np.asarray(following, dtype=int)],
                         item["score"], 1)
    # the line starts one ply earlier than the child's own, so the board the move
    # was played on goes in front of the states that were read. The length of the
    # line is what the verdict quotes, so the two can never fall apart.
    moves = [move] + child["moves_sequence"]
    return plan_with(move, moves,
                     [np.asarray(board, dtype=int)] + list(child["states_sequence"]),
                     item["score"], len(moves))


def first_move_answer(board, player, rules):
    """The answer the tablebase gives on its own, or None when it has none.

    The first move of the stored line, with no ranking of the other moves behind
    it, which is the quick mode the rules can ask for.
    """
    entry = lookup(board, player, rules)
    if entry is None or entry["best_move"] is None:
        return None
    return {
        "move": entry["best_move"],
        "source": SOURCE_FIRST,
        "score": entry["score"],
        "message": f"Tablebase (first move of the line): "
                   f"{format_move(entry['best_move'])}",
    }


def validate_move(board, player, parsed):
    """Check a parsed move against the position, returning the reason or None."""
    grid_size = board.shape[0]
    transfer, add = parsed
    cells = [add] if transfer is None else [transfer[0], neighbour(transfer[0], transfer[1]), add]
    for cell in cells:
        if not on_board(cell, grid_size):
            return f"{cell_name(cell)} is not on a {grid_size}x{grid_size} board."
    if olgf.board_is_full(board):
        return "The board is full, so there is no cell left to add a piece."
    after = board
    if transfer is not None:
        source, target = transfer[0], neighbour(transfer[0], transfer[1])
        opponent = 3 - player
        if board[source] != opponent:
            return f"{cell_name(source)} does not hold a piece of {SYMBOLS[opponent]}."
        if board[target] != 0:
            return f"{cell_name(target)} is not empty, so no piece can be transferred there."
        after = board.copy()
        after[target] = after[source]
        after[source] = 0
    if after[add] != 0:
        return f"{cell_name(add)} is not empty: a piece can only be added to an empty cell."
    return None


def play_best(board, player, rules):
    """Play the move the tool would recommend, or return None."""
    entry = lookup(board, player, rules)
    if entry is None:
        print()
        print(wrap(NOT_STORED))
        return None
    plan = best_move_and_line(board, player, rules, entry)
    if plan.move is None:
        print("  There is no move left to play.")
        return None
    if plan.plies is None:
        verdict = f"{SYMBOLS[player]} to play, and the tablebase holds no line"
    else:
        verdict = print_verdict(plan.score, player, plan.plies)
    note = "" if rules.ranks_moves() else "   (first tablebase move)"
    print(f"  {SYMBOLS[player]} plays {format_move(plan.move)}   ({verdict}){note}")
    return olgf.play_turn(board, plan.move)


def computer_move(board, player, rules):
    """Play the tablebase's choice for the computer and print what it did."""
    answer = None
    if not rules.ranks_moves():
        answer = first_move_answer(board, player, rules)
    if answer is None:
        answer = computer_engine.choose_move(
            board,
            player_turn=player,
            rotation=rules.rotation,
            transfer_allowed=rules.transfer_allowed,
            use_tablebase=True,
            fallback=computer_engine.FALLBACK_SEARCH,
            time_limit=COMPUTER_SECONDS,
            base_dir=TABLEBASE_DIR,
            extra_rotation_allowed=rules.extra_rotation_allowed,
        )
    move = answer.get("move")
    if move is None:
        print(wrap(f"  The computer has no move to make: {answer.get('message', '')}"))
        return None
    note = SOURCE_NOTES.get(answer.get("source"), answer.get("source"))
    following = olgf.play_turn(board, move)
    print(f"  {SYMBOLS[player]} plays {format_move(move)}   ({note})")
    print()
    print(render_board(following))
    return following


def undo(history, mode):
    """Step back in a game, returning False when there is nothing to undo."""
    played = len(history) - 1
    if played == 0:
        print("There is no move to undo.")
        return False
    steps = min(2 if mode != MODE_BOTH else 1, played)
    del history[len(history) - steps:]
    return True


def move_hint(rules):
    """Return the one-line reminder of how to type a move under these rules."""
    size = rules.grid_size
    add = cell_name((min(1, size - 1), min(1, size - 1)))
    if rules.transfer_allowed and size > 1:
        return (
            f"Type {add} to add a piece there, or {cell_name((0, 0))}"
            f"{cell_name((0, 1))} {add} to move an opponent's piece to a "
            "neighbouring cell first. "
        )
    return f"Type {add} to add a piece there. "


def play_game(rules, mode):
    """Play one game against the computer or with the user on both sides.

    Returns ``"over"`` when the game reached an end, and ``"left"`` when the
    user walked away from it before that.
    """
    board = np.zeros((rules.grid_size, rules.grid_size), dtype=int)
    history = [board]
    human = HUMAN_PLAYER[mode]
    print()
    print(f"New game on a {rules.grid_size}x{rules.grid_size} board. {MODE_LABELS[mode]}.")
    print(wrap(
        move_hint(rules)
        + "? analyses the position, * plays the perfect move, h lists the commands."
    ))
    print()
    print(render_board(board))
    shown = True      # whether the board printed above is the one being played
    while True:
        player = 1 if (len(history) - 1) % 2 == 0 else 2
        result = outcome(board, rules)
        if result is None and presses_pending(board, rules):
            # nobody has a move left, the presses are forced and they finish it
            result, pressed = play_presses(board, player, rules)
            history.extend(pressed)
            shown = True
            print()
        if result is not None:
            print()
            head = (f"Game over: {describe_winner(board, result)}, after "
                    f"{print_plies(len(history) - 1)}.")
            if shown:
                print(f"  {head} The board above is the final one.")
            else:
                print(f"  {head} The final board is:")
                print()
                print(render_board(board))
            return "over"
        if mode != MODE_BOTH and player != human:
            following = computer_move(board, player, rules)
            if following is None:
                return "left"
            board = following
            history.append(board)
            shown = True
            continue
        action, value = ask_in_game(board, player, rules, len(history))
        if action == "left":
            print("Left the game.")
            return "left"
        if action == "help":
            print()
            print(GAME_HELP)
            continue
        if action == "board":
            print()
            print(render_board(board))
            shown = True
            continue
        if action == "moves":
            print()
            moves, complete = evaluate(
                board, player, rules, count_traps=rules.ranks_moves())
            best = None
            entry = lookup(board, player, rules)
            if entry is not None:
                best = best_move_and_line(
                    board, player, rules, entry, (moves, complete)).move
            print_move_evaluation(board, player, rules, moves, complete, best)
            continue
        if action == "solve":
            show_analysis(board, value, rules, show_board=False)
            continue
        if action == "undo":
            if undo(history, mode):
                board = history[-1]
                print()
                print(render_board(board))
                shown = True
            continue
        if action == "play":
            following = play_best(board, player, rules)
        else:
            following = olgf.play_turn(board, value)
        if following is None:
            continue
        board = following
        history.append(board)
        # playing against the computer leaves the board off the screen until the
        # next prompt, so the ending has to print it if the game stops right here
        shown = False
        if mode == MODE_BOTH:
            print()
            print(render_board(board))
            shown = True


def solve_position(rules):
    """Ask for a position and print its best move and perfect continuation."""
    print()
    print(wrap(
        f"Type a position of {rules.grid_size * rules.grid_size} cells, row by row, "
        "either with the digits 0, 1 and 2 or with the symbols x, o and .; "
        "`empty` is the untouched board. The side to move follows from the pieces."
    ))
    while True:
        answer = prompt("position > ").strip()
        if answer.lower() in ("q", "quit", "menu", "back", "exit"):
            return
        if not answer:
            continue
        board = parse_state(answer, rules.grid_size)
        if board is None:
            print(f"  A {rules.grid_size}x{rules.grid_size} board is "
                  f"{rules.grid_size * rules.grid_size} cells of 0, 1, 2 or x, o, .")
            continue
        result = outcome(board, rules)
        if result is not None:
            print()
            print(render_board(board))
            print(f"  The game is already over here: {describe_winner(board, result)}.")
            print()
            continue
        player = infer_side_to_move(board)
        if player is None:
            print(wrap(
                "  No game leads to that position: the pieces have to alternate, "
                "starting with x, one per turn."
            ))
            continue
        show_analysis(board, player, rules)
        return


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the tool until the user leaves."""
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help", "help"):
        print("Start the tool with `python orbito_cli.py` and no arguments.")
        print()
        print(GAME_HELP)
        print(MAIN_HELP)
        return 0
    print_banner()
    try:
        rules, mode = ask_rules()
        action = "game"
        while True:
            choice = ask_next(REPEAT_LABELS[action])
            if choice is None:
                continue
            if choice == "quit":
                break
            if choice == "rules":
                rules, mode = ask_rules()
                action = "game"
                continue
            if choice in ("solve", "game"):
                action = choice
            print_rules(rules, mode)
            if action == "solve":
                solve_position(rules)
            else:
                play_game(rules, mode)
    except Quit:
        print()
    print("Bye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
