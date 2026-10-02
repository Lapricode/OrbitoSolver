# ORBITO (Orbital Logic Game)

An abstract two-player strategy game on an `n x n` grid, with a Pygame front-end, a computer
opponent, and tablebases that answer **every** position **exactly** — no heuristics, no
evaluation function. The computer plays perfectly on every grid size that has a tablebase.

Deeper technical notes on the solving machinery live in **[solver.md](solver.md)**.

---

## The game

Each cell is empty, White (`1`) or Black (`2`). A turn has up to three phases:

1. **Transfer** *(optional, if allowed)* — move one **opponent** piece to an orthogonally
   adjacent **empty** cell.
2. **Add** *(mandatory)* — place one **own** piece on an empty cell. The cell vacated by the
   transfer may be re-used.
3. **Rotate** — rotate the whole board `still`, `clockwise` or `counterclockwise`.

First to complete a full row, column or diagonal with their own pieces wins. A board where both
players hold a complete line is a **draw**.

If the board fills up without a line, the **endgame presses** decide it (this rule can be
switched off, in which case a full board without a line is a draw straight away). From then on
there are no transfers and no adds: the side to move presses the button, which turns every ring of
the board one step, up to **five** presses. The first press that leaves a complete line on the
board wins; if all five presses are spent and nobody ever holds a line, the game is a draw.

Because a press turns each ring by one step, the whole board is not turned rigidly: a ring
rotation does **not** carry every line onto a line, so a press can create the winning line out of
a position that held none.

The rotation is what makes the game interesting: the board you are looking at is never the
board you are reasoning about.

## Features

- **Pygame GUI** — two-player, vs-computer, and a position editor for arbitrary boards.
- **Grid sizes 1x1 to 10x10**, with configurable rotation, transfer and endgame-press rules per
  game.
- **Animated moves** — the transferred piece flies to its target, the added piece fades in, then
  everything slides one ring step during the rotation.
- Move builder, undo, restart, move log, winning-line highlighting, resizable window.
- **Two computer buttons** — "Hint (ask the computer)" reports the best move without playing it,
  and "Play Best Move" plays it. The editor offers the same pair, and a drawn position is always
  asked about on its own terms, without the endgame presses of the game played before it.
- **Endgame presses** — a full board without a line is finished by up to five forced presses,
  shown in the status bar as `endgame press 2 of 5`; clicking the board or "Complete Move" plays
  one, and the computer plays them on its own.
- **Computer opponent** with four policies: tablebase-else-random, tablebase-else-search
  (default), random only, search only.
- **Background engine process** — searches never freeze the UI, and replies are held back until
  a minimum "thinking time" so the computer doesn't feel instantaneous.
- **Retrograde tablebase** — built and checked in for 1x1 to 4x4, 77 MiB for both settings of
  the endgame rule.
- **Cross-verification** — stored values are re-checked against the independent search engine.
- **Monte-Carlo statistics** — vectorised random-play win/draw probabilities.

## Installation

CPython 3.12 (tested on 3.12.3, Linux x86_64).

```bash
python3 -m venv orbvenv
orbvenv/bin/pip install -r requirements.txt
orbvenv/bin/python gui.py
```

| Dependency | Version | Used by |
| --- | --- | --- |
| `numpy` | 2.5.3 | board arrays, search buffers, value tables |
| `pygame` | 2.6.1 | `gui.py` |
| `tqdm` | 4.67.1 | progress bars (optional) |

## Usage

```bash
python gui.py                       # graphical interface
python orbito_cli.py                # interactive command line: solve positions and play games

# Look up a position ("0012" is a 2x2 board)
python get_solution.py 0012 -p 1 -r clockwise --transfer-allowed --endgame-presses

# Build a tablebase (one grid size per run, both keep the presses on and off)
python retrograde_tablebase.py -n 4 --workers 8     # value tables (fast, recommended)
python create_game_tablebase.py -n 3                # symmetry-reduced JSON records

# Build, then cross-check against the search engine
python retrograde_tablebase.py -n 4 --verify 4
python retrograde_tablebase.py -n 4 --verify 8 --plies 12-16   # 4x4: the late layers only

# Full search from scratch, and random-play statistics
python solve_game.py
python test_game.py
```

Re-searching a 4x4 from the empty board takes hours, so the cross-check of the big grid is worth
running on its late layers (`--plies`), where every sample is still proved against the independent
search engine in seconds.

### Interactive command line

`python orbito_cli.py` asks for the rules of the game one question at a time, every question with a
default, and then answers from the value tables:

- `s` solves a position: type it as digits (`0012`), as symbols (`xo..`), or `empty` for a fresh
  board. It prints the best move, how many of the legal moves keep that result, and the perfect
  game that follows, ply by ply, to its end.
- `g` plays a game, with the user on `x`, on `o`, or on both sides. Against the computer the moves
  come from the same tables, so every game is played perfectly.
- `?` asks for the best move and the perfect continuation at any point of a game, `*` plays the
  recommended move, `m` values every legal move, `u` undoes, `q` goes back.
- `r` answers the rules again and starts over, `q` leaves.

Moves are typed as the cell to add a piece on, optionally preceded by a transfer of an opponent's
piece: `c3` adds at c3, `a1b1 c3` moves the piece on a1 to the neighbouring cell b1 and adds at
c3, and `a1r c3` does the same with a direction letter (`u`, `d`, `l`, `r`).

A full board without a line is not the end: the forced Orbito presses are played out and the game
is decided by the first press that shows a line, or after the fifth one. Every game ends by naming
the result and showing the board it ended on, so the winning line can be read straight off it.

#### Which move is the best one

Several moves usually reach the same result, so the rules ask how to choose between them, and the
answer changes both the moves the computer plays and the move the tool recommends:

- **r** (the default) ranks every legal move exactly as `computer_engine.best_evaluated_move` does
  for the Pygame front-end, through the key `computer_engine.move_rank_key`. The perfect value comes
  first: among the moves that win, the one that ends the game soonest, and among the moves that lose,
  the one that survives longest. Moves of the same value are then separated by the replies they leave
  the opponent, counted from the opponent's side, in this order:

  1. the most replies that **lose** for the opponent, so the side that wins hands over as many
     chances to slip up as it can;
  2. then the fewest replies that **win** for the opponent, which leaves them the least room to win;
  3. and last the most replies that **draw** for the opponent, the ways they could still hold on.

  The counts are structural, and that decides how much each one can do: a move that wins leaves the
  opponent no reply that draws or wins, so only the first criterion ever applies to it, and a move
  that draws leaves them no reply that wins, so the third is what separates those. All three say
  something only where the value itself is equal, and for a move that loses they are the only things
  left to say. `m` shows the whole ranking, with the result, the length of the game and the three
  reply counts for every move, in that order.
- **f** takes the first move of the stored principal variation and does not rank the moves around it.
  On a 4x4 that is about ten times quicker, at the price of picking whichever of the equally good
  moves the table happens to store first.

Either way the result of the game is the one the tablebase proves; only the choice between the
moves that reach it changes.

Every rule context is a separate table directory, and every lookup asks for one of them:

```
retrograde_game_tablebase/<n>x<n>/<rotation>/<transfer_rule>/<endgame_presses>/retrograde.npz
compressed_game_tablebase/<n>x<n>/<rotation>/<transfer_rule>/<endgame_presses>/<player>/completion_<pieces>.json
```

`counterclockwise` is answered from the `clockwise` table by reflection. The endgame-press flag
has the same meaning everywhere: `extra_rotation_allowed` (on) or `extra_rotation_not_allowed`
(off). Both tables are checked in for the current layout, so a lookup is exact out of the box.

A report names the position it is about by its **state alone**: one digit per cell, row by row,
so a 4x4 position reads as `0012122111222121`. The rules and the side to move are not part of the
name — they are what the tablebase directory is for:

```
- Position ID:         0012122111222121
```

The reported line always runs to the end of the game. A position that reaches a full board without
a line is followed through the presses, and stops on the press that makes a line, or after the
fifth press:

```
State after move 2:                 # the board is now full and still has no line
Move 3 (o) : {... 'rotation_only': True} (endgame press 1 of 5)
State after move 3:                 # still no line: the game is not over yet
Move 4 (x) : {... 'rotation_only': True} (endgame press 2 of 5)
State after move 4:                 # x holds a line: this is the end of the game
```

Python API:

```python
import numpy as np, get_solution, computer_engine, solve_game
import orbital_logic_game_functions as olgf

# Exact lookup (O(1) with a tablebase hit)
board = np.array([[1,0,0],[2,0,0],[0,0,0]])          # Black to move
sol = get_solution.lookup_solution(board, player_turn=2,
                                   rotate_direction="clockwise", transfer_allowed=True,
                                   extra_rotation_allowed=True)
print(sol["score"], sol["game_result"])                # 995 player2_wins

# Play the perfect move
board = olgf.play_turn(board, sol["best_move"])

# From-scratch search
move, value = solve_game.find_best_move(board, rotate_direction="clockwise", player=1)

# Computer policy layer
res = computer_engine.choose_move(board, player_turn=1, rotation="clockwise",
                                  transfer_allowed=True, time_limit=5.0)
print(res["source"], res["move"], res["message"])
```

The endgame rule is a parameter everywhere (`extra_rotation_allowed`, on by default), and the
number of presses already played is passed as `extra_turns`; the board alone cannot say how far
into the presses a position is:

```python
# a full board without a line: the answer is a forced press, then the value
full = np.array([[1, 1, 2], [2, 2, 1], [1, 2, 1]])
move = olgf.rotation_only_move(1, "clockwise")
print(move["rotation_only"], move["add"], move["transfer"])          # True None None
res = computer_engine.choose_move(full, player_turn=1, rotation="clockwise",
                                  use_tablebase=False, fallback="search",
                                  time_limit=1.0, extra_turns=3)
print(res["source"], res["presses_remaining"])         # endgame 2
```

## How it is solved (in brief)

Full details in **[solver.md](solver.md)**.

Every turn adds exactly one piece, so **the number of occupied cells equals the number of
plies played**. The game graph is therefore strictly layered by occupancy and cycle-free, which
lets every value be computed in a single bottom-up pass.

`retrograde_tablebase.py` does exactly that: for each rule context it enumerates the reachable
boards layer by layer from the full board down, pruning children of finished games, and folds
each position's value out of the layer above. Because the ply index *is* the occupancy, the
mate score `WIN_SCORE - plies_to_terminal` is a property of the position alone — the distance to
the end of the game falls out of the sweep for free, so the tables store the value *and* the
optimal line length. Every `(board, side to move)` pair is evaluated exactly once for the whole
context, instead of once per referring position as a normal search does.

The bottom layer of that sweep is the endgame. A full board without a line has no move to
generate, so when the presses are allowed it is finished by replaying them instead: at most five
ring rotations, and the value of the position is whatever the last one leaves on the board
(`WIN_SCORE - presses_needed` for a line, `0` for a draw). The presses do not occupy cells, so
that counter lives beside the board, never inside it — which is why the tables only store the
position before the first press and the rest is replayed on lookup.

Positions are stored in two dense arrays indexed by the base-3 code of the row-major board, so
a query is a single array index. Two isomorphisms halve the data again: a colour swap means
player-2-to-move positions are answered from the swapped board, and a reflection maps the
clockwise game onto the counterclockwise one, so only `still` and `clockwise` tables are built.

The whole 4x4 value tablebase, both settings of the endgame rule, is 77 MiB — a few per cent of
what the equivalent per-position JSON records would need, and it is what makes 4x4 playable.

Fallbacks, used automatically when a position is not in a table:

| Tier | Source | Notes |
| --- | --- | --- |
| 1 | Retrograde value table | exact, O(1) |
| 2 | Symmetry-reduced JSON tablebase | exact, ~15x smaller than one tablebase per position |
| 3 | Negamax + alpha/beta + transposition table | exact but depth/time-bounded |

## Precomputed tablebases

The value tables in `retrograde_game_tablebase/`, all eight contexts of each grid size:

| Grid | Table entries | Positions stored per context | Size per context | Total |
| --- | --- | --- | --- | --- |
| 1x1 | 3 | 2 | 0.8 KiB | 8 KiB |
| 2x2 | 81 | 29 | 0.9 KiB | 8 KiB |
| 3x3 | 19,683 | 5,478 - 6,034 | 7.3 - 8.3 KiB | 64 KiB |
| 4x4 | 43,046,721 | 9,721,176 - 10,161,173 | 5.3 - 13.4 MiB | 77.4 MiB |

and the JSON records in `compressed_game_tablebase/`:

| Grid | Files | Records | Total |
| --- | --- | --- | --- |
| 1x1 | 16 | 24 | 10 KiB |
| 2x2 | 40 | 180 | 100 KiB |
| 3x3 | 80 | 31,524 | 32.5 MiB |

Eight contexts per grid size (`still` and `clockwise` x transfer allowed / not allowed x endgame
presses on / off), so twice as many files and roughly twice the data of a single rule. `still` and
`clockwise` are the only rotations that are built: `counterclockwise` is served from `clockwise`
by reflection. `positions stored` counts the positions a context actually reaches — far fewer than
the table entries, because a finished game is never expanded further — and the compressed records
store one entry per class of symmetric boards instead.

These files are built by the two generators above and match the current code, so `get_solution`
answers 1x1 to 4x4 exactly and the GUI reports the grid sizes it can play on. Rebuild them with

```bash
python retrograde_tablebase.py -n 4 --workers 8        # 1x1 to 4x4, ~77 MiB
python create_game_tablebase.py -n 3 --workers 8       # 1x1 to 3x3, ~33 MiB
```

## Repository layout

| File | Role |
| --- | --- |
| `orbital_logic_game_functions.py` | rules engine: evaluation, rotation, move generation, `play_turn`, state conversions, board statistics |
| `solve_game.py` | negamax + alpha/beta + transposition table solver, PV reconstruction, iterative deepening, parallel root |
| `retrograde_tablebase.py` | retrograde value-table builder, reader and verifier |
| `create_game_tablebase.py` | compressed JSON tablebase generator and the shared symmetry helpers |
| `get_solution.py` | unified lookup API and CLI over the tablebase sources |
| `computer_engine.py` | policy layer (tablebase → endgame presses → search → random) and the background engine process |
| `gui.py` | the entire Pygame front-end |
| `orbito_cli.py` | interactive command line: solve any position, or play a game, straight from the value tables |
| `test_game.py` | vectorised Monte-Carlo random-play statistics |
| `solver.md` | detailed notes on the solving machinery |
| `symmetry_observations.md` | design notes behind the compressed tablebase |

| Directory | Contents |
| --- | --- |
| `retrograde_game_tablebase/` | `retrograde.npz` value tables, **1x1 to 4x4** |
| `compressed_game_tablebase/` | symmetry-reduced JSON records, **1x1 to 3x3** |

The full per-position JSON tablebase that used to live in `game_tablebase/` has been removed: the
compressed records hold the same values and lines in a fraction of the space, so the full variant
was neither generated nor read any more.

## Credits

This project was built and finished with substantial help from
**[opencode](https://opencode.ai)**, the open-source AI coding agent. Most of the code here was
written, debugged, refactored and completed in collaboration with opencode; the commit history
records the point directly at `b05bc3b`, *"made the code much faster using opencode"*, the
commit where the solver went from impractical to usable. The performance work that followed —
numpy vectorisation, the symmetry compression, the retrograde sweep, the multiprocessing
layers, the parallel search root, the cross-verification harness and most of the front-end — was
carried out with opencode driving the implementation. The game design and the solving
algorithms are the author's; opencode implemented them.

Thanks also to the developers of **NumPy**, **Pygame** and **tqdm**.

## Disclaimer

This is an independent programming exercise created for educational purposes. It is not affiliated
with or endorsed by the original game or its developers/publishers. References to the game are
based on publicly available information, and all related intellectual property belongs to its
respective owners.

## License

MIT — Copyright (c) 2025 Printzios Lampros. See [LICENSE](LICENSE).
