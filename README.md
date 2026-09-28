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
players hold a complete line, or a full board with no line, is a **draw**.

The rotation is what makes the game interesting: the board you are looking at is never the
board you are reasoning about.

## Features

- **Pygame GUI** — two-player, vs-computer, and a position editor for arbitrary boards.
- **Grid sizes 1x1 to 10x10**, with configurable rotation and transfer rules per game.
- **Animated moves** — the transferred piece flies to its target, the added piece fades in, then
  everything slides one ring step during the rotation.
- Move builder, undo, restart, move log, winning-line highlighting, hints, resizable window.
- **Computer opponent** with four policies: tablebase-else-random, tablebase-else-search
  (default), random only, search only.
- **Background engine process** — searches never freeze the UI, and replies are held back until
  a minimum "thinking time" so the computer doesn't feel instantaneous.
- **Retrograde tablebase** — complete for 1x1 to 4x4, ~38 MB.
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

# Look up a position ("0012" is a 2x2 board)
python get_solution.py 0012 -p 1 -r clockwise --transfer-allowed

# Build a tablebase
python retrograde_tablebase.py -n 4 --workers 8     # value tables (fast, recommended)
python create_game_tablebase.py -n 3 --tablebase compressed

# Build, then cross-check against the search engine
python retrograde_tablebase.py -n 4 --verify 4

# Full search from scratch, and random-play statistics
python solve_game.py
python test_game.py
```

Python API:

```python
import numpy as np, get_solution, computer_engine, solve_game
import orbital_logic_game_functions as olgf

# Exact lookup (O(1) with a tablebase hit)
board = np.array([[1,0,0],[2,0,0],[0,0,0]])          # Black to move
sol = get_solution.lookup_solution(board, player_turn=2,
                                   rotate_direction="clockwise", transfer_allowed=True)
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

Positions are stored in two dense arrays indexed by the base-3 code of the row-major board, so
a query is a single array index. Two isomorphisms halve the data again: a colour swap means
player-2-to-move positions are answered from the swapped board, and a reflection maps the
clockwise game onto the counterclockwise one, so only `still` and `clockwise` tables are built.

This is 38 MB where the equivalent JSON records could be tens of GB, and it is what makes 4x4 playable.

Fallbacks, used automatically when a position is not in a table:

| Tier | Source | Notes |
| --- | --- | --- |
| 1 | Retrograde value table | exact, O(1) |
| 2 | Symmetry-reduced JSON tablebase | exact, ~15x smaller than the full one |
| 3 | Negamax + alpha/beta + transposition table | exact but depth/time-bounded |

## Precomputed tablebases

| Grid | Table entries | Positions stored | Size |
| --- | --- | --- | --- |
| 1x1 | 3 | 2 | <1 kB |
| 2x2 | 81 | 29 | <1 kB |
| 3x3 | 19,683 | 5,478 - 6,034 | <1 kB |
| 4x4 | 43,046,721 | 9,721,176 - 10,161,173 | 5.6 - 12.5 MB |

Four contexts per grid size (`still` and `clockwise` x transfer allowed / not allowed),
16 files, ~38 MB total. `counterclockwise` is served from `clockwise` by reflection.

## Repository layout

| File | Role |
| --- | --- |
| `orbital_logic_game_functions.py` | rules engine: evaluation, rotation, move generation, `play_turn`, state conversions, board statistics |
| `solve_game.py` | negamax + alpha/beta + transposition table solver, PV reconstruction, iterative deepening, parallel root |
| `retrograde_tablebase.py` | retrograde value-table builder, reader and verifier |
| `create_game_tablebase.py` | JSON tablebase generator (full + compressed) and the shared symmetry helpers |
| `get_solution.py` | unified lookup API and CLI over all tablebase sources |
| `computer_engine.py` | policy layer (tablebase → search → random) and the background engine process |
| `gui.py` | the entire Pygame front-end |
| `test_game.py` | vectorised Monte-Carlo random-play statistics |
| `solver.md` | detailed notes on the solving machinery |
| `symmetry_observations.md` | design notes behind the compressed tablebase |

| Directory | Size | Contents |
| --- | --- | --- |
| `retrograde_game_tablebase/` | 38 MB | `retrograde.npz` value tables, **1x1 to 4x4** |
| `compressed_game_tablebase/` | 17 MB | symmetry-reduced JSON, **1x1 to 3x3** |
| `game_tablebase/` | 248 MB | full JSON, **1x1 to 3x3** |

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

## License

MIT — Copyright (c) 2025 Printzios Lampros. See [LICENSE](LICENSE).
