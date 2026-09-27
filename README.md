# ORBITO (Orbital Logic Game)

A complete, exactly-solved two-player abstract strategy game for the `n x n` grid, with a
Pygame front-end, a computer opponent, three independent solver back-ends and a
cross-verified retrograde tablebase for grids up to 4x4.

Every position the computer is ever asked about in this repository is answered **exactly**,
not heuristically. There is no evaluation function, no material counting, no hand-crafted
weighting — for every board size that has a tablebase, the program knows the game-theoretic
value of every reachable position and can therefore play perfectly. The interesting part of
this repository is *how* that is achieved, which is described at length in
[How the game is solved](#how-the-game-is-solved).

---

## Table of contents

- [The game](#the-game)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Graphical interface](#graphical-interface)
  - [Command line](#command-line)
  - [Python API](#python-api)
- [How the game is solved](#how-the-game-is-solved)
  - [The structural observation that makes it tractable](#the-structural-observation-that-makes-it-tractable)
  - [Tier 1 — Retrograde analysis](#tier-1--retrograde-analysis)
  - [Board coding: base-3 dense tables](#board-coding-base-3-dense-tables)
  - [The sweep: values and distances in one pass](#the-sweep-values-and-distances-in-one-pass)
  - [Storing only half the table: the colour-swap isomorphism](#storing-only-half-the-table-the-colour-swap-isomorphism)
  - [Storing one rotating context: the reflection isomorphism](#storing-one-rotating-context-the-reflection-isomorphism)
  - [Reconstructing the principal variation](#reconstructing-the-principal-variation)
  - [Verification against an independent solver](#verification-against-an-independent-solver)
  - [Tier 2 — Symmetry-reduced JSON tablebase](#tier-2--symmetry-reduced-json-tablebase)
  - [Tier 3 — On-the-fly search](#tier-3--on-the-fly-search)
  - [Parallelism and performance](#parallelism-and-performance)
- [Repository layout](#repository-layout)
- [Precomputed tablebases](#precomputed-tablebases)
- [Development notes](#development-notes)
- [Credits](#credits)
- [License](#license)

---

## The game

Orbital Logic is played on an `n x n` board. Each cell is **empty**, holds a **Black** piece
(`1`) or a **White** piece (`2`).

A turn consists of up to three phases:

1. **Transfer** *(optional)* — move exactly **one opponent piece** to an orthogonally
   adjacent **empty** cell. This phase is skipped entirely when the transfer rule is off.
   Because the piece belongs to the opponent, this is how a player reshapes the opponent's
   own line.
2. **Add** *(mandatory)* — place **one own piece** on an empty cell. The cell vacated by the
   transfer may be re-used, and the transfer's own source cell is a legal target.
3. **Rotation** — rotate the whole board `still`, `clockwise` or `counterclockwise`.

**Win condition.** The first player to complete a full row, column or diagonal with their own
pieces wins. If both players hold a complete line simultaneously the position is scored as a
**draw**, and so is a full board with no line at all.

The rotation is what gives the game its name and its character: the board you reason about is
never quite the board you are looking at, so threats have to be tracked through a permutation
that is applied after every single ply.

---

## Features

### Playing

- **Pygame GUI** (`gui.py`) with a retained-mode widget toolkit written from scratch:
  `Label`, `InputBox`, `Button`, `Checkbox`, `RadioButton`, `TextPanel` and a scrollable
  `Panel` column layout.
- **Three modes** from the menu screen: *2 Players*, *Vs Computer* and a *Position Editor* for
  loading and inspecting arbitrary mid-game positions.
- **Grid sizes 1x1 up to 10x10**, selectable from the menu or the position editor.
- **Configurable rules** per game: rotation direction (*Still* / *Clockwise* /
  *Counterclockwise*) and whether transfer moves are allowed.
- **Animated moves** — the transferred piece physically flies to its target, the added piece
  fades and scales in, and then every piece slides one ring step during the rotation phase. The
  animation derives its cell map by rotating a board of *markers* through the engine's own
  `rotate_board`, so the visuals cannot drift from the rules. Can be switched off.
- **Move builder state machine** — click the opponent piece, then the empty target, then the
  cell to add to. **Manual** or **auto** move completion.
- **Undo** (`Take Back`), `Restart`, `New Game`, a scrolling move log, hover highlight, cell
  legend and a status bar.
- **Hint button** — asks the engine for a perfect move and highlights it, or opens the full
  report in the editor.
- **Winning lines** are detected and drawn for every completed line, and a filled board with
  no line is reported as a draw.
- **Resizable window** that keeps the board square.
- **Non-blocking engine** — the engine lives in a separate daemon process, so a 4x4 search
  can never freeze the UI. Answers are held back until a 1 second minimum "thinking time" and
  the board animation have both finished, so the computer never feels instantaneous.

### Solving

- **Retrograde tablebase** (`retrograde_tablebase.py`) — one bottom-up sweep per rule context;
  every position in the context is evaluated exactly once for the entire context. Complete for
  1x1, 2x2, 3x3 and 4x4. Only the `still` and `clockwise` contexts are built, because a
  reflection of the board answers the counterclockwise game; 4x4 costs **37.5 MB** and the three
  smaller grids together **50 kB**.
- **Symmetry-reduced JSON tablebase** (`create_game_tablebase.py`) — canonical
  representative per symmetry orbit, 4x smaller than the naive table, complete for 1x1 to 3x3
  (4x4 generation exists but is gitignored and was left incomplete).
- **Full JSON tablebase** (`create_game_tablebase.py --tablebase full`) — every one of the
  `3^(n^2)` boards per context, storing score, result, best move and the full move/state
  sequences. Complete for 1x1 to 3x3. Kept as a reference and as a fallback source.
- **Negamax + alpha/beta + transposition table** solver (`solve_game.py`) with an optional
  parallel root and a wall-clock-limited iterative-deepening wrapper.
- **Cross-verification** (`retrograde_tablebase.py --verify`) — samples stored positions per
  layer, re-solves them with the independent search engine, replays the reconstructed line
  through the rules module, asserts the stored distance matches, and cross-checks the other
  rotating context against the mirror. Exits non-zero on any mismatch.
- **Monte-Carlo statistics** (`test_game.py`) — vectorised batched random play producing
  per-start-player win/draw probabilities, plus a slow scalar reference loop for comparison.
- **Board statistics** (`print_game_statistics`) — Burnside-style counts of how many distinct
  boards exist and how many survive symmetry reduction.

### Support

- Board/state <-> base-3 string <-> integer conversions, rotation helpers, and
  programmatically find all rotationally symmetric states.
- `tqdm` progress bars with a dependency-free fallback.
- Multiprocessing everywhere it matters (tablebase generation, retrograde layers, parallel
  search root) with sensible chunking and an automatic serial fallback.
- Small on-disk table cache keyed by `(path, mtime, size)` with FIFO eviction, so repeated
  queries do not re-read the `.npz` files. The four most recently used tables are kept, which
  covers a game at a fixed rule context.

---

## Installation

Requires CPython 3.12 (tested on 3.12.3, Linux x86_64).

```bash
python3 -m venv orbvenv
orbvenv/bin/pip install -r requirements.txt
```

| Dependency | Version | Used by |
| --- | --- | --- |
| `numpy` | 2.5.3 | board arrays, search buffers, dense value tables |
| `pygame` | 2.6.1 | `gui.py` |
| `tqdm` | 4.67.1 | progress bars during tablebase generation (optional) |

```bash
orbvenv/bin/python gui.py
```

---

## Usage

### Graphical interface

```bash
python gui.py
```

The menu lets you pick the grid size, the rotation rule, the transfer rule, the mode, the
engine configuration and the search time limit. The engine radio offers four policies:

| Policy | Behaviour |
| --- | --- |
| Tablebase, else random move | exact when a tablebase hit exists, random otherwise |
| **Tablebase, else search** *(default)* | exact when a tablebase hit exists, time-limited search otherwise |
| Random move only | always random |
| Search only | always a live search, never consults a tablebase |

The **Position Editor** mode takes this further: draw any position with the mouse (left click
places the selected colour, right click erases, middle click places the other colour), set the
side to move and the rules, then press *Ask Computer* to get the full text report — the
"Game rules / Initial game state / Start player" block followed by the perfect game evolution,
board by board.

The engine is a policy layer over the back-ends rather than a solver of its own. On every move
it asks `retrograde_game_tablebase/` first — the value tables are exact for every reachable
position and answer with a single array index — and falls back to `compressed_game_tablebase/`
for any position a value table does not cover, and only then to the configured fallback. The
`--rotations` reduction is invisible from the GUI: selecting *Counterclockwise* is answered from
the clockwise table by reflecting the board and reflecting the line back.

### Command line

```bash
# Perfect play from the empty 4x4 board (see solve_game.py for the hard-coded demo)
python solve_game.py

# Build a tablebase
python create_game_tablebase.py -n 3 --tablebase compressed --workers 8
python create_game_tablebase.py -n 3 --tablebase full

# Build the retrograde value tables, then verify them.
# --base-dir matters: with no argument the tables land in compressed_game_tablebase/,
# the JSON fallback directory, not in the retrograde_game_tablebase/ the engine reads first.
# The default --rotations is "still,clockwise": a reflection answers the counterclockwise
# game, so only two rotation contexts have to be stored.
python retrograde_tablebase.py -n 4 --base-dir retrograde_game_tablebase --workers 8
python retrograde_tablebase.py -n 4 --base-dir retrograde_game_tablebase --verify 4
python retrograde_tablebase.py -n 4 --base-dir retrograde_game_tablebase --rotations all  # redundant full set

# Look up a position ("0012" is a 2x2 board)
python get_solution.py 0012 -p 1 -r clockwise --transfer-allowed

# Random-play statistics
python test_game.py
```

### Python API

```python
import numpy as np
import solve_game
import get_solution
import computer_engine

# 1. Single position, exact, O(1) when a tablebase exists
answer = get_solution.get_solution("0012", player_turn=1,
                                  rotate_direction="clockwise",
                                  transfer_allowed=True)
print(answer)          # "Game rules ... Perfect game evolution ..." as text

# 2. Structured answer: score, result and a directly playable best move
sol = get_solution.lookup_solution(np.array([[0,0,0],[1,0,0],[0,1,0]]), player_turn=1,
                                   rotate_direction="clockwise", transfer_allowed=True)
board = np.array([[0,0,0],[1,0,0],[0,1,0]])
board = orbital_logic_game_functions.play_turn(board, sol["best_move"])

# 3. Full search, from scratch
move, value = solve_game.find_best_move(board, rotate_direction="clockwise", player=2)

# 4. Computer policy layer
result = computer_engine.choose_move(board, player_turn=1,
                                     rotation="clockwise", transfer_allowed=True,
                                     fallback=computer_engine.FALLBACK_SEARCH,
                                     time_limit=5.0)
print(result["source"], result["move"], result["message"])
# -> tablebase {'player': 1, 'transfer': None, 'add': (0, 0), 'rotate': 'clockwise'}
#    Tablebase (perfect play): add (0,0) -> Black wins
```

`computer_engine.EngineProcess` exposes the same thing as a long-lived worker process if you
want it off the main thread.

---

## How the game is solved

This is the core of the repository. There are three back-ends, and at query time the
computer tries them in order of decreasing speed and increasing cost: the retrograde value
table, the symmetry-reduced JSON tablebase, and finally a live search. The first two are
exact; the third is exact as well but bounded by depth or wall-clock time.

### The structural observation that makes it tractable

Every turn adds exactly one piece to the board:

- a **transfer** moves a piece to an empty cell, so it preserves the number of occupied cells;
- the mandatory **add** increases it by one;
- the **rotation** only permutes cells.

So starting from the empty board, **the number of occupied cells is exactly the number of
plies played**. The game graph is therefore *strictly layered by occupancy* and **cannot
contain a cycle**. This is the single fact the whole solver is built on. It means:

- there is a well-defined ply index, and it is readable straight off the board;
- values can be computed in one bottom-up pass instead of requiring fixpoint iteration;
- mate scores do not need any depth bookkeeping, because the ply index *is* the depth.

`create_game_tablebase.py` already walks those layers from the full board downwards — but it
solves every position from scratch, so each board re-searches its entire subtree. The same
waste happens inside a single search: a transposition table collapses repeated positions,
but every distinct position is still expanded once per search. The retrograde module removes
that waste entirely.

### Tier 1 — Retrograde analysis

`retrograde_tablebase.py` runs **one retrograde sweep per rule context**, where a rule context
is a `(grid size, rotation direction, transfer rule)` triple. Within a context the algorithm
has three phases.

**Phase 1 — enumerate the reachable layers, deepest first.**

```python
layers = enumerate_reachable(grid_size, rotation, transfer_allowed)
# layers[ply] is a sorted np.int64 array of base-3 board codes; layers[0] == [0]
```

The enumeration descends from the full board. Crucially, it **prunes the children of terminal
positions**: if a board already has a completed line, nothing below it can ever occur. Without
that pruning the enumeration also produces boards where one player keeps placing pieces after
the opponent has already won — boards the rules score as a draw, but which no real game can
reach. The result is that the table contains only boards that can genuinely occur in a game
played from the empty board.

For the 4x4 `still` + transfer context the layers are:

| ply | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| positions | 1 | 16 | 240 | 1680 | 10920 | 43680 | 160160 | 400400 | 900900 | 1441440 | 2018016 | 2018016 | 1681632 | 960640 | 409920 | 101624 | 11888 |

That is 10,161,173 of the 43,046,721 boards a 4x4 grid admits. The count is not a filter applied
afterwards — at ply *p* a game can only hold `ceil(p/2)` player-1 pieces, which already rules
out 76% of the boards, and terminal pruning removes a further 4,606 that could only be reached by
playing on after somebody had already won. Note that the middle layers sit at the binomial
maximum while the deepest layers are truncated: that is exactly the terminal-position pruning at
work.

**Phase 2 — sweep bottom-up.**

```python
score, dtx = sweep_layers(codec, layers)
```

Iterating `ply` from the deepest layer down to 0, each position's value is folded from the
values of its children, which the previous layer has already determined. Every
`(board, side to move)` pair in the context is therefore evaluated **exactly once for the whole
context**, no matter how many other positions refer to it as a child.

**Phase 3 — store dense arrays.**

```python
np.savez_compressed(file, score=score, dtx=dtx, metadata=np.array(json.dumps(metadata)))
```

### Board coding: base-3 dense tables

Positions are stored in two dense flat arrays indexed by the base-3 code of the row-major
board, so a lookup is a single array index rather than a scan over a JSON array:

```python
code = sum(cell * 3 ** (cell_count - 1 - index) for index, cell in enumerate(row_major_board))
```

The `score` array is `int16` with `UNSOLVED = 32767` marking slots that are not stored (which
also makes "unreachable" and "not computed" impossible to confuse), and `dtx` is `uint8`
holding the distance to the terminal node. The cell count is capped at 255 for exactly that
reason — the distance has to fit in a single byte.

The array is dense over **all** `3^(n^2)` boards, not just the reachable ones, so a lookup never
has to search for the right slot. That looks wasteful and is not: at 3 bytes per slot the 4x4
array is 123.2 MB of raw data, and 76.4% of those slots hold nothing but the `UNSOLVED` sentinel
— identical bytes, which `np.savez_compressed` deflates to 123 kB. Add the fact that the values
themselves are barely information (the whole 4x4 table contains 12 distinct score values: `0` for
the 4,607,791 draws, and `±(1000 - d)` with `d <= 9`) and the 123 MB lands as an 8.78 MB file.
That is 7.1% of raw, and **0.91 bytes per stored position**.

The JSON records work out to about 1.1 kB apiece (247 MB for the 236,196 records of 3x3 in full
mode), so per position the value tables are roughly **1,200x smaller** while answering the same
question. That ratio is why a 4x4 JSON tablebase was never finished — extrapolating the same
1.1 kB over the 39.8M positions of the full six-context set gives tens of gigabytes, against
37.5 MB for the four stored contexts.

The non-obvious part is that computing a child code is not a re-encode. The codec precomputes
a per-cell weight table, and then also the weight each cell takes *after* the rotation:

```python
weights[k]            = 3 ** (cell_count - 1 - k)
destination_weights[k] = weights[inverse_perm[k]]     # the weight k ends up with post-rotation
```

Every position is then split once into a `base` (the part of the code a move cannot touch) and
a `destination_weights` slice, and generating a child is a handful of integer additions:

```python
def child_code(base, destination_weights, move, player):
    code = base + player * destination_weights[move.add_cell]
    if move.transfer:
        code += opponent * (destination_weights[move.target] - destination_weights[move.source])
    return code
```

The transfer term is the elegant bit: the source cell loses the opponent's value and the
target cell gains it, and because the rotation is already folded into `destination_weights`,
both effects are captured by one difference. Move generation itself is driven by
`solve_game._engine_params`, which precomputes the `2n+2` winning lines, the lines through each
cell and the four neighbours of each cell.

### The sweep: values and distances in one pass

`combine_children` folds the `(child_score, child_dtx)` pairs — expressed from the *child's*
side to move — into a value for the current position:

| child (from the child's point of view) | meaning for us | fold |
| --- | --- | --- |
| any negative score | we **win** | keep the **smallest** `dtx`; score = `WIN_SCORE - (dtx+1)` |
| otherwise any positive score | we **lose** | keep the **largest** `dtx`; score = `(dtx+1) - WIN_SCORE` |
| otherwise all draws | draw | keep the smallest `dtx`; score = `0` |
| no children at all | nothing to move | score = `0` |

This is negamax — a losing child for the opponent is a win for us, with the sign flipped — but
the asymmetry in *which* `dtx` survives is what makes the output a full solution rather than
just a value. Among winning continuations we keep the **fastest** mate, and among losing ones
the **slowest** delay. The table therefore does not merely say *whether* a position is won, it
says *how long the game will last* and stores the optimal line's length as a side effect.

Because the ply index is the occupancy, `WIN_SCORE - plies_to_terminal` is a property of the
position alone: no depth fields, no relative-to-root normalisation, nothing that could go stale
when a position is reached from a different root. The distance to the terminal node falls out of
the sweep for free.

Draws need no unknown-value bookkeeping at all: a position is a draw exactly when **none of its
children loses**, and the sweep answers that from the finished layer above. The same fold also
handles the two degenerate cases in one place — a terminal board scores `±WIN_SCORE` with
`dtx = 0` (or `0` when *both* players hold a line), and a board with no legal move scores a
draw.

The sweep is bracketed by a post-condition: `count_nonzero(score != UNSOLVED)` must equal the
number of enumerated positions, otherwise it raises. `combine_children` also raises if it ever
meets an `UNSOLVED` child, which catches a layer ordering mistake immediately rather than
silently producing wrong values.

### Storing only half the table: the colour-swap isomorphism

Only positions of games where **player 1 moved first** are stored. A game where player 2
started is the same game with the two colours exchanged, so `board_index` answers those queries
from the **colour-swapped** board and no second table is needed.

The swap is *not* unconditional, and getting this right matters. At an odd ply the swapped
board is not itself a player-1-started position, so the lookup routes on the **ply parity and
the side to move** rather than on the player alone:

```python
occupied = number of pieces on the board
ones     = number of player-1 pieces
natural  = 1 if occupied % 2 == 0 else 2          # the side to move at that ply

if   ones == (occupied + 1) // 2 and player_turn == natural:   return encode(board)
elif ones == occupied // 2       and player_turn == 3 - natural: return encode(swap(board))
else:                                                          return None
```

The score needs **no negation** when the swap is used, because exchanging the colours is an
isomorphism that preserves the value of the player to move. That is what makes the trick work
at all: a plain "negate the score" colour swap would be wrong, since the side to move changes
identity along with the colours.

The same idea appears a third time in the JSON tablebase, where it is called `swapped` in the
`compressed_representation` dict, and in `get_solution._swap_game_result`, which has to flip
the stored `game_result` string back.

### Storing one rotating context: the reflection isomorphism

The colour swap above is not the only reduction available. Reflecting the board also maps the
`clockwise` game onto the `counterclockwise` game:

- a reflection conjugates a clockwise quarter turn into a counterclockwise one, so the
  mandatory rotation phase of one game becomes the rotation phase of the other;
- the orthogonal adjacency that a **transfer** uses is preserved, so the move set is preserved —
  a transfer that was legal before the mirror is legal after it, with the direction relabelled;
- a **completed line** is mapped onto a completed line, so the win condition and the both-lines
  draw are preserved.

The three phases of a turn are therefore all carried over, and the two games are isomorphic. A
`counterclockwise` position can be answered from the `clockwise` table after reflecting the
board, and the line has to be reflected back on the way out:

```python
_COUNTERPART_ROTATION = {"clockwise": "counterclockwise",
                         "counterclockwise": "clockwise"}
_REFLECTION = (True, 0)          # vertical mirror; any reflection of the square does,
                                  # and this one is its own inverse, which is what lets
                                  # the same mapping send the answer back
```

`build_record` therefore tries the context that was asked for first and only reflects when that
context has no table:

```python
table  = load_table(base_dir, grid_size, rotation, transfer_allowed)
if table is None or table.entry(state, player) is None:
    counterpart = _COUNTERPART_ROTATION.get(rotation)
    reflected   = tablebase._apply_symmetry(state, _REFLECTION)
    entry       = load_table(base_dir, grid_size, counterpart, ...).entry(reflected, player)
    moves       = [tablebase.map_compressed_move(m, _REFLECTION, grid_size, rotation) for m in ...]
```

The mapping is not new code: it is the `create_game_tablebase.map_compressed_move` /
`map_compressed_state` pair that the JSON tablebase already used for exactly this reduction, so
both tablebases spell a reflection the same way — the cell is remapped and the transfer
direction is *recomputed* from the mapped source-to-target delta, which is what keeps
`transfer (0,2) left` from silently becoming an illegal move.

**The reflection has to be applied to the board, not just to the rule name.** This is the one
easy mistake here, and the tables make it visible: for a 4x4 position the clockwise and the
counterclockwise games genuinely have *different* values, and both are player-1 wins of a
different length. This position, player 1 to move, transfer allowed:

```text
. . . .        . . . .        clockwise table, asked for the position on the left
. . . .        . . . .        clockwise          score 989, line 11 plies
. 2 . 2   <->  0 2 0 2        counterclockwise   score 991, line  9 plies
. 1 . 1        0 1 0 1
```

The value on the right is the clockwise table read at `mirror(position)` — which is exactly what
a counterclockwise query needs, and the record comes back tagged `reflected_from: 'clockwise'`.
Answering the *same* board by simply swapping the rule name would have returned 989, which is the
answer to a different question. The identity that actually holds is

```text
value_counterclockwise(board) == value_clockwise(mirror(board))
```

and that is the line `--verify` checks.

`STORED_ROTATIONS = ("still", "clockwise")` is therefore the default for `build_tablebase`, for
`verify_tablebase` and for the command line, and `--rotations all` still produces the redundant
set. `still` is *not* covered by the reflection — a game with no rotation phase is not the same
game as one with a quarter turn after every ply — and the transfer rule is a different rule set
rather than a different orientation, so four of the six original contexts survive: two rotations
times two transfer rules. For 4x4 that is a halving of both the build time and the tables, from
62 MB down to 37.5 MB.

The reduction was checked rather than assumed, and it holds *exactly* rather than approximately:

| check | result |
| --- | --- |
| `cw(b) == ccw(mirror(b))` for stored positions, 1x1–3x3, both transfer rules, all three reflections of the square | 31,629 comparisons, 0 mismatches |
| reflected line replayed with `play_turn`: legality, states, terminal outcome, length | 114 positions, 0 problems |
| 855 lookups against a `still`+`clockwise`-only build vs a full six-context build | 0 unanswered, 0 score differences, 0 illegal lines |
| 4x4 example above, served from the clockwise table | cw 989 (11 plies) direct, ccw 991 (9 plies) via `reflected_from` |

The 4x4 row is deliberately *not* cross-checked against `solve_game`: proving a 4x4 result by
search is the very thing the tablebase exists to avoid. What makes the 4x4 tables trustworthy
instead is that the identical code path is verified exhaustively on 1x1 to 3x3, where an
independent search can confirm every sample, plus the line replay and the terminal evaluation,
which do run at 4x4.

The one visible consequence is benign: where a position has **two equally good moves**, the
reflected path may report the other one, because the mirrored position enumerates its children in
a different order. Scores and line lengths always agree.

### Reconstructing the principal variation

The value tables hold values only — no move sequences. A line is rebuilt by walking the layers
and letting `combine_children` choose the child: for the current position, re-enumerate the
legal moves, skip the children whose `board_index` is `None` (unreachable, hence not stored),
fold the stored `(score, dtx)` pairs with the same function used during the sweep, and follow
the winning/most-delaying outcome. The result is the optimal line, for free, from data that
occupies two bytes per position.

```python
moves, states = principal_variation(table, state, player_turn, rotation)
```

`retrograde_tablebase.build_record` then adapts a value table into the *same* record layout the
JSON tablebase uses (adding a `"source": "retrograde.npz"` field), so `get_solution` has a
single format to deal with and the JSON files remain a genuine fallback rather than a parallel
universe.

### Verification against an independent solver

Because the retrograde sweep and the search engine share no code path beyond the move
generator, they are a genuine cross-check of each other. `retrograde_tablebase.py --verify N`
runs, for every context:

1. **Sample from the stored table**, not from random boards — up to `512 * N` random codes per
   layer filtered by occupancy, so every sample is a position the context really covers. This
   matters: a board with a legal piece split can still be *unreachable* under a rule context, and
   such a board is correctly absent, so generating boards at random would produce false alarms.
2. **Re-solve each sample with `solve_game.solve_game`**, an independent negamax + alpha/beta
   + transposition table implementation, and compare scores.
3. **Rebuild the PV and assert `len(moves) == dtx`** — this checks the distance field, which a
   score comparison alone would never catch.
4. **Replay the line through `orbital_logic_game_functions.play_turn`**, checking legality,
   board shape and that the terminal evaluation matches the stored score.
5. **Generate boards with a deliberately wrong piece split** and assert they are *absent* from
   the table — verifying the reachability pruning and the colour-swap routing.
6. **Cross-check the other rotating context** — assert that it reports the same reachability
   decision, that it carries exactly the value this context stores for the mirrored board, and
   replay the line it returns from the position the caller asked about. When the counterpart
   table is absent this exercises the reflection path, so `--verify` covers the reduction that
   makes the `counterclockwise` folder unnecessary.

Any mismatch prints `MISMATCH: ...` to stderr and the process exits with code 1. This is what
makes the tablebases trustworthy rather than merely fast.

One caveat on 4x4: `solve_game` on a *shallow* 4x4 position can run for hours, so
`--verify 4` on 4x4 is not a practical command — use a small sample count, or verify a specific
context. Every layer is sampled, so a run that does finish covers the full board.

### Tier 2 — Symmetry-reduced JSON tablebase

`create_game_tablebase.py` builds per-position JSON records:
`{id, position, solution: {score, game_result, best_move, moves_sequence, states_sequence}}`,
laid out as

```text
<base_dir>/<n>x<n>/<rotation>/<transfer_allowed|transfer_not_allowed>/<player1|player2>/completion_<pieces>.json
```

A naive build stores all `3^(n^2)` boards per context. The **compressed** variant stores only
the canonical representative of each symmetry orbit, which is a much better use of disk. The
four reductions, recorded in `symmetry_observations.md`, are:

1. **Rotational orbits.** 4-fold-symmetric (invariant under 90 degrees) and 2-fold-symmetric
   (180 degrees) positions are single cases; the best move maps across the group under a
   proper rotation.
2. **Reflection.** With rotation `still`, two reflected cases collapse into one. With
   `clockwise`/`counterclockwise`, the reflected board's stored rotation is swapped — so **only
   the `clockwise` folder is needed** and `counterclockwise` is answered by mapping the move.
3. **Colour swap.** The inverse of a state (1 for 2, 2 for 1) turns "player 1 to move" into
   "player 2 to move", so only a `player1` folder is needed, with `game_result` and the stored
   move's player mapped back.
4. **Transfer rule — not compressible.** The `transfer_allowed` / `transfer_not_allowed`
   folders both have to exist, because the transfer moves are mapped like everything else but
   the rule set genuinely differs.

Implementation-wise, `canonicalize_state` picks the lexicographically smallest row-major tuple
across the orbit, `_apply_symmetry` implements the group element as
`(reflection: bool, turns: int)` (vertical-axis mirror followed by `np.rot90`), and
`map_compressed_move` is the inverse: it maps cell positions, recomputes transfer directions
from the mapped source-to-target delta, and flips the colour fields. `get_solution` looks the
canonical state up and maps the answer back to the caller's orientation.

The payoff on 3x3: **15,714** compressed records versus **236,196** for the full tablebase —
roughly 15x smaller, on top of dropping the `player2` and `counterclockwise` folders. A useful
side benefit of the JSON variant is that it also covers positions that **no game from the empty
board can reach**, which the retrograde tablebase deliberately prunes away.

### Tier 3 — On-the-fly search

`solve_game.py` is the from-scratch solver and the reference implementation. Boards are flat
Python tuples and moves are the compact triple `(transfer_source_flat, transfer_dir, add_flat)`
with `0=up, 1=down, 2=left, 3=right` and `None` for "no transfer" — the numpy conversion only
happens at the public API boundary.

- **Negamax + alpha/beta + transposition table**, with entries tagged `EXACT` / `LOWER` /
  `UPPER` and depth-shifted on reuse via `adj = e_val + (e_depth - depth)`. The table clears
  itself at `TT_MAX = 4_000_000` entries rather than growing without bound.
- **Move ordering** by a cheap static score over the winning lines through the add cell,
  weighted `own^2`, with `10^6` reserved for an immediate win; the TT move is always tried
  first.
- **Iterative deepening with a wall clock.** `find_best_move_within_time` only adopts a move
  from a *completed* iteration, so the answer always comes from a finished search, and it
  stops early once `abs(best_value) >= WIN_SCORE - depth` proves the result. It returns
  `info` with `depth`, `nodes`, `elapsed`, `timed_out`, `reason` and the top 5 moves, and
  `reason == "max_depth"` explicitly flags an answer that is **not** guaranteed.
- **Clock polling is cheap** — the monotonic clock is read once every 2048 nodes.
- **Parallel root** via an `mp.Pool` with the board tables injected through an `initializer`,
  one TT per worker, merged afterwards for the PV reconstruction. The serial path is
  bit-for-bit identical to the single-process original.

`computer_engine.choose_move` is the policy layer over all three tiers, and it is careful to be
honest: a fallback search that hit its depth limit gets a warning that the result is not
guaranteed, and a `depth == 0` answer is reported as "the best ordered move" with no score
rather than being dressed up as an evaluation.

### Parallelism and performance

| Stage | Approach |
| --- | --- |
| `create_game_tablebase` | `mp.Pool` over positions within a context |
| `retrograde_tablebase` enumeration | per-layer fork `Pool`, chunk size `min(4096, ceil(total / (workers * 8)))` |
| `retrograde_tablebase` sweep | per-layer fork `Pool` writing into **lock-free shared memory** |
| `solve_game` | optional parallel root move subtrees |
| `computer_engine` | long-lived daemon worker process, so a long search never blocks the GUI |
| `test_game` | batched vectorised numpy instead of multiprocessing |

The sweep's shared-memory trick is worth calling out: the `score` and `dtx` tables are
allocated as `multiprocessing.Array(ctypes.c_int16, size, lock=False)` and
`Array(ctypes.c_uint8, size, lock=False)` and wrapped as numpy views with `np.frombuffer`, so
workers scatter their results straight into the final arrays with no copies, no locking and no
pickling back through the parent.

Every multiprocessing path in the repository degrades gracefully to a serial one — `_pool_context`
falls back when `fork` is unavailable, and `tqdm` is genuinely optional.

What the tiers actually cost per query, measured on this machine:

| Query | 3x3 | 4x4 |
| --- | --- | --- |
| `retrograde_tablebase` — first load of a table (decompress) | 11 ms | 426 ms |
| `retrograde_tablebase` — value lookup alone, warm | **4.9 us** | **6.8 us** |
| `retrograde_tablebase` — full `lookup_solution`, line rebuilt and report formatted | 0.26 ms | 3.2 ms |
| `retrograde_tablebase` — the same counterclockwise query, answered by reflection | 0.27 ms | 3.2 ms |
| `compressed_game_tablebase` — warm `lookup_solution`, whole formatted report | 0.38 ms | not built |

The value lookup is the part that has to be O(1) and it is: encoding the board and reading two
array slots. The dominant per-move cost on a large board is therefore not the table at all but
re-enumerating the legal moves at every ply of the reconstructed line — 2.8 ms of the 3.2 ms
above. The 426 ms first load is a one-off per table per process, paid once at game start, and is
why the loader keeps the four most recently used tables around. The reflection costs nothing
measurable, which is the point of reducing six contexts to four.

---

## Repository layout

```text
orbital_logic_game_functions.py   Rules engine: evaluation, rotation, move generation,
                                  play_turn, state/string conversions, board statistics
solve_game.py                     Negamax + alpha/beta + TT solver, PV reconstruction,
                                  iterative deepening, parallel root
retrograde_tablebase.py           Retrograde value-table builder, reader and verifier
create_game_tablebase.py          JSON tablebase generator (full + compressed) and the
                                  shared symmetry/canonicalisation helpers
get_solution.py                   Unified lookup API + CLI over all tablebase sources
computer_engine.py                Policy layer (tablebase -> search -> random) and the
                                  background engine process
gui.py                            The entire Pygame front-end and its widget toolkit
test_game.py                      Vectorised Monte-Carlo random-play statistics
symmetry_observations.md           The design notes behind the compressed tablebase
requirements.txt                  Pinned dependencies
LICENSE                           MIT
```

Generated data directories:

| Directory | Size | Contents |
| --- | --- | --- |
| `retrograde_game_tablebase/` | 38 MB | `retrograde.npz` value tables, 1x1 to 4x4, 4 rule contexts per grid |
| `compressed_game_tablebase/` | 3.9 GB | symmetry-reduced JSON, 1x1 to 3x3 complete, 4x4 partial |
| `game_tablebase/` | 248 MB | full JSON, 1x1 to 3x3 |

The two JSON 4x4 directories are gitignored (`game_tablebase/4x4/`,
`compressed_game_tablebase/4x4/`) — they are multi-gigabyte and the generation was left
incomplete, so the committed retrograde tables are the 4x4 source of truth. The retrograde 4x4
tables (37.5 MB) *are* committed, because they are small enough to be worth having in the tree;
the redundant `counterclockwise` pair is not needed and can be dropped.

The `1x1`, `2x2` and `3x3` grids still have all six contexts on disk — 50 kB in total, versus
40 kB for the four that are actually needed. Those extra folders are harmless leftovers of an
earlier build; a counterclockwise query is answered by reflection whether or not the folder
exists, so they can be deleted whenever convenient.

---

## Precomputed tablebases

Retrograde value tables, one file per `(grid size, rotation, transfer rule)`. Only the `still`
and `clockwise` rotations are stored; `counterclockwise` is served by reflection:

| Grid | Table entries | `still` + transfer | `clockwise` + transfer | `still` no transfer | `clockwise` no transfer | Per file |
| --- | --- | --- | --- | --- | --- | --- |
| 1x1 | 3 | 2 | 2 | 2 | 2 | ~0.8 kB |
| 2x2 | 81 | 29 | 29 | 29 | 29 | ~0.9 kB |
| 3x3 | 19,683 | 6,034 | 6,034 | 5,478 | 5,478 | 7.3-8.0 kB |
| 4x4 | 43,046,721 | 10,161,173 | 10,161,161 | 9,722,011 | 9,721,176 | 5.3-12.0 MB |

The four 4x4 contexts come to **37.5 MB** and 39,765,521 positions; the redundant
`counterclockwise` pair would have added another 24.5 MB and 19.9M positions. For 1x1 to 3x3 the
tables are negligible either way — 40 kB for the four that are needed, 50 kB with the leftovers.

For comparison, the JSON tablebases hold 236,196 records for 3x3 in full mode and 15,714 in
compressed mode. `print_game_statistics(4)` reports 15,134,931 distinct 4x4 boards reducing to
3,784,019 rotation/reflection classes (3,783,456 with no symmetry, 544 with 2-fold, 19 with
4-fold).

Historical timings for the *from-scratch* search on 4x4 with `still` rotation, which is what
motivated the tablebase work in the first place: about 171,310 seconds to prove a result from
the empty board. With the retrograde table the same query is a single array index.

---

## Development notes

- `test_game.py` is a **statistics tool**, not a pytest suite — it has no `test_*` functions
  and the pytest cache is empty. It reports random-play win/draw probabilities broken down by
  starting player, and with `verbose_print=True` it replays the games board by board using the
  slow scalar reference loop, which is useful for eyeballing that the batched vectorised
  implementation agrees with the straightforward one.
- The move enumeration order in `orbital_logic_game_functions.get_possible_moves` — all
  add-only moves row-major first, then for each opponent piece, for each of `u,d,l,r` with an
  empty neighbour, for each legal add cell — is part of the contract. `solve_game._gen_moves`
  and `test_game._sample_move_encoded` both reproduce it exactly, including the `e + t*e`
  layout of the encoded move indices, so that random sampling is uniform over the same move set
  the search enumerates.
- Rotation permutations are built from concentric rings (`_build_rings`), which is why the
  centre cell of an odd-sized board maps to itself and needs no special case.
- `_apply_symmetry` / `map_compressed_move` in `create_game_tablebase.py` are the single
  definition of "reflect this board" in the repository. The retrograde module reaches for them
  rather than writing its own, so the two tablebases cannot drift apart on what a reflection
  does to a transfer direction — the bug that this reduction would silently introduce is a
  reflection that maps a legal transfer onto an illegal one.
- There is no GPU code, no Zobrist hashing and no bitboard representation in the current tree.
  Position identity is either the base-3 integer code (retrograde) or the
  `<grid>_<digits>_<context>` string ID (JSON), and the search transposition table is keyed by
  a plain `(board_tuple, player)` pair with a depth-tagged value. An earlier commit explored
  GPU speedups for the tablebase and settled on numpy vectorisation plus `mp.Pool` instead.

---

## Credits

This project was built and finished with substantial help from
**[opencode](https://opencode.ai)** — the open-source AI coding agent. Most of the code in this
repository was written, debugged, refactored and completed in collaboration with opencode, and
the project would not be in its current state without it.

The commit history records this directly — `b05bc3b`, *"made the code much faster using
opencode"*, is the point at which the solver went from impractical to usable, and the
subsequent work on the compressed tablebase (`a6893c4`), the retrograde tablebase (`4bacb3c`,
`87c1152`) and the GUI (`e8261cf`, `38edf1a`) was carried out with opencode driving the
implementation.

opencode in particular was responsible for the performance work that turned this from a
prototype into something usable: the numpy vectorisation, the symmetry compression scheme, the
retrograde sweep, the multiprocessing layers, the parallel search root, the cross-verification
harness, and the large majority of the Pygame front-end. It was also instrumental in tracking
down the bugs that would otherwise have made the tablebases quietly wrong.

The design of the game, the retrograde approach, the distance-to-terminal mate score encoding
and the symmetry reductions are the author's; opencode implemented them.

Thanks also to the developers of **NumPy**, **Pygame** and **tqdm**, on which the whole project
is built.

---

## License

MIT — Copyright (c) 2025 Printzios Lampros. See [LICENSE](LICENSE).
