# How the game is solved — implementation notes

Detailed companion to [README.md](README.md). This file explains the algorithms behind the
three solver back-ends in this repository: why the game is exactly solvable at all, how the
retrograde tablebase is built and queried, how the symmetry reductions shrink the data, how the
fallback search works, and how everything is verified.

---

## Contents

- [The structural observation](#the-structural-observation)
- [Tier 1 — Retrograde analysis](#tier-1--retrograde-analysis)
  - [Phase 1 — enumerate the reachable layers](#phase-1--enumerate-the-reachable-layers)
  - [Board coding: base-3 dense tables](#board-coding-base-3-dense-tables)
  - [Phase 2 — the sweep](#phase-2--the-sweep)
  - [Phase 3 — storage](#phase-3--storage)
  - [Half the table: the colour-swap isomorphism](#half-the-table-the-colour-swap-isomorphism)
  - [One rotating context: the reflection isomorphism](#one-rotating-context-the-reflection-isomorphism)
  - [Querying and reconstructing the principal variation](#querying-and-reconstructing-the-principal-variation)
  - [Verification](#verification)
- [Tier 2 — Symmetry-reduced JSON tablebase](#tier-2--symmetry-reduced-json-tablebase)
- [Tier 3 — On-the-fly search](#tier-3--on-the-fly-search)
- [Parallelism](#parallelism)
- [Measured results](#measured-results)
- [Implementation notes and sharp edges](#implementation-notes-and-sharp-edges)

---

## The structural observation

Every turn adds exactly one piece to the board:

- a **transfer** moves a piece to an empty cell, so it preserves the number of occupied cells;
- the mandatory **add** increases it by one;
- the **rotation** only permutes cells.

So starting from the empty board, **the number of occupied cells equals the number of plies
played**. The game graph is therefore *strictly layered by occupancy* and **cannot contain a
cycle**. This single fact is what the whole solver rests on:

- there is a well-defined ply index and it is readable straight off the board;
- values can be computed in one bottom-up pass instead of needing fixpoint iteration;
- mate scores need no depth bookkeeping, because the ply index *is* the depth.

There is exactly one exception, and it sits at the very bottom of the sweep: the **endgame
presses**. A full board without a line has no move to generate, and the official rule finishes
it with up to `EXTRA_TURNS = 5` forced Orbito presses that add nothing. Presses do not occupy
cells, so they break the "ply = occupancy" identity and cannot be layers of the sweep. They are
instead *replayed*: the bottom layer asks "what does at most five ring rotations of this board
leave?", and the answer is a value plus a distance. Since the presses do not change the occupancy,
the number already spent is **not part of the board** — it is carried beside it as
`extra_turns`, which is why the tables only store the position before the first press.

`create_game_tablebase.py` already walks those layers from the full board downwards, but it
solves every position from scratch, so each board re-searches its entire subtree. The same
waste happens inside a single search: a transposition table collapses repeated positions, but
every distinct position is still expanded once per search. `retrograde_tablebase.py` removes
that waste — one sweep per rule context, where a rule context is a
`(grid size, rotation direction, transfer rule, endgame rule)` quadruple.

---

## Tier 1 — Retrograde analysis

### Phase 1 — enumerate the reachable layers

```python
layers = enumerate_reachable(grid_size, rotation, transfer_allowed, extra_rotation_allowed)
# layers[ply] is a sorted np.int64 array of base-3 board codes; layers[0] == [0]
```

The enumeration descends from the full board. Crucially, it **prunes the children of terminal
positions**: if a board already has a completed line, nothing below it can ever occur. Without
that pruning the enumeration also produces boards where one player keeps placing pieces after
the opponent has already won — boards the rules score as a draw, but which no real game can
reach.

The 4x4 `still` / `transfer_allowed` layers are:

| ply | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| positions | 1 | 16 | 240 | 1680 | 10920 | 43680 | 160160 | 400400 | 900900 | 1441440 | 2018016 | 2018016 | 1681632 | 960640 | 409920 | 101620 | 11880 |

The middle layers sit at the binomial maximum while the deepest layers are truncated — that is
the terminal-position pruning at work.

### Board coding: base-3 dense tables

Positions live in two dense flat arrays indexed by the base-3 code of the row-major board, so
a lookup is a single array index rather than a scan over a JSON array:

```python
code = sum(cell * 3 ** (cell_count - 1 - index) for index, cell in enumerate(row_major_board))
```

`score` is `int16` with `UNSOLVED = 32767` marking slots that are not stored (which also makes
"unreachable" and "not computed" impossible to confuse), and `dtx` is `uint8` holding the
distance to the terminal node. The cell count is capped at 255 for exactly that reason — the
distance has to fit in a single byte.

The non-obvious part is that computing a child code is not a re-encode. The codec precomputes a
per-cell weight table, and then also the weight each cell takes *after* the rotation:

```python
weights[k]              = 3 ** (cell_count - 1 - k)
destination_weights[k]  = weights[inverse_perm[k]]   # the weight k ends up with post-rotation
```

Every position is split once into a `base` (the part of the code a move cannot touch) and a
`destination_weights` slice, and generating a child is a handful of integer additions:

```python
def child_code(base, destination_weights, move, player):
    code = base + player * destination_weights[move.add_cell]
    if move.transfer:
        code += opponent * (destination_weights[move.target] - destination_weights[move.source])
    return code
```

The transfer term is the elegant bit: the source cell loses the opponent's value and the target
cell gains it, and because the rotation is already folded into `destination_weights`, both
effects are captured by one difference.

Move generation itself is driven by `solve_game._engine_params`, which precomputes the `2n+2`
winning lines, the lines through each cell, and the four neighbours of each cell.

### Phase 2 — the sweep

```python
score, dtx = sweep_layers(codec, layers)
```

Iterating `ply` from the deepest layer down to 0, each position's value is folded out of the
values of its children, which the previous layer has already determined. Every
`(board, side to move)` pair in the context is therefore evaluated **exactly once for the whole
context**, no matter how many other positions refer to it as a child.

`_combine_children` folds the `(child_score, child_dtx)` pairs — expressed from the *child's*
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
says *how long the game will last*, and stores the optimal line's length as a side effect.

The last row of that table is where the endgame presses go. A full board without a line has no
children, so the sweep cannot give it a value by folding; `_endgame_value` asks
`endgame_outcome` instead, which replays at most `EXTRA_TURNS` presses through the ring
permutation and returns `(winner, presses_used)`. That is folded in like any other child, with
the press count as the distance — so a position that is won on the third press is stored as
`WIN_SCORE - 3`, and a position that survives all five is stored as a draw with distance `5`.
With the rule switched off the same board is simply the `0` of the last row.

Because the ply index is the occupancy, `WIN_SCORE - plies_to_terminal` is a property of the
position alone: no depth fields, no relative-to-root normalisation, nothing that could go stale
when a position is reached from a different root.

Draws need no unknown-value bookkeeping at all — a position is a draw exactly when **none of its
children loses**, and the sweep answers that from the finished layer above. The same fold also
handles the two degenerate cases in one place: a terminal board scores `±WIN_SCORE` with
`dtx = 0` (or `0` when *both* players hold a line), and a board with no legal move scores a
draw.

The sweep is bracketed by a post-condition: `count_nonzero(score != UNSOLVED)` must equal the
number of enumerated positions, otherwise it raises. `_combine_children` also raises if it ever
meets an `UNSOLVED` child, which catches a layer-ordering mistake immediately rather than
silently producing wrong values.

### Phase 3 — storage

```python
np.savez_compressed(file, score=score, dtx=dtx, metadata=np.array(json.dumps(metadata)))
```

```text
<base_dir>/<n>x<n>/<rotation>/<transfer_allowed|transfer_not_allowed>/<extra_rotation_allowed|extra_rotation_not_allowed>/retrograde.npz
```

`metadata` carries `format_version` (currently `2`), `grid_size`, `rotation`, `transfer_allowed`,
`extra_rotation_allowed`, `starting_player`, the score convention, `positions`, `layer_sizes` and
`table_entries`. A table whose version does not match is refused rather than misread, which is
why tables built before the presses existed have to be rebuilt.

### Half the table: the colour-swap isomorphism

Only positions of games where **player 1 moved first** are stored. A game where player 2
started is the same game with the two colours exchanged, so `board_index` answers those queries
from the **colour-swapped** board and no second table is needed.

The swap is *not* unconditional, and getting this right matters. At an odd ply the swapped
board is not itself a player-1-started position, so the lookup routes on the **ply parity and
the side to move** rather than on the player alone:

```python
occupied = number of pieces on the board
ones     = number of player-1 pieces
natural  = 1 if occupied % 2 == 0 else 2          # who moved on the previous ply

if   ones == (occupied + 1) // 2 and player_turn == natural:    return encode(board)
elif ones == occupied // 2       and player_turn == 3 - natural: return encode(swap(board))
else:                                                           return None
```

The score needs **no negation** when the swap is used, because exchanging the colours is an
isomorphism that preserves the value of the player to move. That is what makes the trick work
at all: a plain "negate the score" colour swap would be wrong, since the side to move changes
identity along with the colours.

The same idea appears twice more — as `swapped` in the JSON tablebase's
`compressed_representation` dict, and in `get_solution._swap_game_result`, which flips the
stored `game_result` string back.

### One rotating context: the reflection isomorphism

A reflection of the board maps the clockwise game onto the counterclockwise one:

- it turns a clockwise quarter turn into a counterclockwise one;
- it maps the orthogonal adjacency a transfer uses onto itself;
- it maps a completed line onto a completed line.

So the two games are isomorphic, and a counterclockwise position is answered from the clockwise
table after reflecting the board, with the moves reflected back on the way out. Only the
`still` and `clockwise` tables are therefore built — 8 contexts per grid size (2 rotations x 2
transfer rules x 2 endgame rules) instead of 12, which is what took the committed tablebase from
62 MB down to 38 MB before the endgame rule doubled the file count.

`_REFLECTION = (True, 0)` is a vertical-axis mirror (its own inverse, which is what lets the
same mapping send the answer back), reused from `create_game_tablebase`'s symmetry machinery so
both tablebases agree on how a reflection is spelled. `build_record` tries the position's own
context first and only then falls back to the counterpart, and tags the record with
`reflected_from` and `symmetry` so the provenance is visible.

### Querying and reconstructing the principal variation

`ValueTable.entry(state, player_turn)` is `(score, dtx)` or `None`, and it is `None` both when
the slot is empty *and* when it holds `UNSOLVED` — a lookup can never confuse "not stored" with
"not computed". `load_table` caches by `(path, mtime, size)` with FIFO eviction at 4 entries, so
repeated queries do not re-read the `.npz`.

The tables hold values only, no move sequences. A line is rebuilt by walking the layers and
letting `_combine_children` choose the child: re-enumerate the legal moves of the current
position, skip children whose `board_index` is `None` (unreachable, hence not stored), fold the
stored `(score, dtx)` pairs with the same function used during the sweep, and follow the
winning / most-delaying outcome:

```python
moves, states = principal_variation(table, state, player_turn, rotation)
```

So the optimal line comes out of two bytes per position.

A line that runs into the endgame is finished by the presses rather than stopping short. The walk
appends them in either of the two ways it can arrive at a full board without a line: when the
position has no children left, and when the last ply the walk is allowed to make filled the board
(its loop is bounded by the cell count, so a game that fills the board on its final ply runs out
of iterations with the presses still to be played). `endgame_outcome` is then replayed and each
press is appended as a `rotation_only` move, up to the press that makes a line or the fifth one.
That is also why the value tables alone answer positions *inside* the presses: the board is the
same board, so a query with `extra_turns = k` is looked up at `k = 0` and the first `k` presses
are skipped.

`build_record` then adapts a value table into the *same* record layout the JSON tablebase uses
(adding a `"source": "retrograde.npz"` field), so `get_solution` has a single format to deal
with and the JSON files remain a genuine fallback rather than a parallel universe.

### Verification

Because the retrograde sweep and the search engine share no code path beyond the move
generator, they are a genuine cross-check of each other. `retrograde_tablebase.py --verify N`
runs, per context:

1. **Sample from the stored table**, not from random boards — up to `512 * N` random codes per
   layer filtered by occupancy, so every sample is a position the context really covers.
2. **Re-solve each sample with `solve_game.solve_game`**, an independent negamax + alpha/beta +
   transposition table implementation, and compare scores.
3. **Rebuild the PV and assert `len(moves) == dtx`** — this checks the distance field, which a
   score comparison alone would never catch.
4. **Replay the line through `orbital_logic_game_functions.play_turn`**, checking legality,
   board shape, and that the terminal evaluation matches the stored score.
5. **Generate boards with a deliberately wrong piece split** and assert they are *absent* from
   the table — verifying the reachability pruning and the colour-swap routing.
6. **Cross-check the counterpart rotating context** (`_check_counterpart`): the `clockwise` and
   `counterclockwise` views must agree on both whether a position is stored and what it scores,
   and the reflected line must stay legal when replayed from the position that was asked about.
   This is the check that justifies storing only one of the two.
7. **Check the endgame presses** (`_mid_endgame_problems`): a full board without a line is
   re-solved by the search engine for every `extra_turns` from `0` to `EXTRA_TURNS` and compared
   with the stored value, so the press replay, the distance and the terminal draw are all
   cross-checked rather than trusted.

Any mismatch prints `MISMATCH: ...` to stderr and the process exits with code 1. This is what
makes the tablebases trustworthy rather than merely fast.

---

## Tier 2 — Symmetry-reduced JSON tablebase

`create_game_tablebase.py` builds per-position JSON records, where `id` is the position itself
written as one base-3 digit per cell, row by row:

```json
{ "id": "00012112", "position": [[0,0,0,1],[2,1,1,2],[2,1,0,1],[1,2,2,1]],
  "solution": { "score": 999, "game_result": "player1_wins",
                "best_move": {...}, "moves_sequence": [...], "states_sequence": [...] } }
```

laid out as

```text
<base_dir>/<n>x<n>/<rotation>/<transfer_allowed|transfer_not_allowed>/<extra_rotation_allowed|extra_rotation_not_allowed>/<player1>/completion_<pieces>.json
```

A naive build stores all `3^(n^2)` boards per context. The **compressed** variant stores only
the canonical representative of each symmetry orbit. The four reductions, recorded in
`symmetry_observations.md`:

1. **Rotational orbits.** 4-fold-symmetric (invariant under 90 degrees) and 2-fold-symmetric
   (180 degrees) positions are single cases; the best move maps across the group under a proper
   rotation.
2. **Reflection.** With rotation `still`, two reflected cases collapse into one. With
   `clockwise`/`counterclockwise`, the reflected board's stored rotation is swapped — so only
   the `clockwise` folder is needed and `counterclockwise` is answered by mapping the move.
3. **Colour swap.** The inverse of a state (1 for 2, 2 for 1) turns "player 1 to move" into
   "player 2 to move", so only a `player1` folder is needed, with `game_result` and the stored
   move's player mapped back.
4. **Transfer rule — not compressible.** Both folders have to exist, because the rule set
   genuinely differs; the transfer moves are then mapped like everything else.

Implementation-wise, `canonicalize_state` picks the lexicographically smallest row-major tuple
across the orbit, `_apply_symmetry` implements a group element as
`(reflection: bool, turns: int)` (vertical-axis mirror followed by `np.rot90`), and
`map_compressed_move` is the inverse: it maps cell positions, recomputes transfer directions
from the mapped source-to-target delta, and flips the colour fields. `get_solution` looks the
canonical state up and maps the answer back to the caller's orientation.

On 3x3 this gives **15,714** compressed records versus **236,196** for the full tablebase —
about 15x smaller, on top of dropping the `player2` and `counterclockwise` folders. A useful
side benefit of the JSON variant is that it also covers positions that **no game from the empty
board can reach**, which the retrograde tablebase deliberately prunes away.

The endgame rule is the one place where a rotation cannot be used to identify two boards, so
`canonicalize_state` has to special-case it. Rotating orbits collapse a board onto its
representative and map the move back across the rotation — that is sound for every position with
an empty cell, but **not** for a full board without a line, where the value is the outcome of
*repeatedly* rotating that exact board. Two orientations of the same full board can therefore
have different press outcomes, and `_turning_changes_the_result` disables the rotational part of
the reduction for `clockwise` (leaving reflection and the colour swap, which are still sound) so
every orientation is stored. With `still` presses, or with the rule switched off, nothing can
change, so the reduction stays on.

---

## Tier 3 — On-the-fly search

`solve_game.py` is the from-scratch solver and the reference implementation. Boards are flat
Python tuples and moves are the compact triple `(transfer_source_flat, transfer_dir, add_flat)`
with `0=up, 1=down, 2=left, 3=right` and `None` for "no transfer" — numpy conversion happens
only at the public API boundary.

- **Negamax + alpha/beta + transposition table**, entries tagged `EXACT` / `LOWER` / `UPPER`
  and depth-shifted on reuse via `adj = e_val + (e_depth - depth)`. The table clears itself at
  `TT_MAX = 4_000_000` entries rather than growing without bound.
- **Move ordering** by a cheap static score over the winning lines through the add cell,
  weighted `own^2`, with `10^6` reserved for an immediate win; the TT move is always tried
  first.
- **Iterative deepening with a wall clock.** `find_best_move_within_time` only adopts a move
  from a *completed* iteration, so the answer always comes from a finished search, and it stops
  early once `abs(best_value) >= WIN_SCORE - depth` proves the result. It returns `info` with
  `depth`, `nodes`, `elapsed`, `timed_out`, `reason` and the top 5 moves, and
  `reason == "max_depth"` explicitly flags an answer that is **not** guaranteed.
- **Clock polling is cheap** — the monotonic clock is read once every 2048 nodes.
- **Parallel root** via an `mp.Pool` with the board tables injected through an `initializer`,
  one TT per worker, merged afterwards for the PV reconstruction. The serial path is
  bit-for-bit identical to the single-process original.

`computer_engine.choose_move` is the policy layer over all three tiers, and it is careful to be
honest: a fallback search that hit its depth limit gets a warning that the result is not
guaranteed, and a `depth == 0` answer is reported as "the best ordered move" with no score
rather than being dressed up as an evaluation.

A position that is already inside the presses is answered before any tablebase is consulted:
the value tables only know the position before the first press, so `choose_move` replays the
presses with `endgame_score` and reports the answer under its own `SOURCE_ENDGAME` with
`presses_remaining`. Once the last press is spent there is no move left to offer, and the engine
says the game is a draw rather than handing out a sixth press.

---

## Parallelism

| Stage | Approach |
| --- | --- |
| `create_game_tablebase` | `mp.Pool` over positions within a context |
| retrograde enumeration | per-layer fork `Pool`, chunk size `min(4096, ceil(total / (workers * 8)))` |
| retrograde sweep | per-layer fork `Pool` writing into **lock-free shared memory** |
| `solve_game` | optional parallel root move subtrees |
| `computer_engine` | long-lived daemon worker process, so a long search never blocks the GUI |
| `test_game` | batched vectorised numpy instead of multiprocessing |

The sweep's shared-memory trick is worth calling out: the `score` and `dtx` tables are allocated
as `multiprocessing.Array(ctypes.c_int16, size, lock=False)` and
`Array(ctypes.c_uint8, size, lock=False)` and wrapped as numpy views with `np.frombuffer`, so
workers scatter their results straight into the final arrays with no copies, no locking and no
pickling back through the parent.

Every multiprocessing path degrades gracefully to a serial one — `_pool_context` falls back when
`fork` is unavailable, and `tqdm` is genuinely optional.

---

## Measured results

Retrograde value tables, one file per `(grid size, rotation, transfer rule, endgame rule)`:

| Grid | Table entries | Positions stored per context | Size per context |
| --- | --- | --- | --- |
| 1x1 | 3 | 2 | 0.8 KiB |
| 2x2 | 81 | 29 | 0.9 KiB |
| 3x3 | 19,683 | 5,478 – 6,034 | 7.3 – 8.3 KiB |
| 4x4 | 43,046,721 | 9,721,176 – 10,161,173 | 5.3 – 13.4 MiB |

Eight contexts per grid size, so 8 files per grid became 32 — 77.4 MiB in total, all of it
generated by the current code and checked in, so the interface reports the grid sizes it can play
on without a build step. Against roughly **4 GB** for the equivalent JSON records the ~60x gap of
the retrograde format is the whole point, and it is why 4x4 is playable at all here.

For comparison, the JSON tablebases hold 236,196 records for 3x3 in full mode and 15,714 in
compressed mode. `print_game_statistics(4)` reports 15,134,931 distinct 4x4 boards reducing to
3,784,019 rotation/reflection classes (3,783,456 asymmetric, 544 with 2-fold, 19 with 4-fold).

Historical timings for the *from-scratch* search on 4x4 with `still` rotation, which is what
motivated the tablebase work: about 171,310 seconds to prove a result from the empty board. With
the retrograde table the same query is a single array index.

---

## Implementation notes and sharp edges

- **Move enumeration order is a contract.** `orbital_logic_game_functions.get_possible_moves`
  emits all add-only moves row-major first, then for each opponent piece, for each of
  `u,d,l,r` with an empty neighbour, for each legal add cell. `solve_game._gen_moves` and
  `test_game._sample_move_encoded` both reproduce it exactly, including the `e + t*e` layout of
  the encoded move indices, so random sampling stays uniform over the same move set the search
  enumerates.
- **Rotation permutations are built from concentric rings** (`_build_rings`), which is why the
  centre cell of an odd-sized board maps to itself and needs no special case. It also means a
  press is *not* a rigid quarter turn of the board: each ring moves one step, so a line is not
  generally carried onto a line, and an endgame press can be the move that completes one. Two
  consequences follow, and both are handled explicitly rather than by luck: a ring rotation does
  not preserve the winning lines, so the compressed tablebase may not identify a full board
  across rotations; and `still` presses return an *equal* board, so `play_turn`'s "returns the
  input unchanged on an illegal move" signal is ambiguous and callers must test
  `olg.is_rotation_only` instead of comparing boards.
- **The press counter is not in the board.** `extra_turns` travels beside the position through
  `solve_game`, `computer_engine` and the GUI, while the tables only store `extra_turns = 0`.
  Anything that answers a position has to know which of the two it is looking at.
- **No GPU code, no Zobrist hashing and no bitboards** are in the current tree. Position
  identity is either the base-3 integer code (retrograde) or the
  `<grid>_<digits>_<context>` string ID (JSON), and the search transposition table is keyed by a
  plain `(board_tuple, player)` pair with a depth-tagged value. An earlier commit explored GPU
  speedups for the tablebase and settled on numpy vectorisation plus `mp.Pool` instead.
- `test_game.py` is a **statistics tool**, not a pytest suite — it has no `test_*` functions
  and the pytest cache is empty. With `verbose_print=True` it replays games board by board using
  a slow scalar reference loop, which is useful for confirming the batched vectorised
  implementation agrees with the straightforward one.
- `play_turn` returns the *unchanged* input state on an illegal move, after printing a message.
  The GUI relies on this to detect a rejected move — with the one exception of a `still` endgame
  press, which is legal and also returns an equal board, so it is checked through
  `olg.is_rotation_only`.
