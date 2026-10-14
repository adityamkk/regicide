# Regicide

I was salty that I lost, so I made a bot to prove that regicide is impossible to win.
Unfortunately, the bot won...

## Installation

You just need plain python (3.9 or newer). No dependencies.

## Run a specific agent

There are two agents. Each script deals a random game, plays it to the end, and prints every
move: the royal's health and attack, each player's hand, the cards played or discarded, and
the random draws.

**Hybrid search** (the stronger agent): expectimax search with MCTS at the frontier.

```
python3 run_hybrid.py                              # 1 player, random seed
python3 run_hybrid.py --players 2 --seed 203       # replay a specific game
python3 run_hybrid.py --sustain                    # value full hands and refills more late in the game
python3 run_hybrid.py --depth 2 --iterations 64    # search harder (slower)
```

| Option | Default | Meaning |
|---|---|---|
| `--players` | 1 | number of players (1 to 4) |
| `--seed` | random | seed for the deal and the luck; the same seed replays the same game |
| `--depth` | 1 | player decisions searched exhaustively before MCTS takes over |
| `--iterations` | 24 | MCTS iterations per frontier position |
| `--rollout-depth` | 6 | player decisions per MCTS simulation |
| `--beam` | 4 | moves searched at deeper levels of the tree |
| `--samples` | 3 | sampled outcomes for large random events (e.g. big Diamonds draws) |
| `--sustain` | off | sustain-aware heuristic and move rules |
| `--backup` | mean | how MCTS scores a position (`mean` or `max`) |
| `--adversarial` | off | the GameAgent picks the worst card for the players instead of a random one |
| `--max-moves` | 2000 | safety limit on player moves |

**Depth-limited minimax** with alpha-beta pruning. It assumes the worst card comes up at every
draw, so it plays very defensively.

```
python3 run_minimax.py --players 2 --seed 5 --depth 3
```

It accepts `--players`, `--seed`, `--depth` (default 3, counting every move including random
events), `--adversarial` and `--max-moves`.

## Run the benchmark

`benchmark.py` plays many seeded games with each agent, in parallel, and compares them:

```
python3 benchmark.py --games 30 --agents minimax hybrid
python3 benchmark.py --games 10 --players 2 --seed 200 --agents hybrid hybrid-sustain
```

| Option | Default | Meaning |
|---|---|---|
| `--games` | 20 | games per agent |
| `--players` | 1 | number of players |
| `--agents` | `minimax hybrid` | agents to compare (see below) |
| `--seed` | 0 | seed of the first game; games use seeds `seed` to `seed + games - 1` |
| `--workers` | all cores | parallel processes |

Every agent plays the same deals, so results are directly comparable. The report shows wins, the
average and best number of royals defeated, decisions per game and seconds per game. It also
shows which kind of royal (J, Q or K) ended each game.

Available agents:

| Agent | What it is |
|---|---|
| `minimax` / `minimax2` | minimax at depth 3 / depth 2 |
| `hybrid` | hybrid search with default settings |
| `hybrid2` | hybrid search at depth 2 (12 MCTS iterations) |
| `hybrid-heuristic` | hybrid search at depth 2 using only the heuristic (no MCTS) |
| `hybrid-sustain` | hybrid search with the sustain-aware heuristic and move rules |
| `hybrid-v3-only` / `hybrid-policy-only` | each half of `hybrid-sustain` on its own |

The hybrid agents take roughly 20 to 50 seconds per game, so 30 games of one agent take a few
minutes on a multi-core machine. Minimax takes a few seconds per game.

## Trajectories

`visualize.py` plays a game and writes a self-contained HTML viewer, then opens it in your
browser:

```
python3 visualize.py --players 2 --sustain --seed 203 --output trajectories/regicide_seed203.html
python3 visualize.py --agent minimax --seed 5
```

The viewer shows:
- the active royal: health (damage still to land is hatched), attack after Spades, and the suit it blocks;
- the sizes of the draw and discard piles, and a royal put back on top of the draw pile;
- the cards played against the royal, and how many Jacks, Queens and Kings are left;
- every player's hand, with the cards about to be played or discarded outlined;
- a game log, where clicking a line jumps to that step.

Step through the game with the Prev and Next buttons, the arrow keys, the slider or Play
(space). Untick "show random events" to step through player moves only. Each step shows the
state just before that move happens. Adding `#step=N` to the file's address jumps to step N.

| Option | Default | Meaning |
|---|---|---|
| `--players` | 1 | number of players |
| `--seed` | random | game seed |
| `--agent` | hybrid | `hybrid` or `minimax` |
| `--sustain` | off | hybrid only: sustain-aware heuristic and move rules |
| `--depth` | 1 (hybrid), 3 (minimax) | search depth |
| `--iterations` | 24 | hybrid only: MCTS iterations per frontier position |
| `--adversarial` | off | the GameAgent picks the worst card |
| `--output` | `regicide_seed<S>.html` | file to write |
| `--no-open` | off | write the file without opening a browser |

[trajectories/regicide_seed203.html](trajectories/regicide_seed203.html) is a recorded win: 2
players, hybrid search with `--sustain`, seed 203. All 12 royals are defeated over 279 steps.
