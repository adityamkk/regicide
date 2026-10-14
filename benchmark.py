"""Compare Regicide agents over many random games.

Usage:
    python3 benchmark.py [--games N] [--players P] [--agents minimax hybrid ...] [--workers W]

Each agent plays the same seeded deals with random luck. Reports win rate, average
royals defeated, and which royals end the games.
"""

import argparse
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from hybrid import HybridSearch, evaluate_v2, evaluate_v3
from minimax import AlphaBetaSearch, play_out
from regicide import new_game

AGENTS = {
    "minimax": lambda seed: AlphaBetaSearch(3),
    "minimax2": lambda seed: AlphaBetaSearch(2),
    "hybrid": lambda seed: HybridSearch(seed=seed),
    "hybrid2": lambda seed: HybridSearch(depth=2, mcts_iterations=12, seed=seed),
    "hybrid-heuristic": lambda seed: HybridSearch(depth=2, mcts_iterations=0, seed=seed),
    # Sustain-aware: evaluate_v3 plus the sustain rollout policy, and each part alone.
    "hybrid-sustain": lambda seed: HybridSearch(sustain=True, seed=seed),
    "hybrid-v3-only": lambda seed: HybridSearch(heuristic=evaluate_v3, seed=seed),
    "hybrid-policy-only": lambda seed: HybridSearch(sustain=True, heuristic=evaluate_v2, seed=seed),
}


def run_one(job):
    agent, players, seed = job
    rng = random.Random(seed)
    start = new_game(players, rng)
    started = time.time()
    final, trajectory, stalemate = play_out(start, AGENTS[agent](seed), rng)
    killer = str(final.active_royal) if final.is_lost and final.active_royal else "-"
    return {
        "agent": agent, "seed": seed, "won": final.is_won,
        "royals": final.royals_defeated, "killer": killer, "stalemate": stalemate,
        "turns": sum(1 for s, _ in trajectory if s.current_agent != -1),
        "seconds": time.time() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--players", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--agents", nargs="+", default=["minimax", "hybrid"], choices=sorted(AGENTS))
    parser.add_argument("--seed", type=int, default=0, help="first deal seed")
    parser.add_argument("--workers", type=int, default=None, help="parallel processes")
    args = parser.parse_args()

    jobs = [(agent, args.players, args.seed + g) for agent in args.agents for g in range(args.games)]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(run_one, jobs))

    print(f"{args.games} games, {args.players} player(s), seeds {args.seed}..{args.seed + args.games - 1}\n")
    print(f"{'agent':<20}{'wins':>6}{'avg royals':>12}{'max':>5}{'avg decisions':>15}{'s/game':>8}")
    for agent in args.agents:
        rows = [r for r in results if r["agent"] == agent]
        wins = sum(r["won"] for r in rows)
        royals = [r["royals"] for r in rows]
        print(f"{agent:<20}{wins:>6}{sum(royals) / len(rows):>12.2f}{max(royals):>5}"
              f"{sum(r['turns'] for r in rows) / len(rows):>15.1f}"
              f"{sum(r['seconds'] for r in rows) / len(rows):>8.1f}")
    print()
    for agent in args.agents:
        rows = [r for r in results if r["agent"] == agent]
        killers = Counter(r["killer"][0] if r["killer"] != "-" else "won" for r in rows)
        print(f"{agent:<20}games ended by: " + ", ".join(f"{k}: {v}" for k, v in sorted(killers.items())))


if __name__ == "__main__":
    main()
