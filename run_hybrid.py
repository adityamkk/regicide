"""Play a random Regicide game with the hybrid (expectimax + MCTS) search and print the trajectory.

Usage:
    python3 run_hybrid.py [--players N] [--depth D] [--iterations I] [--seed S] [--sustain] [--adversarial]

The players choose actions with hybrid.HybridSearch. Luck (deals, reveals, draws,
Hearts) is random by default; with --adversarial the GameAgent picks the worst card.
The output format matches run_minimax.py.
"""

import argparse

from hybrid import HybridSearch
from run_minimax import add_common_arguments, run_game


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--depth", type=int, default=1,
                        help="player decisions searched before the MCTS frontier (default 1)")
    parser.add_argument("--iterations", type=int, default=24, help="MCTS iterations per frontier state")
    parser.add_argument("--rollout-depth", type=int, default=6, help="player decisions per MCTS rollout")
    parser.add_argument("--beam", type=int, default=4, help="actions searched at interior nodes")
    parser.add_argument("--samples", type=int, default=3, help="sampled outcomes for large random events")
    parser.add_argument("--backup", choices=["mean", "max"], default="mean",
                        help="how MCTS values a frontier state")
    parser.add_argument("--sustain", action="store_true",
                        help="value full hands, refills and the draw pile more as the game goes on")
    args = parser.parse_args()

    search = HybridSearch(depth=args.depth, beam_width=args.beam, chance_samples=args.samples,
                          mcts_iterations=args.iterations, rollout_depth=args.rollout_depth,
                          backup=args.backup, sustain=args.sustain, seed=args.seed)
    label = (f"hybrid depth {args.depth}, {args.iterations} MCTS iterations, "
             f"rollout depth {args.rollout_depth}{', sustain-aware' if args.sustain else ''}")
    run_game(search, args.players, args.seed, args.adversarial, args.max_moves, label=label)


if __name__ == "__main__":
    main()
