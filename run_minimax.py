"""Play a random Regicide game with depth-limited minimax players and print the trajectory.

Usage:
    python3 run_minimax.py [--players N] [--depth D] [--seed S] [--adversarial]

The players choose actions with alpha-beta minimax. Luck (deals, reveals, draws,
Hearts) is random by default; with --adversarial the GameAgent plays minimax too.
"""

import argparse
import random
import time

from minimax import AlphaBetaSearch
from regicide import (
    GAME_AGENT, DiscardCards, Phase, apply_action, game_agent_policy, initial_state,
)


def cards_str(cards):
    return " ".join(map(str, cards)) or "-"


def royal_str(state):
    if not state.active_royal:
        return "no active royal"
    extra = ", joker played" if state.joker_played else ""
    return (f"{state.active_royal} hp {state.royal_health}/{state.royal_max_health} "
            f"atk {state.royal_attack}{extra}")


def describe_game_agent(state, action):
    """Human-readable description of a GameAgent action about to be applied."""
    card = action.card
    if state.phase == Phase.REVEAL:
        return f"reveals royal {card}"
    if state.phase == Phase.RESOLVE and state.pending_hearts > 0:
        return f"Hearts returns {card} from the discard pile to the draw pile"
    after = apply_action(state, action)
    drawer = next(p for p in range(state.num_players)
                  if len(after.hands[p]) > len(state.hands[p]))
    verb = "is dealt" if state.phase == Phase.DEAL else "draws"
    return f"P{drawer} {verb} {card}"


def describe_outcome(before, after):
    """Notable effects of a play once its random effects have resolved."""
    notes = []
    if before.phase == Phase.PLAY:
        royal = before.active_royal
        if after.active_royal == royal:
            damage = after.royal_damage - before.royal_damage
            notes.append(f"{royal} takes {damage} damage -> hp {after.royal_health}")
            if after.royal_shield > before.royal_shield:
                notes.append(f"Spades shield {royal}'s attack to {after.royal_attack}")
        elif after.active_royal is None:
            how = ("exact kill: placed on top of the draw pile"
                   if after.draw_pile and after.draw_pile[0] == (royal,) else "removed from the game")
            notes.append(f"{royal} defeated ({how})")
        if after.phase == Phase.DISCARD:
            notes.append(f"{royal} retaliates for {after.royal_attack}")
        elif not after.is_terminal and after.current_player != before.current_player:
            notes.append(f"{royal} has no attack left; turn passes")
    return notes


def add_common_arguments(parser):
    parser.add_argument("--players", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: random)")
    parser.add_argument("--adversarial", action="store_true",
                        help="GameAgent also uses the search (worst case) instead of random luck")
    parser.add_argument("--max-moves", type=int, default=2000, help="safety cap on player moves")


def run_game(search, players, seed=None, adversarial=False, max_moves=2000, label="search"):
    """Deal a random game, play it with `search` (anything with .search(state, history)),
    and print the trajectory. Returns the final state."""
    seed = seed if seed is not None else random.randrange(10**6)
    rng = random.Random(seed)
    print(f"Regicide: {players} player(s), {label}, seed {seed}"
          f"{', adversarial GameAgent' if adversarial else ''}\n")

    # Deal randomly, then show the starting hands.
    state = initial_state(players)
    while state.phase == Phase.DEAL:
        actions, probs = zip(*game_agent_policy(state))
        state = apply_action(state, rng.choices(actions, weights=probs)[0])
    print("Starting hands:")
    for p, hand in enumerate(state.hands):
        print(f"  P{p}: {cards_str(hand)}")
    print()

    moves = turns = 0
    total_nodes = 0
    started = time.time()
    history = set()  # every state the game has passed through, for repetition checks
    stalemate = False

    def game_agent_move(state):
        history.add(state)
        if adversarial:
            action = search.choose_action(state, history)
        else:
            actions, probs = zip(*game_agent_policy(state))
            action = rng.choices(actions, weights=probs)[0]
        print(f"    GameAgent {describe_game_agent(state, action)}")
        return apply_action(state, action)

    while not state.is_terminal and moves < max_moves:
        if state in history:  # only possible if every available move repeats a state
            stalemate = True
            break
        if state.current_agent == GAME_AGENT:
            state = game_agent_move(state)
            continue

        moves += 1
        p = state.current_player
        move_started = time.time()
        result = search.search(state, history)
        history.add(state)
        total_nodes += result.nodes
        verb = "discards" if isinstance(result.action, DiscardCards) else "plays"
        chosen = cards_str(result.action.cards) if result.action.cards else "nothing (yield)"
        if state.phase == Phase.PLAY:
            turns += 1
            print(f"Turn {turns}: [{royal_str(state)} | royals left {len(state.royals_pile)} | "
                  f"draw {state.draw_pile_size} | discard {state.discard_pile_size}]")
        print(f"  P{p} hand {cards_str(state.hand(p))} -> {verb} {chosen} "
              f"(value {result.value:+.3f}, {result.nodes} nodes, {time.time() - move_started:.1f}s)",
              flush=True)
        before = state
        state = apply_action(state, result.action)
        while state.phase == Phase.RESOLVE:  # Hearts / Diamonds luck before the attack lands
            state = game_agent_move(state)
        for note in describe_outcome(before, state):
            print(f"    {note}")

    elapsed = time.time() - started
    print()
    if stalemate:
        print("Result: the players LOSE (reward -1): the game repeated a position (endless yielding)")
    elif state.is_won:
        print("Result: the players WIN (reward +1)")
    elif state.is_lost:
        empty = [f"P{p}" for p, h in enumerate(state.hands) if not h]
        why = f"{', '.join(empty)} has no cards" if empty else f"P{state.current_player} cannot absorb the attack"
        print(f"Result: the players LOSE (reward -1): {why}")
    else:
        print(f"Stopped after {max_moves} moves without finishing (reward 0)")
    print(f"Royals defeated: {state.royals_defeated}/12, turns: {turns}, player decisions: {moves}, "
          f"nodes searched: {total_nodes}, time: {elapsed:.1f}s")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--depth", type=int, default=3, help="search depth in plies (default 3)")
    args = parser.parse_args()
    run_game(AlphaBetaSearch(args.depth), args.players, args.seed, args.adversarial,
             args.max_moves, label=f"minimax depth {args.depth}")


if __name__ == "__main__":
    main()
