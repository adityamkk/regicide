"""Depth-limited minimax with alpha-beta pruning for Regicide.

The players cooperate, so every player node is a MAX node. The GameAgent, whose
choices model luck (deals, draws, reveals, Hearts), is treated as an adversary and
is a MIN node: the search assumes the worst-case card falls at every random event.

Depth is measured in plies of `regicide.apply_action`, so each atomic GameAgent
choice (e.g. one card drawn for Diamonds) uses one ply, like a player decision.

Repetition: once a royal's attack is shielded to 0, the players can yield forever
and the game returns to exactly the same state. Any line of play that revisits a
state (earlier on the search path, or earlier in the actual game via `history`) is
scored as a loss. This loses no wins: if a winning line exists, cutting out its
repeated stretch leaves a winning line without repetition. Scoring a cycle as (just
below) a loss rather than 0 also stops the players stalling forever to avoid a loss.
Since the game has finitely many states, a game with no repeated states must end.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Iterable, List, Optional, Tuple

from regicide import (
    GAME_AGENT, ROYAL_ATTACK, ROYAL_HEALTH, Action, GameState,
    apply_action, game_agent_policy, get_actions, reward,
)

Heuristic = Callable[[GameState], float]

TOTAL_ROYAL_HEALTH = 4 * sum(ROYAL_HEALTH.values())  # 360

# Weights of the heuristic's features; they sum to 1.
W_PROGRESS = 0.55  # how much royal health has been dealt with
W_COVER = 0.20  # can players absorb the royal's retaliation?
W_HAND = 0.15  # how full the players' hands are
W_RESERVE = 0.10  # cards that can still come back via Hearts / Diamonds
RESERVE_TARGET = 20  # draw + discard cards at which the reserve feature saturates
HEURISTIC_SCALE = 0.95  # keeps non-terminal values strictly inside (-1, 1)
# Value of revisiting a state. A cycle can never be won, so it is a loss; scoring it
# just below a real loss also makes the search prefer finishing a lost game over
# stalling, so a repeated state only happens if every available move repeats.
REPETITION_VALUE = -1.01


def _threat(state: GameState) -> int:
    """Attack the players must be able to absorb: the active royal's, else the next one's."""
    if state.active_royal:
        return state.royal_attack
    if state.royals_pile:
        return ROYAL_ATTACK[min(c.rank for c in state.royals_pile)]
    return 0


def evaluate(state: GameState) -> float:
    """Heuristic value of `state` from the players' point of view, in [-1, 1].

    Terminal states score their reward (+1 / -1). Other states score strictly
    between those, combining:
      * progress: fraction of all royal health already dealt (defeated royals,
        damage on the active royal and damage about to be applied),
      * cover: how well each player's hand value covers the royal's attack, since
        a player who cannot discard enough loses the game (an empty hand counts 0),
      * hand fill: cards held relative to the maximum hand sizes,
      * reserve: cards in the draw and discard piles that can refill hands.
    """
    if state.is_terminal:
        return float(reward(state))

    remaining = sum(ROYAL_HEALTH[c.rank] for c in state.royals_pile)
    if state.active_royal:
        remaining += state.royal_max_health - state.royal_damage - state.pending_damage
    progress = min(1.0, max(0.0, 1 - remaining / TOTAL_ROYAL_HEALTH))

    threat = _threat(state)
    covers = []
    for p in range(state.num_players):
        if not state.hands[p]:
            covers.append(0.0)
        elif threat <= 0:
            covers.append(1.0)
        else:
            covers.append(min(1.0, state.hand_value(p) / threat))
    # The current player faces the threat first, so their cover counts double.
    covers.append(covers[state.current_player])
    cover = sum(covers) / len(covers)

    hand_fill = sum(len(h) for h in state.hands) / (state.num_players * state.max_hand_size)
    reserve = min(1.0, (state.draw_pile_size + state.discard_pile_size) / RESERVE_TARGET)

    score = W_PROGRESS * progress + W_COVER * cover + W_HAND * hand_fill + W_RESERVE * reserve
    return HEURISTIC_SCALE * (2 * score - 1)


@dataclass
class SearchResult:
    value: float
    action: Optional[Action]
    nodes: int  # states visited by the search


class AlphaBetaSearch:
    """Depth-limited minimax with alpha-beta pruning.

    `order_moves` sorts children by the heuristic (best first for the side to move),
    which makes cutoffs happen earlier without changing the result.
    `avoid_repetition` scores any revisited state as REPETITION_VALUE (see the
    module docstring); disabling it lets the search choose infinite yield loops.
    """

    def __init__(self, depth: int, heuristic: Heuristic = evaluate, order_moves: bool = True,
                 avoid_repetition: bool = True):
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.depth = depth
        self.heuristic = heuristic
        self.order_moves = order_moves
        self.avoid_repetition = avoid_repetition
        self.nodes = 0
        self._seen = set()  # game history plus the states on the current search path

    def search(self, state: GameState, history: Iterable[GameState] = ()) -> SearchResult:
        """Best action for whoever acts in `state` and its minimax value.

        `history` holds the states the actual game has already passed through;
        moves that lead back to one of them count as repetition.
        """
        self.nodes = 1
        if state.is_terminal:
            return SearchResult(float(reward(state)), None, self.nodes)
        self._seen = set(history) | {state} if self.avoid_repetition else set()
        maximizing = state.current_agent != GAME_AGENT
        alpha, beta = -float("inf"), float("inf")
        best_value = -float("inf") if maximizing else float("inf")
        best_action = None
        for action, child in self._children(state, maximizing):
            value = self._alphabeta(child, self.depth - 1, alpha, beta)
            if maximizing and value > best_value:
                best_value, best_action = value, action
                alpha = max(alpha, value)
            elif not maximizing and value < best_value:
                best_value, best_action = value, action
                beta = min(beta, value)
        return SearchResult(best_value, best_action, self.nodes)

    def choose_action(self, state: GameState, history: Iterable[GameState] = ()) -> Optional[Action]:
        return self.search(state, history).action

    def _children(self, state: GameState, maximizing: bool) -> List[Tuple[Action, GameState]]:
        children = [(a, apply_action(state, a)) for a in get_actions(state)]
        if self.order_moves:
            children.sort(key=lambda ac: self.heuristic(ac[1]), reverse=maximizing)
        return children

    def _alphabeta(self, state: GameState, depth: int, alpha: float, beta: float) -> float:
        self.nodes += 1
        if state.is_terminal:
            return float(reward(state))
        if state in self._seen:
            return REPETITION_VALUE
        if depth == 0:
            return self.heuristic(state)
        if not self.avoid_repetition:
            return self._expand(state, depth, alpha, beta)
        self._seen.add(state)
        try:
            return self._expand(state, depth, alpha, beta)
        finally:
            self._seen.discard(state)

    def _expand(self, state: GameState, depth: int, alpha: float, beta: float) -> float:
        if state.current_agent != GAME_AGENT:
            value = -float("inf")
            for _, child in self._children(state, True):
                value = max(value, self._alphabeta(child, depth - 1, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        value = float("inf")
        for _, child in self._children(state, False):
            value = min(value, self._alphabeta(child, depth - 1, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


def minimax_action(state: GameState, depth: int, heuristic: Heuristic = evaluate,
                   history: Iterable[GameState] = ()) -> Optional[Action]:
    """Convenience wrapper: the alpha-beta action for `state` at `depth`."""
    return AlphaBetaSearch(depth, heuristic).choose_action(state, history)


def play_out(state: GameState, search: AlphaBetaSearch, rng: Optional[random.Random] = None,
             adversarial: bool = False) -> Tuple[GameState, List[Tuple[GameState, Action]], bool]:
    """Play `state` to the end with minimax players.

    Luck is random unless `adversarial`, in which case the GameAgent also uses
    `search`. Returns (final_state, [(state, action), ...], stalemate). If a state
    repeats anyway (only possible if every available move repeats a state), the
    game stops there with stalemate=True and should be counted as a loss.
    """
    rng = rng or random
    history = set()
    trajectory = []
    while not state.is_terminal:
        if state in history:
            return state, trajectory, True
        history.add(state)
        if state.current_agent == GAME_AGENT and not adversarial:
            actions, probs = zip(*game_agent_policy(state))
            action = rng.choices(actions, weights=probs)[0]
        else:
            action = search.choose_action(state, history)
        trajectory.append((state, action))
        state = apply_action(state, action)
    return state, trajectory, False
