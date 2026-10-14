"""Hybrid search for Regicide: depth-limited expectimax with MCTS at the frontier.

This is a separate algorithm from `minimax.AlphaBetaSearch`. Differences:

* Luck is treated as luck. The GameAgent's moves are chance nodes whose value is the
  expectation over its uniformly random choices (expectimax), not the worst case.
  The pure minimax plans for the worst card at every draw, which makes it play
  too defensively and misjudge long fights such as a Queen of Spades.
* Depth counts player decisions only. A whole chain of random events (e.g. the
  8 cards drawn by a Diamonds play) is one chance node. Small chains are expanded
  exactly; large ones are approximated with a few sampled outcomes.
* Alpha-beta generalised to chance nodes (Ballard's Star1): values are bounded in
  [REPETITION_VALUE, 1], so a chance node can be cut off once its remaining
  outcomes cannot change the decision. Player nodes also stop once a forced win
  (value 1) is found.
* At the last depth, instead of a static heuristic, each frontier state is valued
  by a small Monte Carlo Tree Search (UCT with progressive widening, rule-based
  rollouts, heuristic evaluation at the rollout horizon). Frontier values are
  cached for the duration of a search. With the defaults (depth 1) the search
  looks one player decision ahead exhaustively and relies on MCTS beyond that;
  in benchmarks this beat depth 2 at a fraction of the cost.
* A stronger heuristic (`evaluate_v2`) with a resource-planning term: the team's
  hand value against the estimated cost (cards played plus retaliations) of
  beating every remaining royal, each costed with its own suit immunity.
* Interior player nodes keep only the `beam_width` most promising actions.
* The same repetition rule as `minimax` (a revisited state is worse than a loss),
  using both the game history and the current search path.

Usage mirrors `minimax`:
    search = HybridSearch(depth=1, mcts_iterations=24, seed=0)
    result = search.search(state, history)      # SearchResult(value, action, nodes)
    action = search.choose_action(state, history)
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from minimax import REPETITION_VALUE, SearchResult
from regicide import (
    CLUBS, DIAMONDS, GAME_AGENT, HEARTS, ROYAL_ATTACK, ROYAL_HEALTH, SPADES,
    Action, Card, GameState, Phase, _valid_plays,
    apply_action, game_agent_policy, get_actions, reward, total_attack, total_value,
)

Outcomes = List[Tuple[GameState, float]]

LOWER_BOUND = REPETITION_VALUE  # smallest value any state can have
UPPER_BOUND = 1.0  # a win


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

TOTAL_ROYAL_HEALTH = 4 * sum(ROYAL_HEALTH.values())
V2_WEIGHTS = {"progress": 0.45, "plan": 0.35, "cover": 0.15, "reserve": 0.05}
DRAW_CARD_WORTH = 2.0  # hand value a face-down draw pile card is worth to the plan
PLAN_NORM = 500.0  # (resources - cost) margin mapped onto [0, 1]; roughly a full game's cost
RESERVE_TARGET = 20


def _effective_suits(state: GameState) -> set:
    immune = state.immune_suit
    return {s for s in (CLUBS, SPADES, HEARTS, DIAMONDS) if s != immune}


def best_play(hand: Sequence[Card], suits: set) -> Tuple[int, int]:
    """(damage, card value spent) of the most damaging play from `hand`."""
    best = (0, 0)
    for cards in _valid_plays(tuple(hand)):
        power = total_attack(cards)
        if CLUBS in suits and any(c.suit == CLUBS for c in cards):
            power *= 2
        if power > best[0]:
            best = (power, total_value(cards))
    return best


def best_damage(hand: Sequence[Card], suits: set) -> int:
    """Largest damage one play from `hand` deals, given the suits that take effect."""
    return best_play(hand, suits)[0]


def _core_features(state: GameState, draw_credit: Optional[float] = None) -> Dict[str, float]:
    """The progress / plan / cover / reserve features shared by evaluate_v2 and v3.

    `draw_credit` is the hand value credited to the plan for the draw pile
    (default: DRAW_CARD_WORTH per face-down card).
    """
    if state.active_royal:
        health_left = max(0, state.royal_max_health - state.royal_damage - state.pending_damage)
        attack = state.royal_attack
        suits = _effective_suits(state)
    else:
        health_left, attack = 0, 0
        if state.royals_pile:
            attack = ROYAL_ATTACK[min(c.rank for c in state.royals_pile)]
        suits = {CLUBS, SPADES, HEARTS, DIAMONDS}

    remaining = health_left + sum(ROYAL_HEALTH[c.rank] for c in state.royals_pile)
    progress = min(1.0, max(0.0, 1 - remaining / TOTAL_ROYAL_HEALTH))

    n = state.num_players
    all_suits = {CLUBS, SPADES, HEARTS, DIAMONDS}
    rates: Dict[frozenset, Tuple[float, float]] = {}

    def rate(effective: set) -> Tuple[float, float]:
        """Team's average (damage, card value spent) per turn when `effective` suits work."""
        key = frozenset(effective)
        if key not in rates:
            plays = [best_play(h, effective) for h in state.hands]
            rates[key] = (max(1.0, sum(d for d, _ in plays) / n),
                          max(1.0, sum(v for _, v in plays) / n))
        return rates[key]

    def cost(health: int, royal_attack: int, effective: set) -> float:
        damage_per_turn, value_per_turn = rate(effective)
        turns = math.ceil(health / damage_per_turn)
        return turns * value_per_turn + max(0, turns - 1) * royal_attack

    # Each future royal is costed with its own suit immunity.
    total_cost = sum(cost(ROYAL_HEALTH[c.rank], ROYAL_ATTACK[c.rank], all_suits - {c.suit})
                     for c in state.royals_pile)
    if state.active_royal:
        total_cost += cost(health_left, attack, suits)
        if state.phase == Phase.DISCARD:
            total_cost += attack
    team_value = sum(state.hand_value(p) for p in range(n))
    if draw_credit is None:
        draw_credit = DRAW_CARD_WORTH * state.draw_pile_size
    resources = team_value + draw_credit
    # A difference (not a ratio) so that spending v card value to remove v of cost is neutral.
    plan = min(1.0, max(0.0, 1 + (resources - total_cost) / PLAN_NORM))

    covers = []
    for p in range(n):
        if not state.hands[p]:
            covers.append(0.0)
        elif attack <= 0:
            covers.append(1.0)
        else:
            covers.append(min(1.0, state.hand_value(p) / attack))
    covers.append(covers[state.current_player])
    cover = sum(covers) / len(covers)

    reserve = min(1.0, (state.draw_pile_size + state.discard_pile_size) / RESERVE_TARGET)

    return {"progress": progress, "plan": plan, "cover": cover, "reserve": reserve}


def evaluate_v2(state: GameState) -> float:
    """Heuristic value in [-1, 1] (terminals score their reward).

    Features, weighted by V2_WEIGHTS:
      * progress: share of all royal health already dealt with,
      * plan: the team's resources (hand values, plus some value for the draw
        pile) minus the estimated cost of beating every remaining royal. Each
        royal costs the cards played to deal its health (at the team's best damage
        per card) plus one retaliation per turn except the killing one; a
        retaliation already due in the DISCARD phase is added. Costing all
        remaining royals keeps states on either side of a kill comparable,
      * cover: how well each hand covers the immediate attack (current player x2),
      * reserve: cards that can still return via Hearts / Diamonds.
    """
    if state.is_terminal:
        return float(reward(state))

    f = _core_features(state)
    return 0.95 * (2 * sum(V2_WEIGHTS[k] * f[k] for k in V2_WEIGHTS) - 1)


# -- Sustain-aware heuristic (v3) -------------------------------------------------

def game_phase(state: GameState) -> float:
    """0 while fighting Jacks, 0.5 at the Queens, 1 at the Kings (by royals defeated)."""
    return min(1.0, state.royals_defeated / 8)


def refill_power(state: GameState) -> Tuple[float, float]:
    """(cards the team can draw, cards it can recycle) with the Diamonds / Hearts in hand.

    Each Diamond played draws its value in cards (capped by the draw pile); each
    Heart returns its value in cards from the discard pile to the draw pile. Cards
    of the active royal's suit count half: they only work against later royals.
    """
    immune = state.immune_suit
    diamonds = hearts = 0.0
    for hand in state.hands:
        for c in hand:
            if c.suit == DIAMONDS:
                diamonds += c.attack * (0.5 if immune == DIAMONDS else 1.0)
            elif c.suit == HEARTS:
                hearts += c.attack * (0.5 if immune == HEARTS else 1.0)
    return min(diamonds, state.draw_pile_size), min(hearts, state.discard_pile_size)


def cards_to_cover(hand: Sequence[Card], attack: int) -> int:
    """Fewest cards from `hand` whose values add up to at least `attack`."""
    total = count = 0
    for value in sorted((c.value for c in hand), reverse=True):
        if total >= attack:
            break
        total += value
        count += 1
    return count


@dataclass(frozen=True)
class SustainHeuristic:
    """Heuristic that values card supply more as the game goes on (`evaluate_v3`).

    It blends the v2 features with three sustain features, whose weights move
    linearly from their (early, late) values as `game_phase` goes from 0 to 1:
      * hand: average fraction of each hand that is filled (card count). In the
        DISCARD phase the cards about to be discarded (at least `cards_to_cover`)
        are already subtracted, so a yield is not mistaken for keeping cards,
      * refill: Diamonds that can draw plus half the Hearts that can recycle,
        relative to REFILL_TARGET cards,
      * deck: draw pile size relative to DECK_TARGET cards.
    The v2 features share the remaining weight in their usual proportions. The plan
    term only credits draw pile cards the team can actually draw with its Diamonds
    (at `reachable_draw_worth` each), instead of the whole draw pile.
    """

    hand: Tuple[float, float] = (0.05, 0.25)
    refill: Tuple[float, float] = (0.05, 0.20)
    deck: Tuple[float, float] = (0.02, 0.10)
    reachable_draw_worth: float = 4.0
    refill_target: float = 12.0
    deck_target: float = 20.0

    def weights(self, phase: float) -> Dict[str, float]:
        def lerp(pair):
            return pair[0] + (pair[1] - pair[0]) * phase
        return {"hand": lerp(self.hand), "refill": lerp(self.refill), "deck": lerp(self.deck)}

    def __call__(self, state: GameState) -> float:
        if state.is_terminal:
            return float(reward(state))
        draws, recycles = refill_power(state)
        core = _core_features(state, draw_credit=self.reachable_draw_worth * draws)
        held = sum(len(h) for h in state.hands)
        if state.phase == Phase.DISCARD:
            held -= cards_to_cover(state.current_hand, state.royal_attack)
        sustain = {
            "hand": held / (state.num_players * state.max_hand_size),
            "refill": min(1.0, (draws + 0.5 * recycles) / self.refill_target),
            "deck": min(1.0, state.draw_pile_size / self.deck_target),
        }
        w = self.weights(game_phase(state))
        core_share = 1.0 - sum(w.values())
        score = core_share * sum(V2_WEIGHTS[k] * core[k] for k in V2_WEIGHTS)
        score += sum(w[k] * sustain[k] for k in w)
        return 0.95 * (2 * score - 1)


evaluate_v3 = SustainHeuristic()


# ---------------------------------------------------------------------------
# Rule-based policy (MCTS rollouts and action priors)
# ---------------------------------------------------------------------------


def _play_components(state: GameState, cards: Tuple) -> Dict[str, float]:
    """What a play would do, estimated without applying it."""
    suits = _effective_suits(state)
    played_suits = {c.suit for c in cards if not c.is_joker}
    power = total_attack(cards)
    damage = power * (2 if CLUBS in suits and CLUBS in played_suits else 1)
    shield = power if SPADES in suits and SPADES in played_suits else 0
    draws = power if DIAMONDS in suits and DIAMONDS in played_suits else 0
    draws = min(draws, state.draw_pile_size,
                sum(state.max_hand_size - len(h) for h in state.hands) + len(cards))
    hearts = min(power, state.discard_pile_size) if HEARTS in suits and HEARTS in played_suits else 0
    hand_left = state.hand_value(state.current_player) - total_value(cards)
    health = state.royal_health
    attack_after = max(0, state.royal_attack - shield)
    return {
        "damage": damage, "shield": shield, "draws": draws, "hearts": hearts,
        "value": total_value(cards), "cards": len(cards),
        "kill": damage >= health, "exact": damage == health,
        "loses": hand_left + 5 * draws < attack_after or (hand_left == 0 and draws == 0),
    }


def play_score(state: GameState, cards: Tuple) -> float:
    """Fast score of a play without applying it; higher is better."""
    c = _play_components(state, cards)
    if c["kill"]:
        # Killing is great; an exact kill also returns the royal to the draw pile.
        return 100.0 + (5.0 if c["exact"] else 0.0) - 0.1 * c["value"]
    if c["loses"]:
        return -100.0 + c["damage"]  # this play loses at the retaliation
    return (c["damage"] + 1.2 * c["shield"] + 0.8 * c["draws"] + 0.3 * c["hearts"]
            - 0.5 * c["value"])


def sustain_play_score(state: GameState, cards: Tuple) -> float:
    """Like play_score, but values refilling and spending few cards more as the game goes on.

    Refill effects count even on killing plays (effects resolve before the attack).
    """
    c = _play_components(state, cards)
    phase = game_phase(state)
    w_draw, w_hearts = 0.8 + 1.7 * phase, 0.3 + 0.7 * phase
    refill = w_draw * c["draws"] + w_hearts * c["hearts"]
    if c["kill"]:
        return (100.0 + (5.0 if c["exact"] else 0.0) - 0.1 * c["value"]
                + 0.2 * refill - 0.5 * phase * c["cards"])
    if c["loses"]:
        return -100.0 + c["damage"]
    return (c["damage"] + 1.2 * c["shield"] + refill
            - (0.5 + 0.3 * phase) * c["value"] - 1.0 * phase * c["cards"])


def discard_score(state: GameState, cards: Tuple) -> float:
    """Discard as little value as possible, then keep as many cards as possible."""
    return -total_value(cards) - 0.5 * len(cards)


def sustain_discard_score(state: GameState, cards: Tuple) -> float:
    """Like discard_score, but later in the game keep cards (count) and refill suits."""
    phase = game_phase(state)
    refill_cards = sum(c.value for c in cards if c.suit in (DIAMONDS, HEARTS))
    return -total_value(cards) - (0.5 + 1.5 * phase) * len(cards) - 0.3 * phase * refill_cards


def ordered_actions(state: GameState, sustain: bool = False) -> List[Action]:
    """Actions for the player to act, best first according to the rule-based scores."""
    actions = get_actions(state)
    if state.phase == Phase.PLAY:
        score = sustain_play_score if sustain else play_score
        actions.sort(key=lambda a: score(state, a.cards), reverse=True)
    elif state.phase == Phase.DISCARD:
        score = sustain_discard_score if sustain else discard_score
        actions.sort(key=lambda a: score(state, a.cards), reverse=True)
    return actions


def rollout_action(state: GameState, rng: random.Random, epsilon: float,
                   sustain: bool = False) -> Action:
    actions = ordered_actions(state, sustain)
    if rng.random() < epsilon:
        return rng.choice(actions)
    return actions[0]


def sample_game_agent(state: GameState, rng: random.Random) -> GameState:
    """Resolve GameAgent moves by uniform random sampling."""
    while state.current_agent == GAME_AGENT:
        actions, probs = zip(*game_agent_policy(state))
        state = apply_action(state, rng.choices(actions, weights=probs)[0], check=False)
    return state


# ---------------------------------------------------------------------------
# Leaf MCTS
# ---------------------------------------------------------------------------


class _Node:
    __slots__ = ("visits", "total", "untried", "stats", "children")

    def __init__(self, state: GameState, sustain: bool = False):
        self.visits = 0  # iterations through this node, including its first rollout
        self.total = 0.0
        self.untried = [] if state.is_terminal else ordered_actions(state, sustain)
        self.stats: Dict[Action, List[float]] = {}  # action -> [visits, total value]
        self.children: Dict[Tuple[Action, GameState], "_Node"] = {}


class LeafMCTS:
    """UCT over player decisions; chance is sampled (closed loop: children per outcome).

    Selection uses UCB1 on mean values. The returned value backs up the tree the
    way expectimax does: the best action at player nodes, the visit-weighted
    average over sampled luck below each action, and the plain mean of rollouts at
    nodes that were never expanded. Plain UCT averages would undervalue a forced
    win, since exploratory and noisy rollouts drag the mean down.
    """

    def __init__(self, iterations: int, rollout_depth: int, exploration: float = 0.7,
                 widening: float = 1.5, epsilon: float = 0.15, heuristic=evaluate_v2,
                 backup: str = "mean", sustain: bool = False):
        if backup not in ("max", "mean"):
            raise ValueError("backup must be 'max' or 'mean'")
        self.backup = backup
        self.sustain = sustain  # use the sustain-aware rule-based policy
        self.iterations = iterations
        self.rollout_depth = rollout_depth
        self.exploration = exploration
        self.widening = widening
        self.epsilon = epsilon
        self.heuristic = heuristic

    def value(self, state: GameState, seen: Iterable[GameState], rng: random.Random) -> float:
        if state.is_terminal:
            return float(reward(state))
        seen = set(seen)
        root = _Node(state, self.sustain)
        for _ in range(self.iterations):
            self._iterate(root, state, seen, rng)
        if not root.stats:
            return self.heuristic(state)
        if self.backup == "mean":  # classic UCT: mean of the most visited action
            visits, total = max(root.stats.values(), key=lambda st: (st[0], st[1]))
            return total / visits
        return self._backed_up(root)

    def _backed_up(self, node: _Node) -> float:
        if not node.stats:
            return node.total / node.visits
        by_action: Dict[Action, List[_Node]] = {}
        for (action, _), child in node.children.items():
            by_action.setdefault(action, []).append(child)
        best = -float("inf")
        for action, children in by_action.items():
            weight = sum(c.visits for c in children)
            if weight:
                best = max(best, sum(c.visits * self._backed_up(c) for c in children) / weight)
        return best if best > -float("inf") else node.total / node.visits

    def _iterate(self, root: _Node, root_state: GameState, seen: set, rng: random.Random):
        node, state = root, root_state
        visited = set(seen)
        path: List[Tuple[_Node, Action]] = []
        while True:
            if state.is_terminal:
                value = float(reward(state))
                break
            if state in visited:
                value = REPETITION_VALUE
                break
            visited.add(state)
            limit = max(1, int(self.widening * math.sqrt(node.visits + 1)))
            if node.untried and len(node.stats) < limit:
                action = node.untried.pop(0)
            else:
                action = self._select(node)
            path.append((node, action))
            nxt = sample_game_agent(apply_action(state, action, check=False), rng)
            child = node.children.get((action, nxt))
            if child is None:
                child = node.children[(action, nxt)] = _Node(nxt, self.sustain)
                value = self._rollout(nxt, visited, rng)
                child.visits, child.total = 1, value
                break
            node, state = child, nxt
        for n, a in path:
            n.visits += 1
            n.total += value
            st = n.stats.setdefault(a, [0, 0.0])
            st[0] += 1
            st[1] += value

    def _select(self, node: _Node) -> Action:
        log_n = math.log(node.visits + 1)
        return max(node.stats, key=lambda a: node.stats[a][1] / node.stats[a][0]
                   + self.exploration * math.sqrt(log_n / node.stats[a][0]))

    def _rollout(self, state: GameState, visited: set, rng: random.Random) -> float:
        visited = set(visited)
        for _ in range(self.rollout_depth):
            if state.is_terminal:
                return float(reward(state))
            if state in visited:
                return REPETITION_VALUE
            visited.add(state)
            action = rollout_action(state, rng, self.epsilon, self.sustain)
            state = sample_game_agent(apply_action(state, action, check=False), rng)
        return self.heuristic(state)


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------


class HybridSearch:
    """Depth-limited expectimax (Star1 pruning) with MCTS leaf evaluation.

    depth            player decisions to search exhaustively (chance chains are free)
    beam_width       max actions searched at interior player nodes (root: all)
    max_exact        expand a chance chain exactly when it has at most this many outcomes
    chance_samples   otherwise, average this many sampled outcomes
    mcts_iterations  MCTS iterations per frontier state (0: use the heuristic only)
    rollout_depth    player decisions per MCTS rollout before applying the heuristic
    backup           how MCTS values its root: "max" (expectimax-style) or "mean" (UCT)
    sustain          value card supply more as the game goes on: uses evaluate_v3 (unless
                     `heuristic` is given) and the sustain-aware rollout policy
    heuristic        frontier / rollout-horizon evaluation (default evaluate_v2, or
                     evaluate_v3 with sustain)
    seed             seeds sampling, so a search is reproducible
    """

    def __init__(self, depth: int = 1, beam_width: int = 4, max_exact: int = 6,
                 chance_samples: int = 3, mcts_iterations: int = 24, rollout_depth: int = 6,
                 exploration: float = 0.7, backup: str = "mean", sustain: bool = False,
                 heuristic=None, seed: Optional[int] = 0):
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.depth = depth
        self.beam_width = beam_width
        self.max_exact = max_exact
        self.chance_samples = chance_samples
        self.mcts_iterations = mcts_iterations
        if heuristic is None:
            heuristic = evaluate_v3 if sustain else evaluate_v2
        self.heuristic = heuristic
        self.sustain = sustain
        self.mcts = LeafMCTS(mcts_iterations, rollout_depth, exploration, heuristic=heuristic,
                             backup=backup, sustain=sustain)
        self.seed = seed
        self.rng = random.Random(seed)
        self.nodes = 0
        self._seen: set = set()
        self._leaf_cache: Dict[GameState, float] = {}
        self._chance_cache: Dict[GameState, Outcomes] = {}

    # -- public interface ----------------------------------------------------

    def search(self, state: GameState, history: Iterable[GameState] = ()) -> SearchResult:
        self.nodes = 1
        self._leaf_cache.clear()
        self._chance_cache.clear()
        if state.is_terminal:
            return SearchResult(float(reward(state)), None, self.nodes)
        self._seen = set(history) | {state}

        if state.current_agent == GAME_AGENT:
            # Asked to act for the GameAgent (e.g. an adversarial runner): pick the worst
            # outcome for the players, valuing each by the expectimax below.
            best = (float("inf"), None)
            for action in get_actions(state):
                v = self._chance_value(self._outcomes(apply_action(state, action, check=False)),
                                       self.depth, LOWER_BOUND, UPPER_BOUND)
                if v < best[0]:
                    best = (v, action)
            return SearchResult(best[0], best[1], self.nodes)

        best_value, best_action = -float("inf"), None
        for action, outcomes in self._children(state, beam=False):
            alpha = max(LOWER_BOUND, best_value)
            value = self._chance_value(outcomes, self.depth - 1, alpha, UPPER_BOUND)
            if value > best_value:
                best_value, best_action = value, action
            if best_value >= UPPER_BOUND:
                break
        return SearchResult(best_value, best_action, self.nodes)

    def choose_action(self, state: GameState, history: Iterable[GameState] = ()) -> Optional[Action]:
        return self.search(state, history).action

    # -- internals -------------------------------------------------------------

    def _outcomes(self, state: GameState) -> Outcomes:
        """Distribution over the player/terminal states reached by resolving luck."""
        if state.current_agent != GAME_AGENT:
            return [(state, 1.0)]
        cached = self._chance_cache.get(state)
        if cached is not None:
            return cached
        exact = self._exact_outcomes(state)
        if exact is None:
            counts = Counter(sample_game_agent(state, self.rng) for _ in range(self.chance_samples))
            exact = [(s, c / self.chance_samples) for s, c in counts.items()]
        self._chance_cache[state] = exact
        return exact

    def _exact_outcomes(self, state: GameState) -> Optional[Outcomes]:
        frontier = {state: 1.0}
        result: Dict[GameState, float] = {}
        while frontier:
            nxt: Dict[GameState, float] = {}
            for s, p in frontier.items():
                if s.current_agent == GAME_AGENT:
                    for a, q in game_agent_policy(s):
                        t = apply_action(s, a, check=False)
                        nxt[t] = nxt.get(t, 0.0) + p * q
                else:
                    result[s] = result.get(s, 0.0) + p
            if len(nxt) + len(result) > self.max_exact:
                return None
            frontier = nxt
        return list(result.items())

    def _children(self, state: GameState, beam: bool) -> List[Tuple[Action, Outcomes]]:
        children = []
        for action in get_actions(state):
            outcomes = self._outcomes(apply_action(state, action, check=False))
            prior = sum(p * self.heuristic(s) for s, p in outcomes)
            children.append((prior, action, outcomes))
        children.sort(key=lambda c: c[0], reverse=True)
        if beam:
            children = children[:self.beam_width]
        return [(a, o) for _, a, o in children]

    def _chance_value(self, outcomes: Outcomes, depth: int, alpha: float, beta: float) -> float:
        """Expected value of `outcomes` with Star1 cutoffs against the (alpha, beta) window."""
        if len(outcomes) == 1:
            return self._value(outcomes[0][0], depth, alpha, beta)
        lo, hi = LOWER_BOUND, UPPER_BOUND
        done = 0.0  # sum of p * value over searched outcomes
        left = 1.0  # probability mass not yet searched
        for state, p in sorted(outcomes, key=lambda sp: -sp[1]):
            rest = left - p
            child_alpha = max(lo, (alpha - done - rest * hi) / p)
            child_beta = min(hi, (beta - done - rest * lo) / p)
            value = self._value(state, depth, child_alpha, child_beta)
            done += p * value
            left = rest
            if done + left * hi <= alpha:
                return done + left * hi
            if done + left * lo >= beta:
                return done + left * lo
        return done

    def _value(self, state: GameState, depth: int, alpha: float, beta: float) -> float:
        self.nodes += 1
        if state.is_terminal:
            return float(reward(state))
        if state in self._seen:
            return REPETITION_VALUE
        if depth == 0:
            return self._leaf(state)
        self._seen.add(state)
        try:
            best = LOWER_BOUND
            for _, outcomes in self._children(state, beam=True):
                best = max(best, self._chance_value(outcomes, depth - 1, max(alpha, best), beta))
                if best >= beta or best >= UPPER_BOUND:
                    break
            return best
        finally:
            self._seen.discard(state)

    def _leaf(self, state: GameState) -> float:
        cached = self._leaf_cache.get(state)
        if cached is None:
            if self.mcts_iterations > 0:
                cached = self.mcts.value(state, self._seen, self.rng)
            else:
                cached = self.heuristic(state)
            self._leaf_cache[state] = cached
        return cached


def hybrid_action(state: GameState, depth: int = 1, history: Iterable[GameState] = (),
                  **kwargs) -> Optional[Action]:
    """Convenience wrapper: the hybrid search's action for `state`."""
    return HybridSearch(depth, **kwargs).choose_action(state, history)
