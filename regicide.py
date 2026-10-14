'''
Regicide is a cooperative multiplayer game of up to 4 players.
The game consists of a standard deck of 52 playing cards. (plus some Jokers)
Unlike the normal game, assume agents have perfect information about the hands of everyone else.
Player agents cannot see the face down cards however.

The 12 face cards (Jacks, Queens, Kings) are shuffled and stored in a "royals" pile face down.
The face cards are shuffled such that all the Jacks are above all the Queens, which are above all the Kings.
That is, when drawing from this pile, all 4 Jacks must be drawn before any Queens can be drawn, etc.
There is also an active royal card, which is initially empty.
Jacks have 20 health and 10 attack
Queens have 30 health and 15 attack
Kings have 40 health and 20 attack

The remaining cards are shuffled and kept face down in a draw pile.
There is also a discard pile, which is initially empty.

Each player has a hand of 0 or more cards, with some maximum hand size per player.
If the game has 1 player, the maximum hand size is 8 (and 0 Jokers are added to the standard deck)
If the game has 2 players, the maximum hand size is 7 (and 0 Jokers are added to the standard deck)
If the game has 3 players, the maximum hand size is 6 (and 1 Joker is added to the standard deck)
If the game has 4 players, the maximum hand size is 5 (and 2 Jokers are added to the standard deck)

The players win the game if the royals pile is empty and the active royal card is empty.
The players lose the game if at least one player has 0 cards in their hand.

For ease of implementation and description, assume luck based events (e.g. card draw, shuffling, royals pile draw) are actions taken by a GameAgent (rather than a PlayerAgent)
The GameAgent loses if the players win, and wins if the players lose.

In the initial state, the GameAgent selects NUM_PLAYERS groups of cards of size MAX_HAND_SIZE and gives them to each player.

Gameplay:
If there is no card in the active royal card slot, the GameAgent selects a card to place into that slot from the royals pile (provided that all Jacks are placed before all Queens, etc.)
If there is a card in the active royal card slot, the next player (player X) goes.
Player X first chooses a set of cards to play from their hand, provided that the set of cards is valid (see below for which sets are valid).
Upon selection, a few things happen:
1. Effect Resolution: A set of cards selected also have a set of effects {hearts, diamonds, spades, clubs}, which trigger depending on the suit of the cards played.
A set of cards also has a total value, denoted by the sum of the values of the cards (J,Q,K have value = their health).
Each effect resolves in order of [clubs, spades, hearts, diamonds].
    Clubs: Doubles all damage dealt to the active royal.
    Spades: Reduces attack of the active royal by the total value of played cards.
    Hearts: Moves cards in the discard pile to the draw pile. The number of cards moves is equal to the total value of played cards.
    Diamonds: All players draw cards from the draw pile in player order, starting from the current player. A player draws only if the draw pile is not empty and they are not at their max hand size. Draw cards until no players can draw or the number of drawn cards equals the total value of the played cards.
Any effects which are the same suit as the active royal do not take effect, unless a player previously played a Joker against the same royal.
2. Attack Phase: Players deal damage based on the total value of their cards (if the clubs effect is on, the damage doubles).
This subtracts from the active royal's total health. 
If the active royal's total health goes strictly below zero, that royal is permanently removed from the game.
If the active royal's total health goes to exactly zero, that royal is placed face down at the top of the draw pile.
In either of these cases, the current player then becomes the GameAgent, which selects the next royal from the royals pile into the active royal slot.
3. Retaliation Phase: The active royal attacks the current player with their attack stat (which may be decreased via effects of spades suit).
The current player must discard cards whose total value is at least the attack value.
The current player becomes the player after the current player.

VALID CARD PLAYS:
Any set of cards is valid to play as long as the combined sum of the played cards does not exceed 10,
and, when more than one card is played, every card in the set has the same value (e.g. 2S 2D is valid, 3S 6D is not).
Royals (J, Q, K) are an exception to this rule: a single royal may be played, but only on its own.
If a joker is played, the active royal can no longer negate effects of its suit.
Jokers have 10 health, but 0 attack.
'''

# ---------------------------------------------------------------------------
# Implementation
#
# Interpretation notes (where the description above leaves room):
#   * Card values: A=1, 2-10 face value, J/Q/K = their health (20/30/40), Joker = 10.
#     Card values count toward the "sum <= 10" play limit and toward discarding.
#     A royal in hand is exempt from that limit but may only be played alone; it then
#     deals damage and powers its suit's effect with its attack (J=10, Q=15, K=20).
#   * Damage / effect strength of a play uses each card's *attack*, which equals its
#     value except for royals (their attack) and Jokers (0). So a Joker can only be played alone, deals
#     no damage, and removes the active royal's suit immunity.
#   * Playing the empty set ("yielding") is a valid play: it has sum 0 <= 10.
#   * Cards played against a royal stay in a "played" pile and move to the discard
#     pile when that royal is defeated. Cards discarded during retaliation go
#     straight to the discard pile.
#   * After a royal is defeated there is no retaliation; the GameAgent reveals the
#     next royal and the same player takes the next turn.
#   * The players lose if the current player cannot discard enough value to cover
#     the royal's attack, or if any player holds 0 cards when a player is about to
#     make a decision (checked after an action and all of its effects resolve).
#   * The draw pile is face down: it is stored as a stack of segments, each an
#     unordered multiset whose internal order nobody knows. A royal placed on top
#     after an exact kill is its own (known) segment; cards returned by Hearts form a
#     new segment at the bottom. Draws come uniformly from the top non-empty segment.
#   * Luck is modelled as explicit, atomic GameAgent actions (deal one card, reveal
#     one royal, move one card for Hearts, draw one card for Diamonds).
# ---------------------------------------------------------------------------

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, fields
from enum import Enum
from typing import Iterable, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union

CLUBS, SPADES, HEARTS, DIAMONDS = "C", "S", "H", "D"
SUITS = (CLUBS, SPADES, HEARTS, DIAMONDS)  # also the effect resolution order
JOKER_SUIT = "*"

JACK, QUEEN, KING = 11, 12, 13
ROYAL_RANKS = (JACK, QUEEN, KING)
ROYAL_HEALTH = {JACK: 20, QUEEN: 30, KING: 40}
ROYAL_ATTACK = {JACK: 10, QUEEN: 15, KING: 20}
JOKER_HEALTH = 10
JOKER_ATTACK = 0
MAX_PLAY_VALUE = 10

MAX_HAND_SIZE = {1: 8, 2: 7, 3: 6, 4: 5}
NUM_JOKERS = {1: 0, 2: 0, 3: 1, 4: 2}

GAME_AGENT = -1

_RANK_NAMES = {0: "Jo", 1: "A", JACK: "J", QUEEN: "Q", KING: "K"}
_NAME_RANKS = {"A": 1, "J": JACK, "Q": QUEEN, "K": KING}


class Card(NamedTuple):
    rank: int  # 1 (Ace) .. 13 (King); 0 for a Joker
    suit: str  # one of SUITS, or JOKER_SUIT

    @property
    def is_joker(self) -> bool:
        return self.suit == JOKER_SUIT

    @property
    def is_royal(self) -> bool:
        return not self.is_joker and self.rank in ROYAL_HEALTH

    @property
    def value(self) -> int:
        """Value used for the play limit and for discarding."""
        if self.is_joker:
            return JOKER_HEALTH
        if self.is_royal:
            return ROYAL_HEALTH[self.rank]
        return self.rank

    @property
    def attack(self) -> int:
        """Contribution of this card to damage and effect strength when played."""
        if self.is_joker:
            return JOKER_ATTACK
        if self.is_royal:
            return ROYAL_ATTACK[self.rank]
        return self.value

    def __repr__(self) -> str:
        if self.is_joker:
            return "JK"
        return f"{_RANK_NAMES.get(self.rank, str(self.rank))}{self.suit}"

    __str__ = __repr__


JOKER = Card(0, JOKER_SUIT)
Cards = Tuple[Card, ...]


def parse_card(text: str) -> Card:
    """Parse a card such as '5H', '10C', 'AS', 'KD' or 'JK' (Joker)."""
    text = text.strip().upper()
    if text in ("JK", "JOKER"):
        return JOKER
    rank_text, suit = text[:-1], text[-1]
    if suit not in SUITS:
        raise ValueError(f"bad suit in card {text!r}")
    rank = _NAME_RANKS[rank_text] if rank_text in _NAME_RANKS else int(rank_text)
    if not 1 <= rank <= 13:
        raise ValueError(f"bad rank in card {text!r}")
    return Card(rank, suit)


def parse_cards(cards: Union[str, Iterable[Union[Card, str]]]) -> Cards:
    """Parse '5H 3C KS' (or an iterable of Cards / strings) into a sorted tuple."""
    if isinstance(cards, str):
        cards = cards.split()
    return tuple(sorted(c if isinstance(c, Card) else parse_card(c) for c in cards))


def total_value(cards: Iterable[Card]) -> int:
    return sum(c.value for c in cards)


def total_attack(cards: Iterable[Card]) -> int:
    return sum(c.attack for c in cards)


def full_deck(num_players: int) -> Cards:
    """Every card in a game with this many players (number cards, jokers, royals)."""
    numbers = [Card(r, s) for s in SUITS for r in range(1, 11)]
    royals = [Card(r, s) for r in ROYAL_RANKS for s in SUITS]
    return tuple(sorted(numbers + royals + [JOKER] * NUM_JOKERS[num_players]))


class Phase(Enum):
    DEAL = "deal"  # GameAgent deals the initial hands
    REVEAL = "reveal"  # GameAgent reveals the next royal
    PLAY = "play"  # current player chooses cards to play
    RESOLVE = "resolve"  # GameAgent resolves Hearts / Diamonds randomness
    DISCARD = "discard"  # current player discards to survive retaliation
    WON = "won"
    LOST = "lost"


GAME_AGENT_PHASES = (Phase.DEAL, Phase.REVEAL, Phase.RESOLVE)
TERMINAL_PHASES = (Phase.WON, Phase.LOST)


@dataclass(frozen=True)
class PlayCards:
    """Player action: play these cards against the active royal."""

    cards: Cards

    def __post_init__(self):
        object.__setattr__(self, "cards", tuple(sorted(self.cards)))


@dataclass(frozen=True)
class DiscardCards:
    """Player action: discard these cards to survive the royal's retaliation."""

    cards: Cards

    def __post_init__(self):
        object.__setattr__(self, "cards", tuple(sorted(self.cards)))


@dataclass(frozen=True)
class GameAgentAction:
    """GameAgent action: the card chosen by luck (dealt, revealed, moved or drawn)."""

    card: Card


Action = Union[PlayCards, DiscardCards, GameAgentAction]


@dataclass(frozen=True)
class GameState:
    """An immutable, hashable Regicide game state.

    All card collections are sorted tuples (multisets). `draw_pile` is a tuple of
    segments ordered top first; see the interpretation notes above.
    """

    num_players: int
    hands: Tuple[Cards, ...]
    draw_pile: Tuple[Cards, ...]
    discard_pile: Cards
    played_cards: Cards
    royals_pile: Cards
    removed_cards: Cards
    active_royal: Optional[Card]
    royal_damage: int  # damage dealt to the active royal so far
    royal_shield: int  # total Spades reduction applied to the active royal's attack
    joker_played: bool  # a Joker was played against the active royal
    current_player: int
    phase: Phase
    # Bookkeeping for a play whose random effects are still being resolved.
    pending_damage: int = 0
    pending_hearts: int = 0
    pending_draws: int = 0
    next_drawer: int = 0

    def __hash__(self) -> int:
        # States are hashed constantly by search code, so cache the hash.
        h = self.__dict__.get("_hash")
        if h is None:
            h = hash(tuple(self.__dict__[f] for f in _STATE_FIELDS))
            object.__setattr__(self, "_hash", h)
        return h

    # -- agents / terminal status ------------------------------------------

    @property
    def max_hand_size(self) -> int:
        return MAX_HAND_SIZE[self.num_players]

    @property
    def current_agent(self) -> Optional[int]:
        """Index of the player to act, GAME_AGENT for luck, or None if the game is over."""
        if self.phase in TERMINAL_PHASES:
            return None
        if self.phase in GAME_AGENT_PHASES:
            return GAME_AGENT
        return self.current_player

    @property
    def is_terminal(self) -> bool:
        return self.phase in TERMINAL_PHASES

    @property
    def is_won(self) -> bool:
        return self.phase == Phase.WON

    @property
    def is_lost(self) -> bool:
        return self.phase == Phase.LOST

    # -- active royal --------------------------------------------------------

    @property
    def royal_max_health(self) -> Optional[int]:
        return ROYAL_HEALTH[self.active_royal.rank] if self.active_royal else None

    @property
    def royal_health(self) -> Optional[int]:
        """Remaining health of the active royal."""
        return self.royal_max_health - self.royal_damage if self.active_royal else None

    @property
    def royal_base_attack(self) -> Optional[int]:
        return ROYAL_ATTACK[self.active_royal.rank] if self.active_royal else None

    @property
    def royal_attack(self) -> Optional[int]:
        """Attack of the active royal after Spades reductions (never below 0)."""
        if not self.active_royal:
            return None
        return max(0, self.royal_base_attack - self.royal_shield)

    @property
    def royal_suit(self) -> Optional[str]:
        return self.active_royal.suit if self.active_royal else None

    @property
    def immune_suit(self) -> Optional[str]:
        """Suit whose effects the active royal currently negates, if any."""
        if not self.active_royal or self.joker_played:
            return None
        return self.active_royal.suit

    # -- hands and piles -----------------------------------------------------

    def hand(self, player: int) -> Cards:
        return self.hands[player]

    def hand_value(self, player: int) -> int:
        return total_value(self.hands[player])

    @property
    def current_hand(self) -> Cards:
        return self.hands[self.current_player]

    @property
    def draw_pile_cards(self) -> Cards:
        return tuple(c for segment in self.draw_pile for c in segment)

    @property
    def draw_pile_size(self) -> int:
        return sum(len(segment) for segment in self.draw_pile)

    @property
    def discard_pile_size(self) -> int:
        return len(self.discard_pile)

    @property
    def royals_remaining(self) -> int:
        """Royals not yet defeated (pile plus active)."""
        return len(self.royals_pile) + (1 if self.active_royal else 0)

    @property
    def royals_defeated(self) -> int:
        return len(ROYAL_RANKS) * len(SUITS) - self.royals_remaining

    def all_cards(self) -> Cards:
        cards = [c for h in self.hands for c in h]
        cards += self.draw_pile_cards + self.discard_pile + self.played_cards
        cards += self.royals_pile + self.removed_cards
        if self.active_royal:
            cards.append(self.active_royal)
        return tuple(sorted(cards))

    # -- validity ------------------------------------------------------------

    def validate(self) -> bool:
        """Raise ValueError if this state breaks a game invariant, else return True."""

        def check(cond: bool, msg: str):
            if not cond:
                raise ValueError(f"invalid state: {msg}\n{self}")

        n = self.num_players
        check(n in MAX_HAND_SIZE, "1-4 players")
        check(len(self.hands) == n, "one hand per player")
        check(0 <= self.current_player < n, "current player in range")
        collections = list(self.hands) + list(self.draw_pile)
        collections += [self.discard_pile, self.played_cards, self.royals_pile, self.removed_cards]
        check(all(list(c) == sorted(c) for c in collections), "card collections are sorted")
        check(Counter(self.all_cards()) == Counter(full_deck(n)), "every card is in exactly one place")
        check(all(len(h) <= self.max_hand_size for h in self.hands), "hands within max size")
        check(all(c.is_royal for c in self.royals_pile), "royals pile holds only royals")

        empty_segments = [i for i, seg in enumerate(self.draw_pile) if not seg]
        hearts_open = self.phase == Phase.RESOLVE and self.pending_hearts > 0
        allowed_empty = [len(self.draw_pile) - 1] if hearts_open else []
        check(all(i in allowed_empty for i in empty_segments), "no empty draw pile segments")

        in_fight = self.phase in (Phase.PLAY, Phase.RESOLVE, Phase.DISCARD)
        if in_fight:
            check(self.active_royal is not None and self.active_royal.is_royal, "active royal present")
            check(all(c.rank >= self.active_royal.rank for c in self.royals_pile), "royal order")
            check(0 <= self.royal_damage < self.royal_max_health, "royal still alive")
            check(self.royal_shield >= 0, "shield non-negative")
        elif self.phase != Phase.LOST:  # a lost game keeps the royal that beat the players
            check(self.active_royal is None, "no active royal outside a fight")
            check(not self.played_cards, "no played cards outside a fight")
            check(self.royal_damage == 0 and self.royal_shield == 0 and not self.joker_played,
                  "royal state reset")

        if self.phase != Phase.RESOLVE:
            check(self.pending_damage == self.pending_hearts == self.pending_draws == 0,
                  "no pending effects outside resolution")
        else:
            check(self.pending_damage >= 0 and self.pending_hearts >= 0 and self.pending_draws >= 0,
                  "pending effects non-negative")
            check(self.pending_hearts <= len(self.discard_pile), "hearts limited by discard pile")
            check(_chance_source(self) != (), "resolution has a random event to resolve")

        if self.phase in (Phase.PLAY, Phase.DISCARD):
            check(all(self.hands), "no player has an empty hand at a decision point")
        if self.phase == Phase.DISCARD:
            check(0 < self.royal_attack <= self.hand_value(self.current_player), "discard is possible")
        if self.phase == Phase.DEAL:
            check(_next_to_deal(self) is not None, "someone still needs cards")
        if self.phase == Phase.REVEAL:
            check(bool(self.royals_pile), "a royal is left to reveal")
        if self.phase == Phase.WON:
            check(not self.royals_pile and self.active_royal is None, "all royals defeated")
        return True

    def __str__(self) -> str:
        royal = "none"
        if self.active_royal:
            royal = (f"{self.active_royal} hp {self.royal_health}/{self.royal_max_health} "
                     f"atk {self.royal_attack}" + (" (joker)" if self.joker_played else ""))
        lines = [f"phase={self.phase.value} current_player={self.current_player} royal={royal}"]
        for p, h in enumerate(self.hands):
            lines.append(f"  P{p}: {' '.join(map(str, h)) or '-'}")
        lines.append(f"  draw={self.draw_pile_size} discard={self.discard_pile_size} "
                     f"played={list(self.played_cards)} royals_left={len(self.royals_pile)}")
        return "\n".join(lines)


_STATE_FIELDS = tuple(f.name for f in fields(GameState))


def _replace(state: GameState, **changes) -> GameState:
    """Fast equivalent of dataclasses.replace for GameState (no re-validation)."""
    new = object.__new__(GameState)
    d = new.__dict__
    d.update(state.__dict__)
    d.pop("_hash", None)
    d.update(changes)
    return new


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def initial_state(num_players: int) -> GameState:
    """The state before anything is dealt; the GameAgent acts first."""
    if num_players not in MAX_HAND_SIZE:
        raise ValueError("Regicide supports 1-4 players")
    deck = tuple(c for c in full_deck(num_players) if not c.is_royal)
    royals = tuple(c for c in full_deck(num_players) if c.is_royal)
    return GameState(
        num_players=num_players,
        hands=((),) * num_players,
        draw_pile=(deck,),
        discard_pile=(),
        played_cards=(),
        royals_pile=royals,
        removed_cards=(),
        active_royal=None,
        royal_damage=0,
        royal_shield=0,
        joker_played=False,
        current_player=0,
        phase=Phase.DEAL,
    )


def new_game(num_players: int, rng: Optional[random.Random] = None) -> GameState:
    """A dealt game with the first Jack revealed, sampling the GameAgent's choices."""
    return resolve_game_agent(initial_state(num_players), rng)


def make_state(
    hands: Sequence[Union[str, Iterable[Card]]],
    active_royal: Union[str, Card, None] = None,
    *,
    royal_damage: int = 0,
    royal_shield: int = 0,
    joker_played: bool = False,
    current_player: int = 0,
    discard_pile: Union[str, Iterable[Card]] = (),
    played_cards: Union[str, Iterable[Card]] = (),
    draw_pile: Union[str, Iterable[Card], None] = None,
    draw_top: Union[str, Iterable[Card]] = (),
    royals_pile: Union[str, Iterable[Card], None] = None,
) -> GameState:
    """Build a specific mid-game state (handy for tests and analysis).

    Unspecified cards are placed as follows: royals ranked at or above the active
    royal go to the royals pile, lower royals are treated as already removed, and
    number cards / Jokers go to the draw pile (or, if `draw_pile` is given, are
    removed from the game). `draw_top` cards sit on top of the draw pile in the
    given order, each as a known card.
    """
    n = len(hands)
    hands_t = tuple(parse_cards(h) for h in hands)
    royal = parse_card(active_royal) if isinstance(active_royal, str) else active_royal
    discard_t, played_t = parse_cards(discard_pile), parse_cards(played_cards)
    top = [parse_card(c) if isinstance(c, str) else c
           for c in (draw_top.split() if isinstance(draw_top, str) else draw_top)]

    used = Counter(c for h in hands_t for c in h) + Counter(discard_t) + Counter(played_t) + Counter(top)
    if royal:
        used[royal] += 1
    if draw_pile is not None:
        used += Counter(parse_cards(draw_pile))
    if royals_pile is not None:
        used += Counter(parse_cards(royals_pile))
    remaining = Counter(full_deck(n))
    remaining.subtract(used)
    if any(v < 0 for v in remaining.values()):
        raise ValueError("a card is used more times than it exists in the deck")
    remaining = list(remaining.elements())

    min_rank = royal.rank if royal else JACK
    if royals_pile is None:
        royals_t = parse_cards(c for c in remaining if c.is_royal and c.rank >= min_rank)
    else:
        royals_t = parse_cards(royals_pile)
    leftover = Counter(remaining)
    leftover.subtract(royals_t)
    leftover = list(leftover.elements())
    if draw_pile is None:
        main = parse_cards(c for c in leftover if not c.is_royal)
        removed = parse_cards(c for c in leftover if c.is_royal)
    else:
        main = parse_cards(draw_pile)
        removed = parse_cards(leftover)

    segments = tuple((c,) for c in top) + ((main,) if main else ())
    state = GameState(
        num_players=n,
        hands=hands_t,
        draw_pile=segments,
        discard_pile=discard_t,
        played_cards=played_t,
        royals_pile=royals_t,
        removed_cards=removed,
        active_royal=royal,
        royal_damage=royal_damage,
        royal_shield=royal_shield,
        joker_played=joker_played,
        current_player=current_player,
        phase=Phase.PLAY if royal else Phase.REVEAL,
    )
    if royal:
        state = _enter_play(state)
    state.validate()
    return state


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def _sub_multisets(cards: Cards) -> Iterator[Cards]:
    """All distinct sub-multisets of a sorted card tuple, each sorted."""
    counts = sorted(Counter(cards).items())

    def rec(i: int) -> Iterator[Cards]:
        if i == len(counts):
            yield ()
            return
        card, n = counts[i]
        for rest in rec(i + 1):
            for k in range(n + 1):
                yield (card,) * k + rest

    return rec(0)


def _is_valid_play(cards: Cards) -> bool:
    if len(cards) == 1 and cards[0].is_royal:
        return True
    if len({c.value for c in cards}) > 1:  # multiple cards must share one value
        return False
    return total_value(cards) <= MAX_PLAY_VALUE


def _valid_plays(hand: Cards) -> List[Cards]:
    """Every valid play from `hand`: yield, single royals, and same-value sets worth <= 10."""
    plays = [()]
    by_value = {}
    for c in hand:
        if c.is_royal:
            if (c,) not in plays:
                plays.append((c,))
        else:
            by_value.setdefault(c.value, []).append(c)
    for value, group in sorted(by_value.items()):
        plays.extend(s for s in _sub_multisets(tuple(group))
                     if s and total_value(s) <= MAX_PLAY_VALUE)
    return plays


def get_actions(state: GameState) -> List[Action]:
    """Available actions for whoever acts in `state` (empty if the game is over)."""
    if state.phase == Phase.PLAY:
        return [PlayCards(s) for s in _valid_plays(state.current_hand)]
    if state.phase == Phase.DISCARD:
        need = state.royal_attack
        return [DiscardCards(s) for s in _sub_multisets(state.current_hand)
                if total_value(s) >= need]
    if state.phase in GAME_AGENT_PHASES:
        return [GameAgentAction(c) for c in sorted(set(_chance_source(state)))]
    return []


def game_agent_policy(state: GameState) -> List[Tuple[GameAgentAction, float]]:
    """The GameAgent's (uniformly random) choice: each physical card equally likely."""
    source = _chance_source(state)
    counts = Counter(source)
    return [(GameAgentAction(c), n / len(source)) for c, n in sorted(counts.items())]


def apply_action(state: GameState, action: Action, check: bool = True) -> GameState:
    """Apply one action deterministically. Raises ValueError if it is not available.

    Pass check=False to skip the availability check when the caller took the action
    from `get_actions` / `game_agent_policy` (much faster inside search).
    """
    if check and action not in get_actions(state):
        raise ValueError(f"action {action} is not available in phase {state.phase.value}")
    phase = state.phase
    if phase == Phase.PLAY:
        return _play(state, action.cards)
    if phase == Phase.DISCARD:
        return _discard(state, action.cards)
    if phase == Phase.DEAL:
        return _deal(state, action.card)
    if phase == Phase.REVEAL:
        return _reveal(state, action.card)
    if state.pending_hearts > 0:
        return _hearts_step(state, action.card)
    return _draw_step(state, action.card)


def successors(state: GameState, action: Action) -> List[Tuple[GameState, float]]:
    """Distribution over the next states in which a player acts (or the game ends).

    The action is applied, then every GameAgent decision that follows is expanded
    with the GameAgent choosing uniformly at random. Identical states are merged.
    The number of outcomes grows combinatorially with the number of random cards
    involved (e.g. the initial deal), so prefer `step` for sampling large branches.
    """
    frontier = {apply_action(state, action): 1.0}
    result = {}
    while frontier:
        nxt = {}
        for s, p in frontier.items():
            if s.current_agent == GAME_AGENT:
                for a, q in game_agent_policy(s):
                    t = apply_action(s, a)
                    nxt[t] = nxt.get(t, 0.0) + p * q
            else:
                result[s] = result.get(s, 0.0) + p
        frontier = nxt
    return list(result.items())


def reward(state: GameState) -> int:
    """+1 for a won game, -1 for a lost game, 0 otherwise."""
    if state.is_won:
        return 1
    if state.is_lost:
        return -1
    return 0


def resolve_game_agent(state: GameState, rng: Optional[random.Random] = None) -> GameState:
    """Let the GameAgent act (uniformly at random) until a player acts or the game ends."""
    rng = rng or random
    while state.current_agent == GAME_AGENT:
        actions, probs = zip(*game_agent_policy(state))
        state = apply_action(state, rng.choices(actions, weights=probs)[0])
    return state


def step(state: GameState, action: Action,
         rng: Optional[random.Random] = None) -> Tuple[GameState, int]:
    """Apply `action`, sample the GameAgent's responses, and return (next_state, reward).

    The returned state is always one where a player acts, or a terminal state.
    """
    next_state = resolve_game_agent(apply_action(state, action), rng)
    return next_state, reward(next_state)


# ---------------------------------------------------------------------------
# Transition internals
# ---------------------------------------------------------------------------


def _remove(cards: Cards, to_remove: Iterable[Card]) -> Cards:
    remaining = list(cards)
    for c in to_remove:
        remaining.remove(c)
    return tuple(remaining)


def _add(cards: Cards, to_add: Iterable[Card]) -> Cards:
    return tuple(sorted(cards + tuple(to_add)))


def _with_hand(state: GameState, player: int, hand: Cards) -> Tuple[Cards, ...]:
    return state.hands[:player] + (hand,) + state.hands[player + 1:]


def _top_segment(state: GameState) -> Cards:
    for segment in state.draw_pile:
        if segment:
            return segment
    return ()


def _draw_card(state: GameState, card: Card) -> Tuple[Cards, ...]:
    """Draw pile with `card` removed from the top non-empty segment."""
    segments = [s for s in state.draw_pile if s]
    segments[0] = _remove(segments[0], [card])
    return tuple(s for s in segments if s)


def _next_to_deal(state: GameState) -> Optional[int]:
    if not state.draw_pile_size:
        return None
    candidates = [p for p in range(state.num_players) if len(state.hands[p]) < state.max_hand_size]
    return min(candidates, key=lambda p: (len(state.hands[p]), p)) if candidates else None


def _next_drawer(state: GameState) -> Optional[int]:
    for i in range(state.num_players):
        p = (state.next_drawer + i) % state.num_players
        if len(state.hands[p]) < state.max_hand_size:
            return p
    return None


def _chance_source(state: GameState) -> Cards:
    """The cards the GameAgent chooses among in this state (with multiplicity)."""
    if state.phase == Phase.DEAL:
        return _top_segment(state)
    if state.phase == Phase.REVEAL:
        if not state.royals_pile:
            return ()
        lowest = min(c.rank for c in state.royals_pile)
        return tuple(c for c in state.royals_pile if c.rank == lowest)
    if state.phase == Phase.RESOLVE:
        if state.pending_hearts > 0:
            return state.discard_pile
        if state.pending_draws > 0 and _next_drawer(state) is not None:
            return _top_segment(state)
    return ()


def _deal(state: GameState, card: Card) -> GameState:
    p = _next_to_deal(state)
    state = _replace(state, hands=_with_hand(state, p, _add(state.hands[p], [card])),
                    draw_pile=_draw_card(state, card))
    if _next_to_deal(state) is None:
        state = _replace(state, phase=Phase.REVEAL)
    return state


def _reveal(state: GameState, card: Card) -> GameState:
    state = _replace(state, royals_pile=_remove(state.royals_pile, [card]), active_royal=card,
                    royal_damage=0, royal_shield=0, joker_played=False)
    return _enter_play(state)


def _enter_play(state: GameState) -> GameState:
    if not all(state.hands):
        return _replace(state, phase=Phase.LOST)
    return _replace(state, phase=Phase.PLAY)


def _play(state: GameState, cards: Cards) -> GameState:
    p = state.current_player
    power = total_attack(cards)
    joker = state.joker_played or any(c.is_joker for c in cards)
    immune = None if joker else state.active_royal.suit
    played_suits = {c.suit for c in cards if not c.is_joker}
    effects = [s for s in SUITS if s in played_suits and s != immune]

    hearts = min(power, len(state.discard_pile)) if HEARTS in effects else 0
    draw_pile = state.draw_pile + ((),) if hearts else state.draw_pile  # Hearts' new bottom segment
    state = _replace(
        state,
        hands=_with_hand(state, p, _remove(state.hands[p], cards)),
        played_cards=_add(state.played_cards, cards),
        draw_pile=draw_pile,
        joker_played=joker,
        royal_shield=state.royal_shield + (power if SPADES in effects else 0),
        pending_damage=power * (2 if CLUBS in effects else 1),
        pending_hearts=hearts,
        pending_draws=power if DIAMONDS in effects else 0,
        next_drawer=p,
        phase=Phase.RESOLVE,
    )
    return _continue_resolution(state)


def _hearts_step(state: GameState, card: Card) -> GameState:
    *rest, bottom = state.draw_pile
    state = _replace(state, discard_pile=_remove(state.discard_pile, [card]),
                    draw_pile=tuple(rest) + (_add(bottom, [card]),),
                    pending_hearts=state.pending_hearts - 1)
    return _continue_resolution(state)


def _draw_step(state: GameState, card: Card) -> GameState:
    p = _next_drawer(state)
    state = _replace(state, hands=_with_hand(state, p, _add(state.hands[p], [card])),
                    draw_pile=_draw_card(state, card),
                    pending_draws=state.pending_draws - 1,
                    next_drawer=(p + 1) % state.num_players)
    return _continue_resolution(state)


def _continue_resolution(state: GameState) -> GameState:
    if _chance_source(state):
        return state
    return _attack(state)


def _attack(state: GameState) -> GameState:
    damage = state.royal_damage + state.pending_damage
    state = _replace(state, pending_damage=0, pending_hearts=0, pending_draws=0, next_drawer=0)
    if damage < state.royal_max_health:
        return _enter_discard(_replace(state, royal_damage=damage))

    royal = state.active_royal
    draw_pile, removed = state.draw_pile, state.removed_cards
    if damage == state.royal_max_health:
        draw_pile = ((royal,),) + draw_pile
    else:
        removed = _add(removed, [royal])
    state = _replace(
        state,
        draw_pile=draw_pile,
        removed_cards=removed,
        discard_pile=_add(state.discard_pile, state.played_cards),
        played_cards=(),
        active_royal=None,
        royal_damage=0,
        royal_shield=0,
        joker_played=False,
    )
    return _replace(state, phase=Phase.REVEAL if state.royals_pile else Phase.WON)


def _enter_discard(state: GameState) -> GameState:
    if not all(state.hands):
        return _replace(state, phase=Phase.LOST)
    attack = state.royal_attack
    if attack == 0:
        return _end_turn(state)
    if state.hand_value(state.current_player) < attack:
        return _replace(state, phase=Phase.LOST)
    return _replace(state, phase=Phase.DISCARD)


def _discard(state: GameState, cards: Cards) -> GameState:
    p = state.current_player
    state = _replace(state, hands=_with_hand(state, p, _remove(state.hands[p], cards)),
                    discard_pile=_add(state.discard_pile, cards))
    return _end_turn(state)


def _end_turn(state: GameState) -> GameState:
    state = _replace(state, current_player=(state.current_player + 1) % state.num_players)
    return _enter_play(state)