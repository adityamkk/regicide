"""Tests for minimax.py on small Regicide game trees near terminal states.

Run with `python3 -m unittest -v test_minimax`.
"""

import random
import unittest

from minimax import REPETITION_VALUE, AlphaBetaSearch, evaluate, minimax_action, play_out
from regicide import (
    GAME_AGENT, GameAgentAction, PlayCards, apply_action, get_actions, make_state,
    new_game, parse_card, parse_cards, reward, step,
)


def play(cards):
    return PlayCards(parse_cards(cards))


def plain_minimax(state, depth, heuristic=evaluate, path=frozenset()):
    """Reference minimax without pruning, with the same depth and repetition rules.

    `path` holds the states above this one; revisiting one of them is a loss.
    """
    if state.is_terminal:
        return float(reward(state))
    if state in path:
        return REPETITION_VALUE
    if depth == 0:
        return heuristic(state)
    path = path | {state}
    values = [plain_minimax(apply_action(state, a), depth - 1, heuristic, path)
              for a in get_actions(state)]
    return min(values) if state.current_agent == GAME_AGENT else max(values)


def count_nodes(state, depth):
    """Number of states plain minimax visits."""
    if state.is_terminal or depth == 0:
        return 1
    return 1 + sum(count_nodes(apply_action(state, a), depth - 1) for a in get_actions(state))


# Small states a few plies from the end of the game.
def win_in_one():
    # Last King on 10 health: 10C deals 20 (Clubs) and wins.
    return make_state(["10C 2D"], "KH", royal_damage=30, royals_pile="")


def win_in_two():
    # Last King on 25 health with no attack left: needs two plays, e.g. 10C then 6D.
    return make_state(["10C 6D 3S"], "KH", royal_damage=15, royal_shield=20,
                      royals_pile="", draw_pile="")


def forced_loss():
    # Jack attacks for 10; no play leaves enough hand value to discard.
    return make_state(["2C 3D"], "JS", draw_pile="")


def single_survivor():
    # Only 9S (shielding the Jack's attack to 1) survives the retaliation with cards left.
    return make_state(["9S 2C 3D"], "JH", draw_pile="")


def luck_dependent():
    # Playing AD alone draws one card: 10C wins next turn, 2H does not, so the
    # adversary deals 2H. Playing AD AS together draws both cards.
    return make_state(["AD AS"], "KS", royal_damage=30, royal_shield=20,
                      royals_pile="", draw_pile="10C 2H")


def two_player_fight():
    return make_state(["5C 3H 9S", "4D 2S 6H"], "JD", discard_pile="2C 3C", draw_pile="7C 8D")


SMALL_STATES = [win_in_one, win_in_two, forced_loss, single_survivor, luck_dependent, two_player_fight]


class HeuristicTests(unittest.TestCase):
    def test_terminal_values(self):
        won, _ = step(win_in_one(), play("10C"))
        lost, _ = step(forced_loss(), play(""))
        self.assertEqual(evaluate(won), 1.0)
        self.assertEqual(evaluate(lost), -1.0)

    def test_non_terminal_values_strictly_inside_bounds(self):
        rng = random.Random(7)
        for game in range(30):
            s = new_game(game % 4 + 1, rng)
            while not s.is_terminal:
                self.assertGreater(evaluate(s), -1.0)
                self.assertLess(evaluate(s), 1.0)
                s, _ = step(s, rng.choice(get_actions(s)), rng)

    def test_more_damage_is_better(self):
        values = [evaluate(make_state(["5C 3H 9S"], "QD", royal_damage=d)) for d in (0, 10, 20, 29)]
        self.assertEqual(values, sorted(values))
        self.assertLess(values[0], values[-1])

    def test_later_royals_are_better(self):
        jack = evaluate(make_state(["5C 3H 9S"], "JD"))
        queen = evaluate(make_state(["5C 3H 9S"], "QD"))
        self.assertLess(jack, queen)

    def test_weak_hand_is_worse(self):
        strong = evaluate(make_state(["9C 9H 8S 7D"], "JD"))
        weak = evaluate(make_state(["2C 3H"], "JD"))
        self.assertLess(weak, strong)


class SearchTests(unittest.TestCase):
    def test_terminal_state(self):
        won, _ = step(win_in_one(), play("10C"))
        result = AlphaBetaSearch(3).search(won)
        self.assertEqual(result.value, 1.0)
        self.assertIsNone(result.action)

    def test_invalid_depth(self):
        with self.assertRaises(ValueError):
            AlphaBetaSearch(0)

    def test_finds_win_in_one(self):
        result = AlphaBetaSearch(1).search(win_in_one())
        self.assertEqual(result.value, 1.0)
        self.assertEqual(result.action, play("10C"))

    def test_win_in_two_needs_depth_two(self):
        shallow = AlphaBetaSearch(1).search(win_in_two())
        self.assertLess(shallow.value, 1.0)
        deep = AlphaBetaSearch(2).search(win_in_two())
        self.assertEqual(deep.value, 1.0)
        # Following the search's choices actually wins.
        s = win_in_two()
        while not s.is_terminal:
            s, r = step(s, minimax_action(s, 2))
        self.assertEqual(r, 1)

    def test_forced_loss(self):
        for depth in (1, 2, 3):
            self.assertEqual(AlphaBetaSearch(depth).search(forced_loss()).value, -1.0)

    def test_avoids_loss(self):
        result = AlphaBetaSearch(2).search(single_survivor())
        self.assertEqual(result.action, play("9S"))
        self.assertGreater(result.value, -1.0)

    def test_game_agent_minimizes(self):
        resolve = apply_action(luck_dependent(), play("AD"))
        self.assertEqual(resolve.current_agent, GAME_AGENT)
        result = AlphaBetaSearch(3).search(resolve)
        self.assertEqual(result.action, GameAgentAction(parse_card("2H")))
        self.assertLess(result.value, 1.0)
        # Had the GameAgent dealt 10C, the players would win.
        good_draw = apply_action(resolve, GameAgentAction(parse_card("10C")))
        self.assertEqual(AlphaBetaSearch(2).search(good_draw).value, 1.0)

    def test_players_play_around_adversarial_luck(self):
        # Playing AD and AS together draws both cards, guaranteeing 10C: a forced win.
        result = AlphaBetaSearch(4).search(luck_dependent())
        self.assertEqual(result.value, 1.0)
        s = luck_dependent()
        while not s.is_terminal:
            s = apply_action(s, minimax_action(s, 4))  # both sides play minimax
        self.assertTrue(s.is_won)

    def test_matches_plain_minimax(self):
        for make in SMALL_STATES:
            for depth in (1, 2, 3, 4):
                with self.subTest(state=make.__name__, depth=depth):
                    s = make()
                    result = AlphaBetaSearch(depth).search(s)
                    self.assertAlmostEqual(result.value, plain_minimax(s, depth))
                    # The chosen action achieves the minimax value.
                    child = apply_action(s, result.action)
                    self.assertAlmostEqual(plain_minimax(child, depth - 1, path=frozenset([s])),
                                           result.value)

    def test_move_ordering_does_not_change_value(self):
        for make in SMALL_STATES:
            for depth in (1, 2, 3):
                ordered = AlphaBetaSearch(depth).search(make())
                unordered = AlphaBetaSearch(depth, order_moves=False).search(make())
                self.assertAlmostEqual(ordered.value, unordered.value)

    def test_pruning_visits_fewer_nodes(self):
        total_pruned = total_full = 0
        for make in SMALL_STATES:
            s = make()
            pruned = AlphaBetaSearch(4, order_moves=False).search(s).nodes
            full = count_nodes(s, 4)
            self.assertLessEqual(pruned, full)
            total_pruned += pruned
            total_full += full
        self.assertLess(total_pruned, total_full)

    def test_custom_heuristic_with_repetition_off(self):
        # A zero heuristic still finds forced results inside the horizon. (With
        # repetition on, the yield in win_in_two would score -1 instead of 0 below.)
        search = AlphaBetaSearch(1, heuristic=lambda s: 0.0, avoid_repetition=False)
        self.assertEqual(search.search(win_in_two()).value, 0.0)

    def test_custom_heuristic(self):
        # A zero heuristic still finds forced results inside the horizon.
        self.assertEqual(AlphaBetaSearch(1, heuristic=lambda s: 0.0).search(win_in_one()).value, 1.0)
        self.assertEqual(AlphaBetaSearch(1, heuristic=lambda s: 0.0).search(win_in_two()).value, 0.0)


def shielded_jack(players=1):
    # 10S shielded the Jack's attack to 0: yielding costs nothing and changes nothing.
    hands = ["2C 3D", "4H 5H", "6C", "7D"][:players]
    return make_state(hands, "JH", royal_damage=10, royal_shield=10, draw_pile="")


class RepetitionTests(unittest.TestCase):
    def test_yield_returns_to_the_same_state(self):
        # The loop this guards against: with 1 player, yielding is a no-op.
        s = shielded_jack()
        self.assertEqual(apply_action(s, play("")), s)
        # With 2 players, two yields bring the game back around.
        s = shielded_jack(2)
        self.assertEqual(apply_action(apply_action(s, play("")), play("")), s)

    def test_without_repetition_check_search_stalls(self):
        # Every real play leads towards a loss, so a plain search prefers yielding forever.
        for depth in (1, 2, 3):
            search = AlphaBetaSearch(depth, avoid_repetition=False)
            self.assertEqual(search.choose_action(shielded_jack()), play(""))

    def test_search_never_yields_into_a_repeat(self):
        for depth in (1, 2, 3, 4):
            result = AlphaBetaSearch(depth).search(shielded_jack())
            self.assertNotEqual(result.action, play(""))

    def test_history_detects_cycles_longer_than_the_horizon(self):
        # Two players at depth 1: P0's yield is new, but P1's yield would return to
        # the state P0 faced, which is in the game history.
        start = shielded_jack(2)
        after_p0 = apply_action(start, play(""))
        search = AlphaBetaSearch(1)
        self.assertEqual(search.choose_action(after_p0), play(""))  # no history: looks free
        result = search.search(after_p0, history=[start])
        self.assertNotEqual(result.action, play(""))

    def test_repeated_state_scores_as_loss(self):
        start = shielded_jack(2)
        after_p0 = apply_action(start, play(""))
        search = AlphaBetaSearch(2)
        search.search(after_p0, history=[start])
        search._seen = {start}
        self.assertEqual(search._alphabeta(start, 2, -2.0, 2.0), REPETITION_VALUE)

    def test_play_out_terminates_from_shielded_states(self):
        for players in (1, 2, 3, 4):
            for depth in (1, 2, 3):
                with self.subTest(players=players, depth=depth):
                    final, trajectory, stalemate = play_out(
                        shielded_jack(players), AlphaBetaSearch(depth), random.Random(0))
                    self.assertTrue(final.is_terminal)
                    self.assertFalse(stalemate)
                    states = [st for st, _ in trajectory]
                    self.assertEqual(len(states), len(set(states)))  # no state repeated

    def test_play_out_random_games_terminate(self):
        rng = random.Random(11)
        for game in range(6):
            s = new_game(game % 4 + 1, rng)
            final, trajectory, stalemate = play_out(s, AlphaBetaSearch(1), rng)
            self.assertTrue(final.is_terminal)
            self.assertFalse(stalemate)
            self.assertTrue(trajectory)


if __name__ == "__main__":
    unittest.main()
