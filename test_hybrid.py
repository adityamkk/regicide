"""Tests for hybrid.py (expectimax + Star1 pruning + MCTS leaves) on small game trees.

Run with `python3 -m unittest -v test_hybrid`.
"""

import random
import unittest

from hybrid import (
    LOWER_BOUND, UPPER_BOUND, HybridSearch, LeafMCTS, SustainHeuristic, cards_to_cover, evaluate_v2,
    evaluate_v3, game_phase, hybrid_action, ordered_actions, play_score, refill_power,
    sustain_discard_score, sustain_play_score,
)
from minimax import REPETITION_VALUE, play_out
from regicide import (
    GAME_AGENT, GameAgentAction, apply_action, get_actions, make_state, new_game,
    parse_card, parse_cards, reward, step, successors,
)
from test_minimax import (
    forced_loss, luck_dependent, play, shielded_jack, single_survivor, two_player_fight,
    win_in_one, win_in_two,
)

SMALL_STATES = [win_in_one, win_in_two, forced_loss, single_survivor, luck_dependent,
                two_player_fight, shielded_jack]


def win_in_one_small():
    # win_in_one with an empty draw pile, so exact expectimax stays small.
    return make_state(["10C 2D"], "KH", royal_damage=30, royals_pile="", draw_pile="")


# States whose luck can be expanded exactly in a test.
EXACT_STATES = [win_in_one_small, win_in_two, forced_loss, single_survivor, luck_dependent,
                two_player_fight, shielded_jack]


def exact_search(depth, **kwargs):
    """Hybrid search with no sampling, no beam and a static heuristic leaf: deterministic."""
    kwargs.setdefault("mcts_iterations", 0)
    return HybridSearch(depth, beam_width=1000, max_exact=10**6, **kwargs)


def expectimax(state, depth, heuristic=evaluate_v2, path=frozenset()):
    """Reference expectimax: depth counts player decisions, luck is averaged exactly."""
    if state.is_terminal:
        return float(reward(state))
    if state in path:
        return REPETITION_VALUE
    if depth == 0:
        return heuristic(state)
    path = path | {state}
    return max(sum(p * expectimax(t, depth - 1, heuristic, path) for t, p in successors(state, a))
               for a in get_actions(state))


class HeuristicTests(unittest.TestCase):
    def test_terminal_values(self):
        won, _ = step(win_in_one(), play("10C"))
        lost, _ = step(forced_loss(), play(""))
        self.assertEqual(evaluate_v2(won), 1.0)
        self.assertEqual(evaluate_v2(lost), -1.0)

    def test_values_strictly_inside_bounds(self):
        rng = random.Random(3)
        for game in range(20):
            s = new_game(game % 4 + 1, rng)
            while not s.is_terminal:
                self.assertTrue(-1.0 < evaluate_v2(s) < 1.0)
                s, _ = step(s, rng.choice(get_actions(s)), rng)

    def test_more_damage_is_better(self):
        values = [evaluate_v2(make_state(["5C 3H 9S"], "QD", royal_damage=d)) for d in (0, 10, 20, 29)]
        self.assertEqual(values, sorted(values))

    def test_plan_term_prefers_hands_that_can_finish_the_royal(self):
        # Same hand size, but one hand can pay for the Queen's retaliations.
        strong = evaluate_v2(make_state(["9C 9H 9D 8S 8C"], "QS"))
        weak = evaluate_v2(make_state(["AC AH 2D 2S 3C"], "QS"))
        self.assertLess(weak, strong)


    def test_killing_beats_yielding(self):
        # Regression: the heuristic once preferred yielding (and paying 10) over killing
        # this Jack, because it ignored the pending retaliation and costed every later
        # royal with the next royal's suit immunity.
        s = make_state(["AC 2H 5S 7C 7S 8C 8H"], "JS", royal_damage=13, played_cards="3D JD",
                       discard_pile="4S 7H 6H AH 5H 9D 10C 9H",
                       royals_pile="JC JH QC QD QH QS KC KD KH KS")

        def expected(cards):
            return sum(p * evaluate_v2(t) for t, p in successors(s, play(cards)))

        for kill in ("7C", "7S", "8H"):
            self.assertGreater(expected(kill), expected(""))
        self.assertNotEqual(HybridSearch(seed=0).choose_action(s), play(""))


class SustainHeuristicTests(unittest.TestCase):
    def test_game_phase(self):
        self.assertEqual(game_phase(make_state(["5C"], "JH")), 0.0)
        self.assertEqual(game_phase(make_state(["5C"], "QH")), 0.5)  # 4 Jacks defeated
        self.assertEqual(game_phase(make_state(["5C"], "KH")), 1.0)

    def test_refill_power(self):
        s = make_state(["9D 4D 3H 2C"], "KS", royals_pile="", discard_pile="5C 6C",
                       draw_pile="7C 8C 9C 10C 2S 3S 4S 5S 6S 7S 8S 9S 10S")
        self.assertEqual(refill_power(s), (13, 2))  # Hearts capped by the 2 discards
        # Cards of the royal's suit only help against later royals: half credit.
        s = make_state(["9D 4D"], "KD", royals_pile="", draw_pile="7C 8C 9C 10C 2S 3S 4S 5S 6S 7S")
        self.assertEqual(refill_power(s), (6.5, 0))
        # Draws are capped by the draw pile.
        s = make_state(["9D 4D"], "KS", royals_pile="", draw_pile="7C 8C")
        self.assertEqual(refill_power(s)[0], 2)

    def test_terminal_values_and_bounds(self):
        won, _ = step(win_in_one(), play("10C"))
        lost, _ = step(forced_loss(), play(""))
        self.assertEqual((evaluate_v3(won), evaluate_v3(lost)), (1.0, -1.0))
        rng = random.Random(9)
        for game in range(20):
            s = new_game(game % 4 + 1, rng)
            while not s.is_terminal:
                self.assertTrue(-1.0 < evaluate_v3(s) < 1.0)
                s, _ = step(s, rng.choice(get_actions(s)), rng)

    def test_weights_interpolate(self):
        h = SustainHeuristic(hand=(0.1, 0.3), refill=(0.0, 0.2), deck=(0.0, 0.0))
        self.assertEqual(h.weights(0.0), {"hand": 0.1, "refill": 0.0, "deck": 0.0})
        self.assertAlmostEqual(h.weights(0.5)["hand"], 0.2)
        self.assertAlmostEqual(h.weights(1.0)["refill"], 0.2)

    def test_full_hands_matter_more_later(self):
        gaps = []
        for royal in ("JH", "QH", "KH"):
            few = make_state(["10C 10S"], royal)
            many = make_state(["5C 5S 5D 4C 3S 2D"], royal)  # same hand value, more cards
            gaps.append(evaluate_v3(many) - evaluate_v3(few))
        self.assertGreater(gaps[0], 0)
        self.assertEqual(gaps, sorted(gaps))
        self.assertGreater(gaps[2], 2 * gaps[0])

    def test_pending_discard_counts_against_hand(self):
        # Yielding into a 10-point retaliation must not look like keeping cards.
        s = make_state(["AC AH 3D 7D 7H"], "QD", royal_damage=19)
        yielded = apply_action(s, play(""))
        self.assertEqual(yielded.current_hand, s.current_hand)  # same cards, 15 now due
        self.assertEqual(cards_to_cover(yielded.current_hand, yielded.royal_attack), 3)
        hand_only = SustainHeuristic(hand=(0.5, 0.5), refill=(0.0, 0.0), deck=(0.0, 0.0))
        self.assertLess(hand_only(yielded), hand_only(s))
        self.assertLess(evaluate_v3(yielded), evaluate_v3(s))

    def test_reachable_draws_are_valued(self):
        # Same hand value and size; only one hand can draw from the deck.
        kwargs = dict(royals_pile="", draw_pile="2C 3C 4C 5C 6C 7C 8C 9C 10C")
        with_diamonds = make_state(["8D 2S"], "KH", **kwargs)
        without = make_state(["8S 2D"], "KH", **kwargs)  # 2D draws only 2
        self.assertGreater(evaluate_v3(with_diamonds), evaluate_v3(without))

    def test_v2_unchanged_by_sustain_features(self):
        # evaluate_v2 does not reward card count on its own.
        few = make_state(["10C 10S"], "KH")
        many = make_state(["5C 5S 5D 4C 3S 2D"], "KH")
        self.assertLess(abs(evaluate_v2(many) - evaluate_v2(few)), 0.05)


class SustainPolicyTests(unittest.TestCase):
    def test_refill_valued_more_late(self):
        kwargs = dict(draw_pile="5C 6C 7C 8C 9C 10C 2S 3S 4S")
        early = make_state(["9D 9S 2C 3C"], "JH", **kwargs)
        late = make_state(["9D 9S 2C 3C"], "KH", royals_pile="", **kwargs)
        for s in (early, late):
            # The sustain policy never ranks the Diamond refill lower than the plain one does.
            self.assertGreaterEqual(sustain_play_score(s, play("9D").cards), play_score(s, play("9D").cards))
        early_gain = sustain_play_score(early, play("9D").cards) - play_score(early, play("9D").cards)
        late_gain = sustain_play_score(late, play("9D").cards) - play_score(late, play("9D").cards)
        self.assertGreater(late_gain, early_gain)
        # Late in the game the Diamond (draws 9) beats the Spade (shields 9).
        self.assertEqual(ordered_actions(late, sustain=True)[0], play("9D"))

    def test_kills_still_first(self):
        self.assertEqual(ordered_actions(win_in_one(), sustain=True)[0], play("10C"))

    def test_late_discards_keep_card_count(self):
        # Attack 10 against the King: discard one 10 rather than 4 + 6.
        s = apply_action(make_state(["10S 4C 6C 9H 8H"], "KH", royals_pile="", royal_shield=10), play(""))
        self.assertEqual(s.royal_attack, 10)
        self.assertEqual(ordered_actions(s, sustain=True)[0].cards, parse_cards("10S"))
        self.assertLess(sustain_discard_score(s, parse_cards("4C 6C")),
                        sustain_discard_score(s, parse_cards("10S")))


class PolicyTests(unittest.TestCase):
    def test_kill_is_ranked_first(self):
        s = win_in_one()
        self.assertEqual(ordered_actions(s)[0], play("10C"))
        self.assertGreater(play_score(s, play("10C").cards), 90)  # kills score ~100

    def test_suicidal_play_ranked_low(self):
        # Yielding against attack 10 with hand value 5 loses at once.
        s = forced_loss()
        self.assertLess(play_score(s, ()), 0)

    def test_discards_ordered_cheapest_first(self):
        s = apply_action(make_state(["2C 3D 5S 9H 9C"], "JS"), play(""))
        first = ordered_actions(s)[0]
        self.assertEqual(sum(c.value for c in first.cards), 10)


class LeafMCTSTests(unittest.TestCase):
    def test_terminal_and_bounds(self):
        mcts = LeafMCTS(iterations=16, rollout_depth=4)
        won, _ = step(win_in_one(), play("10C"))
        self.assertEqual(mcts.value(won, (), random.Random(0)), 1.0)
        for make in SMALL_STATES:
            v = mcts.value(make(), (), random.Random(0))
            self.assertGreaterEqual(v, LOWER_BOUND)
            self.assertLessEqual(v, UPPER_BOUND)

    def test_finds_wins_with_max_backup(self):
        mcts = LeafMCTS(iterations=32, rollout_depth=4, backup="max")
        self.assertEqual(mcts.value(win_in_one(), (), random.Random(0)), 1.0)
        self.assertEqual(mcts.value(win_in_two(), (), random.Random(0)), 1.0)

    def test_mean_backup_rates_wins_highly(self):
        # The mean is diluted by exploratory rollouts but stays well above other states.
        mcts = LeafMCTS(iterations=32, rollout_depth=4, backup="mean")
        self.assertEqual(mcts.value(win_in_one(), (), random.Random(0)), 1.0)
        self.assertGreater(mcts.value(win_in_two(), (), random.Random(0)), 0.5)

    def test_invalid_backup(self):
        with self.assertRaises(ValueError):
            LeafMCTS(iterations=8, rollout_depth=2, backup="median")

    def test_forced_loss(self):
        mcts = LeafMCTS(iterations=32, rollout_depth=4)
        self.assertLessEqual(mcts.value(forced_loss(), (), random.Random(0)), -1.0)

    def test_reproducible(self):
        mcts = LeafMCTS(iterations=24, rollout_depth=5)
        s = new_game(2, random.Random(5))
        self.assertEqual(mcts.value(s, (), random.Random(1)), mcts.value(s, (), random.Random(1)))


class SearchTests(unittest.TestCase):
    def test_terminal_state(self):
        won, _ = step(win_in_one(), play("10C"))
        result = HybridSearch(2).search(won)
        self.assertEqual((result.value, result.action), (1.0, None))

    def test_invalid_depth(self):
        with self.assertRaises(ValueError):
            HybridSearch(0)

    def test_finds_win_in_one(self):
        result = HybridSearch(1).search(win_in_one())
        self.assertEqual((result.value, result.action), (1.0, play("10C")))

    def test_win_in_two(self):
        self.assertLess(exact_search(1).search(win_in_two()).value, 1.0)  # beyond the horizon
        self.assertEqual(exact_search(2).search(win_in_two()).value, 1.0)
        # MCTS at the frontier sees past the horizon even at depth 1.
        self.assertEqual(HybridSearch(1).search(win_in_two()).value, 1.0)
        s, search = win_in_two(), HybridSearch(1)
        while not s.is_terminal:
            s, r = step(s, search.choose_action(s))
        self.assertEqual(r, 1)

    def test_forced_loss(self):
        for depth in (1, 2):
            self.assertLessEqual(HybridSearch(depth).search(forced_loss()).value, -1.0)

    def test_avoids_loss(self):
        self.assertEqual(HybridSearch(1).choose_action(single_survivor()), play("9S"))

    def test_luck_is_averaged_not_worst_case(self):
        # Playing AD alone draws 10C (win next turn) or 2H with equal chance.
        s = luck_dependent()
        after_ad = successors(s, play("AD"))
        self.assertEqual(len(after_ad), 2)
        search = exact_search(2)
        value_ad = search._chance_value(after_ad, 1, LOWER_BOUND, UPPER_BOUND)
        expected = sum(p * expectimax(t, 1, path=frozenset([s])) for t, p in after_ad)
        self.assertAlmostEqual(value_ad, expected)
        win_branch = [t for t, _ in after_ad if parse_card("10C") in t.hand(0)][0]
        self.assertEqual(expectimax(win_branch, 1), 1.0)
        # The forced win (AD AS draws both cards) is still found and preferred.
        result = exact_search(2).search(s)
        self.assertEqual((result.value, result.action), (1.0, play("AD AS")))

    def test_star1_matches_expectimax(self):
        for make in EXACT_STATES:
            for depth in (1, 2, 3):
                with self.subTest(state=make.__name__, depth=depth):
                    s = make()
                    result = exact_search(depth).search(s)
                    self.assertAlmostEqual(result.value, expectimax(s, depth))
                    # The chosen action achieves that value.
                    achieved = sum(p * expectimax(t, depth - 1, path=frozenset([s]))
                                   for t, p in successors(s, result.action))
                    self.assertAlmostEqual(achieved, result.value)

    def test_avoids_repetition(self):
        for depth in (1, 2):
            self.assertNotEqual(HybridSearch(depth).choose_action(shielded_jack()), play(""))
        # The history catches the 2-player yield cycle.
        start = shielded_jack(2)
        after_p0 = apply_action(start, play(""))
        self.assertNotEqual(HybridSearch(1).choose_action(after_p0, history=[start]), play(""))

    def test_game_agent_root_picks_worst_outcome(self):
        resolve = apply_action(luck_dependent(), play("AD"))
        self.assertEqual(resolve.current_agent, GAME_AGENT)
        result = exact_search(2).search(resolve)
        self.assertEqual(result.action, GameAgentAction(parse_card("2H")))
        self.assertLess(result.value, 1.0)

    def test_reproducible_with_seed(self):
        s = new_game(2, random.Random(8))
        a = HybridSearch(2, seed=4).search(s)
        b = HybridSearch(2, seed=4).search(s)
        self.assertEqual((a.value, a.action), (b.value, b.action))

    def test_sustain_search(self):
        search = HybridSearch(sustain=True)
        self.assertIs(search.heuristic, evaluate_v3)
        self.assertIs(HybridSearch().heuristic, evaluate_v2)
        self.assertEqual(search.search(win_in_one()).value, 1.0)
        self.assertEqual(HybridSearch(sustain=True).choose_action(single_survivor()), play("9S"))
        self.assertNotEqual(HybridSearch(sustain=True).choose_action(shielded_jack()), play(""))
        for depth in (1, 2):
            with self.subTest(depth=depth):
                s = luck_dependent()
                result = HybridSearch(depth, beam_width=1000, max_exact=10**6, mcts_iterations=0,
                                      sustain=True).search(s)
                self.assertAlmostEqual(result.value, expectimax(s, depth, heuristic=evaluate_v3))

    def test_wrapper(self):
        self.assertEqual(hybrid_action(win_in_one(), depth=1), play("10C"))

    def test_play_out_terminates(self):
        for players in (1, 2, 3, 4):
            final, trajectory, stalemate = play_out(
                shielded_jack(players), HybridSearch(1, mcts_iterations=8), random.Random(0))
            self.assertTrue(final.is_terminal)
            self.assertFalse(stalemate)
        rng = random.Random(21)
        for game in range(3):
            s = new_game(game % 4 + 1, rng)
            final, _, stalemate = play_out(
                s, HybridSearch(1, mcts_iterations=8, rollout_depth=3, seed=game), rng)
            self.assertTrue(final.is_terminal)
            self.assertFalse(stalemate)


if __name__ == "__main__":
    unittest.main()
