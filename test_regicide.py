"""Tests for the Regicide GameState. Run with `python3 -m unittest -v test_regicide`."""

import random
import unittest
from collections import Counter

import regicide as R
from regicide import (
    GAME_AGENT, JOKER, DiscardCards, GameAgentAction, Phase, PlayCards,
    apply_action, get_actions, make_state, new_game, parse_card, parse_cards,
    step, successors, total_value,
)


def play(cards):
    return PlayCards(parse_cards(cards))


def discard(cards):
    return DiscardCards(parse_cards(cards))


def only_successor(state, action):
    outcomes = successors(state, action)
    assert len(outcomes) == 1, outcomes
    return outcomes[0][0]


class CardTests(unittest.TestCase):
    def test_values_and_attack(self):
        self.assertEqual(parse_card("AS").value, 1)
        self.assertEqual(parse_card("7D").value, 7)
        self.assertEqual(parse_card("10C").value, 10)
        self.assertEqual(parse_card("JH").value, 20)
        self.assertEqual(parse_card("QH").value, 30)
        self.assertEqual(parse_card("KH").value, 40)
        self.assertEqual(JOKER.value, 10)
        self.assertEqual(JOKER.attack, 0)
        self.assertEqual(parse_card("9S").attack, 9)

    def test_parse_round_trip(self):
        for card in R.full_deck(4):
            self.assertEqual(parse_card(str(card)), card)

    def test_full_deck_sizes(self):
        for n, jokers in R.NUM_JOKERS.items():
            deck = R.full_deck(n)
            self.assertEqual(len(deck), 52 + jokers)
            self.assertEqual(sum(c.is_royal for c in deck), 12)
            self.assertEqual(deck.count(JOKER), jokers)


class InitialStateTests(unittest.TestCase):
    def test_initial_state(self):
        for n in range(1, 5):
            s = R.initial_state(n)
            s.validate()
            self.assertEqual(s.phase, Phase.DEAL)
            self.assertEqual(s.current_agent, GAME_AGENT)
            self.assertEqual(s.hands, ((),) * n)
            self.assertEqual(s.draw_pile_size, 40 + R.NUM_JOKERS[n])
            self.assertEqual(len(s.royals_pile), 12)
            self.assertIsNone(s.active_royal)
            self.assertEqual(s.royals_remaining, 12)

    def test_invalid_player_count(self):
        with self.assertRaises(ValueError):
            R.initial_state(5)

    def test_new_game_features(self):
        for n in range(1, 5):
            s = new_game(n, random.Random(n))
            s.validate()
            max_hand = R.MAX_HAND_SIZE[n]
            self.assertEqual(s.max_hand_size, max_hand)
            self.assertEqual(s.phase, Phase.PLAY)
            self.assertEqual(s.current_player, 0)
            self.assertEqual(s.current_agent, 0)
            self.assertEqual(s.current_hand, s.hand(0))
            self.assertTrue(all(len(h) == max_hand for h in s.hands))
            self.assertEqual(s.hand_value(0), total_value(s.hand(0)))
            self.assertEqual(s.draw_pile_size, 40 + R.NUM_JOKERS[n] - n * max_hand)
            self.assertEqual(len(s.draw_pile_cards), s.draw_pile_size)
            self.assertEqual(s.discard_pile_size, 0)
            # The first royal is a Jack at full strength.
            self.assertEqual(s.active_royal.rank, R.JACK)
            self.assertEqual(s.royal_health, 20)
            self.assertEqual(s.royal_max_health, 20)
            self.assertEqual(s.royal_attack, 10)
            self.assertEqual(s.immune_suit, s.active_royal.suit)
            self.assertEqual(len(s.royals_pile), 11)
            self.assertEqual(s.royals_defeated, 0)
            self.assertFalse(s.is_terminal)

    def test_deal_successors(self):
        # Dealing the last card of a nearly complete deal is a uniform choice.
        s = R.initial_state(1)
        rng = random.Random(0)
        for _ in range(7):
            s = apply_action(s, rng.choice(get_actions(s)))
        self.assertEqual(len(s.hand(0)), 7)
        action = get_actions(s)[0]
        # The next action finishes the deal; then the GameAgent reveals one of 4 Jacks.
        outcomes = successors(s, action)
        self.assertEqual(len(outcomes), 4)
        for t, p in outcomes:
            self.assertAlmostEqual(p, 0.25)
            self.assertEqual(t.active_royal.rank, R.JACK)


class ActionTests(unittest.TestCase):
    def test_play_actions_are_valid_subsets(self):
        s = make_state(["AS 2D 2H 5S 5C 9S 9C"], "JD")
        actions = get_actions(s)
        self.assertIn(PlayCards(()), actions)  # yielding is allowed
        self.assertEqual(len(actions), len(set(actions)))
        hand = Counter(s.current_hand)
        for a in actions:
            self.assertIsInstance(a, PlayCards)
            self.assertLessEqual(total_value(a.cards), R.MAX_PLAY_VALUE)
            self.assertLessEqual(len({c.value for c in a.cards}), 1)  # one shared value
            self.assertFalse(Counter(a.cards) - hand)
        self.assertIn(play("2D 2H"), actions)
        self.assertIn(play("5S 5C"), actions)
        self.assertNotIn(play("9S 9C"), actions)  # same value but sums to 18

    def test_multiple_cards_must_share_value(self):
        s = make_state(["AS 2D 3S 6D 7H"], "JD")
        # Only single cards (and yielding) are possible: no two cards share a value.
        self.assertEqual(set(get_actions(s)),
                         {play(""), play("AS"), play("2D"), play("3S"), play("6D"), play("7H")})
        for invalid in ("3S 6D", "AS 2D", "AS 2D 3S"):
            self.assertNotIn(play(invalid), get_actions(s))
            with self.assertRaises(ValueError):
                apply_action(s, play(invalid))

    def test_sets_of_equal_values_up_to_ten(self):
        s = make_state(["3C 3D 3H 3S 2C"], "JD")
        actions = get_actions(s)
        self.assertIn(play("3C 3D"), actions)
        self.assertIn(play("3C 3D 3H"), actions)
        self.assertNotIn(play("3C 3D 3H 3S"), actions)  # 12 > 10
        self.assertNotIn(play("2C 3C"), actions)
        # Each of the 4 threes alone, 6 pairs, 4 triples, the 2C, and yielding.
        self.assertEqual(len(actions), 4 + 6 + 4 + 1 + 1)

    def test_set_of_equal_values_resolves_all_suits(self):
        # 4C 4D against a Jack of Spades: 8 damage doubled by Clubs, Diamonds draws 8.
        s = make_state(["4C 4D 9H", "2S"], "JS", draw_pile="5H 6H")
        outcomes = successors(s, play("4C 4D"))
        self.assertEqual(len(outcomes), 2)  # which player gets 5H vs 6H
        for t, p in outcomes:
            self.assertAlmostEqual(p, 0.5)
            self.assertEqual(t.royal_health, 4)
            self.assertEqual(t.draw_pile_size, 0)
            self.assertEqual((len(t.hand(0)), len(t.hand(1))), (2, 2))

    def test_royals_in_hand_only_played_alone(self):
        s = make_state(["JC 2D"], "JD")
        self.assertEqual(set(get_actions(s)), {play(""), play("2D"), play("JC")})

    def test_royal_play_uses_attack_value(self):
        self.assertEqual(parse_card("JS").attack, 10)
        self.assertEqual(parse_card("QS").attack, 15)
        self.assertEqual(parse_card("KS").attack, 20)
        # A Queen of Spades against a Jack of Hearts deals 15 and shields 15 (attack 10 -> 0).
        s = make_state(["QS 2D", "3C"], "JH")
        t = only_successor(s, play("QS"))
        self.assertEqual(t.royal_health, 5)
        self.assertEqual(t.royal_attack, 0)
        self.assertEqual(t.phase, Phase.PLAY)
        self.assertEqual(t.current_player, 1)
        # A King of Clubs deals 20, doubled to 40, overkilling a Jack.
        s = make_state(["KC 2D"], "JS")
        t = apply_action(s, play("KC"))
        self.assertIn(parse_card("JS"), t.removed_cards)
        self.assertIn(parse_card("KC"), t.discard_pile)
        # Royals cannot be combined with other cards.
        self.assertNotIn(play("KC 2D"), get_actions(s))

    def test_joker_only_played_alone(self):
        s = make_state(["JK AS 5C", "2C", "3C"], "JD")
        joker_plays = [a for a in get_actions(s) if JOKER in a.cards]
        self.assertEqual(joker_plays, [PlayCards((JOKER,))])

    def test_duplicate_jokers_counted_once(self):
        s = make_state(["JK JK 2C", "2D", "3D", "4D"], "JD")
        actions = get_actions(s)
        self.assertEqual(len(actions), len(set(actions)))
        self.assertNotIn(play("JK JK"), actions)

    def test_discard_actions_cover_attack(self):
        s = only_successor(make_state(["2C 3D 5S 9H 9C"], "JS"), play(""))
        self.assertEqual(s.phase, Phase.DISCARD)
        actions = get_actions(s)
        self.assertTrue(actions)
        for a in actions:
            self.assertIsInstance(a, DiscardCards)
            self.assertGreaterEqual(total_value(a.cards), 10)
        self.assertIn(discard("9H 2C"), actions)
        self.assertNotIn(discard("9H"), actions)

    def test_invalid_action_raises(self):
        s = make_state(["2C 3D"], "JS")
        with self.assertRaises(ValueError):
            apply_action(s, play("4C"))
        with self.assertRaises(ValueError):
            apply_action(s, discard("2C"))
        with self.assertRaises(ValueError):
            apply_action(s, GameAgentAction(parse_card("2C")))

    def test_terminal_state_has_no_actions(self):
        s = only_successor(make_state(["2C 3D"], "JS"), play(""))
        self.assertTrue(s.is_lost)
        self.assertEqual(get_actions(s), [])
        self.assertIsNone(s.current_agent)

    def test_apply_does_not_mutate(self):
        s = make_state(["2C 2D 9S 8S"], "JH")
        before = (s.hands, s.royal_damage, s.phase)
        apply_action(s, play("2C 2D"))
        self.assertEqual((s.hands, s.royal_damage, s.phase), before)


class EffectTests(unittest.TestCase):
    def test_clubs_double_damage(self):
        s = only_successor(make_state(["5C 9S 9H"], "JS"), play("5C"))
        self.assertEqual(s.royal_health, 10)
        self.assertEqual(s.played_cards, parse_cards("5C"))

    def test_royal_negates_own_suit(self):
        s = only_successor(make_state(["5C 9S 9H"], "JC"), play("5C"))
        self.assertEqual(s.royal_health, 15)

    def test_spades_reduce_attack(self):
        s = make_state(["4S 9C 9H"], "JH")
        t = only_successor(s, play("4S"))
        self.assertEqual(t.royal_attack, 6)
        self.assertEqual(t.royal_base_attack, 10)
        self.assertEqual(t.phase, Phase.DISCARD)
        self.assertTrue(all(total_value(a.cards) >= 6 for a in get_actions(t)))

    def test_spades_accumulate_and_floor_at_zero(self):
        s = make_state(["9S 8C", "2C"], "JH", royal_shield=5)
        t = only_successor(s, play("9S"))
        self.assertEqual(t.royal_attack, 0)
        # No retaliation needed: the turn passes straight to the next player.
        self.assertEqual(t.phase, Phase.PLAY)
        self.assertEqual(t.current_player, 1)
        self.assertEqual(t.hand(0), parse_cards("8C"))

    def test_hearts_move_discard_to_bottom_of_draw_pile(self):
        s = make_state(["3H 9C 9D"], "JS", discard_pile="2C 3C 4C 5C 6C 7C")
        outcomes = successors(s, play("3H"))
        self.assertEqual(len(outcomes), 20)  # C(6, 3)
        self.assertAlmostEqual(sum(p for _, p in outcomes), 1.0)
        for t, p in outcomes:
            t.validate()
            self.assertAlmostEqual(p, 1 / 20)
            self.assertEqual(t.discard_pile_size, 3)
            self.assertEqual(t.draw_pile_size, s.draw_pile_size + 3)
            self.assertEqual(len(t.draw_pile[-1]), 3)  # returned cards form the bottom segment
            self.assertEqual(t.royal_health, 17)

    def test_hearts_limited_by_discard_size(self):
        s = make_state(["9H 9C 9D"], "JS", discard_pile="2C")
        (t, p), = successors(s, play("9H"))
        self.assertEqual(p, 1.0)
        self.assertEqual(t.discard_pile_size, 0)
        self.assertEqual(t.draw_pile[-1], parse_cards("2C"))

    def test_diamonds_draw_in_player_order(self):
        s = make_state(["3D 2C 9S", "4C"], "JC", draw_pile="5S 6S 7S 8S")
        outcomes = successors(s, play("3D"))
        # P0, P1, P0 each draw once: 4 choices for P1's card x C(3, 2) for P0's.
        self.assertEqual(len(outcomes), 12)
        self.assertAlmostEqual(sum(p for _, p in outcomes), 1.0)
        for t, p in outcomes:
            t.validate()
            self.assertAlmostEqual(p, 1 / 12)
            self.assertEqual(len(t.hand(0)), 4)  # 2C 9S + 2 drawn
            self.assertEqual(len(t.hand(1)), 2)
            self.assertEqual(t.draw_pile_size, 1)

    def test_diamonds_draw_starts_at_current_player(self):
        s = make_state(["2C", "AD 9S"], "JC", current_player=1, draw_pile="5S")
        (t, p), = successors(s, play("AD"))
        self.assertEqual(t.hand(1), parse_cards("5S 9S"))
        self.assertEqual(t.hand(0), parse_cards("2C"))

    def test_diamonds_respect_max_hand_size(self):
        s = make_state(["4D 4H AC 2C 3C 6C 7C 8C"], "JC", draw_pile="2S 3S 4S 5S 6S 7S 8S 9S 10S")
        outcomes = successors(s, play("4D 4H"))
        for t, _ in outcomes:
            self.assertEqual(len(t.hand(0)), 8)
            self.assertEqual(t.draw_pile_size, 7)

    def test_effects_resolve_with_empty_draw_pile(self):
        s = make_state(["5D 9C"], "JC", draw_pile="")
        t = only_successor(s, play("5D"))
        self.assertEqual(t.hand(0), parse_cards("9C"))
        self.assertEqual(t.royal_health, 15)

    def test_joker_cancels_immunity(self):
        s = make_state(["JK 9C 10C", "2C", "3C"], "JS")
        t = only_successor(s, PlayCards((JOKER,)))
        self.assertTrue(t.joker_played)
        self.assertIsNone(t.immune_suit)
        self.assertEqual(t.royal_health, 20)  # Jokers deal no damage
        self.assertEqual(t.phase, Phase.DISCARD)

        s = make_state(["5S 9C 9H", "2C", "3C"], "JS", joker_played=True)
        t = only_successor(s, play("5S"))
        self.assertEqual(t.royal_attack, 5)


class RoyalDefeatTests(unittest.TestCase):
    def test_exact_kill_puts_royal_on_draw_pile(self):
        s = make_state(["5H 2C"], "JS", royal_damage=15)
        t = apply_action(s, play("5H"))
        self.assertEqual(t.phase, Phase.REVEAL)
        self.assertEqual(t.current_agent, GAME_AGENT)
        self.assertEqual(t.draw_pile[0], parse_cards("JS"))
        self.assertIn(parse_card("5H"), t.discard_pile)
        self.assertEqual(t.played_cards, ())

        outcomes = successors(s, play("5H"))
        self.assertEqual(len(outcomes), 3)  # one of the 3 remaining Jacks
        for u, p in outcomes:
            u.validate()
            self.assertAlmostEqual(p, 1 / 3)
            self.assertEqual(u.active_royal.rank, R.JACK)
            self.assertEqual(u.royal_health, 20)
            self.assertEqual(u.current_player, 0)  # same player continues
            self.assertEqual(u.royals_defeated, 1)

    def test_known_royal_is_drawn_first(self):
        s = make_state(["AD 9C"], "JC", draw_top="JS")
        t = only_successor(s, play("AD"))
        self.assertIn(parse_card("JS"), t.hand(0))

    def test_overkill_removes_royal(self):
        s = make_state(["9C 2D"], "JS", royal_damage=5)
        t = apply_action(s, play("9C"))
        self.assertIn(parse_card("JS"), t.removed_cards)
        self.assertNotIn(parse_card("JS"), t.draw_pile_cards)
        self.assertEqual(t.phase, Phase.REVEAL)

    def test_queens_only_after_jacks(self):
        s = make_state(["9C 2D"], "JS", royal_damage=5, royals_pile="QC QD QH QS KC KD KH KS")
        outcomes = successors(s, play("9C"))
        self.assertEqual({u.active_royal for u, _ in outcomes}, set(parse_cards("QC QD QH QS")))
        for u, _ in outcomes:
            self.assertEqual(u.royal_health, 30)
            self.assertEqual(u.royal_attack, 15)


class TerminalTests(unittest.TestCase):
    def test_win_after_last_king(self):
        s = make_state(["10C 2D"], "KH", royal_damage=30, royals_pile="")
        t, r = step(s, play("10C"))
        self.assertTrue(t.is_won)
        self.assertEqual(r, 1)
        t.validate()

    def test_lose_when_discard_impossible(self):
        s = make_state(["2C 3D"], "JS")
        t, r = step(s, play(""))
        self.assertTrue(t.is_lost)
        self.assertEqual(r, -1)

    def test_lose_when_hand_empty(self):
        s = make_state(["5C", "2D"], "JS")
        t, r = step(s, play("5C"))
        self.assertTrue(t.is_lost)
        self.assertEqual(r, -1)

    def test_non_terminal_reward_is_zero(self):
        s = make_state(["2C 9D 9H"], "JS")
        t, r = step(s, play("2C"), random.Random(0))
        self.assertFalse(t.is_terminal)
        self.assertEqual(r, 0)


class TransitionConsistencyTests(unittest.TestCase):
    def test_step_matches_successor_distribution(self):
        s = make_state(["3D 2C 9S", "4C"], "JC", draw_pile="5S 6S 7S 8S")
        action = play("3D")
        expected = dict(successors(s, action))
        rng = random.Random(42)
        trials = 3000
        counts = Counter(step(s, action, rng)[0] for _ in range(trials))
        self.assertLessEqual(set(counts), set(expected))
        for state, p in expected.items():
            self.assertAlmostEqual(counts[state] / trials, p, delta=0.03)

    def test_random_rollouts_stay_valid(self):
        rng = random.Random(1234)
        for game in range(200):
            n = game % 4 + 1
            s = new_game(n, rng)
            s.validate()
            for _ in range(10_000):
                if s.is_terminal:
                    break
                self.assertIn(s.current_agent, range(n))
                actions = get_actions(s)
                self.assertTrue(actions)
                action = rng.choice(actions)
                s, r = step(s, action, rng)
                s.validate()
                self.assertIn(r, (-1, 0, 1))
                self.assertEqual(r != 0, s.is_terminal)
                self.assertNotEqual(s.current_agent, GAME_AGENT)
            self.assertTrue(s.is_terminal, "game did not finish")

    def test_successors_on_random_states(self):
        rng = random.Random(99)
        checked = 0
        for game in range(40):
            s = new_game(game % 4 + 1, rng)
            while not s.is_terminal:
                action = rng.choice(get_actions(s))
                # Keep enumeration small: few random cards drawn or returned.
                if not isinstance(action, PlayCards) or R.total_attack(action.cards) <= 2:
                    outcomes = successors(s, action)
                    self.assertAlmostEqual(sum(p for _, p in outcomes), 1.0)
                    for t, p in outcomes:
                        self.assertGreater(p, 0)
                        t.validate()
                        self.assertNotEqual(t.current_agent, GAME_AGENT)
                    checked += 1
                s, _ = step(s, action, rng)
        self.assertGreater(checked, 50)


if __name__ == "__main__":
    unittest.main()
