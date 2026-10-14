"""Tests for visualize.py (trajectory recording and HTML output).

Run with `python3 -m unittest -v test_visualize`.
"""

import json
import re
import unittest
from collections import Counter

from minimax import AlphaBetaSearch
from regicide import MAX_HAND_SIZE
from visualize import build_html, record_game


def embedded_game(html):
    match = re.search(r"const GAME = (.*?);\nconst FRAMES", html, re.S)
    return json.loads(match.group(1).replace("<\\/", "</"))


class RecordGameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.games = {players: record_game(AlphaBetaSearch(1), players, seed=players)
                     for players in (1, 2, 4)}

    def test_frame_kinds(self):
        for frames in self.games.values():
            self.assertEqual(frames[-1]["kind"], "end")
            self.assertTrue(all(f["kind"] in ("player", "luck") for f in frames[:-1]))
            self.assertIn("player", {f["kind"] for f in frames})
            self.assertEqual(frames[0]["kind"], "luck")  # the first royal is revealed

    def test_hands_and_highlights(self):
        for players, frames in self.games.items():
            for f in frames:
                self.assertEqual(len(f["hands"]), players)
                self.assertTrue(all(len(h) <= MAX_HAND_SIZE[players] for h in f["hands"]))
                if f["kind"] == "player":
                    hand = Counter(f["hands"][f["current"]])
                    self.assertFalse(Counter(f["highlight"]) - hand, "played cards come from the hand")
                    self.assertIsNotNone(f["value"])
                elif f["kind"] == "luck":
                    self.assertEqual(len(f["highlight"]), 1)
                    self.assertTrue(f["action"])

    def test_royal_and_piles(self):
        for frames in self.games.values():
            for f in frames:
                self.assertGreaterEqual(f["draw"], 0)
                self.assertGreaterEqual(f["discard"], 0)
                self.assertEqual(sum(f["royalsLeft"].values()) + f["defeated"] + (1 if f["royal"] else 0), 12)
                r = f["royal"]
                if r:
                    self.assertTrue(0 < r["hp"] <= r["maxHp"])
                    self.assertTrue(0 <= r["attack"] <= r["baseAttack"])
                    self.assertGreaterEqual(r["incoming"], 0)

    def test_turns_never_decrease(self):
        for frames in self.games.values():
            turns = [f["turn"] for f in frames]
            self.assertEqual(turns, sorted(turns))

    def test_end_frame(self):
        for frames in self.games.values():
            end = frames[-1]
            self.assertIn("won", end)
            self.assertEqual(end["won"], end["action"].startswith("The players win"))

    def test_outcome_notes_after_random_effects(self):
        # A play that triggered draws still gets its damage note once luck resolves.
        for frames in self.games.values():
            for i, f in enumerate(frames[:-1]):
                if f["kind"] == "player" and frames[i + 1]["kind"] == "luck" \
                        and frames[i + 1]["phase"] == "resolve":
                    self.assertTrue(f["notes"])


class HtmlTests(unittest.TestCase):
    def test_round_trip_and_escaping(self):
        frames = record_game(AlphaBetaSearch(1), 1, seed=7)
        frames[0]["notes"] = ["</script><b>not markup</b>"]
        html = build_html(frames, "label </script>", 8)
        self.assertEqual(html.count("</script>"), 1)  # only the real closing tag
        game = embedded_game(html)
        self.assertEqual(game["frames"], frames)
        self.assertEqual(game["maxHand"], 8)
        self.assertIn("<title>Regicide trajectory</title>", html)


if __name__ == "__main__":
    unittest.main()
