"""Record a Regicide game and view its trajectory in a browser.

Usage:
    python3 visualize.py [--players N] [--seed S] [--agent hybrid|minimax] [--sustain]
                         [--depth D] [--iterations I] [--output FILE] [--no-open]

Plays one game with the chosen agent, then writes a self-contained HTML viewer
(no network access needed) and opens it. The viewer shows the active royal (health,
attack, suit immunity), the draw / discard pile sizes, the cards played against the
royal, every player's hand, and steps through the game with buttons, a slider or
the arrow keys.
"""

import argparse
import json
import os
import random
import webbrowser

from hybrid import HybridSearch
from minimax import AlphaBetaSearch
from regicide import (
    GAME_AGENT, JACK, KING, QUEEN, DiscardCards, Phase, apply_action, game_agent_policy,
    initial_state,
)
from run_minimax import describe_game_agent, describe_outcome

SUIT_NAMES = {"C": "Clubs", "S": "Spades", "H": "Hearts", "D": "Diamonds"}


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def snapshot(state, turn):
    """The parts of a state the viewer displays, as JSON-friendly data."""
    royal = None
    if state.active_royal:
        royal = {
            "card": str(state.active_royal),
            "hp": state.royal_health,
            "incoming": state.pending_damage,  # damage of a play whose luck is still resolving
            "maxHp": state.royal_max_health,
            "attack": state.royal_attack,
            "baseAttack": state.royal_base_attack,
            "shield": state.royal_shield,
            "immune": SUIT_NAMES.get(state.immune_suit),
            "joker": state.joker_played,
        }
    known_top = None
    if state.draw_pile and len(state.draw_pile[0]) == 1 and state.draw_pile[0][0].is_royal:
        known_top = str(state.draw_pile[0][0])  # a royal placed back after an exact kill
    pile = state.royals_pile
    return {
        "turn": turn,
        "phase": state.phase.value,
        "current": state.current_player,
        "royal": royal,
        "royalsLeft": {"J": sum(c.rank == JACK for c in pile), "Q": sum(c.rank == QUEEN for c in pile),
                       "K": sum(c.rank == KING for c in pile)},
        "defeated": state.royals_defeated,
        "draw": state.draw_pile_size,
        "drawTop": known_top,
        "discard": state.discard_pile_size,
        "played": [str(c) for c in state.played_cards],
        "hands": [[str(c) for c in hand] for hand in state.hands],
    }


def record_game(search, players, seed, adversarial=False, max_moves=2000):
    """Play a game and return the list of frames the viewer steps through.

    Each frame is the state *before* a step, plus what happens in that step:
    a player decision (kind "player") or one GameAgent random event (kind "luck").
    The last frame (kind "end") is the final state with the result.
    """
    rng = random.Random(seed)
    state = initial_state(players)
    while state.phase == Phase.DEAL:  # the deal itself is not shown step by step
        actions, probs = zip(*game_agent_policy(state))
        state = apply_action(state, rng.choices(actions, weights=probs)[0])

    frames = []
    history = set()
    turn = moves = 0
    stalemate = False

    def game_agent_step(state):
        """Record one GameAgent random event and return the next state."""
        history.add(state)
        if adversarial:
            action = search.choose_action(state, history)
        else:
            actions, probs = zip(*game_agent_policy(state))
            action = rng.choices(actions, weights=probs)[0]
        frame = snapshot(state, turn)
        frame.update(kind="luck", actor="GameAgent", value=None,
                     action=describe_game_agent(state, action), highlight=[str(action.card)],
                     notes=[])
        frames.append(frame)
        return apply_action(state, action)

    while not state.is_terminal and moves < max_moves:
        if state in history:  # only possible if every available move repeats a state
            stalemate = True
            break
        if state.current_agent == GAME_AGENT:
            state = game_agent_step(state)
            continue

        moves += 1
        if state.phase == Phase.PLAY:
            turn += 1
        result = search.search(state, history)
        history.add(state)
        cards = result.action.cards
        verb = "discards" if isinstance(result.action, DiscardCards) else "plays"
        text = f"{verb} {' '.join(map(str, cards))}" if cards else "yields (plays nothing)"
        frame = snapshot(state, turn)
        frame.update(kind="player", actor=f"P{state.current_player}", value=round(result.value, 3),
                     action=text, highlight=[str(c) for c in cards], notes=[])
        frames.append(frame)
        before = state
        state = apply_action(state, result.action)
        while state.phase == Phase.RESOLVE:  # Hearts / Diamonds luck before the attack lands
            state = game_agent_step(state)
        frame["notes"] = describe_outcome(before, state)

    final = snapshot(state, turn)
    if stalemate:
        result = "The players lose: the game repeated a position."
    elif state.is_won:
        result = "The players win! All 12 royals defeated."
    elif state.is_lost:
        empty = [f"P{p}" for p, h in enumerate(state.hands) if not h]
        result = ("The players lose: " + (f"{', '.join(empty)} has no cards." if empty else
                  f"P{state.current_player} cannot absorb the attack."))
    else:
        result = f"Stopped after {max_moves} moves."
    final.update(kind="end", actor="", value=None, action=result, highlight=[], notes=[],
                 won=state.is_won)
    frames.append(final)
    return frames


# ---------------------------------------------------------------------------
# HTML viewer
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Regicide trajectory</title>
<style>
:root {
  --bg: #f4f1ea; --panel: #fffdf8; --ink: #23201b; --muted: #6d665b; --line: #d9d2c3;
  --card: #ffffff; --card-edge: #cfc6b4; --red: #b3261e; --black: #23201b;
  --accent: #2f5d8a; --accent-soft: #dbe7f3; --good: #2e7d32; --bad: #b3261e;
  --hp: #c0392b; --hp-track: #eadfd2; --shield: #3b6ea5; --hl: #e0a800;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #1b1a17; --panel: #24221e; --ink: #ece6da; --muted: #a39b8c; --line: #3a372f;
    --card: #f3efe6; --card-edge: #8c8474; --red: #c62828; --black: #1f1d1a;
    --accent: #8fb6dc; --accent-soft: #2c3a48; --good: #81c784; --bad: #ef9a9a;
    --hp: #e57373; --hp-track: #3a332c; --shield: #8fb6dc; --hl: #ffca28;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
       font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 16px; }
header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 16px; margin-bottom: 12px; }
h1 { font-size: 20px; margin: 0; }
.meta { color: var(--muted); }
.badge { padding: 2px 10px; border-radius: 999px; font-weight: 600; font-size: 13px; }
.badge.won { background: color-mix(in srgb, var(--good) 20%, transparent); color: var(--good); }
.badge.lost { background: color-mix(in srgb, var(--bad) 18%, transparent); color: var(--bad); }
.grid { display: grid; grid-template-columns: 300px minmax(0, 1fr); gap: 12px; }
.grid > *, .stack > *, .stats { min-width: 0; }
@media (max-width: 820px) { .grid { grid-template-columns: minmax(0, 1fr); } }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px; }
.panel h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted);
            margin: 0 0 8px; font-weight: 600; }
.stack { display: flex; flex-direction: column; gap: 12px; }

/* Cards */
.card { width: 46px; height: 66px; border-radius: 6px; background: var(--card);
        border: 1px solid var(--card-edge); position: relative; flex: none;
        box-shadow: 0 1px 2px rgba(0,0,0,.15); color: var(--black); user-select: none; }
.card.red { color: var(--red); }
.card .corner { position: absolute; top: 3px; left: 5px; font-weight: 700; font-size: 13px; line-height: 1; }
.card .corner small { display: block; font-size: 12px; }
.card .pip { position: absolute; inset: 0; display: grid; place-items: center; font-size: 24px; }
.card.big { width: 92px; height: 132px; border-radius: 10px; }
.card.big .corner { font-size: 20px; top: 6px; left: 8px; }
.card.big .corner small { font-size: 18px; }
.card.big .pip { font-size: 48px; }
.card.hl { outline: 3px solid var(--hl); outline-offset: 1px; transform: translateY(-6px); }
.card.joker .pip { font-size: 20px; }
.card.royal-in-hand { background: color-mix(in srgb, var(--card) 85%, #c9a227); }
.cards { display: flex; flex-wrap: wrap; gap: 6px; min-height: 66px; align-items: flex-end; padding-top: 6px; }
.empty { color: var(--muted); font-style: italic; align-self: center; }

/* Royal */
.royal { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start; }
.royal .stats { min-width: 150px; }
.stats { flex: 1; display: flex; flex-direction: column; gap: 8px; }
.bar { height: 10px; background: var(--hp-track); border-radius: 5px; overflow: hidden; }
.bar > div { height: 100%; background: var(--hp); transition: width .25s; }
.bar { display: flex; }
.bar > div.incoming { background: repeating-linear-gradient(45deg, var(--hp), var(--hp) 3px, transparent 3px, transparent 6px); }
.bar.attack > div { background: var(--shield); }
.stat-label { display: flex; justify-content: space-between; font-size: 13px; }
.chip { display: inline-block; padding: 1px 8px; border-radius: 999px; background: var(--accent-soft);
        color: var(--accent); font-size: 12px; }

/* Piles */
.piles { display: flex; gap: 16px; }
.pile { display: flex; flex-direction: column; align-items: center; gap: 4px; }
.pile-stack { position: relative; width: 50px; height: 92px; }
.pile-stack span { position: absolute; left: 0; width: 46px; height: 66px; border-radius: 6px;
                   border: 1px solid var(--card-edge); }
.pile-stack.draw span { background: repeating-linear-gradient(45deg, #2f5d8a, #2f5d8a 4px, #3d6f9e 4px, #3d6f9e 8px); }
.pile-stack.discard span { background: var(--card); }
.pile b { font-size: 16px; }
.castle { display: flex; gap: 10px; font-size: 13px; }
.castle div { text-align: center; }
.castle b { display: block; font-size: 18px; }

/* Players */
.player { border-top: 1px solid var(--line); padding: 10px 0 6px; }
.player:first-of-type { border-top: 0; padding-top: 0; }
.player-head { display: flex; gap: 8px; align-items: baseline; }
.player-head b { font-size: 15px; }
.player.active .player-head b { color: var(--accent); }
.player.active { background: linear-gradient(90deg, var(--accent-soft), transparent 70%);
                 border-radius: 8px; padding-left: 8px; margin-left: -8px; }

/* Step panel and controls */
.step-head { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px 16px; align-items: flex-start; }
.step-cards { min-height: 0; padding-top: 0; }
.step-title { font-size: 16px; font-weight: 600; }
.step-title .who { color: var(--accent); }
.notes { margin: 6px 0 0; padding-left: 18px; color: var(--muted); }
.controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 12px 0; }
button { font: inherit; padding: 6px 12px; border-radius: 8px; border: 1px solid var(--line);
         background: var(--panel); color: var(--ink); cursor: pointer; }
button:hover { border-color: var(--accent); }
button.primary { background: var(--accent); color: var(--panel); border-color: var(--accent); }
input[type=range] { flex: 1; min-width: 160px; accent-color: var(--accent); }
label { color: var(--muted); display: inline-flex; gap: 4px; align-items: center; white-space: nowrap; }
@media (max-width: 520px) { input[type=range] { flex-basis: 100%; } }
.log { max-height: 260px; overflow-y: auto; font-size: 13px; position: relative; }
.log div { padding: 3px 6px; border-radius: 6px; cursor: pointer; display: flex; gap: 8px; }
.log div:hover { background: var(--accent-soft); }
.log div.cur { background: var(--accent-soft); font-weight: 600; }
.log .t { color: var(--muted); min-width: 52px; }
.log .luck { color: var(--muted); }
.result { font-size: 18px; font-weight: 700; }
.result.won { color: var(--good); } .result.lost { color: var(--bad); }
.value { color: var(--muted); font-size: 13px; }
kbd { border: 1px solid var(--line); border-bottom-width: 2px; border-radius: 4px; padding: 0 4px; font-size: 12px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Regicide trajectory</h1>
    <span class="meta" id="meta"></span>
    <span class="badge" id="badge"></span>
  </header>

  <div class="panel">
    <div class="step-head">
      <div>
        <div class="step-title" id="stepTitle"></div>
        <div class="value" id="stepValue"></div>
      </div>
      <div class="cards step-cards" id="stepCards"></div>
    </div>
    <ul class="notes" id="stepNotes"></ul>
    <div class="controls">
      <button id="first" title="First step (Home)">&#x23EE;</button>
      <button id="prev" title="Previous step (&larr;)">&#x25C0; Prev</button>
      <button id="play" class="primary" title="Play / pause (space)">&#x25B6; Play</button>
      <button id="next" title="Next step (&rarr;)">Next &#x25B6;</button>
      <button id="last" title="Last step (End)">&#x23ED;</button>
      <input type="range" id="slider" min="0" value="0" aria-label="Step">
      <span id="counter" class="meta"></span>
      <label><input type="checkbox" id="luck" checked> show random events</label>
      <label>speed <select id="speed"><option value="1400">slow</option><option value="700" selected>normal</option><option value="250">fast</option></select></label>
    </div>
    <div class="meta"><kbd>&larr;</kbd> <kbd>&rarr;</kbd> step &middot; <kbd>space</kbd> play/pause &middot; <kbd>Home</kbd>/<kbd>End</kbd> jump</div>
  </div>

  <div class="grid" style="margin-top:12px">
    <div class="stack">
      <div class="panel">
        <h2>Active royal</h2>
        <div id="royal"></div>
      </div>
      <div class="panel">
        <h2>Piles</h2>
        <div class="piles" id="piles"></div>
      </div>
      <div class="panel">
        <h2>Castle (royals left)</h2>
        <div class="castle" id="castle"></div>
      </div>
    </div>
    <div class="stack">
      <div class="panel">
        <h2>Played against the royal</h2>
        <div class="cards" id="played"></div>
      </div>
      <div class="panel">
        <h2>Hands</h2>
        <div id="hands"></div>
      </div>
      <div class="panel">
        <h2>Game log</h2>
        <div class="log" id="log"></div>
      </div>
    </div>
  </div>
</div>

<script>
const GAME = __GAME_JSON__;
const FRAMES = GAME.frames;
const SUITS = { C: "♣", S: "♠", H: "♥", D: "♦" };
const RED = new Set(["H", "D"]);
const $ = (id) => document.getElementById(id);
const esc = (t) => String(t).replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch]);
let idx = 0, timer = null;

function parseCard(s) {
  if (s === "JK") return { rank: "JK", suit: "*" };
  return { rank: s.slice(0, -1), suit: s.slice(-1) };
}

function cardEl(s, opts = {}) {
  const c = parseCard(s);
  const el = document.createElement("div");
  el.className = "card" + (RED.has(c.suit) ? " red" : "") + (opts.big ? " big" : "")
    + (opts.hl ? " hl" : "") + (c.rank === "JK" ? " joker" : "")
    + (!opts.big && ["J", "Q", "K"].includes(c.rank) ? " royal-in-hand" : "");
  el.title = s;
  if (c.rank === "JK") {
    el.innerHTML = '<span class="corner">JK</span><span class="pip">★</span>';
  } else {
    el.innerHTML = `<span class="corner">${c.rank}<small>${SUITS[c.suit]}</small></span>`
      + `<span class="pip">${["J", "Q", "K"].includes(c.rank) ? c.rank + SUITS[c.suit] : SUITS[c.suit]}</span>`;
  }
  return el;
}

function fillCards(container, cards, highlight = []) {
  container.innerHTML = "";
  const left = [...highlight];
  if (!cards.length) {
    container.innerHTML = '<span class="empty">none</span>';
    return;
  }
  for (const s of cards) {
    const i = left.indexOf(s);
    if (i >= 0) left.splice(i, 1);
    container.appendChild(cardEl(s, { hl: i >= 0 }));
  }
}

function pileEl(kind, count, label, extra) {
  const wrap = document.createElement("div");
  wrap.className = "pile";
  const stack = document.createElement("div");
  stack.className = "pile-stack " + kind;
  const layers = count === 0 ? 0 : Math.min(8, 1 + Math.floor(count / 6));
  for (let i = 0; i < layers; i++) {
    const s = document.createElement("span");
    s.style.top = (layers - 1 - i) * 3 + "px";
    stack.appendChild(s);
  }
  if (!layers) stack.innerHTML = '<span style="border-style:dashed;background:none"></span>';
  wrap.appendChild(stack);
  wrap.insertAdjacentHTML("beforeend", `<b>${count}</b><span class="meta">${label}</span>${extra || ""}`);
  return wrap;
}

function visibleFrames() {
  if ($("luck").checked) return FRAMES.map((_, i) => i);
  return FRAMES.map((f, i) => (f.kind === "luck" ? -1 : i)).filter((i) => i >= 0);
}

function describe(f) {
  if (f.kind === "end") return f.action;
  return `${f.actor} ${f.action}`;
}

function render() {
  const f = FRAMES[idx];
  const vis = visibleFrames();
  const pos = Math.max(0, vis.indexOf(idx));
  $("slider").max = vis.length - 1;
  $("slider").value = pos;
  $("counter").textContent = `step ${pos + 1} / ${vis.length}`;

  // Step description
  if (f.kind === "end") {
    $("stepTitle").innerHTML = `<span class="result ${f.won ? "won" : "lost"}">${esc(f.action)}</span>`;
  } else {
    const turn = f.turn ? `Turn ${f.turn} · ` : "";
    $("stepTitle").innerHTML = `${turn}<span class="who">${esc(f.actor)}</span> ${esc(f.action)}`;
  }
  $("stepValue").textContent = f.value === null || f.value === undefined ? ""
    : `search value ${f.value >= 0 ? "+" : ""}${f.value.toFixed(3)} (−1 loss … +1 win)`;
  $("stepNotes").innerHTML = (f.notes || []).map((n) => `<li>${esc(n)}</li>`).join("");
  const stepCards = $("stepCards");
  stepCards.innerHTML = "";
  for (const s of f.highlight || []) stepCards.appendChild(cardEl(s, { hl: false }));
  try { history.replaceState(null, "", "#step=" + idx); } catch (e) { /* file:// may refuse */ }

  // Royal
  const r = f.royal;
  const royal = $("royal");
  royal.innerHTML = "";
  if (!r) {
    royal.innerHTML = `<span class="empty">${f.kind === "end" ? "none \u2014 every royal has been defeated"
      : "none (the next royal is about to be revealed)"}</span>`;
  } else {
    const box = document.createElement("div");
    box.className = "royal";
    box.appendChild(cardEl(r.card, { big: true }));
    const incoming = Math.min(r.incoming || 0, r.hp);
    const attackPct = r.baseAttack ? 100 * r.attack / r.baseAttack : 0;
    box.insertAdjacentHTML("beforeend", `<div class="stats">
      <div><div class="stat-label"><span>Health</span><b>${r.hp} / ${r.maxHp}</b></div>
        <div class="bar" title="hatched: damage about to land"><div style="width:${100 * (r.hp - incoming) / r.maxHp}%"></div><div class="incoming" style="width:${100 * incoming / r.maxHp}%"></div></div>
        ${r.incoming ? `<div class="meta">${r.incoming} damage lands after the random effects</div>` : ""}</div>
      <div><div class="stat-label"><span>Attack</span><b>${r.attack}${r.shield ? ` <span class="meta">(${r.baseAttack} − ${Math.min(r.shield, r.baseAttack)} Spades)</span>` : ""}</b></div>
        <div class="bar attack"><div style="width:${attackPct}%"></div></div></div>
      <div>${r.immune ? `<span class="chip">blocks ${r.immune}</span>` : '<span class="chip">no immunity (Joker)</span>'}</div>
    </div>`);
    royal.appendChild(box);
  }

  // Piles
  const piles = $("piles");
  piles.innerHTML = "";
  piles.appendChild(pileEl("draw", f.draw, "draw pile",
    f.drawTop ? `<span class="meta">top: ${f.drawTop}</span>` : ""));
  piles.appendChild(pileEl("discard", f.discard, "discard"));

  // Castle
  $("castle").innerHTML = ["J", "Q", "K"].map((k) =>
    `<div><b>${f.royalsLeft[k]}</b>${{ J: "Jacks", Q: "Queens", K: "Kings" }[k]}</div>`).join("")
    + `<div><b>${f.defeated}</b>defeated</div>`;

  // Played cards
  fillCards($("played"), f.played);

  // Hands
  const hands = $("hands");
  hands.innerHTML = "";
  f.hands.forEach((hand, p) => {
    const acting = f.kind === "player" && f.current === p;
    const turnHolder = f.kind !== "end" && f.current === p && f.royal !== null;
    const value = hand.reduce((sum, s) => sum + cardValue(s), 0);
    const div = document.createElement("div");
    div.className = "player" + (turnHolder ? " active" : "");
    div.innerHTML = `<div class="player-head"><b>P${p}</b>
      <span class="meta">${hand.length} / ${GAME.maxHand} cards · value ${value}${acting ? " · to act" : turnHolder ? " · their turn" : ""}</span></div>`;
    const cards = document.createElement("div");
    cards.className = "cards";
    fillCards(cards, hand, acting ? f.highlight : []);
    div.appendChild(cards);
    hands.appendChild(div);
  });

  // Log
  [...$("log").children].forEach((el) => {
    el.classList.toggle("cur", Number(el.dataset.i) === idx);
    el.hidden = !$("luck").checked && FRAMES[Number(el.dataset.i)].kind === "luck";
  });
  // Keep the current entry visible by scrolling the log box only (not the page).
  const log = $("log"), cur = log.querySelector(".cur");
  if (cur) {
    const top = cur.offsetTop - log.offsetTop;
    if (top < log.scrollTop) log.scrollTop = top;
    else if (top + cur.offsetHeight > log.scrollTop + log.clientHeight)
      log.scrollTop = top + cur.offsetHeight - log.clientHeight;
  }
}

function cardValue(s) {
  const c = parseCard(s);
  if (c.rank === "JK") return 10;
  return { A: 1, J: 20, Q: 30, K: 40 }[c.rank] || Number(c.rank);
}

function go(i) { idx = i; render(); }
function step(delta) {
  const vis = visibleFrames();
  let pos = vis.indexOf(idx);
  if (pos < 0) pos = vis.findIndex((i) => i > idx) - (delta > 0 ? 1 : 0);
  const next = Math.min(vis.length - 1, Math.max(0, pos + delta));
  go(vis[next]);
  return next !== pos;
}
function togglePlay() {
  if (timer) { clearInterval(timer); timer = null; $("play").innerHTML = "▶ Play"; return; }
  if (idx === FRAMES.length - 1) go(0);
  $("play").innerHTML = "⏸ Pause";
  timer = setInterval(() => { if (!step(1)) togglePlay(); }, Number($("speed").value));
}

function buildLog() {
  const log = $("log");
  FRAMES.forEach((f, i) => {
    const div = document.createElement("div");
    div.dataset.i = i;
    div.innerHTML = `<span class="t">${f.kind === "end" ? "end" : f.turn ? "T" + f.turn : ""}</span>`
      + `<span class="${f.kind === "luck" ? "luck" : ""}">${esc(describe(f))}</span>`;
    div.onclick = () => go(i);
    log.appendChild(div);
  });
}

function init() {
  $("meta").textContent = GAME.label;  // textContent: no markup
  const last = FRAMES[FRAMES.length - 1];
  $("badge").textContent = last.won ? "won" : `lost · ${last.defeated} / 12 royals`;
  $("badge").className = "badge " + (last.won ? "won" : "lost");
  buildLog();
  $("first").onclick = () => go(visibleFrames()[0]);
  $("last").onclick = () => go(FRAMES.length - 1);
  $("prev").onclick = () => step(-1);
  $("next").onclick = () => step(1);
  $("play").onclick = togglePlay;
  $("slider").oninput = (e) => go(visibleFrames()[Number(e.target.value)]);
  $("luck").onchange = () => { if (!$("luck").checked && FRAMES[idx].kind === "luck") step(1); render(); };
  $("speed").onchange = () => { if (timer) { togglePlay(); togglePlay(); } };
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT") return;
    if (e.key === "ArrowRight") { step(1); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { step(-1); e.preventDefault(); }
    else if (e.key === " ") { togglePlay(); e.preventDefault(); }
    else if (e.key === "Home") go(visibleFrames()[0]);
    else if (e.key === "End") go(FRAMES.length - 1);
  });
  const m = /step=(\d+)/.exec(location.hash);
  if (m) idx = Math.min(FRAMES.length - 1, Number(m[1]));
  render();
}
init();
</script>
</body>
</html>
"""


def build_html(frames, label, max_hand):
    data = {"frames": frames, "label": label, "maxHand": max_hand}
    payload = json.dumps(data).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__GAME_JSON__", payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--players", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: random)")
    parser.add_argument("--agent", choices=["hybrid", "minimax"], default="hybrid")
    parser.add_argument("--sustain", action="store_true", help="hybrid: sustain-aware heuristic and policy")
    parser.add_argument("--depth", type=int, default=None,
                        help="search depth (default: 1 for hybrid, 3 for minimax)")
    parser.add_argument("--iterations", type=int, default=24, help="hybrid: MCTS iterations per frontier state")
    parser.add_argument("--adversarial", action="store_true", help="GameAgent picks the worst card")
    parser.add_argument("--output", default=None, help="HTML file to write (default: regicide_seed<S>.html)")
    parser.add_argument("--no-open", action="store_true", help="do not open the viewer in a browser")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randrange(10**6)
    if args.agent == "hybrid":
        depth = args.depth or 1
        search = HybridSearch(depth=depth, mcts_iterations=args.iterations, sustain=args.sustain, seed=seed)
        label = f"hybrid depth {depth}{', sustain' if args.sustain else ''}"
    else:
        depth = args.depth or 3
        search = AlphaBetaSearch(depth)
        label = f"minimax depth {depth}"
    label = f"{args.players} player(s) · {label} · seed {seed}"

    print(f"Playing: {label} ...", flush=True)
    frames = record_game(search, args.players, seed, args.adversarial)
    output = args.output or f"regicide_seed{seed}.html"
    with open(output, "w") as f:
        f.write(build_html(frames, label, initial_state(args.players).max_hand_size))
    print(f"{frames[-1]['action']} Wrote {len(frames)} steps to {output}")
    if not args.no_open:
        webbrowser.open("file://" + os.path.abspath(output))


if __name__ == "__main__":
    main()
