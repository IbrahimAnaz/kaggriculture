"""Ground-truth parity harness for the Kaggriculture CUDA simulator.

This module wraps the official ``kaggle-environments`` Kaggriculture engine as an
oracle so the CUDA simulator can be validated against the real game rules.

Modes:

* ``generate`` -> a deterministic golden replay (JSON), keyed off ``--seed``.
* ``check`` -> replay a replay's recorded actions through the official engine and
  confirm the engine reproduces every observation exactly (parity).
* ``play`` -> replay a recorded replay and print a step-by-step summary.

Parity is defined on the deterministic transition function only. The single
stochastic element in the game (weed spawns and shop-unlock draws at end of day)
is keyed off ``env.info["seed"]``, so it is *patched* by passing the recorded seed
through rather than re-implemented. Verifying with a different seed diverges only
on weed/shop state, never on the deterministic mechanics (actions, prices, growth,
money).

Usage:
    python kaggriculture_oracle.py generate --seed 42 --steps 720 -o replay_golden.json
    python kaggriculture_oracle.py check --replay replay.json
    python kaggriculture_oracle.py play --replay replay.json --every 24
"""

from __future__ import annotations

import argparse
import json

from kaggle_environments import make


def _build_agent_from_action_stream(action_stream):
    """Return an agent function that replays a pre-recorded action stream.

    ``action_stream`` is a list (one entry per step) of dicts shaped exactly like
    the action contract: ``{"farmer": [...], "hands": [...], "market": [...]}``.
    Each agent is invoked exactly once per turn, so a monotonic call counter is
    used to index the stream (the observation's own step field is not reliable
    across framework versions).
    """
    state = {"i": 0}

    def agent(obs):
        i = state["i"]
        state["i"] += 1
        if i >= len(action_stream):
            return {"farmer": ["PASS"], "hands": [], "market": []}
        return action_stream[i]

    return agent


def generate(agents, seed: int, steps: int = 720) -> dict:
    """Run the official engine and return the full replay JSON (deterministic)."""
    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed}, debug=False)
    env.run(agents)
    return env.toJSON()


def check_self_consistency(replay_json: dict) -> dict:
    """Replay a golden replay's own actions and confirm the engine reproduces its
    observations exactly. Returns a summary of mismatches (empty == parity).

    Replay-record indexing: ``steps[0].action`` is a placeholder no-op; the real
    action sequence begins at ``steps[1].action``, whose effect appears in
    ``steps[1].observation``. In general ``steps[t].action`` (t >= 1) transforms
    ``steps[t-1].observation`` into ``steps[t].observation``, so the agent must be
    fed ``steps[t+1].action`` on its t-th call (there are ``n_steps - 1`` calls).
    """
    steps = replay_json["steps"]
    n_steps = len(steps)
    players = 2

    action_streams = [[None] * (n_steps - 1) for _ in range(players)]
    for t in range(n_steps - 1):
        for p in range(players):
            action_streams[p][t] = steps[t + 1][p].get("action", {"farmer": ["PASS"]})

    seed = replay_json.get("info", {}).get("seed", 0)
    env = make("kaggriculture", configuration={"episodeSteps": n_steps, "seed": seed}, debug=False)
    agents = [_build_agent_from_action_stream(action_streams[p]) for p in range(players)]
    env.run(agents)

    mismatches = []
    for t in range(n_steps):
        rec_obs = steps[t]
        got = env.steps[t]
        for p in range(players):
            want = rec_obs[p].get("observation")
            have = got[p].get("observation")
            if want != have:
                mismatches.append({"step": t, "player": p})
    return {"steps": n_steps, "mismatches": mismatches, "parity": len(mismatches) == 0}


def play(replay_json: dict, every: int = 24) -> dict:
    """Replay a recorded replay through the official engine and emit a step-by-step
    summary so the game can be "watched". Returns the parity summary plus a list of
    per-interval snapshots."""
    result = check_self_consistency(replay_json)
    steps = replay_json["steps"]
    snapshots = []
    for t in range(0, len(steps), every):
        rec = steps[t]
        row = {"step": t, "day": t // 24, "hour": t % 24}
        for p in range(2):
            obs = rec[p].get("observation", {})
            row[f"p{p}_money"] = obs.get("farms", [{}])[p]["money"] if obs else None
        snapshots.append(row)
    result["snapshots"] = snapshots
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Kaggriculture parity oracle")
    ap.add_argument("mode", choices=["generate", "check", "play"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--every", type=int, default=24)
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--replay", default=None)
    args = ap.parse_args()

    if args.mode == "generate":
        agents = ["starter", "pass"]
        rp = generate(agents, args.seed, args.steps)
        out = args.output or "replay_golden.json"
        with open(out, "w") as f:
            json.dump(rp, f)
        print(f"GENERATED {out} seed={args.seed} steps={len(rp['steps'])}")
        print("spec:", rp.get("specification", {}).get("name", "kaggriculture"))
        return 0

    if args.mode == "check":
        with open(args.replay, "r") as f:
            rp = json.load(f)
        result = check_self_consistency(rp)
        print(f"CHECK steps={result['steps']} parity={result['parity']} "
              f"mismatches={len(result['mismatches'])}")
        if result["mismatches"]:
            print("first mismatches:", result["mismatches"][:10])
            return 1
        return 0

    if args.mode == "play":
        with open(args.replay, "r") as f:
            rp = json.load(f)
        result = play(rp, args.every)
        print(f"PLAY parity={result['parity']} mismatches={len(result['mismatches'])}")
        for row in result["snapshots"]:
            print(f"  step={row['step']:>3} day={row['day']:>2} hour={row['hour']:>2} "
                  f"p0_money={row.get('p0_money')} p1_money={row.get('p1_money')}")
        return 0 if result["parity"] else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
