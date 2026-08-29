#!/usr/bin/env python3
"""Dump a Kaggriculture replay's per-step states and actions to a compact
little-endian binary so the CUDA engine can be validated against it.

Wire format (all little-endian):
  header:  <IIi   magic=0x4B475230 'KG0' reserved, num_steps, seed
  then num_steps records; record t is (state_t, actions_t -> state_{t+1}):
    state (see _dump_state)
    actions (see _dump_actions)
"""

from __future__ import annotations

import argparse
import json
import struct
import sys

MAGIC = 0x4B475230  # "KG0\x00"

# Item order for shed/inventory: 9 products then 3 animals (GOOSE, COW, SHEEP).
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
ITEMS = PRODUCTS + ANIMALS
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
SHOPS = ["BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET"]

OPS = {"PASS":0,"NORTH":1,"SOUTH":2,"EAST":3,"WEST":4,"PICKUP":5,"PLANT":6,"WATER":7,"HARVEST":8,
       "FERTILIZE":9,"BUILD_COOP":10,"BUILD_PASTURE":11,"DIG":12,"PLACE":13,"FEED":14,
       "COLLECT_FERTILIZER":15,"CARE":16,"BUY_SEED":17,"BUY_PRODUCT":18,"BUY_ANIMAL":19,
       "SELL":20,"HIRE":21,"BUY_LAND":22,"DROP":23}

TILE_KINDS = {"EMPTY":0,"LOCKED":1,"WEED":2,"PLANT":3,"COOP":4,"PASTURE":5}
QUAD_BITS = {"NW":1,"NE":2,"SW":4,"SE":8}


def item_index(name: str) -> int:
    return ITEMS.index(name)


def crop_index(name: str) -> int:
    return CROPS.index(name)


def animal_index(name: str) -> int:
    return ANIMALS.index(name)


def action_op(name: str) -> int:
    return OPS[name]


def dump_tile(tile, out):
    if tile is None:
        out.write(struct.pack("<7B7h", TILE_KINDS["EMPTY"], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        return
    if tile == "LOCKED":
        out.write(struct.pack("<7B7h", TILE_KINDS["LOCKED"], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        return
    kind = tile.get("kind")
    if kind == "WEED":
        out.write(struct.pack("<7B7h", TILE_KINDS["WEED"], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        return
    if kind == "PLANT":
        out.write(struct.pack(
            "<7B7h",
            TILE_KINDS["PLANT"], crop_index(tile["crop"]), 0,
            1 if tile.get("watered_today") else 0, 0, 0, 0,
            tile.get("planted_day", 0), tile.get("yield_units", 0),
            tile.get("max_lifespan_step", -1), tile.get("fertilized_until_day", -1),
            tile.get("consecutive_unwatered", 0), 0, 0))
        return
    # COOP / PASTURE
    animal = tile.get("animal")
    out.write(struct.pack(
        "<7B7h",
        TILE_KINDS[kind], 0, (animal_index(animal) + 1) if animal else 0,
        0, 1 if tile.get("fed_today") else 0, 1 if tile.get("cared_today") else 0,
        1 if tile.get("fertilizer_available") else 0,
        tile.get("placed_day", 0), tile.get("yield_units", 0), 0, 0,
        0, tile.get("consecutive_unfed", 0), tile.get("pending_care_bonus", 0)))


def dump_farm(farm, out):
    out.write(struct.pack("<fhhBB", farm["money"], farm["farmer"][0], farm["farmer"][1],
                          farm["hand_count"] if "hand_count" in farm else len(farm["hands"]),
                          farm.get("hires_today", 0)))
    quad = 0
    for q in farm.get("unlocked_quadrants", []):
        quad |= QUAD_BITS[q]
    out.write(struct.pack("<B", quad))
    hands = farm.get("hands", [])
    hx = [h[0] for h in hands] + [0] * (16 - len(hands))
    hy = [h[1] for h in hands] + [0] * (16 - len(hands))
    out.write(struct.pack("<16h", *hx[:16]))
    out.write(struct.pack("<16h", *hy[:16]))
    for row in farm["tiles"]:
        for tile in row:
            dump_tile(tile, out)


def dump_private(private, out):
    shed = private.get("shed", {})
    out.write(struct.pack("<12h", *[shed.get(i, 0) for i in ITEMS]))
    seeds = private.get("seeds", {})
    out.write(struct.pack("<5h", *[seeds.get(c, 0) for c in CROPS]))
    inventories = private.get("inventories", [{}])
    inv = inventories[:17]
    inv += [{}] * (17 - len(inv))
    for u in range(17):
        d = inv[u]
        out.write(struct.pack("<12h", *[d.get(i, 0) for i in ITEMS]))


def dump_market(market, out):
    inv = market["inventory"]
    out.write(struct.pack("<9q", *[inv[p] for p in PRODUCTS]))
    prices = market["prices"]
    out.write(struct.pack("<9h", *[prices[p] for p in PRODUCTS]))


def dump_town(town, out):
    shops = town.get("unlocked_shops", [])
    out.write(struct.pack("<B", len(shops)))
    ids = [SHOPS.index(s) for s in shops] + [0] * (8 - len(shops))
    out.write(struct.pack("<8B", *ids[:8]))


def dump_state(obs, out):
    out.write(struct.pack("<hhiI", obs["day"], obs["hour"], obs.get("player", 0), obs.get("step", 0)))
    for f in obs["farms"]:
        dump_farm(f, out)
    # private is only on the player's own observation; the oracle keeps both.
    privates = obs.get("private")
    # For replay records, each player record has its own observation.private.
    # The dumper passes each player's private separately (see main).
    dump_market(obs["market"], out)
    dump_town(obs["town"], out)


def dump_private_record(private, out):
    dump_private(private, out)


def parse_unit_action(a):
    if not isinstance(a, list) or not a:
        return (0, -1, 0)  # PASS
    op = action_op(a[0])
    item = -1
    quantity = 1
    if op in (OPS["PICKUP"], OPS["PLANT"], OPS["PLACE"]):
        # item is product/animal/crop depending on op
        name = a[1] if len(a) > 1 else ""
        if op == OPS["PLANT"]:
            item = crop_index(name) if name in CROPS else -1
        elif name in ITEMS:
            item = item_index(name)
        else:
            item = -1
        quantity = int(a[2]) if len(a) > 2 else 1
    elif op == OPS["FEED"]:
        item = item_index("WHEAT")
    return (op, item, quantity)


def parse_market_order(o):
    op = action_op(o[0])
    item = -1
    quantity = 0
    if op in (OPS["BUY_SEED"], OPS["BUY_PRODUCT"], OPS["BUY_ANIMAL"], OPS["SELL"]):
        name = o[1]
        if op == OPS["BUY_SEED"]:
            item = crop_index(name) if name in CROPS else -1
        elif op == OPS["BUY_ANIMAL"]:
            item = animal_index(name) if name in ANIMALS else -1
        else:
            item = item_index(name) if name in PRODUCTS else -1
        quantity = int(o[2]) if len(o) > 2 else 0
    return (op, item, quantity)


def dump_actions(action, out):
    farmer = parse_unit_action(action.get("farmer"))
    out.write(struct.pack("<bbh", *farmer))
    hands = action.get("hands", [])
    out.write(struct.pack("<b", len(hands)))
    hlist = [parse_unit_action(h) for h in hands]
    hlist += [(0, -1, 0)] * (16 - len(hlist))
    for h in hlist[:16]:
        out.write(struct.pack("<bbh", *h))
    market = action.get("market", [])
    out.write(struct.pack("<b", len(market)))
    mlist = [parse_market_order(o) for o in market]
    mlist += [(0, -1, 0)] * (10 - len(mlist))
    for m in mlist[:10]:
        out.write(struct.pack("<bbh", *m))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("replay")
    ap.add_argument("output")
    args = ap.parse_args()

    replay = json.load(open(args.replay))
    steps = replay["steps"]
    n = len(steps)
    seed = replay.get("info", {}).get("seed", 0)

    out = open(args.output, "wb")
    out.write(struct.pack("<IIi", MAGIC, n, seed))

    # Replay the recorded actions through the official engine to obtain the
    # correct per-step state (including both players' private state).
    from kaggle_environments import make
    streams = [[steps[t + 1][p].get("action", {"farmer": ["PASS"]}) for t in range(n - 1)] for p in range(2)]
    def mk(a):
        st = {"i": 0}
        def ag(obs):
            i = st["i"]; st["i"] += 1
            return a[i] if i < len(a) else {"farmer": ["PASS"], "hands": [], "market": []}
        return ag
    env = make("kaggriculture", configuration={"episodeSteps": n, "seed": seed}, debug=False)
    env.run([mk(streams[0]), mk(streams[1])])

    for t in range(n):
        # state_t: use the engine's replayed observation (has private for both players)
        obs = env.steps[t][0]["observation"]
        # Emit state_t, then actions_t (which produce state_{t+1}).
        dump_state(obs, out)
        # Private for player 0 and 1.
        dump_private(env.steps[t][0]["observation"]["private"], out)
        dump_private(env.steps[t][1]["observation"]["private"], out)
        if t < n - 1:
            dump_actions(steps[t + 1][0].get("action", {}), out)
            dump_actions(steps[t + 1][1].get("action", {}), out)

    out.close()
    print(f"DUMPED {args.output} steps={n} seed={seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
