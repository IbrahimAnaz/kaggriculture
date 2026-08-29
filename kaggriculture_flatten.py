"""Flatten an official Kaggriculture replay into a fixed-width little-endian stream."""
import argparse
import json
import struct
from pathlib import Path

MAGIC = b"KAGRST01"
VERSION = 1
BOARD = 10
PLAYERS = 2
MAX_HANDS = 16
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
CROPS = {name: index + 1 for index, name in enumerate(PRODUCTS[:5])}
SHOP_NAMES = ["BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET"]
HEADER = struct.Struct("<8sHHHHI")
FRAME = struct.Struct("<IHHB")
FARM = struct.Struct("<fhhBBH")
TILE = struct.Struct("<BBHhhhhh")


def number(value, default=0):
    return int(value or default)


def tile_record(tile):
    if tile is None:
        return TILE.pack(0, 0, 0, -1, 0, 0, -1, -1)
    if tile == "LOCKED":
        return TILE.pack(1, 0, 0, -1, 0, 0, -1, -1)
    kind = tile.get("kind")
    if kind == "WEED":
        return TILE.pack(3, 0, 0, -1, 0, 0, -1, -1)
    if kind == "PLANT":
        flags = 1 if tile.get("watered_today") else 0
        return TILE.pack(2, CROPS.get(tile.get("crop"), 0), flags,
                         number(tile.get("planted_day"), -1), number(tile.get("yield_units")),
                         number(tile.get("max_lifespan_step"), -1),
                         number(tile.get("fertilized_until_day"), -1),
                         number(tile.get("consecutive_unwatered")))
    if kind in ("COOP", "PASTURE"):
        animal = {"GOOSE": 1, "COW": 2, "SHEEP": 3}.get(tile.get("animal"), 0)
        flags = (1 if tile.get("fed_today") else 0) | (2 if tile.get("cared_today") else 0)
        return TILE.pack(4 if kind == "COOP" else 5, animal, flags,
                         number(tile.get("placed_day"), -1), number(tile.get("yield_units")),
                         number(tile.get("consecutive_unfed")), number(tile.get("pending_care_bonus")),
                         1 if tile.get("fertilizer_available") else 0)
    return TILE.pack(0, 0, 0, -1, 0, 0, -1, -1)


def farm_record(farm, out):
    hands = farm.get("hands", [])[:MAX_HANDS]
    quadrants = {"NW": 1, "NE": 2, "SW": 4, "SE": 8}
    unlocked = sum(quadrants.get(q, 0) for q in farm.get("unlocked_quadrants", []))
    out.write(FARM.pack(float(farm.get("money", 0)), *farm.get("farmer", [0, 0]),
                        unlocked, len(hands), number(farm.get("hires_today"))))
    for hand in hands:
        out.write(struct.pack("<bb", *hand))
    out.write(b"\x00\x00" * (MAX_HANDS - len(hands)))
    for row in farm.get("tiles", []):
        for tile in row[:BOARD]:
            out.write(tile_record(tile))


def counts(mapping, names):
    return [number(mapping.get(name)) for name in names]


def write_replay(source, target):
    replay = json.loads(Path(source).read_text(encoding="utf-8"))
    steps = replay.get("steps", [])
    with Path(target).open("wb") as out:
        out.write(HEADER.pack(MAGIC, VERSION, BOARD, PLAYERS, MAX_HANDS, len(steps)))
        for step_index, step in enumerate(steps):
            observation = step[0].get("observation", {})
            out.write(FRAME.pack(step_index, number(observation.get("day")), number(observation.get("hour")), number(observation.get("player"))))
            for farm in observation.get("farms", [])[:PLAYERS]:
                farm_record(farm, out)
            private = observation.get("private", {})
            out.write(struct.pack("<9h", *counts(private.get("shed", {}), PRODUCTS)))
            out.write(struct.pack("<5h", *counts(private.get("seeds", {}), PRODUCTS[:5])))
            inventories = private.get("inventories", [])[:MAX_HANDS]
            for inventory in inventories:
                out.write(struct.pack("<9h", *counts(inventory, PRODUCTS)))
            for _ in range(MAX_HANDS - len(inventories)):
                out.write(b"\x00" * struct.calcsize("<9h"))
            market = observation.get("market", {})
            out.write(struct.pack("<9q", *counts(market.get("inventory", {}), PRODUCTS)))
            out.write(struct.pack("<9h", *counts(market.get("prices", {}), PRODUCTS)))
            shops = observation.get("town", {}).get("unlocked_shops", [])
            out.write(struct.pack("<8B", *[shops.count(name) for name in SHOP_NAMES]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replay")
    parser.add_argument("output")
    args = parser.parse_args()
    write_replay(args.replay, args.output)


if __name__ == "__main__":
    main()
