// ============================================================================
// kaggriculture_engine.cuh
// Faithful CUDA port of the official Kaggriculture game engine
// (kaggle-environments envs/kaggriculture/kaggriculture.py, module 1.32.7).
//
// This header mirrors the official transition function exactly so the GPU
// simulator reproduces the deterministic game rules. The single stochastic
// element (weed spawns + shop-unlock draws at end of day) is exposed through a
// `rng_next()` hook so callers can patch it (e.g. cuRAND or a replay-supplied
// stream) without affecting deterministic parity.
//
// Turn order (mirrors interpreter()):
//   unit actions -> process_market -> town_consume -> decay_plants -> end_of_day
// ============================================================================
#pragma once

#include <cstdint>
#include <cmath>

// Allow the header to compile with a plain host compiler (g++/clang) for
// fast CPU-side validation, in addition to nvcc for GPU execution.
#if !defined(__CUDACC__) && !defined(__NVCC__)
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#endif

namespace kgr {

// --- Board / game constants -------------------------------------------------
constexpr int BOARD_SIZE      = 10;
constexpr int BOARD_TILES     = BOARD_SIZE * BOARD_SIZE;
constexpr int NUM_PLAYERS     = 2;
constexpr int TURNS_PER_DAY   = 24;
constexpr int SEASON_DAYS     = 30;
constexpr int EPISODE_STEPS   = TURNS_PER_DAY * SEASON_DAYS;  // 720
constexpr int MAX_HANDS       = 16;
constexpr int MAX_MARKET_ORDERS = 10;
constexpr int SHED_CAPACITY   = 100;
constexpr int MARKET_I0       = 10000;
constexpr int PRICE_FLOOR     = 1;
constexpr float WEED_SPAWN_CHANCE = 0.005f;
constexpr int TOWN_SHOP_SELL_INTERVAL  = 4;
constexpr int TOWN_CENTER_SELL_INTERVAL = 24;
constexpr int TOWN_SHOP_UNLOCK_INTERVAL = 3;
constexpr int MAX_SHOP_INSTANCES = 8;
constexpr int STARTING_MONEY   = 3000;
constexpr int FARM_HAND_COST_MULT = 1;
constexpr float HINGE_GAIN     = 8.0f;

// --- Product / crop / animal enums -----------------------------------------
enum Product : uint8_t {
    P_WHEAT = 0, P_CARROT, P_TOMATO, P_STRAWBERRY, P_MELON,
    P_EGG, P_MILK, P_WOOL, P_FERTILIZER, kProducts = 9
};

enum Crop : uint8_t { C_WHEAT = 0, C_CARROT, C_TOMATO, C_STRAWBERRY, C_MELON, kCrops = 5 };

enum Animal : uint8_t { A_GOOSE = 0, A_COW, A_SHEEP, kAnimals = 3 };

enum Shop : uint8_t {
    S_BAKERY = 0, S_PIZZA_SHOP, S_BRUNCH_SPOT, S_YARN_STORE,
    S_ICE_CREAM_SHOP, S_PET_CAFE, S_SMOOTHIE_SHOP, S_FARMERS_MARKET, kShops = 8
};

// Official action operations (order matters: this is the wire contract).
enum Op : uint8_t {
    OP_PASS = 0, OP_NORTH, OP_SOUTH, OP_EAST, OP_WEST,
    OP_PICKUP, OP_PLANT, OP_WATER, OP_HARVEST, OP_FERTILIZE,
    OP_BUILD_COOP, OP_BUILD_PASTURE, OP_DIG, OP_PLACE, OP_FEED,
    OP_COLLECT_FERTILIZER, OP_CARE, OP_BUY_SEED, OP_BUY_PRODUCT,
    OP_BUY_ANIMAL, OP_SELL, OP_HIRE, OP_BUY_LAND, OP_DROP, kOps = 24
};

// Tile kinds.
enum TileKind : uint8_t { T_EMPTY = 0, T_LOCKED, T_WEED, T_PLANT, T_COOP, T_PASTURE };

// Shape functions used by the market price model.
enum Shape : uint8_t { F_LINEAR = 0, F_SQ, F_SQRT, F_LOG, F_LOG10, F_HINGE };

// --- Crop parameters --------------------------------------------------------
struct CropParams {
    int16_t seed;
    int16_t first_yield_day;
    int16_t max_yield_day;
    int16_t interval;
    int16_t max_yield;
    bool    ongoing;
};

__device__ __host__ constexpr CropParams CROPS[kCrops] = {
    {  10,  2,  4, 0,  6, false },  // WHEAT
    {  20,  2,  3, 0,  4, false },  // CARROT
    {  50,  8,  8, 1,  4, true  },  // TOMATO (ongoing)
    { 100, 10, 10, 2,  4, true  },  // STRAWBERRY (ongoing)
    {  80, 10, 12, 0,  6, false },  // MELON
};

// --- Animal parameters ------------------------------------------------------
struct AnimalParams {
    int16_t cost;
    TileKind structure;   // COOP or PASTURE
    int16_t first_yield_day;
    int16_t interval;
    int16_t max_held;
    Product product;
};

__device__ __host__ constexpr AnimalParams ANIMALS[kAnimals] = {
    { 300, T_COOP,    4, 1, 4, P_EGG  },  // GOOSE
    { 400, T_PASTURE, 8, 2, 6, P_MILK },  // COW
    { 500, T_PASTURE, 6, 3, 6, P_WOOL },  // SHEEP
};

// --- Market price parameters ------------------------------------------------
struct MarketParams {
    double base;
    double I0;
    double T;
    Shape below_func;
    double below_target;
    Shape above_func;
    double above_target;
};

__device__ __host__ constexpr MarketParams MARKET_PARAMS[kProducts] = {
    // WHEAT
    {  25, 10000, 400, F_SQRT,   0.80f, F_LOG,   0.20f },
    // CARROT
    {  35, 10000, 450, F_HINGE,  1.00f, F_SQRT,  0.70f },
    // TOMATO
    {  60, 10000, 200, F_HINGE,  0.40f, F_SQRT,  0.60f },
    // STRAWBERRY
    { 120, 10000, 100, F_SQRT,   0.70f, F_LINEAR, 1.60f },
    // MELON
    { 250, 10000, 300, F_LOG,    0.20f, F_SQ,    3.60f },
    // EGG
    {  50, 10000, 332, F_HINGE,  0.40f, F_LOG,   0.20f },
    // MILK
    { 160, 10000, 122, F_SQRT,   0.60f, F_LINEAR, 1.60f },
    // WOOL
    { 200, 10000, 105, F_LOG,    0.20f, F_SQ,    3.20f },
    // FERTILIZER
    { 100, 10000, 200, F_LINEAR, 0.40f, F_LINEAR, 0.40f },
};

// --- Shops: products each shop instance consumes ----------------------------
// Represented as a flattened demand list. Single-product shops consume 2x.
__device__ __host__ constexpr uint8_t SHOP_DEMAND_COUNT[kShops] = {
    2, 3, 3, 1, 3, 1, 2, 4,
};

__device__ __host__ constexpr uint8_t SHOP_DEMAND[kShops][4] = {
    { P_EGG,        P_WHEAT,     0xFF,         0xFF },         // BAKERY
    { P_MILK,       P_TOMATO,    P_WHEAT,      0xFF },         // PIZZA_SHOP
    { P_EGG,        P_WHEAT,     P_STRAWBERRY, 0xFF },         // BRUNCH_SPOT
    { P_WOOL,       0xFF,        0xFF,         0xFF },         // YARN_STORE
    { P_STRAWBERRY, P_MILK,      P_WHEAT,      0xFF },         // ICE_CREAM_SHOP
    { P_CARROT,     0xFF,        0xFF,         0xFF },         // PET_CAFE
    { P_STRAWBERRY, P_MILK,      0xFF,         0xFF },         // SMOOTHIE_SHOP
    { P_WHEAT,      P_CARROT,    P_TOMATO,     P_STRAWBERRY }, // FARMERS_MARKET
};

// --- Land order / prices ----------------------------------------------------
__device__ __host__ constexpr uint8_t LAND_ORDER[3] = { 1 /*NE*/, 2 /*SW*/, 3 /*SE*/ };
__device__ __host__ constexpr int LAND_PRICES[3] = { 1000, 2000, 4000 };

// ============================================================================
// PRICE MODEL
//   price(inv) = base + sign * amp * f(|inv - I0|), floored at PRICE_FLOOR.
//   amp = target * base / f(T)
// ============================================================================
__device__ __host__ __forceinline__ double shape_fn(Shape f, double x, double T) {
    x = fmax(0.0, x);
    switch (f) {
        case F_LINEAR: return x;
        case F_SQ:     return x * x;
        case F_SQRT:   return sqrt(x);
        case F_LOG:    return log(1.0 + x);
        case F_LOG10:  return log10(1.0 + x);
        case F_HINGE: {
            if (T <= 0.0) return x;
            double u = x / T;
            return u + HINGE_GAIN * fmax(0.0, u - 1.0) * fmax(0.0, u - 1.0);
        }
        default: return x;
    }
}

__device__ __host__ __forceinline__ int market_price(int item, int inventory) {
    const MarketParams& p = MARKET_PARAMS[item];
    double price;
    if (inventory < p.I0) {
        double amp = p.below_target * p.base / shape_fn(p.below_func, p.T, p.T);
        price = p.base + amp * shape_fn(p.below_func, p.I0 - inventory, p.T);
    } else {
        double amp = p.above_target * p.base / shape_fn(p.above_func, p.T, p.T);
        price = p.base - amp * shape_fn(p.above_func, inventory - p.I0, p.T);
    }
    int rounded = static_cast<int>(lround(price));
    return rounded < PRICE_FLOOR ? PRICE_FLOOR : rounded;
}

// ============================================================================
// STATE MODEL
// ============================================================================
// Item space: 0..8 products (WHEAT..FERTILIZER), 9..11 animals (GOOSE, COW, SHEEP).
constexpr int kItemCount = kProducts + kAnimals;  // 12

struct Tile {
    uint8_t  kind;                 // TileKind
    uint8_t  crop;                 // Crop (valid if kind == T_PLANT)
    uint8_t  animal;               // Animal+1 (0 = no animal on structure)
    uint8_t  watered_today;
    uint8_t  fed_today;
    uint8_t  cared_today;
    uint8_t  fertilizer_available;
    int16_t  planted_day;          // plant: planted_day ; animal: placed_day
    int16_t  yield_units;
    int16_t  max_lifespan_step;    // plant only, -1 for ongoing crops
    int16_t  fertilized_until_day; // plant only, -1
    int16_t  consecutive_unwatered;
    int16_t  consecutive_unfed;
    int16_t  pending_care_bonus;
};

struct Farm {
    float    money;
    Tile     tiles[BOARD_TILES];   // row-major: tiles[y*BOARD_SIZE + x]
    int16_t  farmer_x, farmer_y;
    int16_t  hand_x[MAX_HANDS];
    int16_t  hand_y[MAX_HANDS];
    uint8_t  hand_count;
    uint8_t  unlocked_quadrants;   // bit0=NW(1) bit1=NE(2) bit2=SW(4) bit3=SE(8)
    uint8_t  hires_today;
};

struct Private {
    int16_t shed[kItemCount];                 // products + animals
    int16_t seeds[kCrops];
    int16_t inventory[MAX_HANDS + 1][kItemCount]; // [0] = main farmer, [1..] = hands
};

struct Market {
    int64_t inventory[kProducts];
    int16_t prices[kProducts];
};

struct Town {
    uint8_t unlocked_shops[MAX_SHOP_INSTANCES];
    uint8_t shop_count;
};

struct GameState {
    Farm    farms[NUM_PLAYERS];
    Private privates[NUM_PLAYERS];
    Market  market;
    Town    town;
    int16_t day;
    int16_t hour;
    uint8_t active_player;
    uint32_t step;
};

// ============================================================================
// GEOMETRY / INIT
// ============================================================================
__device__ __host__ __forceinline__ bool in_nw(int x, int y) {
    int half = BOARD_SIZE / 2;
    return y < half && x < half;
}

__device__ __host__ __forceinline__ bool is_shed_adjacent(int x, int y) {
    // Shed-access tiles: (4,4),(5,4),(4,5),(5,5) for a 10x10 board.
    int half = BOARD_SIZE / 2;
    return (y == half - 1 || y == half) && (x == half - 1 || x == half);
}

__device__ __host__ __forceinline__ int shed_access_tile_index(int slot) {
    // 0..3 -> (4,4),(5,4),(4,5),(5,5) in NWSE order.
    int half = BOARD_SIZE / 2;
    int x = (slot == 0 || slot == 2) ? half - 1 : half;
    int y = (slot == 0 || slot == 1) ? half - 1 : half;
    return y * BOARD_SIZE + x;
}

__device__ __host__ __forceinline__ void tile_init_empty(Tile& t) { t = Tile{}; }

__device__ __host__ __forceinline__ void tile_init_locked(Tile& t) {
    t = Tile{};
    t.kind = T_LOCKED;
}

__device__ __host__ __forceinline__ void tile_init_weed(Tile& t) {
    t = Tile{};
    t.kind = T_WEED;
}

__device__ __host__ __forceinline__ void tile_init_plant(Tile& t, int crop, int day, int turns_per_day) {
    t = Tile{};
    t.kind = T_PLANT;
    t.crop = static_cast<uint8_t>(crop);
    t.planted_day = static_cast<int16_t>(day);
    t.watered_today = 0;
    t.consecutive_unwatered = 1;  // planting day counts as unwatered
    t.yield_units = CROPS[crop].ongoing ? 0 : 1;
    t.max_lifespan_step = CROPS[crop].ongoing ? -1
        : static_cast<int16_t>((day + CROPS[crop].max_yield_day + 1) * turns_per_day);
    t.fertilized_until_day = -1;
    t.consecutive_unfed = 0;
}

__device__ __host__ __forceinline__ void tile_init_animal(Tile& t, int animal, int day) {
    t = Tile{};
    t.kind = ANIMALS[animal].structure;
    t.animal = static_cast<uint8_t>(animal + 1);
    t.planted_day = static_cast<int16_t>(day);
    t.yield_units = 0;
    t.consecutive_unfed = 0;
    t.fed_today = 0;
    t.cared_today = 0;
    t.fertilizer_available = 0;
    t.pending_care_bonus = 0;
}

__device__ __host__ __forceinline__ void init_farm(Farm& farm, int starting_money) {
    farm.money = static_cast<float>(starting_money);
    for (int y = 0; y < BOARD_SIZE; ++y)
        for (int x = 0; x < BOARD_SIZE; ++x) {
            Tile& t = farm.tiles[y * BOARD_SIZE + x];
            in_nw(x, y) ? tile_init_empty(t) : tile_init_locked(t);
        }
    // Default spawn: first free shed-access tile in NWSE order (always (4,4)).
    int half = BOARD_SIZE / 2;
    farm.farmer_x = static_cast<int16_t>(half - 1);
    farm.farmer_y = static_cast<int16_t>(half - 1);
    farm.hand_count = 0;
    farm.unlocked_quadrants = 1;  // NW
    farm.hires_today = 0;
}

__device__ __host__ __forceinline__ void init_private(Private& p) {
    for (int i = 0; i < kItemCount; ++i) p.shed[i] = 0;
    for (int c = 0; c < kCrops; ++c) p.seeds[c] = 0;
    for (int u = 0; u <= MAX_HANDS; ++u)
        for (int i = 0; i < kItemCount; ++i) p.inventory[u][i] = 0;
}

__device__ __host__ __forceinline__ void init_market(Market& m) {
    for (int p = 0; p < kProducts; ++p) {
        m.inventory[p] = MARKET_PARAMS[p].I0;
        m.prices[p] = static_cast<int16_t>(market_price(p, static_cast<int>(m.inventory[p])));
    }
}

__device__ __host__ __forceinline__ void init_town(Town& t) { t.shop_count = 0; }

__device__ __host__ __forceinline__ void init_state(GameState& s, int starting_money = STARTING_MONEY) {
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        init_farm(s.farms[p], starting_money);
        init_private(s.privates[p]);
    }
    init_market(s.market);
    init_town(s.town);
    s.day = 0;
    s.hour = 0;
    s.active_player = 0;
    s.step = 0;
}

__device__ __host__ __forceinline__ void refresh_prices(Market& m) {
    for (int p = 0; p < kProducts; ++p)
        m.prices[p] = static_cast<int16_t>(market_price(p, static_cast<int>(m.inventory[p])));
}

// ============================================================================
// INVENTORY HELPERS
// ============================================================================
__device__ __host__ __forceinline__ int16_t* unit_inventory(Private& p, int idx) {
    return p.inventory[idx];  // idx 0 = farmer, 1.. = hands
}

__device__ __host__ __forceinline__ void inv_add(int16_t* inv, int item, int n) {
    inv[item] = static_cast<int16_t>(inv[item] + n);
}

__device__ __host__ __forceinline__ bool inv_take(int16_t* inv, int item, int n) {
    if (inv[item] < n) return false;
    inv[item] = static_cast<int16_t>(inv[item] - n);
    return true;
}

__device__ __host__ __forceinline__ int shed_total(const Private& p) {
    int s = 0;
    for (int i = 0; i < kItemCount; ++i) s += p.shed[i];
    return s;
}

// ============================================================================
// UNIT ACTION (farmer / hand). Mirrors _apply_unit_action exactly.
// ============================================================================
__device__ __host__ __forceinline__ void apply_unit_action(
    Farm& farm, Private& priv, int idx,
    int op, int item, int quantity,  // quantity = n arg (default 1)
    int day, int turns_per_day, int shed_capacity)
{
    int fx, fy;
    if (idx == 0) { fx = farm.farmer_x; fy = farm.farmer_y; }
    else {
        int h = idx - 1;
        if (h >= farm.hand_count) return;
        fx = farm.hand_x[h]; fy = farm.hand_y[h];
    }
    int16_t* inv = unit_inventory(priv, idx);

    // Movement
    if (op == OP_NORTH || op == OP_SOUTH || op == OP_EAST || op == OP_WEST) {
        int nx = fx, ny = fy;
        if (op == OP_NORTH) ny--; else if (op == OP_SOUTH) ny++;
        else if (op == OP_EAST) nx++; else nx--;
        if (nx < 0 || nx >= BOARD_SIZE || ny < 0 || ny >= BOARD_SIZE) return;
        if (idx == 0) { farm.farmer_x = static_cast<int16_t>(nx); farm.farmer_y = static_cast<int16_t>(ny); }
        else { farm.hand_x[idx - 1] = static_cast<int16_t>(nx); farm.hand_y[idx - 1] = static_cast<int16_t>(ny); }
        return;
    }
    if (op == OP_PASS) return;

    Tile& tile = farm.tiles[fy * BOARD_SIZE + fx];

    // Shed operations resolve before the LOCKED guard.
    if (op == OP_DROP) {
        if (!is_shed_adjacent(fx, fy)) return;
        for (int i = 0; i < kItemCount; ++i) {
            int n = inv[i];
            if (n <= 0) { inv[i] = 0; continue; }
            int room = shed_capacity - shed_total(priv);
            if (room < 0) room = 0;
            int take = n < room ? n : room;
            if (take > 0) priv.shed[i] = static_cast<int16_t>(priv.shed[i] + take);
            inv[i] = 0;
        }
        return;
    }
    if (op == OP_PICKUP) {
        if (!is_shed_adjacent(fx, fy)) return;
        if (item < 0 || item >= kItemCount) return;
        int n = quantity;
        if (n <= 0) return;
        int available = priv.shed[item];
        n = n < available ? n : available;
        if (n <= 0) return;
        priv.shed[item] = static_cast<int16_t>(priv.shed[item] - n);
        inv_add(inv, item, n);
        return;
    }
    if (op == OP_PLACE) {
        if (item < 0) return;
        // Animal placement: standing on matching unoccupied structure.
        if (item >= kProducts && item < kItemCount) {
            int animal = item - kProducts;
            if (tile.kind == ANIMALS[animal].structure && tile.animal == 0) {
                if (inv_take(inv, item, 1)) {
                    tile_init_animal(tile, animal, day);
                }
                return;
            }
        }
        // Shed drop: adjacent to shed, obey capacity.
        if (is_shed_adjacent(fx, fy)) {
            int n = quantity;
            if (n <= 0) return;
            if (inv[item] < n) n = inv[item];
            if (n <= 0) return;
            int room = shed_capacity - shed_total(priv);
            if (room < 0) room = 0;
            if (n > room) n = room;
            if (n <= 0) return;
            inv[item] = static_cast<int16_t>(inv[item] - n);
            priv.shed[item] = static_cast<int16_t>(priv.shed[item] + n);
        }
        return;
    }

    // Everything below requires an owned tile.
    if (tile.kind == T_LOCKED) return;

    if (op == OP_PLANT) {
        if (item < 0 || item >= kCrops) return;
        if (tile.kind != T_EMPTY) return;
        if (priv.seeds[item] <= 0) return;
        priv.seeds[item]--;
        tile_init_plant(tile, item, day, turns_per_day);
        return;
    }
    if (op == OP_WATER) {
        if (tile.kind != T_PLANT) return;
        if (tile.watered_today) return;
        tile.watered_today = 1;
        const CropParams& cd = CROPS[tile.crop];
        if (!cd.ongoing) {
            int age_days = day - tile.planted_day;
            int window_start = (cd.max_yield_day + 1) / 2;
            if (window_start <= age_days && age_days <= cd.max_yield_day) {
                int bonus = (tile.fertilized_until_day >= day) ? 2 : 1;
                int y = tile.yield_units + bonus;
                if (y > cd.max_yield) y = cd.max_yield;
                tile.yield_units = static_cast<int16_t>(y);
            }
        }
        return;
    }
    if (op == OP_HARVEST) {
        if (tile.yield_units <= 0) return;
        if (tile.kind == T_PLANT) {
            const CropParams& cd = CROPS[tile.crop];
            if (day - tile.planted_day < cd.first_yield_day) return;
            int units = tile.yield_units;
            tile.yield_units = 0;
            inv_add(inv, tile.crop, units);
            if (!cd.ongoing) tile_init_empty(tile);
        } else if (tile.animal > 0) {
            int animal = tile.animal - 1;
            int units = tile.yield_units;
            tile.yield_units = 0;
            inv_add(inv, ANIMALS[animal].product, units);
        }
        return;
    }
    if (op == OP_FERTILIZE) {
        if (tile.kind != T_PLANT) return;
        if (!inv_take(inv, P_FERTILIZER, 1)) return;
        int until = day + 2;
        if (tile.fertilized_until_day < until) tile.fertilized_until_day = static_cast<int16_t>(until);
        return;
    }
    if (op == OP_DIG) {
        if (tile.kind == T_EMPTY) return;
        if (tile.animal > 0) return;  // cannot dig a structure with an animal
        tile_init_empty(tile);
        return;
    }
    if (op == OP_BUILD_COOP) {
        if (tile.kind != T_EMPTY) return;
        tile = Tile{};
        tile.kind = T_COOP;
        return;
    }
    if (op == OP_BUILD_PASTURE) {
        if (tile.kind != T_EMPTY) return;
        tile = Tile{};
        tile.kind = T_PASTURE;
        return;
    }
    if (op == OP_FEED) {
        if (tile.animal == 0) return;
        if (tile.fed_today) return;
        if (!inv_take(inv, P_WHEAT, 1)) return;
        tile.fed_today = 1;
        return;
    }
    if (op == OP_COLLECT_FERTILIZER) {
        if (tile.animal == 0) return;
        if (!tile.fertilizer_available) return;
        tile.fertilizer_available = 0;
        inv_add(inv, P_FERTILIZER, 1);
        return;
    }
    if (op == OP_CARE) {
        if (tile.animal == 0) return;
        if (tile.cared_today) return;
        tile.cared_today = 1;
        return;
    }
}

// ============================================================================
// HIRING / LAND
// ============================================================================
__device__ __host__ __forceinline__ int fib(int n) {
    // fib(0)=1, fib(1)=1, fib(2)=2, fib(3)=3, fib(4)=5...
    int a = 1, b = 1;
    for (int i = 0; i < n; ++i) { int t = a + b; a = b; b = t; }
    return a;
}

__device__ __host__ __forceinline__ int hire_cost(int hires_today, int mult) {
    return mult * fib(hires_today);
}

__device__ __host__ __forceinline__ void do_hire(Farm& farm, Private& priv, int mult) {
    int cost = hire_cost(farm.hires_today, mult);
    if (farm.money < cost) return;
    farm.money -= cost;
    farm.hires_today++;
    // Spawn hand at first free shed-access tile (NWSE), tie-broken by min occupancy.
    int half = BOARD_SIZE / 2;
    int xs[4] = { half - 1, half, half - 1, half };
    int ys[4] = { half - 1, half - 1, half, half };
    int best_slot = 0, best_occ = 100;
    for (int slot = 0; slot < 4; ++slot) {
        int occ = 0;
        if (farm.farmer_x == xs[slot] && farm.farmer_y == ys[slot]) occ++;
        for (int h = 0; h < farm.hand_count; ++h)
            if (farm.hand_x[h] == xs[slot] && farm.hand_y[h] == ys[slot]) occ++;
        if (occ < best_occ) { best_occ = occ; best_slot = slot; }
    }
    int idx = farm.hand_count;
    farm.hand_x[idx] = static_cast<int16_t>(xs[best_slot]);
    farm.hand_y[idx] = static_cast<int16_t>(ys[best_slot]);
    farm.hand_count++;
    // New inventory slot for the hand (already zeroed in init; clear defensively).
    for (int i = 0; i < kItemCount; ++i) priv.inventory[idx + 1][i] = 0;
}

__device__ __host__ __forceinline__ void do_buy_land(Farm& farm) {
    // unlocked_quadrants bits: NW=1 always; NE=2, SW=4, SE=8 in LAND_ORDER.
    int n_unlocked_extra = 0;
    if (farm.unlocked_quadrants & 2) n_unlocked_extra++;
    if (farm.unlocked_quadrants & 4) n_unlocked_extra++;
    if (farm.unlocked_quadrants & 8) n_unlocked_extra++;
    if (n_unlocked_extra >= 3) return;
    int cost = LAND_PRICES[n_unlocked_extra];
    if (farm.money < cost) return;
    farm.money -= cost;
    uint8_t qbit = (uint8_t)(2 << n_unlocked_extra);  // NE=2, SW=4, SE=8
    farm.unlocked_quadrants |= qbit;
    // Unlock the quadrant's LOCKED tiles.
    for (int y = 0; y < BOARD_SIZE; ++y)
        for (int x = 0; x < BOARD_SIZE; ++x) {
            bool in = false;
            if (n_unlocked_extra == 0) in = (y < 5 && x >= 5);          // NE
            else if (n_unlocked_extra == 1) in = (y >= 5 && x < 5);     // SW
            else in = (y >= 5 && x >= 5);                               // SE
            if (in && farm.tiles[y * BOARD_SIZE + x].kind == T_LOCKED)
                tile_init_empty(farm.tiles[y * BOARD_SIZE + x]);
        }
}

// ============================================================================
// RNG HOOK (patched: simple xorshift, NOT Python's MT19937). Deterministic
// transitions do not depend on the exact sequence; weeds/shops are loaded from
// the replay when validating recorded state.
// ============================================================================
__device__ __host__ __forceinline__ uint32_t rng_next(uint32_t& s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s;
}

__device__ __host__ __forceinline__ uint32_t rng_seed(uint32_t episode_seed, int day) {
    uint32_t s = (episode_seed * 1000003u) ^ static_cast<uint32_t>(day);
    if (s == 0) s = 1u;
    return s;
}

// ============================================================================
// MARKET ORDER PROCESSING. Mirrors _process_market / _commit_unit / _parse_order.
// ============================================================================
struct MarketOrder {
    int8_t  op;       // OP_BUY_SEED / OP_BUY_PRODUCT / OP_BUY_ANIMAL / OP_SELL / OP_HIRE / OP_BUY_LAND
    int8_t  item;     // crop (BUY_SEED), product (BUY_PRODUCT/SELL), animal (BUY_ANIMAL)
    int16_t quantity; // n
};

__device__ __host__ __forceinline__ bool commit_unit(
    int op, int item, int price,
    Farm& farm, Private& priv, Market& market, int shed_capacity)
{
    if (op == OP_SELL) {
        if (priv.shed[item] <= 0) return false;
        priv.shed[item]--;
        farm.money += price;
        if (price > 1) market.inventory[item]++;
        return true;
    }
    if (op == OP_BUY_PRODUCT) {
        if (farm.money < price) return false;
        if (shed_total(priv) >= shed_capacity) return false;
        farm.money -= price;
        priv.shed[item]++;
        market.inventory[item]--;
        return true;
    }
    if (op == OP_BUY_SEED) {
        if (farm.money < price) return false;
        farm.money -= price;
        priv.seeds[item]++;
        return true;
    }
    if (op == OP_BUY_ANIMAL) {
        if (farm.money < price) return false;
        if (shed_total(priv) >= shed_capacity) return false;
        farm.money -= price;
        priv.shed[kProducts + item]++;  // animals live in shed indices 9..11
        return true;
    }
    return false;
}

__device__ __host__ __forceinline__ int market_order_price(int op, int item, const Market& market) {
    if (op == OP_SELL)        return market_price(item, static_cast<int>(market.inventory[item]));
    if (op == OP_BUY_PRODUCT) return market_price(item, static_cast<int>(market.inventory[item]) - 1);
    if (op == OP_BUY_SEED)    return CROPS[item].seed;
    if (op == OP_BUY_ANIMAL)  return ANIMALS[item].cost;
    return 0;
}

__device__ __host__ __forceinline__ bool market_order_valid(int op, int item) {
    if (op == OP_SELL)        return item >= 0 && item < kProducts;
    if (op == OP_BUY_PRODUCT) return item == P_WHEAT || item == P_FERTILIZER;
    if (op == OP_BUY_SEED)    return item >= 0 && item < kCrops;
    if (op == OP_BUY_ANIMAL)  return item >= 0 && item < kAnimals;
    return false;
}

__device__ __host__ __forceinline__ void process_market(
    GameState& s,
    const MarketOrder orders[NUM_PLAYERS][MAX_MARKET_ORDERS],
    const int8_t order_count[NUM_PLAYERS],
    int max_orders, int hire_mult, int shed_capacity)
{
    // Build per-player queues (capped at max_orders).
    int8_t qlen[NUM_PLAYERS];
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        qlen[p] = order_count[p] < max_orders ? order_count[p] : max_orders;
        if (qlen[p] < 0) qlen[p] = 0;
    }
    int max_len = qlen[0] > qlen[1] ? qlen[0] : qlen[1];

    for (int i = 0; i < max_len; ++i) {
        // Parse the i-th order of each player.
        struct OrderState { int8_t type; int8_t item; int16_t remaining; } ost[NUM_PLAYERS];
        for (int p = 0; p < NUM_PLAYERS; ++p) {
            ost[p].type = -1; ost[p].item = 0; ost[p].remaining = 0;
            if (i >= qlen[p]) continue;
            const MarketOrder& o = orders[p][i];
            int op = o.op; int item = o.item; int n = o.quantity;
            if (op == OP_HIRE) { ost[p].type = OP_HIRE; }
            else if (op == OP_BUY_LAND) { ost[p].type = OP_BUY_LAND; }
            else if (market_order_valid(op, item) && n > 0) {
                ost[p].type = op; ost[p].item = static_cast<int8_t>(item); ost[p].remaining = static_cast<int16_t>(n);
            }
        }
        // Atomic orders first, in player order.
        for (int p = 0; p < NUM_PLAYERS; ++p) {
            if (ost[p].type == OP_HIRE) { do_hire(s.farms[p], s.privates[p], hire_mult); ost[p].type = -1; }
            else if (ost[p].type == OP_BUY_LAND) { do_buy_land(s.farms[p]); ost[p].type = -1; }
        }
        // Per-unit lockstep for SELL / BUY_*.
        for (int guard = 0; guard < 100000; ++guard) {
            int quoted_op[NUM_PLAYERS], quoted_item[NUM_PLAYERS], quoted_price[NUM_PLAYERS];
            bool any = false;
            for (int p = 0; p < NUM_PLAYERS; ++p) {
                quoted_op[p] = -1; quoted_item[p] = 0; quoted_price[p] = 0;
                if (ost[p].type < 0 || ost[p].remaining <= 0) continue;
                if (!market_order_valid(ost[p].type, ost[p].item)) { ost[p].type = -1; continue; }
                quoted_op[p] = ost[p].type;
                quoted_item[p] = ost[p].item;
                quoted_price[p] = market_order_price(ost[p].type, ost[p].item, s.market);
                any = true;
            }
            if (!any) break;
            bool committed_any = false;
            for (int p = 0; p < NUM_PLAYERS; ++p) {
                if (quoted_op[p] < 0) continue;
                bool ok = commit_unit(quoted_op[p], quoted_item[p], quoted_price[p],
                                      s.farms[p], s.privates[p], s.market, shed_capacity);
                if (ok) { ost[p].remaining--; committed_any = true; }
                else ost[p].type = -1;
            }
            if (!committed_any) break;
        }
    }
    refresh_prices(s.market);
}

// ============================================================================
// TOWN CONSUMPTION
// ============================================================================
__device__ __host__ __forceinline__ void town_consume(GameState& s, int step, int shop_interval, int center_interval) {
    if (step % shop_interval == 0) {
        for (int k = 0; k < s.town.shop_count; ++k) {
            int shop = s.town.unlocked_shops[k];
            int cnt = SHOP_DEMAND_COUNT[shop];
            int multiplier = (cnt == 1) ? 2 : 1;
            for (int d = 0; d < cnt; ++d) {
                int item = SHOP_DEMAND[shop][d];
                s.market.inventory[item] -= multiplier;
            }
        }
    }
    if (step % center_interval == 0) {
        // Town center demands 1 of every non-fertilizer product.
        for (int item = 0; item < kProducts - 1; ++item)
            s.market.inventory[item] -= 1;
    }
    refresh_prices(s.market);
}

// ============================================================================
// PLANT DECAY / DAILY REFRESH
// ============================================================================
__device__ __host__ __forceinline__ void decay_plants(Farm& farm, int step) {
    for (int t = 0; t < BOARD_TILES; ++t) {
        Tile& tile = farm.tiles[t];
        if (tile.kind != T_PLANT) continue;
        int mls = tile.max_lifespan_step;
        if (mls < 0 || step < mls) continue;
        if ((step - mls) % 2 != 0) continue;
        tile.yield_units--;
        if (tile.yield_units <= 0) tile_init_weed(tile);
    }
}

__device__ __host__ __forceinline__ void daily_refresh_plants(Farm& farm, int current_day, int turns_per_day) {
    int next_day = current_day + 1;
    for (int t = 0; t < BOARD_TILES; ++t) {
        Tile& tile = farm.tiles[t];
        if (tile.kind != T_PLANT) continue;
        bool was_watered = tile.watered_today != 0;
        if (was_watered) tile.consecutive_unwatered = 0;
        else tile.consecutive_unwatered++;
        tile.watered_today = 0;
        if (tile.consecutive_unwatered >= 2) { tile_init_weed(tile); continue; }
        const CropParams& cd = CROPS[tile.crop];
        if (!cd.ongoing) continue;
        int days_since_first = next_day - tile.planted_day - cd.first_yield_day;
        if (days_since_first < 0) continue;
        if (cd.interval == 0) continue;
        if (days_since_first % cd.interval != 0) continue;
        int production_count = days_since_first / cd.interval + 1;
        if (production_count > cd.max_yield) continue;
        bool fertilized = was_watered && tile.fertilized_until_day >= current_day;
        int add = fertilized ? 2 : 1;
        int y = tile.yield_units + add;
        if (y > cd.max_yield) y = cd.max_yield;
        tile.yield_units = static_cast<int16_t>(y);
        if (production_count == cd.max_yield)
            tile.max_lifespan_step = static_cast<int16_t>((next_day + 1) * turns_per_day);
    }
}

__device__ __host__ __forceinline__ void daily_refresh_animals(Farm& farm, int day) {
    int next_day = day + 1;
    for (int t = 0; t < BOARD_TILES; ++t) {
        Tile& tile = farm.tiles[t];
        if (tile.animal == 0) continue;
        if (tile.fed_today) tile.consecutive_unfed = 0;
        else tile.consecutive_unfed++;
        if (tile.consecutive_unfed >= 2) {
            // Animal escapes; structure remains.
            TileKind k = (TileKind)((tile.animal == 1) ? T_COOP : T_PASTURE);
            tile = Tile{};
            tile.kind = k;
            continue;
        }
        int animal = tile.animal - 1;
        const AnimalParams& a = ANIMALS[animal];
        int days_since_first = next_day - tile.planted_day - a.first_yield_day;
        if (days_since_first >= 0 && a.interval > 0 && days_since_first % a.interval == 0) {
            int base = 1;
            int bonus = tile.fed_today ? tile.pending_care_bonus : 0;
            int y = tile.yield_units + base + bonus;
            if (y > a.max_held) y = a.max_held;
            tile.yield_units = static_cast<int16_t>(y);
            tile.pending_care_bonus = 0;
        }
        if (tile.cared_today && tile.fed_today) tile.pending_care_bonus++;
        tile.fertilizer_available = 1;
        tile.fed_today = 0;
        tile.cared_today = 0;
    }
}

// ============================================================================
// END OF DAY
// ============================================================================
__device__ __host__ __forceinline__ void drop_inventories_to_shed(Private& priv, int capacity) {
    for (int u = 0; u <= MAX_HANDS; ++u) {
        int16_t* inv = priv.inventory[u];
        for (int i = 0; i < kItemCount; ++i) {
            int n = inv[i];
            if (n <= 0) { inv[i] = 0; continue; }
            int room = capacity - shed_total(priv);
            if (room < 0) room = 0;
            int take = n < room ? n : room;
            if (take > 0) priv.shed[i] = static_cast<int16_t>(priv.shed[i] + take);
            inv[i] = 0;
        }
    }
}

__device__ __host__ __forceinline__ void spawn_weeds(Farm& farm, float weed_chance, uint32_t& rng) {
    for (int t = 0; t < BOARD_TILES; ++t) {
        if (farm.tiles[t].kind != T_EMPTY) continue;
        // Patched RNG: uniform-ish draw from xorshift, top 24 bits.
        float draw = static_cast<float>(rng_next(rng) >> 8) / 16777216.0f;
        if (draw < weed_chance) tile_init_weed(farm.tiles[t]);
    }
}

__device__ __host__ __forceinline__ void end_of_day(
    GameState& s, int day, int turns_per_day, float weed_chance,
    int shed_capacity, int shop_interval, uint32_t& rng)
{
    int half = BOARD_SIZE / 2;
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        Farm& farm = s.farms[p];
        Private& priv = s.privates[p];
        daily_refresh_plants(farm, day, turns_per_day);
        daily_refresh_animals(farm, day);
        spawn_weeds(farm, weed_chance, rng);
        drop_inventories_to_shed(priv, shed_capacity);
        farm.farmer_x = static_cast<int16_t>(half - 1);
        farm.farmer_y = static_cast<int16_t>(half - 1);
        farm.hand_count = 0;
        farm.hires_today = 0;
        for (int h = 0; h < MAX_HANDS; ++h) { farm.hand_x[h] = 0; farm.hand_y[h] = 0; }
        for (int u = 0; u <= MAX_HANDS; ++u)
            for (int i = 0; i < kItemCount; ++i) priv.inventory[u][i] = 0;
    }
    int next_day = day + 1;
    if (next_day > 0 && next_day % shop_interval == 0) {
        if (s.town.shop_count < MAX_SHOP_INSTANCES) {
            // Patched RNG: draw a shop index in [0, kShops) from xorshift.
            int shop = static_cast<int>(rng_next(rng) % kShops);
            s.town.unlocked_shops[s.town.shop_count++] = static_cast<uint8_t>(shop);
        }
    }
}

// ============================================================================
// INTERPRETER (one full turn). Mirrors interpreter().
// ============================================================================
struct UnitAction {
    int8_t op;
    int8_t item;
    int16_t quantity;
};

struct TurnAction {
    UnitAction  farmer;
    UnitAction  hands[MAX_HANDS];
    int8_t      hand_count;
    MarketOrder market[MAX_MARKET_ORDERS];
    int8_t      market_count;
};

__device__ __host__ __forceinline__ void interpreter(
    GameState& s,
    const TurnAction actions[NUM_PLAYERS],
    uint32_t episode_seed,
    int turns_per_day, int board_size, int shed_capacity,
    int max_orders, int hire_mult)
{
    int step = static_cast<int>(s.step);
    int day = step / turns_per_day;

    for (int p = 0; p < NUM_PLAYERS; ++p) {
        Farm& farm = s.farms[p];
        Private& priv = s.privates[p];
        const TurnAction& a = actions[p];

        // Atomic PLANT validation: if total PLANT requests for a crop exceed
        // available seeds, drop ALL PLANT requests for that crop this turn.
        // The seed count is snapshotted once, before any planting, so later
        // units in the same turn cannot invalidate earlier-validated plants.
        int plant_demand[kCrops] = {0,0,0,0,0};
        int seeds_snapshot[kCrops];
        for (int c = 0; c < kCrops; ++c) seeds_snapshot[c] = priv.seeds[c];
        int unit_count = 1 + (a.hand_count > 0 ? a.hand_count : 0);
        for (int u = 0; u < unit_count; ++u) {
            const UnitAction& ua = (u == 0) ? a.farmer : a.hands[u - 1];
            if (ua.op == OP_PLANT && ua.item >= 0 && ua.item < kCrops) plant_demand[ua.item]++;
        }
        auto allowed_op = [&](const UnitAction& ua) -> UnitAction {
            UnitAction r = ua;
            if (ua.op == OP_PLANT && ua.item >= 0 && ua.item < kCrops && plant_demand[ua.item] > seeds_snapshot[ua.item])
                r.op = OP_PASS;
            return r;
        };

        UnitAction fa = allowed_op(a.farmer);
        apply_unit_action(farm, priv, 0, fa.op, fa.item, fa.quantity, day, turns_per_day, shed_capacity);
        for (int h = 0; h < a.hand_count && h < MAX_HANDS; ++h) {
            UnitAction ha = allowed_op(a.hands[h]);
            apply_unit_action(farm, priv, h + 1, ha.op, ha.item, ha.quantity, day, turns_per_day, shed_capacity);
        }
    }

    MarketOrder orders[NUM_PLAYERS][MAX_MARKET_ORDERS];
    int8_t order_count[NUM_PLAYERS];
    for (int p = 0; p < NUM_PLAYERS; ++p) {
        order_count[p] = actions[p].market_count;
        for (int m = 0; m < MAX_MARKET_ORDERS; ++m)
            orders[p][m] = actions[p].market[m];
    }
    process_market(s, orders, order_count, max_orders, hire_mult, shed_capacity);
    town_consume(s, step, TOWN_SHOP_SELL_INTERVAL, TOWN_CENTER_SELL_INTERVAL);

    for (int p = 0; p < NUM_PLAYERS; ++p) decay_plants(s.farms[p], step);

    if ((step + 1) % turns_per_day == 0) {
        uint32_t rng = rng_seed(episode_seed, day);
        end_of_day(s, day, turns_per_day, WEED_SPAWN_CHANCE, shed_capacity,
                   TOWN_SHOP_UNLOCK_INTERVAL, rng);
    }

    int next_step = step + 1;
    s.day = static_cast<int16_t>(next_step / turns_per_day);
    s.hour = static_cast<int16_t>(next_step % turns_per_day);
    s.step = static_cast<uint32_t>(next_step);
}

} // namespace kgr
