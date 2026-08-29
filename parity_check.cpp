// parity_check.cpp - validate kaggriculture_engine.cuh against a golden binary
// dump produced by dump_replay_binary.py. Compiles with g++ (host) or nvcc.
#include "kaggriculture_engine.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace kgr;

static bool rd(void* p, size_t n, FILE* f) { return fread(p, 1, n, f) == n; }

static uint8_t u8(FILE* f){ uint8_t v; rd(&v,1,f); return v; }
static int16_t i16(FILE* f){ int16_t v; rd(&v,2,f); return v; }
static int32_t i32(FILE* f){ int32_t v; rd(&v,4,f); return v; }
static uint32_t u32(FILE* f){ uint32_t v; rd(&v,4,f); return v; }
static int64_t i64(FILE* f){ int64_t v; rd(&v,8,f); return v; }
static float f32(FILE* f){ float v; rd(&v,4,f); return v; }

static int g_mismatches = 0;
static bool g_first = true;

static void check(int step, int player, const char* what, int64_t got, int64_t want) {
    if (got != want) {
        g_mismatches++;
        if (g_first) {
            g_first = false;
            std::fprintf(stderr, "FIRST MISMATCH step=%d player=%d field=%s got=%lld want=%lld\n",
                         step, player, what, (long long)got, (long long)want);
        }
    }
}

// Read a full state record and compare it to `s`. Layout mirrors dump_replay_binary.py.
static void compare_state(FILE* f, const GameState& s, int step) {
    int day = i16(f), hour = i16(f);
    (void)i32(f);  // player
    int32_t wstep = (int32_t)u32(f);
    check(step, -1, "day", day, s.day);
    check(step, -1, "hour", hour, s.hour);
    check(step, -1, "step", wstep, (int32_t)s.step);

    for (int p = 0; p < NUM_PLAYERS; ++p) {
        float money = f32(f);
        int fx = i16(f), fy = i16(f);
        int hand_count = u8(f), hires = u8(f);
        int quad = u8(f);
        check(step, p, "money", (int64_t)money, (int64_t)s.farms[p].money);
        check(step, p, "farmer_x", fx, s.farms[p].farmer_x);
        check(step, p, "farmer_y", fy, s.farms[p].farmer_y);
        check(step, p, "hand_count", hand_count, s.farms[p].hand_count);
        check(step, p, "hires_today", hires, s.farms[p].hires_today);
        check(step, p, "quadrants", quad, s.farms[p].unlocked_quadrants);
        for (int h = 0; h < MAX_HANDS; ++h) {
            int16_t hx = i16(f);
            check(step, p, "hand_x", hx, s.farms[p].hand_x[h]);
        }
        for (int h = 0; h < MAX_HANDS; ++h) {
            int16_t hy = i16(f);
            check(step, p, "hand_y", hy, s.farms[p].hand_y[h]);
        }
        for (int t = 0; t < BOARD_TILES; ++t) {
            int kind = u8(f), crop = u8(f), animal = u8(f);
            int watered = u8(f), fed = u8(f), cared = u8(f), fert = u8(f);
            int pd = i16(f), yu = i16(f), mls = i16(f), fu = i16(f);
            int cu = i16(f), cf = i16(f), cb = i16(f);
            const Tile& tile = s.farms[p].tiles[t];
            char tag[64];
            std::snprintf(tag, sizeof(tag), "tile[%d].kind", t);
            check(step, p, tag, kind, tile.kind);
            if (kind != tile.kind) continue;  // don't spam sub-field diffs on a kind mismatch
            std::snprintf(tag, sizeof(tag), "tile[%d].crop", t);
            check(step, p, tag, crop, tile.crop);
            std::snprintf(tag, sizeof(tag), "tile[%d].animal", t);
            check(step, p, tag, animal, tile.animal);
            std::snprintf(tag, sizeof(tag), "tile[%d].watered", t);
            check(step, p, tag, watered, tile.watered_today);
            std::snprintf(tag, sizeof(tag), "tile[%d].fed", t);
            check(step, p, tag, fed, tile.fed_today);
            std::snprintf(tag, sizeof(tag), "tile[%d].cared", t);
            check(step, p, tag, cared, tile.cared_today);
            std::snprintf(tag, sizeof(tag), "tile[%d].fert", t);
            check(step, p, tag, fert, tile.fertilizer_available);
            std::snprintf(tag, sizeof(tag), "tile[%d].planted_day", t);
            check(step, p, tag, pd, tile.planted_day);
            std::snprintf(tag, sizeof(tag), "tile[%d].yield", t);
            check(step, p, tag, yu, tile.yield_units);
            std::snprintf(tag, sizeof(tag), "tile[%d].max_lifespan", t);
            check(step, p, tag, mls, tile.max_lifespan_step);
            std::snprintf(tag, sizeof(tag), "tile[%d].fertilized_until", t);
            check(step, p, tag, fu, tile.fertilized_until_day);
            std::snprintf(tag, sizeof(tag), "tile[%d].consec_unwatered", t);
            check(step, p, tag, cu, tile.consecutive_unwatered);
            std::snprintf(tag, sizeof(tag), "tile[%d].consec_unfed", t);
            check(step, p, tag, cf, tile.consecutive_unfed);
            std::snprintf(tag, sizeof(tag), "tile[%d].pending_care", t);
            check(step, p, tag, cb, tile.pending_care_bonus);
        }
    }
    for (int p = 0; p < kProducts; ++p) {
        int64_t inv = i64(f);
        check(step, -1, "market_inventory", inv, s.market.inventory[p]);
    }
    for (int p = 0; p < kProducts; ++p) {
        int16_t pr = i16(f);
        check(step, -1, "market_price", pr, s.market.prices[p]);
    }
    int shop_count = u8(f);
    uint8_t shops[8];
    for (int i = 0; i < 8; ++i) shops[i] = u8(f);
    check(step, -1, "shop_count", shop_count, s.town.shop_count);
    for (int i = 0; i < 8; ++i) check(step, -1, "shop_id", shops[i], s.town.unlocked_shops[i]);

    for (int p = 0; p < NUM_PLAYERS; ++p) {
        const Private& pr = s.privates[p];
        char tag[64];
        for (int i = 0; i < kItemCount; ++i) { std::snprintf(tag, sizeof(tag), "shed[%d]", i); check(step, p, tag, i16(f), pr.shed[i]); }
        for (int i = 0; i < kCrops; ++i) { std::snprintf(tag, sizeof(tag), "seeds[%d]", i); check(step, p, tag, i16(f), pr.seeds[i]); }
        for (int u = 0; u <= MAX_HANDS; ++u)
            for (int i = 0; i < kItemCount; ++i) { std::snprintf(tag, sizeof(tag), "inventory[u%d][i%d]", u, i); check(step, p, tag, i16(f), pr.inventory[u][i]); }
    }
}

static void read_actions(FILE* f, TurnAction& a) {
    a.farmer.op = (int8_t)u8(f);
    a.farmer.item = (int8_t)u8(f);
    a.farmer.quantity = i16(f);
    a.hand_count = (int8_t)u8(f);
    for (int h = 0; h < MAX_HANDS; ++h) {
        a.hands[h].op = (int8_t)u8(f);
        a.hands[h].item = (int8_t)u8(f);
        a.hands[h].quantity = i16(f);
    }
    a.market_count = (int8_t)u8(f);
    for (int m = 0; m < MAX_MARKET_ORDERS; ++m) {
        a.market[m].op = (int8_t)u8(f);
        a.market[m].item = (int8_t)u8(f);
        a.market[m].quantity = i16(f);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: parity_check <golden.bin>\n"); return 2; }
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }

    uint32_t magic = u32(f);
    uint32_t n = u32(f);
    int32_t seed = i32(f);
    if (magic != 0x4B475230) { std::fprintf(stderr, "bad magic\n"); return 2; }
    std::fprintf(stderr, "steps=%u seed=%d\n", n, seed);

    GameState s;
    init_state(s, STARTING_MONEY);

    // Step 0: read initial state and init the engine from it.
    // (compare_state reads into checks against `s`, but we need to first LOAD it.)
    // Read and discard: we reconstruct via a load pass below instead.
    // Simpler: load the full state into `s` directly.
    std::fseek(f, 12, SEEK_SET);

    // Load initial state (step 0) into `s`.
    {
        int day = i16(f), hour = i16(f);
        (void)i32(f); (void)u32(f);
        s.day = day; s.hour = hour;
        for (int p = 0; p < NUM_PLAYERS; ++p) {
            s.farms[p].money = f32(f);
            s.farms[p].farmer_x = i16(f);
            s.farms[p].farmer_y = i16(f);
            s.farms[p].hand_count = u8(f);
            s.farms[p].hires_today = u8(f);
            s.farms[p].unlocked_quadrants = u8(f);
            for (int h = 0; h < MAX_HANDS; ++h) s.farms[p].hand_x[h] = i16(f);
            for (int h = 0; h < MAX_HANDS; ++h) s.farms[p].hand_y[h] = i16(f);
            for (int t = 0; t < BOARD_TILES; ++t) {
                Tile& tile = s.farms[p].tiles[t];
                tile.kind = u8(f); tile.crop = u8(f); tile.animal = u8(f);
                tile.watered_today = u8(f); tile.fed_today = u8(f); tile.cared_today = u8(f); tile.fertilizer_available = u8(f);
                tile.planted_day = i16(f); tile.yield_units = i16(f); tile.max_lifespan_step = i16(f);
                tile.fertilized_until_day = i16(f); tile.consecutive_unwatered = i16(f);
                tile.consecutive_unfed = i16(f); tile.pending_care_bonus = i16(f);
            }
        }
        for (int p = 0; p < kProducts; ++p) s.market.inventory[p] = i64(f);
        for (int p = 0; p < kProducts; ++p) s.market.prices[p] = i16(f);
        s.town.shop_count = u8(f);
        for (int i = 0; i < 8; ++i) s.town.unlocked_shops[i] = u8(f);
        for (int p = 0; p < NUM_PLAYERS; ++p) {
            for (int i = 0; i < kItemCount; ++i) s.privates[p].shed[i] = i16(f);
            for (int i = 0; i < kCrops; ++i) s.privates[p].seeds[i] = i16(f);
            for (int u = 0; u <= MAX_HANDS; ++u)
                for (int i = 0; i < kItemCount; ++i) s.privates[p].inventory[u][i] = i16(f);
            // Rebuild insertion order from the (sparse) inventory, preserving
            // the order in which non-zero items appear left-to-right.
            for (int u = 0; u <= MAX_HANDS; ++u) {
                s.privates[p].inv_order_len[u] = 0;
                for (int i = 0; i < kItemCount; ++i)
                    if (s.privates[p].inventory[u][i] != 0) {
                        s.privates[p].inv_order[u][s.privates[p].inv_order_len[u]] = (uint8_t)i;
                        s.privates[p].inv_order_len[u]++;
                    }
            }
        }
        s.step = 0; s.active_player = 0;
    }

    for (uint32_t t = 0; t < n - 1; ++t) {
        TurnAction actions[NUM_PLAYERS] = {};
        read_actions(f, actions[0]);
        read_actions(f, actions[1]);

        interpreter(s, actions, (uint32_t)seed, TURNS_PER_DAY, BOARD_SIZE, SHED_CAPACITY,
                    MAX_MARKET_ORDERS, FARM_HAND_COST_MULT);

        // The next state record should now match s.
        compare_state(f, s, (int)(t + 1));
    }

    std::fclose(f);
    if (g_mismatches == 0) {
        std::printf("PARITY PASS steps=%u\n", n);
        return 0;
    }
    std::printf("PARITY FAIL mismatches=%d\n", g_mismatches);
    return 1;
}
