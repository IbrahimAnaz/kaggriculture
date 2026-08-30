/*
 * KAGGRICULTURE UNIFIED CUDA PORT v1.0
 * =====================================
 * Single-file build: 
 *   nvcc -O3 -arch=sm_86 -std=c++17 kaggriculture_unified.cu -lcurand -o kaggriculture_unified
 *
 * Multi-arch build (for Colab/Kaggle):
 *   nvcc -O3 -gencode arch=compute_75,code=sm_75 \
 *          -gencode arch=compute_80,code=sm_80 \
 *          -gencode arch=compute_86,code=sm_86 \
 *          -std=c++17 kaggriculture_unified.cu -lcurand -o kaggriculture_unified
 *
 * Usage:
 *   ./kaggriculture_unified encode replay.json replay.kagrbin
 *   ./kaggriculture_unified validate replay.kagrbin
 *   ./kaggriculture_unified train 100 720 replay.kagrbin
 *   ./kaggriculture_unified smoke
 */
// Catfish actions affect the market state (shift prices, consume inventory)
// This creates the adversarial game: farmers adapt to catfish, catfish adapts to farmers
// Place this right above your kernel definitions


// 2. Now include your actual dependencies or project headers below this line
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <map>
#include <vector>
#include <algorithm>

// ============================================================================
// CUDA ERROR CHECKING
// ============================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA FATAL at %s:%d: %s | %s\n", \
                     __FILE__, __LINE__, cudaGetErrorString(_err), #call); \
        std::exit(1); \
    } \
} while(0)

#define CUDA_CHECK_LAST(msg) do { \
    cudaError_t _err = cudaGetLastError(); \
    if (_err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA KERNEL FATAL at %s:%d: %s | %s\n", \
                     __FILE__, __LINE__, cudaGetErrorString(_err), msg); \
        std::exit(1); \
    } \
} while(0)

// ============================================================================
// PROTOCOL CONSTANTS
// ============================================================================
namespace proto {
constexpr int kBoardSize      = 10;
constexpr int kBoardTiles     = 100;
constexpr int kMaskWords      = 4;   // ceil(100/32)
constexpr int kPlayers        = 2;
constexpr int kSteps          = 720;
constexpr int kDays           = 30;
constexpr int kHoursPerDay    = 24;
constexpr int kProducts       = 9;
constexpr int kCrops          = 5;
constexpr int kAnimals        = 3;
constexpr int kShops          = 8;
constexpr int kMaxHands       = 16;
constexpr int kMarketOrders   = 10;
constexpr int kMaxActionTokens = 8;
constexpr char kReplayName[]   = "kaggriculture";
constexpr char kReplayModule[] = "1.32.7";
constexpr int kStartingMoney  = 3000;

// Official actions (must match replay JSON exactly)
enum Action : uint8_t {
    Pass = 0, North, South, East, West,
    Pickup, Plant, Water, Harvest, Fertilize,
    BuildCoop, BuildPasture, Dig, Place,
    Feed, CollectFertilizer, Care,
    BuySeed, BuyProduct, BuyAnimal,
    Sell, Hire, BuyLand,
    Drop,
    UnknownAction = 255
};

enum TileKind : uint8_t { Empty = 0, Locked, Weed, TilePlant, Coop, Pasture };
enum Product : uint8_t { Wheat = 0, Carrot, Tomato, Strawberry, Melon, Egg, Milk, Wool, Fertilizer };
enum Animal : uint8_t { NoAnimal = 0, Goose, Cow, Sheep };

static const char* const kProductNames[kProducts] = {
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER"
};
static const char* const kShopNames[kShops] = {
    "BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
    "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET"
};

// Quadrant bitmasks
__device__ constexpr uint8_t kQuadNW = 1, kQuadNE = 2, kQuadSW = 4, kQuadSE = 8;

// Seed costs (from official game)
__device__ __host__ constexpr int kSeedCost[kCrops] = {10, 20, 50, 80, 100};

// Shop unlock costs
__device__ __host__ constexpr int kShopCost[kShops] = {25, 40, 55, 70, 85, 100, 125, 150};

// Crop growth parameters
__device__ __host__ constexpr int kCropGrowthDays[kCrops] = {3, 4, 5, 6, 7};  // days to mature
__device__ __host__ constexpr int kCropMaxYield[kCrops]   = {2, 3, 4, 5, 6};   // max yield at harvest
__device__ __host__ constexpr int kUnwateredDeathDays    = 2;  // days without water before death

// Base market prices
 __constant__ float kBasePrice[kProducts] = {
    25.0f, 35.0f, 60.0f, 120.0f, 250.0f,  // crops
    50.0f, 160.0f, 200.0f, 100.0f          // animal products + fertilizer
};
 __constant__ float kThroughput[kProducts] = {
    400.0f, 450.0f, 200.0f, 100.0f, 300.0f,
    332.0f, 122.0f, 105.0f, 200.0f
};
} // namespace proto

// ============================================================================
// BINARY REPLAY FORMAT
// ============================================================================
#pragma pack(push, 1)

struct ReplayTile {
    uint8_t  kind;        // TileKind
    uint8_t  item;        // Product or Animal
    uint8_t  flags;       // bit0=watered, bit1=fed, bit2=cared, bit3=fertilizer
    int16_t  planted_day;
    int16_t  yield;
    int16_t  lifetime;    // max_lifespan or consecutive_unfed
    int16_t  care;        // fertilized_until or pending_care_bonus
    int16_t  extra;       // consecutive_unwatered or fertilizer_available
};

struct ReplayPlayer {
    float    money;
    int16_t  x, y;
    uint8_t  unlocked_quadrants;
    uint8_t  hand_count;
    uint16_t hires_today;
};

struct EncodedAction {
    uint8_t  operation;
    uint8_t  item;
    uint8_t  animal;
    uint8_t  token_count;
    int16_t  quantity;
    int16_t  x, y;
};

struct FlatHeader {
    char     magic[8];      // "KAGRST01"
    uint16_t version;
    uint16_t board;
    uint16_t players;
    uint16_t max_hands;
    uint32_t frames;
};

struct FlatFrame {
    uint32_t step;
    uint16_t day, hour;
    uint8_t  active_player;
};

#pragma pack(pop)

// ============================================================================
// TRAINING HYPERPARAMETERS
// ============================================================================
__device__ constexpr int kPopulation    = 4096;
__device__ constexpr int kGenerations   = 1000;
__device__ constexpr int kFeatures      = 32;   // Input features
__device__ constexpr int kHidden        = 64;   // Hidden layer units
__device__ constexpr int kActions       = 37;   // Internal action space
__device__ constexpr int kEliteCount    = 128;  // Top agents preserved

// Internal action IDs (dense, for neural network output)
enum InternalAction : int {
    IA_Pass = 0,
    IA_BuyWheatSeed, IA_BuyCarrotSeed, IA_BuyTomatoSeed,
    IA_BuyStrawberrySeed, IA_BuyMelonSeed,
    IA_Unused6, IA_Unused7, IA_Unused8, IA_Unused9, IA_Unused10,
    IA_SellWheat, IA_SellCarrot, IA_SellTomato, IA_SellStrawberry, IA_SellMelon,
    IA_Unused16, IA_Unused17, IA_Unused18,
    IA_North, IA_South, IA_East, IA_West,
    IA_PlantWheat, IA_PlantCarrot, IA_PlantTomato, IA_PlantStrawberry, IA_PlantMelon,
    IA_Water, IA_Harvest,
    IA_Bakery, IA_PizzaShop, IA_BrunchSpot, IA_YarnStore,
    IA_IceCreamShop, IA_PetCafe, IA_SmoothieShop, IA_FarmersMarket
};

// Action mapping: internal -> official
__device__ __host__ inline int internal_to_official(int ia) {
    switch (ia) {
        case IA_Pass: return proto::Pass;
        case IA_North: return proto::North;
        case IA_South: return proto::South;
        case IA_East: return proto::East;
        case IA_West: return proto::West;
        case IA_PlantWheat: case IA_PlantCarrot: case IA_PlantTomato:
        case IA_PlantStrawberry: case IA_PlantMelon: return proto::Plant;
        case IA_Water: return proto::Water;
        case IA_Harvest: return proto::Harvest;
        case IA_BuyWheatSeed: case IA_BuyCarrotSeed: case IA_BuyTomatoSeed:
        case IA_BuyStrawberrySeed: case IA_BuyMelonSeed: return proto::BuySeed;
        case IA_SellWheat: case IA_SellCarrot: case IA_SellTomato:
        case IA_SellStrawberry: case IA_SellMelon: return proto::Sell;
        case IA_Bakery: case IA_PizzaShop: case IA_BrunchSpot: case IA_YarnStore:
        case IA_IceCreamShop: case IA_PetCafe: case IA_SmoothieShop: case IA_FarmersMarket:
            return proto::BuyProduct;  // Shops are buy_product in official
        default: return -1;
    }
}

// Official -> internal (for replay matching)
__device__ __host__ inline int official_to_internal(proto::Action op, uint8_t item) {
    switch (op) {
        case proto::Pass: return IA_Pass;
        case proto::North: return IA_North;
        case proto::South: return IA_South;
        case proto::East: return IA_East;
        case proto::West: return IA_West;
        case proto::Plant:
            if (item < proto::kCrops) return IA_PlantWheat + item;
            return IA_Pass;
        case proto::Water: return IA_Water;
        case proto::Harvest: return IA_Harvest;
        case proto::BuySeed:
            if (item < proto::kCrops) return IA_BuyWheatSeed + item;
            return IA_Pass;
        case proto::Sell:
            if (item < proto::kCrops) return IA_SellWheat + item;
            return IA_Pass;
        case proto::BuyProduct:
            if (item < proto::kShops) return IA_Bakery + item;
            return IA_Pass;
        default: return -1;  // Unsupported for now
    }
}

// ============================================================================
// BITPLANE STATE (GPU-optimized)
// ============================================================================
struct HotFarmState {
    uint32_t occupied[proto::kMaskWords];
    uint32_t plant[proto::kMaskWords];
    uint32_t structure[proto::kMaskWords];
    uint32_t watered_today[proto::kMaskWords];
    uint32_t fed_today[proto::kMaskWords];
    uint32_t cared_today[proto::kMaskWords];
    int16_t  yield[proto::kBoardTiles];
};

struct ColdFarmState {
    uint32_t locked[proto::kMaskWords];
    uint32_t weed[proto::kMaskWords];
    uint32_t crop[proto::kCrops][proto::kMaskWords];
    uint32_t animal[proto::kAnimals][proto::kMaskWords];
    int16_t  planted_day[proto::kBoardTiles];
    int16_t  fertilized_until_day[proto::kBoardTiles];
    int8_t   consecutive_unwatered[proto::kBoardTiles];
    int8_t   consecutive_unfed[proto::kBoardTiles];
    int8_t   pending_care_bonus[proto::kBoardTiles];
};

struct FarmBitplanes {
    HotFarmState  hot;
    ColdFarmState cold;
};

struct HotEconomyState {
    int32_t  market_inventory[proto::kProducts];
    int16_t  market_prices[proto::kProducts];
    int16_t  shed[proto::kProducts];
    int16_t  seeds[proto::kCrops];
    int16_t  farmer_x, farmer_y;
    int16_t  day;
    int8_t   hour;
    int8_t   hands_today;
    float    money;
    float    revenue;
    float    breadcrumb;
};

struct ColdEconomyState {
    uint8_t  unlocked_shop_counts[proto::kShops];
    uint8_t  unlocked_quadrants;
    uint8_t  hires_today;
    uint8_t  reserved;
    int32_t  money_cents;
};

struct SimulatorState {
    FarmBitplanes    farm;
    HotEconomyState  hot_econ;
    ColdEconomyState cold_econ;
};

// ============================================================================
// WEIGHT STRUCTURE (SoA layout for coalesced access)
// ============================================================================
struct AgentWeights {
    // Layout: [feature * kHidden + hidden_unit] per agent, stored as SoA:
    // d_input[feature * kHidden * kPopulation + hidden_unit * kPopulation + agent]
    // But for simplicity, we'll use AoS per-agent and let compiler handle it
    // For true SoA, we'd use separate arrays. Let's use AoS for now (4096 agents * 9KB = 36MB)
    float input[kFeatures * kHidden];
    float output[kHidden * kActions];
};

// ============================================================================
// BITPLANE PRIMITIVES
// ============================================================================
__device__ __forceinline__ void bp_set(uint32_t* plane, int tile) {
    plane[tile >> 5] |= (1u << (tile & 31));
}

__device__ __forceinline__ void bp_clear(uint32_t* plane, int tile) {
    plane[tile >> 5] &= ~(1u << (tile & 31));
}
__device__ __forceinline__ bool bp_test(const uint32_t* plane, int tile) {
    return (plane[tile >> 5] & (1u << (tile & 31))) != 0;
}

__device__ __forceinline__ int bp_popcount(const uint32_t* plane, int words) {
    int count = 0;
    for (int i = 0; i < words; ++i) count += __popc(plane[i]);
    return count;
}

// ============================================================================
// TILE OPERATIONS
// ============================================================================
__device__ void tile_clear(SimulatorState* state, int tile) {
    FarmBitplanes* farm = &state->farm;
    bp_clear(farm->hot.occupied, tile);
    bp_clear(farm->cold.locked, tile);
    bp_clear(farm->cold.weed, tile);
    bp_clear(farm->hot.plant, tile);
    bp_clear(farm->hot.structure, tile);
    bp_clear(farm->hot.watered_today, tile);
    bp_clear(farm->hot.fed_today, tile);
    bp_clear(farm->hot.cared_today, tile);
    for (int c = 0; c < proto::kCrops; ++c) bp_clear(farm->cold.crop[c], tile);
    for (int a = 0; a < proto::kAnimals; ++a) bp_clear(farm->cold.animal[a], tile);
    farm->cold.planted_day[tile] = -1;
    farm->hot.yield[tile] = 0;
    farm->cold.fertilized_until_day[tile] = -1;
    farm->cold.consecutive_unwatered[tile] = 0;
    farm->cold.consecutive_unfed[tile] = 0;
    farm->cold.pending_care_bonus[tile] = 0;
}

__device__ void tile_plant(SimulatorState* state, int tile, int crop_type, int day) {
    tile_clear(state, tile);
    FarmBitplanes* farm = &state->farm;
    bp_set(farm->hot.occupied, tile);
    bp_set(farm->hot.plant, tile);
    bp_set(farm->cold.crop[crop_type], tile);
    farm->cold.planted_day[tile] = static_cast<int16_t>(day);
    farm->hot.yield[tile] = 1;
    farm->cold.consecutive_unwatered[tile] = 0;
}

__device__ void tile_water(SimulatorState* state, int tile) {
    bp_set(state->farm.hot.watered_today, tile);
    state->farm.cold.consecutive_unwatered[tile] = 0;
}

__device__ void tile_harvest(SimulatorState* state, int tile, int* out_crop, int* out_yield) {
    FarmBitplanes* farm = &state->farm;
    *out_yield = farm->hot.yield[tile];
    *out_crop = -1;
    for (int c = 0; c < proto::kCrops; ++c) {
        if (bp_test(farm->cold.crop[c], tile)) { *out_crop = c; break; }
    }
    tile_clear(state, tile);
}

// ============================================================================
// STATE INITIALIZATION
// ============================================================================
__device__ void state_reset(SimulatorState* state) {
    FarmBitplanes* farm = &state->farm;
    for (int i = 0; i < proto::kMaskWords; ++i) {
        farm->hot.occupied[i] = 0;
        farm->hot.plant[i] = 0;
        farm->hot.structure[i] = 0;
        farm->hot.watered_today[i] = 0;
        farm->hot.fed_today[i] = 0;
        farm->hot.cared_today[i] = 0;
        farm->cold.locked[i] = 0;
        farm->cold.weed[i] = 0;
        for (int c = 0; c < proto::kCrops; ++c) farm->cold.crop[c][i] = 0;
        for (int a = 0; a < proto::kAnimals; ++a) farm->cold.animal[a][i] = 0;
    }
    for (int t = 0; t < proto::kBoardTiles; ++t) {
        farm->hot.yield[t] = 0;
        farm->cold.planted_day[t] = -1;
        farm->cold.fertilized_until_day[t] = -1;
        farm->cold.consecutive_unwatered[t] = 0;
        farm->cold.consecutive_unfed[t] = 0;
        farm->cold.pending_care_bonus[t] = 0;
    }
    HotEconomyState* econ = &state->hot_econ;
    for (int p = 0; p < proto::kProducts; ++p) {
        econ->market_inventory[p] = 10000;
        econ->shed[p] = 0;
    }
    for (int c = 0; c < proto::kCrops; ++c) econ->seeds[c] = 0;
    econ->farmer_x = 4;
    econ->farmer_y = 4;
    econ->day = 0;
    econ->hour = 0;
    econ->hands_today = 0;
    econ->money = static_cast<float>(proto::kStartingMoney);
    econ->revenue = 0.0f;
    econ->breadcrumb = 0.0f;

    ColdEconomyState* cold = &state->cold_econ;
    cold->unlocked_quadrants = 0;
    cold->hires_today = 0;
    cold->reserved = 0;
    cold->money_cents = proto::kStartingMoney * 100;
    for (int s = 0; s < proto::kShops; ++s) cold->unlocked_shop_counts[s] = (s == 0) ? 1 : 0;
}

// ============================================================================
// MARKET PRICE
// ============================================================================
__device__ float compute_market_price(int product, int inventory) {
    float displacement = (10000.0f - static_cast<float>(inventory)) / proto::kThroughput[product];
    return fmaxf(1.0f, roundf(proto::kBasePrice[product] * (1.0f + 0.2f * displacement)));
}

__device__ void update_all_prices(SimulatorState* state) {
    for (int p = 0; p < proto::kProducts; ++p) {
        state->hot_econ.market_prices[p] = static_cast<int16_t>(
            compute_market_price(p, state->hot_econ.market_inventory[p]));
    }
}

// ============================================================================
// DAY TRANSITION (biological timers)
// ============================================================================
__device__ void end_of_day(SimulatorState* state) {
    FarmBitplanes* farm = &state->farm;
    HotEconomyState* econ = &state->hot_econ;

    for (int tile = 0; tile < proto::kBoardTiles; ++tile) {
        if (!bp_test(farm->hot.occupied, tile)) continue;

        if (bp_test(farm->hot.plant, tile)) {
            // Plant logic
            if (!bp_test(farm->hot.watered_today, tile)) {
                // Not watered today
                farm->cold.consecutive_unwatered[tile]++;
                if (farm->cold.consecutive_unwatered[tile] >= proto::kUnwateredDeathDays) {
                    // Plant dies -> becomes weed
                    tile_clear(state, tile);
                    bp_set(farm->cold.weed, tile);
                    bp_set(farm->hot.occupied, tile);
                    econ->breadcrumb -= 0.35f;  // weed penalty
                }
            } else {
                // Watered: grow if not mature
                int crop = -1;
                for (int c = 0; c < proto::kCrops; ++c) {
                    if (bp_test(farm->cold.crop[c], tile)) { crop = c; break; }
                }
                if (crop >= 0) {
                    int age = econ->day - farm->cold.planted_day[tile];
                    int max_yield = proto::kCropMaxYield[crop];
                    if (farm->hot.yield[tile] < max_yield && age >= proto::kCropGrowthDays[crop]) {
                        farm->hot.yield[tile]++;
                    }
                }
                farm->cold.consecutive_unwatered[tile] = 0;
            }
        }

        // Animal structures
        if (bp_test(farm->hot.structure, tile)) {
            if (!bp_test(farm->hot.fed_today, tile)) {
                farm->cold.consecutive_unfed[tile]++;
            } else {
                farm->cold.consecutive_unfed[tile] = 0;
            }
        }
    }

    // Reset daily flags
    for (int i = 0; i < proto::kMaskWords; ++i) {
        farm->hot.watered_today[i] = 0;
        farm->hot.fed_today[i] = 0;
        farm->hot.cared_today[i] = 0;
    }

    // Advance clock
    econ->day++;
    econ->hour = 0;
    state->cold_econ.hires_today = 0;

    // Update prices
    update_all_prices(state);
}

// ============================================================================
// NEURAL NETWORK FEATURES
// ============================================================================
__device__ void make_features(const SimulatorState* state, float* features) {
    const HotEconomyState* econ = &state->hot_econ;
    const FarmBitplanes* farm = &state->farm;

    // Zero-initialize
    for (int i = 0; i < kFeatures; ++i) features[i] = 0.0f;

    // [0] Normalized money
    features[0] = econ->money / 10000.0f;
    // [1] Normalized revenue
    features[1] = econ->revenue / 10000.0f;
    // [2] Day progress
    features[2] = static_cast<float>(econ->day) / static_cast<float>(proto::kDays);
    // [3] Hour progress
    features[3] = static_cast<float>(econ->hour) / static_cast<float>(proto::kHoursPerDay);
    // [4-5] Farmer position
    features[4] = static_cast<float>(econ->farmer_x) / static_cast<float>(proto::kBoardSize - 1);
    features[5] = static_cast<float>(econ->farmer_y) / static_cast<float>(proto::kBoardSize - 1);
    // [6] Farm utilization ratio
    features[6] = static_cast<float>(bp_popcount(farm->hot.occupied, proto::kMaskWords)) 
                  / static_cast<float>(proto::kBoardTiles);
    // [7-15] Market prices
    for (int p = 0; p < proto::kProducts; ++p) {
        features[7 + p] = static_cast<float>(econ->market_prices[p]) / 500.0f;
    }
    // [16-24] Shed inventory
    for (int p = 0; p < proto::kProducts; ++p) {
        features[16 + p] = static_cast<float>(econ->shed[p]) / 100.0f;
    }
    // [25-29] Seed inventory
    for (int c = 0; c < proto::kCrops; ++c) {
        features[25 + c] = static_cast<float>(econ->seeds[c]) / 20.0f;
    }
    // [30] Has revenue flag
    features[30] = (econ->revenue > 0.0f) ? 1.0f : 0.0f;
    // [31] Has buffer money flag
    features[31] = (econ->money >= 1000.0f) ? 1.0f : 0.0f;
}

// ============================================================================
// ACTION LEGALITY (deterministic, hard constraints)
// ============================================================================
__device__ void compute_legal_mask(const SimulatorState* state, bool* legal) {
    const HotEconomyState* econ = &state->hot_econ;
    const FarmBitplanes* farm = &state->farm;
    int tile = econ->farmer_y * proto::kBoardSize + econ->farmer_x;

    for (int a = 0; a < kActions; ++a) legal[a] = false;

    // Always legal
    legal[IA_Pass] = true;

    // Movement
    legal[IA_North] = (econ->farmer_y > 0);
    legal[IA_South] = (econ->farmer_y < proto::kBoardSize - 1);
    legal[IA_East]  = (econ->farmer_x < proto::kBoardSize - 1);
    legal[IA_West]  = (econ->farmer_x > 0);

    // Buy seeds
    for (int c = 0; c < proto::kCrops; ++c) {
        legal[IA_BuyWheatSeed + c] = (econ->money >= proto::kSeedCost[c]);
    }

    // Sell products (only crops for now)
    for (int c = 0; c < proto::kCrops; ++c) {
        legal[IA_SellWheat + c] = (econ->shed[c] > 0);
    }

    // Plant (need seeds and empty tile)
    bool tile_empty = !bp_test(farm->hot.occupied, tile);
    for (int c = 0; c < proto::kCrops; ++c) {
        legal[IA_PlantWheat + c] = (econ->seeds[c] > 0) && tile_empty;
    }

    // Water (need plant, not watered, not weed)
    legal[IA_Water] = bp_test(farm->hot.plant, tile) 
                      && !bp_test(farm->hot.watered_today, tile)
                      && !bp_test(farm->cold.weed, tile);

    // Harvest (need plant with sufficient yield)
    int crop_type = -1;
    for (int c = 0; c < proto::kCrops; ++c) {
        if (bp_test(farm->cold.crop[c], tile)) { crop_type = c; break; }
    }
    int min_yield = (crop_type >= 0) ? proto::kCropMaxYield[crop_type] : 999;
    legal[IA_Harvest] = bp_test(farm->hot.plant, tile) 
                        && farm->hot.yield[tile] >= min_yield;

    // Shops (simplified: just unlock next shop)
    for (int s = 0; s < proto::kShops; ++s) {
        legal[IA_Bakery + s] = (state->cold_econ.unlocked_shop_counts[s] > 0) 
                                && (econ->money >= proto::kShopCost[s]);
    }
}

// ============================================================================
// NEURAL NETWORK FORWARD PASS
// ============================================================================
__device__ int choose_action(const AgentWeights* weights, const SimulatorState* state) {
    float features[kFeatures];
    float hidden[kHidden];
    float logits[kActions];

    make_features(state, features);

    // Hidden layer: ReLU(features @ W_input)
    for (int h = 0; h < kHidden; ++h) {
        float sum = 0.0f;
        for (int f = 0; f < kFeatures; ++f) {
            sum += features[f] * weights->input[f * kHidden + h];
        }
        hidden[h] = fmaxf(0.0f, sum);
    }

    // Output layer: hidden @ W_output
    for (int a = 0; a < kActions; ++a) {
        float sum = 0.0f;
        for (int h = 0; h < kHidden; ++h) {
            sum += hidden[h] * weights->output[h * kActions + a];
        }
        logits[a] = sum;
    }

    // Apply legal mask and select best
    bool legal[kActions];
    compute_legal_mask(state, legal);

    int best_action = IA_Pass;
    float best_logit = -1e30f;
    for (int a = 0; a < kActions; ++a) {
        if (legal[a] && logits[a] > best_logit) {
            best_logit = logits[a];
            best_action = a;
        }
    }
    return best_action;
}

// ============================================================================
// ACTION EXECUTION
// ============================================================================
__device__ void execute_action(SimulatorState* state, int action) {
    HotEconomyState* econ = &state->hot_econ;
    FarmBitplanes* farm = &state->farm;
    int tile = econ->farmer_y * proto::kBoardSize + econ->farmer_x;

    switch (action) {
        case IA_Pass:
            break;

        case IA_BuyWheatSeed:
            econ->money -= proto::kSeedCost[proto::Wheat];
            econ->seeds[proto::Wheat]++;
            econ->breadcrumb += 0.09f;
            break;
        case IA_BuyCarrotSeed:
            econ->money -= proto::kSeedCost[proto::Carrot];
            econ->seeds[proto::Carrot]++;
            econ->breadcrumb += 0.09f;
            break;
        case IA_BuyTomatoSeed:
            econ->money -= proto::kSeedCost[proto::Tomato];
            econ->seeds[proto::Tomato]++;
            econ->breadcrumb += 0.09f;
            break;
        case IA_BuyStrawberrySeed:
            econ->money -= proto::kSeedCost[proto::Strawberry];
            econ->seeds[proto::Strawberry]++;
            econ->breadcrumb += 0.09f;
            break;
        case IA_BuyMelonSeed:
            econ->money -= proto::kSeedCost[proto::Melon];
            econ->seeds[proto::Melon]++;
            econ->breadcrumb += 0.09f;
            break;

        case IA_SellWheat: {
            int qty = econ->shed[proto::Wheat];
            float proceeds = qty * econ->market_prices[proto::Wheat];
            econ->money += proceeds;
            econ->revenue += proceeds;
            econ->market_inventory[proto::Wheat] += qty;
            econ->shed[proto::Wheat] = 0;
            break;
        }
        case IA_SellCarrot: {
            int qty = econ->shed[proto::Carrot];
            float proceeds = qty * econ->market_prices[proto::Carrot];
            econ->money += proceeds;
            econ->revenue += proceeds;
            econ->market_inventory[proto::Carrot] += qty;
            econ->shed[proto::Carrot] = 0;
            break;
        }
        case IA_SellTomato: {
            int qty = econ->shed[proto::Tomato];
            float proceeds = qty * econ->market_prices[proto::Tomato];
            econ->money += proceeds;
            econ->revenue += proceeds;
            econ->market_inventory[proto::Tomato] += qty;
            econ->shed[proto::Tomato] = 0;
            break;
        }
        case IA_SellStrawberry: {
            int qty = econ->shed[proto::Strawberry];
            float proceeds = qty * econ->market_prices[proto::Strawberry];
            econ->money += proceeds;
            econ->revenue += proceeds;
            econ->market_inventory[proto::Strawberry] += qty;
            econ->shed[proto::Strawberry] = 0;
            break;
        }
        case IA_SellMelon: {
            int qty = econ->shed[proto::Melon];
            float proceeds = qty * econ->market_prices[proto::Melon];
            econ->money += proceeds;
            econ->revenue += proceeds;
            econ->market_inventory[proto::Melon] += qty;
            econ->shed[proto::Melon] = 0;
            break;
        }

        case IA_North:
            econ->farmer_y--;
            econ->breadcrumb += 0.0001f;
            break;
        case IA_South:
            econ->farmer_y++;
            econ->breadcrumb += 0.0001f;
            break;
        case IA_East:
            econ->farmer_x++;
            econ->breadcrumb += 0.0001f;
            break;
        case IA_West:
            econ->farmer_x--;
            econ->breadcrumb += 0.0001f;
            break;

        case IA_PlantWheat:
            econ->seeds[proto::Wheat]--;
            tile_plant(state, tile, proto::Wheat, econ->day);
            econ->breadcrumb += 0.25f;
            break;
        case IA_PlantCarrot:
            econ->seeds[proto::Carrot]--;
            tile_plant(state, tile, proto::Carrot, econ->day);
            econ->breadcrumb += 0.25f;
            break;
        case IA_PlantTomato:
            econ->seeds[proto::Tomato]--;
            tile_plant(state, tile, proto::Tomato, econ->day);
            econ->breadcrumb += 0.25f;
            break;
        case IA_PlantStrawberry:
            econ->seeds[proto::Strawberry]--;
            tile_plant(state, tile, proto::Strawberry, econ->day);
            econ->breadcrumb += 0.25f;
            break;
        case IA_PlantMelon:
            econ->seeds[proto::Melon]--;
            tile_plant(state, tile, proto::Melon, econ->day);
            econ->breadcrumb += 0.25f;
            break;

        case IA_Water:
            if (bp_test(farm->hot.plant, tile) && !bp_test(farm->hot.watered_today, tile)) {
                tile_water(state, tile);
                econ->breadcrumb += 0.5f;
            }
            break;

        case IA_Harvest: {
            int crop, yield;
            tile_harvest(state, tile, &crop, &yield);
            if (crop >= 0 && crop < proto::kProducts) {
                econ->shed[crop] += yield;
            }
            econ->breadcrumb += 2.0f;
            break;
        }

        case IA_Bakery:
        case IA_PizzaShop:
        case IA_BrunchSpot:
        case IA_YarnStore:
        case IA_IceCreamShop:
        case IA_PetCafe:
        case IA_SmoothieShop:
        case IA_FarmersMarket: {
            int shop = action - IA_Bakery;
            if (state->cold_econ.unlocked_shop_counts[shop] > 0 && 
                econ->money >= proto::kShopCost[shop]) {
                econ->money -= proto::kShopCost[shop];
                econ->revenue += proto::kShopCost[shop] * 1.1f;
                econ->money += proto::kShopCost[shop] * 1.1f;
                econ->breadcrumb += 0.05f;
                if (shop + 1 < proto::kShops) {
                    state->cold_econ.unlocked_shop_counts[shop + 1] = 1;
                }
            }
            break;
        }

        default:
            break;
    }

    // Advance hour
    econ->hour++;
    if (econ->hour >= proto::kHoursPerDay) {
        end_of_day(state);
    }
}

// ============================================================================
// FITNESS FUNCTION
// ============================================================================
__device__ float compute_fitness(const SimulatorState* state) {
    const HotEconomyState* econ = &state->hot_econ;
    constexpr float kStartMoney = static_cast<float>(proto::kStartingMoney);
    constexpr float kAllowedLoss = 1000.0f;

    float excess_loss = fmaxf(0.0f, kStartMoney - econ->money - kAllowedLoss);
    float loss_units = excess_loss / 50.0f;
    float revenue_exponent = 2.0f / (1.0f + loss_units);
    float revenue_term = powf(fmaxf(econ->revenue / 50.0f, 0.0f), revenue_exponent);

    return econ->money + revenue_term - excess_loss + econ->breadcrumb;
}

// ============================================================================
// REPLAY LOADING (device-side, from binary frame)
// ============================================================================
// The binary frame layout (after FlatFrame header):
// For each player:
//   ReplayPlayer player
//   int8_t hand_coords[kMaxHands * 2]
//   ReplayTile tiles[kBoardTiles]
//   int16_t shed[kProducts]
//   int16_t seeds[kCrops]
//   int16_t hand_inventory[kMaxHands][kProducts]
// Global:
//   int64_t market_inventory[kProducts]
//   int16_t prices[kProducts]
//   uint8_t shops[kShops]
// Actions:
//   EncodedAction farmer_actions[kPlayers]
//   EncodedAction hand_actions[kPlayers][kMaxHands]
//   EncodedAction market_actions[kPlayers][kMarketOrders]
//   uint8_t hand_action_count[kPlayers]
//   uint8_t market_action_count[kPlayers]

struct FullReplayFrame {
    FlatFrame header;
    // Player 0
    ReplayPlayer p0;
    int8_t p0_hands[proto::kMaxHands * 2];
    ReplayTile p0_tiles[proto::kBoardTiles];
    int16_t p0_shed[proto::kProducts];
    int16_t p0_seeds[proto::kCrops];
    int16_t p0_hand_inv[proto::kMaxHands][proto::kProducts];
    // Player 1
    ReplayPlayer p1;
    int8_t p1_hands[proto::kMaxHands * 2];
    ReplayTile p1_tiles[proto::kBoardTiles];
    int16_t p1_shed[proto::kProducts];
    int16_t p1_seeds[proto::kCrops];
    int16_t p1_hand_inv[proto::kMaxHands][proto::kProducts];
    // Global
    int64_t market_inventory[proto::kProducts];
    int16_t prices[proto::kProducts];
    uint8_t shops[proto::kShops];
    // Actions
    EncodedAction farmer_actions[proto::kPlayers];
    EncodedAction hand_actions[proto::kPlayers][proto::kMaxHands];
    EncodedAction market_actions[proto::kPlayers][proto::kMarketOrders];
    uint8_t hand_action_count[proto::kPlayers];
    uint8_t market_action_count[proto::kPlayers];
};

__device__ void load_replay_state(const FullReplayFrame* frame, int player, SimulatorState* state) {
    state_reset(state);

    const ReplayPlayer* rp = (player == 0) ? &frame->p0 : &frame->p1;
    const ReplayTile* tiles = (player == 0) ? frame->p0_tiles : frame->p1_tiles;
    const int16_t* shed = (player == 0) ? frame->p0_shed : frame->p1_shed;
    const int16_t* seeds = (player == 0) ? frame->p0_seeds : frame->p1_seeds;

    state->hot_econ.money = rp->money;
    state->hot_econ.farmer_x = rp->x;
    state->hot_econ.farmer_y = rp->y;
    state->hot_econ.day = frame->header.day;
    state->hot_econ.hour = static_cast<int8_t>(frame->header.hour);
    state->cold_econ.unlocked_quadrants = rp->unlocked_quadrants;
    state->cold_econ.hires_today = rp->hires_today;

    for (int p = 0; p < proto::kProducts; ++p) {
        state->hot_econ.market_inventory[p] = static_cast<int32_t>(frame->market_inventory[p]);
        state->hot_econ.market_prices[p] = frame->prices[p];
        state->hot_econ.shed[p] = shed[p];
    }
    for (int c = 0; c < proto::kCrops; ++c) {
        state->hot_econ.seeds[c] = seeds[c];
    }
    for (int s = 0; s < proto::kShops; ++s) {
        state->cold_econ.unlocked_shop_counts[s] = frame->shops[s];
    }

    // Load tiles
    for (int t = 0; t < proto::kBoardTiles; ++t) {
        const ReplayTile& rt = tiles[t];
        if (rt.kind == proto::TilePlant) {
            bp_set(state->farm.hot.occupied, t);
            bp_set(state->farm.hot.plant, t);
            if (rt.item > 0 && rt.item <= proto::kCrops) {
                bp_set(state->farm.cold.crop[rt.item - 1], t);
            }
            state->farm.hot.yield[t] = rt.yield;
            state->farm.cold.planted_day[t] = rt.planted_day;
            if (rt.flags & 1) bp_set(state->farm.hot.watered_today, t);
            state->farm.cold.consecutive_unwatered[t] = static_cast<int8_t>(rt.extra);
        } else if (rt.kind == proto::Weed) {
            bp_set(state->farm.cold.weed, t);
            bp_set(state->farm.hot.occupied, t);
        } else if (rt.kind == proto::Locked) {
            bp_set(state->farm.cold.locked, t);
        } else if (rt.kind == proto::Coop || rt.kind == proto::Pasture) {
            bp_set(state->farm.hot.occupied, t);
            bp_set(state->farm.hot.structure, t);
            if (rt.item > 0 && rt.item <= proto::kAnimals) {
                bp_set(state->farm.cold.animal[rt.item - 1], t);
            }
            if (rt.flags & 1) bp_set(state->farm.hot.fed_today, t);
            if (rt.flags & 2) bp_set(state->farm.hot.cared_today, t);
            state->farm.hot.yield[t] = rt.yield;
        }
    }
}

// ============================================================================
// EVALUATION KERNEL
// ============================================================================
__global__ void evaluate_kernel(
    const AgentWeights* population,
    const FullReplayFrame* replay,
    float* fitness,
    int episode_steps,
    bool use_replay
) {
    int agent = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent >= kPopulation) return;

    SimulatorState state;
    state_reset(&state);

    float action_matches = 0.0f;
    float action_total = 0.0f;

    const AgentWeights* weights = &population[agent];

    for (int step = 0; step < episode_steps; ++step) {
        if (use_replay) {
            const FullReplayFrame& frame = replay[step % proto::kSteps];

            for (int player = 0; player < proto::kPlayers; ++player) {
                load_replay_state(&frame, player, &state);
                int predicted = choose_action(weights, &state);
                int expected_op = static_cast<int>(frame.farmer_actions[player].operation);
                int predicted_op = internal_to_official(predicted);

                if (predicted_op == expected_op) action_matches += 1.0f;
                action_total += 1.0f;
            }
        } else {
            // Self-play simulation
            state.hot_econ.day = step / proto::kHoursPerDay;
            state.hot_econ.hour = step % proto::kHoursPerDay;

            if (state.hot_econ.money < 0.0f) break;

            int action = choose_action(weights, &state);
            execute_action(&state, action);
        }
    }

    if (use_replay) {
        fitness[agent] = 100000.0f * (action_matches / fmaxf(action_total, 1.0f));
    } else {
        fitness[agent] = compute_fitness(&state);
    }
}

// ============================================================================
// MUTATION KERNEL
// ============================================================================
__global__ void mutate_kernel(
    AgentWeights* population,
    const int* elite_indices,
    unsigned int seed
) {
    int agent = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent >= kPopulation || agent < kEliteCount) return;

    curandState rng;
    curand_init(seed, agent, 0, &rng);

    int parent = elite_indices[curand(&rng) % kEliteCount];
    AgentWeights* child = &population[agent];
    const AgentWeights* source = &population[parent];

    constexpr float mutation_prob = 0.08f;
    constexpr float mutation_std = 0.05f;

    for (int i = 0; i < kFeatures * kHidden; ++i) {
        child->input[i] = source->input[i] + 
            (curand_uniform(&rng) < mutation_prob ? curand_normal(&rng) * mutation_std : 0.0f);
    }
    for (int i = 0; i < kHidden * kActions; ++i) {
        child->output[i] = source->output[i] + 
            (curand_uniform(&rng) < mutation_prob ? curand_normal(&rng) * mutation_std : 0.0f);
    }
}

// ============================================================================
// SMOKE TEST KERNEL
// ============================================================================
__global__ void smoke_test_kernel(int* results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid != 0) return;

    SimulatorState state;
    state_reset(&state);

    // Test 1: Initial state
    results[0] = (state.hot_econ.money == 3000.0f) ? 1 : 0;

    // Test 2: Plant wheat at (4,4)
    int tile = 4 * proto::kBoardSize + 4;
    tile_plant(&state, tile, proto::Wheat, 0);
    results[1] = bp_test(state.farm.hot.plant, tile) ? 1 : 0;

    // Test 3: Water it
    tile_water(&state, tile);
    results[2] = bp_test(state.farm.hot.watered_today, tile) ? 1 : 0;

    // Test 4: End of day
    end_of_day(&state);
    results[3] = (state.hot_econ.day == 1 && state.hot_econ.hour == 0) ? 1 : 0;

    // Test 5: Plant should still be there (was watered)
    results[4] = bp_test(state.farm.hot.plant, tile) ? 1 : 0;

    // Test 6: Water flag should be reset
    results[5] = !bp_test(state.farm.hot.watered_today, tile) ? 1 : 0;

    // Test 7: Buy seed action (money between wheat cost [10] and melon cost [100])
    state.hot_econ.money = 80.0f;
    bool legal[kActions];
    compute_legal_mask(&state, legal);
    results[6] = legal[IA_BuyWheatSeed] ? 1 : 0;

    // Test 8: Can't buy expensive seed
    results[7] = !legal[IA_BuyMelonSeed] ? 1 : 0;
}

// ============================================================================
// HOST-SIDE REPLAY ENCODER (JSON -> binary)
// ============================================================================

class JsonError : public std::runtime_error {
public:
    explicit JsonError(const std::string& msg) : std::runtime_error(msg) {}
};

struct JsonValue {
    enum Type : uint8_t { Null, Bool, Number, String, Array, Object } type = Null;
    double number = 0.0;
    bool boolean = false;
    std::string string;
    std::vector<JsonValue> array;
    std::map<std::string, JsonValue> object;

    const JsonValue& get(const char* key) const {
        if (type != Object) throw JsonError("expected object for: " + std::string(key));
        auto it = object.find(key);
        if (it == object.end()) throw JsonError("missing key: " + std::string(key));
        return it->second;
    }
    const JsonValue* find(const char* key) const {
        if (type != Object) return nullptr;
        auto it = object.find(key);
        return (it == object.end()) ? nullptr : &it->second;
    }
    const JsonValue& operator[](size_t idx) const { return array.at(idx); }
    bool empty() const { return type == Array ? array.empty() : object.empty(); }
};

class JsonParser {
    const std::string& text;
    size_t pos = 0;

    [[noreturn]] void error(const std::string& msg) const {
        throw JsonError("JSON byte " + std::to_string(pos) + ": " + msg);
    }
    void skip_ws() {
        while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\n' || 
               text[pos] == '\r' || text[pos] == '\t')) ++pos;
    }
    void expect(char c) {
        if (pos >= text.size() || text[pos++] != c) 
            error(std::string("expected '") + c + "'");
    }

public:
    explicit JsonParser(const std::string& input) : text(input) {}

    JsonValue parse() {
        skip_ws();
        JsonValue v = parse_value();
        skip_ws();
        if (pos != text.size()) error("trailing data");
        return v;
    }

private:
    JsonValue parse_value() {
        skip_ws();
        if (pos >= text.size()) error("unexpected end");
        char c = text[pos];
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string();
        if (c == 't') { expect('t'); expect('r'); expect('u'); expect('e'); return JsonValue{JsonValue::Bool, 0, true}; }
        if (c == 'f') { expect('f'); expect('a'); expect('l'); expect('s'); expect('e'); return JsonValue{JsonValue::Bool, 0, false}; }
        if (c == 'n') { expect('n'); expect('u'); expect('l'); expect('l'); return JsonValue{JsonValue::Null}; }
        return parse_number();
    }

    JsonValue parse_number() {
        size_t start = pos;
        if (text[pos] == '-') ++pos;
        while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') ++pos;
        if (pos < text.size() && text[pos] == '.') {
            ++pos;
            while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') ++pos;
        }
        if (pos < text.size() && (text[pos] == 'e' || text[pos] == 'E')) {
            ++pos;
            if (pos < text.size() && (text[pos] == '+' || text[pos] == '-')) ++pos;
            while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') ++pos;
        }
        JsonValue v; v.type = JsonValue::Number;
        v.number = std::stod(text.substr(start, pos - start));
        return v;
    }

    JsonValue parse_string() {
        expect('"');
        std::string result;
        while (pos < text.size()) {
            char c = text[pos++];
            if (c == '"') { JsonValue v; v.type = JsonValue::String; v.string = result; return v; }
            if (c == '\\') {
                if (pos >= text.size()) error("unterminated escape");
                char e = text[pos++];
                switch (e) {
                    case '"': case '\\': case '/': result += e; break;
                    case 'b': result += '\b'; break;
                    case 'f': result += '\f'; break;
                    case 'n': result += '\n'; break;
                    case 'r': result += '\r'; break;
                    case 't': result += '\t'; break;
                    default: error("unsupported escape");
                }
            } else {
                if (static_cast<unsigned char>(c) < 0x20) error("control char");
                result += c;
            }
        }
        error("unterminated string");
    }

    JsonValue parse_array() {
        expect('['); skip_ws();
        JsonValue v; v.type = JsonValue::Array;
        if (pos < text.size() && text[pos] == ']') { ++pos; return v; }
        for (;;) {
            v.array.push_back(parse_value()); skip_ws();
            if (pos < text.size() && text[pos] == ']') { ++pos; return v; }
            expect(',');
        }
    }

    JsonValue parse_object() {
        expect('{'); skip_ws();
        JsonValue v; v.type = JsonValue::Object;
        if (pos < text.size() && text[pos] == '}') { ++pos; return v; }
        for (;;) {
            skip_ws(); expect('"');
            std::string key = parse_string().string; skip_ws(); expect(':');
            v.object.emplace(std::move(key), parse_value()); skip_ws();
            if (pos < text.size() && text[pos] == '}') { ++pos; return v; }
            expect(',');
        }
    }
};

static int host_enum_product(const std::string& name) {
    for (int i = 0; i < proto::kProducts; ++i)
        if (name == proto::kProductNames[i]) return i;
    throw JsonError("unknown product: " + name);
}

static int host_enum_shop(const std::string& name) {
    for (int i = 0; i < proto::kShops; ++i)
        if (name == proto::kShopNames[i]) return i;
    throw JsonError("unknown shop: " + name);
}

static proto::Action host_enum_action(const std::string& name) {
    if (name == "PASS") return proto::Pass;
    if (name == "NORTH") return proto::North;
    if (name == "SOUTH") return proto::South;
    if (name == "EAST") return proto::East;
    if (name == "WEST") return proto::West;
    if (name == "PICKUP") return proto::Pickup;
    if (name == "PLANT") return proto::Plant;
    if (name == "WATER") return proto::Water;
    if (name == "HARVEST") return proto::Harvest;
    if (name == "FERTILIZE") return proto::Fertilize;
    if (name == "BUILD_COOP") return proto::BuildCoop;
    if (name == "BUILD_PASTURE") return proto::BuildPasture;
    if (name == "DIG") return proto::Dig;
    if (name == "PLACE") return proto::Place;
    if (name == "FEED") return proto::Feed;
    if (name == "COLLECT_FERTILIZER") return proto::CollectFertilizer;
    if (name == "CARE") return proto::Care;
    if (name == "BUY_SEED") return proto::BuySeed;
    if (name == "BUY_PRODUCT") return proto::BuyProduct;
    if (name == "BUY_ANIMAL") return proto::BuyAnimal;
    if (name == "SELL") return proto::Sell;
    if (name == "HIRE") return proto::Hire;
    if (name == "BUY_LAND") return proto::BuyLand;
    if (name == "DROP") return proto::Drop;
    return proto::UnknownAction;
}

static int host_enum_animal(const std::string& name) {
    if (name == "GOOSE") return proto::Goose;
    if (name == "COW") return proto::Cow;
    if (name == "SHEEP") return proto::Sheep;
    throw JsonError("unknown animal: " + name);
}

static EncodedAction host_parse_action(const JsonValue& value, int step, int player) {
    if (value.type != JsonValue::Array || value.empty() || value[0].type != JsonValue::String)
        throw JsonError("malformed action at step " + std::to_string(step));
    if (value.array.size() > proto::kMaxActionTokens)
        throw JsonError("too many tokens at step " + std::to_string(step));

    EncodedAction action{};
    action.operation = static_cast<uint8_t>(host_enum_action(value[0].string));
    action.token_count = static_cast<uint8_t>(value.array.size());

    if (action.operation == static_cast<uint8_t>(proto::UnknownAction))
        throw JsonError("unknown action: " + value[0].string);

    for (size_t t = 1; t < value.array.size(); ++t) {
        const JsonValue& arg = value[t];
        if (arg.type == JsonValue::String) {
            bool mapped = false;
            for (int p = 0; p < proto::kProducts; ++p) {
                if (arg.string == proto::kProductNames[p]) {
                    action.item = static_cast<uint8_t>(p); mapped = true; break;
                }
            }
            if (!mapped) {
                try { action.animal = static_cast<uint8_t>(host_enum_animal(arg.string)); mapped = true; }
                catch (const JsonError&) {}
            }
            if (!mapped && !arg.string.empty())
                throw JsonError("unknown arg: " + arg.string);
        } else if (arg.type == JsonValue::Number) {
            if (arg.number != static_cast<int64_t>(arg.number))
                throw JsonError("non-integral arg");
            int n = static_cast<int>(arg.number);
            if (t == 1) action.quantity = static_cast<int16_t>(n);
            else if (t == 2) action.x = static_cast<int16_t>(n);
            else if (t == 3) action.y = static_cast<int16_t>(n);
            else throw JsonError("too many numeric args");
        }
    }
    return action;
}

static ReplayTile host_parse_tile(const JsonValue& value, int step, int tile_idx) {
    ReplayTile tile{};
    if (value.type == JsonValue::Null) return tile;
    if (value.type == JsonValue::String) {
        if (value.string != "LOCKED") throw JsonError("unknown tile string");
        tile.kind = proto::Locked;
        return tile;
    }
    const std::string kind = value.get("kind").string;
    if (kind == "WEED") {
        tile.kind = proto::Weed;
    } else if (kind == "PLANT") {
        tile.kind = proto::TilePlant;
        tile.item = static_cast<uint8_t>(host_enum_product(value.get("crop").string) + 1);
        const JsonValue* w = value.find("watered_today");
        if (w && w->boolean) tile.flags |= 1;
        const JsonValue* pd = value.find("planted_day");
        tile.planted_day = pd ? static_cast<int16_t>(pd->number) : -1;
        const JsonValue* yu = value.find("yield_units");
        tile.yield = yu ? static_cast<int16_t>(yu->number) : 0;
        const JsonValue* ls = value.find("max_lifespan_step");
        tile.lifetime = ls ? static_cast<int16_t>(ls->number) : -1;
        const JsonValue* fd = value.find("fertilized_until_day");
        tile.care = fd ? static_cast<int16_t>(fd->number) : -1;
        const JsonValue* cu = value.find("consecutive_unwatered");
        tile.extra = cu ? static_cast<int16_t>(cu->number) : 0;
    } else if (kind == "COOP") {
        tile.kind = proto::Coop;
        const JsonValue* a = value.find("animal");
        if (a) tile.item = static_cast<uint8_t>(host_enum_animal(a->string));
        const JsonValue* f = value.find("fed_today");
        if (f && f->boolean) tile.flags |= 1;
        const JsonValue* c = value.find("cared_today");
        if (c && c->boolean) tile.flags |= 2;
    } else if (kind == "PASTURE") {
        tile.kind = proto::Pasture;
        const JsonValue* a = value.find("animal");
        if (a) tile.item = static_cast<uint8_t>(host_enum_animal(a->string));
        const JsonValue* f = value.find("fed_today");
        if (f && f->boolean) tile.flags |= 1;
        const JsonValue* c = value.find("cared_today");
        if (c && c->boolean) tile.flags |= 2;
    } else {
        throw JsonError("unknown tile kind: " + kind);
    }
    return tile;
}

// ============================================================================
// HOST-SIDE: Player parser, frame assembler, binary I/O
// ============================================================================

static void host_parse_player(const JsonValue& farm_json, ReplayPlayer& player,
                               std::vector<int8_t>& hand_coords,
                               std::vector<ReplayTile>& tiles,
                               int step, int player_idx) {
    const JsonValue& farmer = farm_json.get("farmer");
    if (farmer.type != JsonValue::Array || farmer.array.size() != 2)
        throw JsonError("invalid farmer coords at step " + std::to_string(step));

    player.money = static_cast<float>(farm_json.get("money").number);
    player.x = static_cast<int16_t>(farmer[0].number);
    player.y = static_cast<int16_t>(farmer[1].number);

    if (player.x < 0 || player.x >= proto::kBoardSize || 
        player.y < 0 || player.y >= proto::kBoardSize)
        throw JsonError("farmer OOB at step " + std::to_string(step));

    const JsonValue& quads = farm_json.get("unlocked_quadrants");
    player.unlocked_quadrants = static_cast<uint8_t>(quads.array.size());

    const JsonValue& hands = farm_json.get("hands");
    if (hands.type != JsonValue::Array || hands.array.size() > proto::kMaxHands)
        throw JsonError("too many hands");
    player.hand_count = static_cast<uint8_t>(hands.array.size());
    player.hires_today = static_cast<uint16_t>(farm_json.get("hires_today").number);

    hand_coords.clear();
    for (const JsonValue& h : hands.array) {
        if (h.type != JsonValue::Array || h.array.size() != 2)
            throw JsonError("invalid hand coord");
        hand_coords.push_back(static_cast<int8_t>(h[0].number));
        hand_coords.push_back(static_cast<int8_t>(h[1].number));
    }
    while (hand_coords.size() < proto::kMaxHands * 2) hand_coords.push_back(0);

    const JsonValue& rows = farm_json.get("tiles");
    if (rows.type != JsonValue::Array || rows.array.size() != proto::kBoardSize)
        throw JsonError("invalid tile rows");
    tiles.clear();
    for (int y = 0; y < proto::kBoardSize; ++y) {
        const JsonValue& row = rows[y];
        if (row.type != JsonValue::Array || row.array.size() != proto::kBoardSize)
            throw JsonError("invalid tile row");
        for (int x = 0; x < proto::kBoardSize; ++x) {
            tiles.push_back(host_parse_tile(row[x], step, y * proto::kBoardSize + x));
        }
    }
}

static void write_frame(std::ofstream& out, const FullReplayFrame& frame) {
    out.write(reinterpret_cast<const char*>(&frame), sizeof(frame));
}

static void read_frame(std::ifstream& in, FullReplayFrame& frame, int step) {
    in.read(reinterpret_cast<char*>(&frame), sizeof(frame));
    if (!in || in.gcount() != sizeof(frame))
        throw std::runtime_error("truncated frame at step " + std::to_string(step));
}

static void encode_replay(const std::string& json_path, const std::string& bin_path) {
    std::ifstream in(json_path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open: " + json_path);

    std::string text((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());

    JsonValue root = JsonParser(text).parse();
    if (root.get("name").string != proto::kReplayName)
        throw JsonError("wrong replay name");
    if (root.get("module_version").string != proto::kReplayModule)
        throw JsonError("wrong module version: " + root.get("module_version").string);

    const JsonValue& steps = root.get("steps");
    if (steps.type != JsonValue::Array || steps.array.size() != proto::kSteps)
        throw JsonError("expected " + std::to_string(proto::kSteps) + " steps, got " +
                       std::to_string(steps.array.size()));

    std::ofstream out(bin_path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot create: " + bin_path);

    FlatHeader header{};
    std::memcpy(header.magic, "KAGRST01", 8);
    header.version = 1;
    header.board = proto::kBoardSize;
    header.players = proto::kPlayers;
    header.max_hands = proto::kMaxHands;
    header.frames = proto::kSteps;
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    ReplayPlayer players[proto::kPlayers];
    std::vector<int8_t> hand_coords[proto::kPlayers];
    std::vector<ReplayTile> tiles[proto::kPlayers];
    int16_t shed[proto::kPlayers][proto::kProducts];
    int16_t seeds[proto::kPlayers][proto::kCrops];
    int16_t hand_inv[proto::kPlayers][proto::kMaxHands][proto::kProducts];
    int64_t market_inventory[proto::kProducts];
    int16_t prices[proto::kProducts];
    uint8_t shops[proto::kShops];

    EncodedAction farmer_actions[proto::kPlayers];
    EncodedAction hand_actions[proto::kPlayers][proto::kMaxHands];
    EncodedAction market_actions[proto::kPlayers][proto::kMarketOrders];
    uint8_t hand_action_count[proto::kPlayers];
    uint8_t market_action_count[proto::kPlayers];

    for (int step = 0; step < proto::kSteps; ++step) {
        const JsonValue& step_val = steps[step];
        if (step_val.type != JsonValue::Array || step_val.array.size() != proto::kPlayers)
            throw JsonError("step " + std::to_string(step) + ": expected 2 records");

        FullReplayFrame frame{};
        frame.header.step = static_cast<uint32_t>(step);
        frame.header.day = static_cast<uint16_t>(step / proto::kHoursPerDay);
        frame.header.hour = static_cast<uint16_t>(step % proto::kHoursPerDay);

        const JsonValue* observation = nullptr;

        for (int player = 0; player < proto::kPlayers; ++player) {
            const JsonValue& record = step_val[player];
            const JsonValue& action = record.get("action");
            observation = &record.get("observation");

            farmer_actions[player] = host_parse_action(action.get("farmer"), step, player);

            const JsonValue& ha = action.get("hands");
            if (ha.type != JsonValue::Array || ha.array.size() > proto::kMaxHands)
                throw JsonError("too many hand actions");
            hand_action_count[player] = static_cast<uint8_t>(ha.array.size());
            for (size_t h = 0; h < ha.array.size(); ++h)
                hand_actions[player][h] = host_parse_action(ha[h], step, player);

            const JsonValue& ma = action.get("market");
            if (ma.type != JsonValue::Array || ma.array.size() > proto::kMarketOrders)
                throw JsonError("too many market actions");
            market_action_count[player] = static_cast<uint8_t>(ma.array.size());
            for (size_t m = 0; m < ma.array.size(); ++m)
                market_actions[player][m] = host_parse_action(ma[m], step, player);

            host_parse_player(observation->get("farms").array[player],
                             players[player], hand_coords[player], tiles[player],
                             step, player);

            const JsonValue& priv = observation->get("private");
            const JsonValue& shed_val = priv.get("shed");
            const JsonValue& seeds_val = priv.get("seeds");
            for (int p = 0; p < proto::kProducts; ++p) {
                const JsonValue* v = shed_val.find(proto::kProductNames[p]);
                shed[player][p] = v ? static_cast<int16_t>(v->number) : 0;
                if (p < proto::kCrops) {
                    v = seeds_val.find(proto::kProductNames[p]);
                    seeds[player][p] = v ? static_cast<int16_t>(v->number) : 0;
                }
            }

            const JsonValue& invs = priv.get("inventories");
            int inv_count = 0;
            if (invs.type == JsonValue::Array) {
                inv_count = static_cast<int>(invs.array.size());
                for (int i = 0; i < inv_count && i < proto::kMaxHands; ++i) {
                    for (int p = 0; p < proto::kProducts; ++p) {
                        const JsonValue* v = invs[i].find(proto::kProductNames[p]);
                        hand_inv[player][i][p] = v ? static_cast<int16_t>(v->number) : 0;
                    }
                }
            }
            for (int i = inv_count; i < proto::kMaxHands; ++i)
                for (int p = 0; p < proto::kProducts; ++p)
                    hand_inv[player][i][p] = 0;

            if (player == 0) {
                frame.header.day = static_cast<uint16_t>(observation->get("day").number);
                frame.header.hour = static_cast<uint16_t>(observation->get("hour").number);
                frame.header.active_player = static_cast<uint8_t>(observation->get("player").number);
            }
        }

        if (frame.header.day != step / proto::kHoursPerDay || 
            frame.header.hour != step % proto::kHoursPerDay)
            throw JsonError("clock mismatch at step " + std::to_string(step));

        frame.p0 = players[0];
        std::memcpy(frame.p0_hands, hand_coords[0].data(), proto::kMaxHands * 2);
        std::memcpy(frame.p0_tiles, tiles[0].data(), proto::kBoardTiles * sizeof(ReplayTile));
        std::memcpy(frame.p0_shed, shed[0], proto::kProducts * sizeof(int16_t));
        std::memcpy(frame.p0_seeds, seeds[0], proto::kCrops * sizeof(int16_t));
        std::memcpy(frame.p0_hand_inv, hand_inv[0], proto::kMaxHands * proto::kProducts * sizeof(int16_t));

        frame.p1 = players[1];
        std::memcpy(frame.p1_hands, hand_coords[1].data(), proto::kMaxHands * 2);
        std::memcpy(frame.p1_tiles, tiles[1].data(), proto::kBoardTiles * sizeof(ReplayTile));
        std::memcpy(frame.p1_shed, shed[1], proto::kProducts * sizeof(int16_t));
        std::memcpy(frame.p1_seeds, seeds[1], proto::kCrops * sizeof(int16_t));
        std::memcpy(frame.p1_hand_inv, hand_inv[1], proto::kMaxHands * proto::kProducts * sizeof(int16_t));

        const JsonValue& market = observation->get("market");
        for (int p = 0; p < proto::kProducts; ++p) {
            market_inventory[p] = static_cast<int64_t>(
                market.get("inventory").get(proto::kProductNames[p]).number);
            prices[p] = static_cast<int16_t>(
                market.get("prices").get(proto::kProductNames[p]).number);
        }
        std::memcpy(frame.market_inventory, market_inventory, proto::kProducts * sizeof(int64_t));
        std::memcpy(frame.prices, prices, proto::kProducts * sizeof(int16_t));

        std::fill(shops, shops + proto::kShops, 0);
        const JsonValue& town = observation->get("town");
        const JsonValue& unlocked = town.get("unlocked_shops");
        if (unlocked.type == JsonValue::Array) {
            for (const JsonValue& s : unlocked.array)
                shops[host_enum_shop(s.string)] = 1;
        }
        std::memcpy(frame.shops, shops, proto::kShops);

        std::memcpy(frame.farmer_actions, farmer_actions, proto::kPlayers * sizeof(EncodedAction));
        std::memcpy(frame.hand_actions, hand_actions, 
                   proto::kPlayers * proto::kMaxHands * sizeof(EncodedAction));
        std::memcpy(frame.market_actions, market_actions,
                   proto::kPlayers * proto::kMarketOrders * sizeof(EncodedAction));
        std::memcpy(frame.hand_action_count, hand_action_count, proto::kPlayers);
        std::memcpy(frame.market_action_count, market_action_count, proto::kPlayers);

        write_frame(out, frame);
    }

    if (!out) throw std::runtime_error("short write to " + bin_path);

    std::printf("ENCODE_OK steps=%d path=%s size=%lld\n",
                proto::kSteps, bin_path.c_str(),
                static_cast<long long>(sizeof(FlatHeader) + 
                                       proto::kSteps * sizeof(FullReplayFrame)));
}

static void validate_binary(const std::string& bin_path) {
    std::ifstream in(bin_path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open: " + bin_path);

    auto file_size = in.tellg();
    in.seekg(0);

    FlatHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in) throw std::runtime_error("cannot read header");

    if (std::memcmp(header.magic, "KAGRST01", 8) != 0)
        throw std::runtime_error("bad magic");
    if (header.version != 1 || header.board != proto::kBoardSize ||
        header.players != proto::kPlayers || header.max_hands != proto::kMaxHands)
        throw std::runtime_error("bad header fields");

    auto expected_size = static_cast<std::streamoff>(sizeof(FlatHeader)) +
                         static_cast<std::streamoff>(header.frames) * sizeof(FullReplayFrame);
    if (file_size != expected_size)
        throw std::runtime_error("size mismatch");

    std::vector<FullReplayFrame> frames(header.frames);
    for (uint32_t i = 0; i < header.frames; ++i)
        read_frame(in, frames[i], static_cast<int>(i));

    std::printf("VALIDATE_OK frames=%d size=%lld\n", header.frames,
                static_cast<long long>(file_size));
}

// ============================================================================
// TRAINING LOOP (fixed elite tracking)
// ============================================================================

static void run_training(int generations, int episode_steps, const std::string& replay_path) {
    const auto t_start = std::chrono::steady_clock::now();

    std::vector<FullReplayFrame> host_replay;
    bool use_replay = !replay_path.empty();

    if (use_replay) {
        std::ifstream in(replay_path, std::ios::binary | std::ios::ate);
        if (!in) {
            std::fprintf(stderr, "cannot open replay: %s\n", replay_path.c_str());
            std::exit(2);
        }
        auto size = in.tellg();
        auto expected = static_cast<std::streamoff>(sizeof(FlatHeader)) +
                       static_cast<std::streamoff>(proto::kSteps) * sizeof(FullReplayFrame);
        if (size != expected) {
            std::fprintf(stderr, "replay size mismatch: expected %lld, got %lld\n",
                        static_cast<long long>(expected), static_cast<long long>(size));
            std::exit(2);
        }
        in.seekg(sizeof(FlatHeader));
        host_replay.resize(proto::kSteps);
        for (int i = 0; i < proto::kSteps; ++i)
            read_frame(in, host_replay[i], i);
        std::printf("REPLAY_LOADED steps=%d\n", proto::kSteps);
    }

    AgentWeights* host_pop = new AgentWeights[kPopulation]();
    float* host_fitness = new float[kPopulation];
    int host_elites[kEliteCount];

    for (int a = 0; a < kPopulation; ++a) {
        for (int i = 0; i < kFeatures * kHidden; ++i)
            host_pop[a].input[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
        for (int i = 0; i < kHidden * kActions; ++i)
            host_pop[a].output[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
    }

    AgentWeights* d_pop = nullptr;
    float* d_fitness = nullptr;
    FullReplayFrame* d_replay = nullptr;
    int* d_elites = nullptr;

    CUDA_CHECK(cudaMalloc(&d_pop, kPopulation * sizeof(AgentWeights)));
    CUDA_CHECK(cudaMalloc(&d_fitness, kPopulation * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_pop, host_pop, kPopulation * sizeof(AgentWeights),
                          cudaMemcpyHostToDevice));

    if (use_replay) {
        CUDA_CHECK(cudaMalloc(&d_replay, proto::kSteps * sizeof(FullReplayFrame)));
        CUDA_CHECK(cudaMemcpy(d_replay, host_replay.data(),
                              proto::kSteps * sizeof(FullReplayFrame),
                              cudaMemcpyHostToDevice));
    }

    int threads = 128;
    int blocks = (kPopulation + threads - 1) / threads;
    int best_agent_idx = 0;

    for (int gen = 0; gen < generations; ++gen) {
        evaluate_kernel<<<blocks, threads>>>(d_pop, d_replay, d_fitness, episode_steps, use_replay);
        CUDA_CHECK_LAST("evaluate_kernel");
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(host_fitness, d_fitness, kPopulation * sizeof(float),
                              cudaMemcpyDeviceToHost));

        for (int a = 0; a < kPopulation; ++a) {
            if (!std::isfinite(host_fitness[a])) {
                std::fprintf(stderr, "INVALID_FITNESS gen=%d agent=%d val=%f\n",
                            gen, a, host_fitness[a]);
                std::exit(1);
            }
        }

        std::vector<int> sorted(kPopulation);
        for (int i = 0; i < kPopulation; ++i) sorted[i] = i;
        std::sort(sorted.begin(), sorted.end(),
                  [&](int a, int b) { return host_fitness[a] > host_fitness[b]; });

        for (int r = 0; r < kEliteCount; ++r) host_elites[r] = sorted[r];
        best_agent_idx = host_elites[0];

        if (gen % 10 == 0 || gen == generations - 1) {
            std::printf("GEN=%d best=%.1f elite0=%d elite127=%.1f\n",
                       gen, host_fitness[best_agent_idx],
                       best_agent_idx, host_fitness[host_elites[kEliteCount-1]]);
        }

        CUDA_CHECK(cudaMalloc(&d_elites, kEliteCount * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_elites, host_elites, kEliteCount * sizeof(int),
                              cudaMemcpyHostToDevice));

        mutate_kernel<<<blocks, threads>>>(d_pop, d_elites, 42u + static_cast<unsigned int>(gen));
        CUDA_CHECK_LAST("mutate_kernel");
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(d_elites);
    }

    evaluate_kernel<<<blocks, threads>>>(d_pop, d_replay, d_fitness, episode_steps, use_replay);
    CUDA_CHECK_LAST("final_evaluate");
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_fitness, d_fitness, kPopulation * sizeof(float),
                          cudaMemcpyDeviceToHost));

    best_agent_idx = 0;
    for (int a = 1; a < kPopulation; ++a)
        if (host_fitness[a] > host_fitness[best_agent_idx]) best_agent_idx = a;

    AgentWeights best_weights;
    CUDA_CHECK(cudaMemcpy(&best_weights, &d_pop[best_agent_idx], sizeof(AgentWeights),
                          cudaMemcpyDeviceToHost));

    char versioned[128];
    std::snprintf(versioned, sizeof(versioned),
                 "kaggriculture_weights_gen_%03d.bin", generations - 1);

    FILE* fp_v = std::fopen(versioned, "wb");
    FILE* fp_l = std::fopen("kaggriculture_weights_latest.bin", "wb");

    if (!fp_v || !fp_l) {
        std::fprintf(stderr, "weight export failed\n");
        std::exit(1);
    }

    size_t wbytes = sizeof(AgentWeights);
    if (std::fwrite(&best_weights, 1, wbytes, fp_v) != wbytes ||
        std::fwrite(&best_weights, 1, wbytes, fp_l) != wbytes) {
        std::fprintf(stderr, "weight export short write\n");
        std::exit(1);
    }

    std::fclose(fp_v);
    std::fclose(fp_l);

    cudaFree(d_replay);
    cudaFree(d_fitness);
    cudaFree(d_pop);
    delete[] host_fitness;
    delete[] host_pop;

    const auto t_end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(t_end - t_start).count();
    double agent_steps = static_cast<double>(kPopulation) * generations * episode_steps;

    std::printf("TRAINING_DONE gen=%d best_agent=%d best_fitness=%.1f\n",
                generations, best_agent_idx, host_fitness[best_agent_idx]);
    std::printf("WEIGHTS %s kaggriculture_weights_latest.bin\n", versioned);
    std::printf("TIME %.3fs throughput=%.0f agent_steps/sec\n", seconds, agent_steps / seconds);
}

// ============================================================================
// SMOKE TEST
// ============================================================================

static void run_smoke_test() {
    int* d_results = nullptr;
    int h_results[8] = {0};

    CUDA_CHECK(cudaMalloc(&d_results, 8 * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_results, 0, 8 * sizeof(int)));

    smoke_test_kernel<<<1, 1>>>(d_results);
    CUDA_CHECK_LAST("smoke_test_kernel");
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_results, d_results, 8 * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_results);

    bool pass = true;
    const char* tests[] = {
        "initial_money", "plant_wheat", "water_tile", "end_of_day",
        "plant_survives", "water_reset", "can_afford_seed", "cant_afford_melon"
    };
    for (int i = 0; i < 8; ++i) {
        if (h_results[i] != 1) {
            std::printf("SMOKE_FAIL test=%s result=%d\n", tests[i], h_results[i]);
            pass = false;
        }
    }

    if (pass) std::printf("SMOKE_PASS all=8/8\n");
    else std::exit(1);
}

// ============================================================================
// MAIN
// ============================================================================

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Kaggriculture Unified CUDA Port\n"
        "Usage:\n"
        "  %s smoke                          - Run bitplane smoke tests\n"
        "  %s encode <replay.json> <out.bin> - Encode replay to binary\n"
        "  %s validate <replay.bin>           - Validate binary frames\n"
        "  %s train <gens> <steps> [replay]  - Run evolution\n"
        "  %s benchmark <gens> <steps>       - Benchmark throughput\n",
        prog, prog, prog, prog, prog);
}

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 2; }

    std::string cmd = argv[1];

    try {
        if (cmd == "smoke") {
            run_smoke_test();
            return 0;
        }
        else if (cmd == "encode") {
            if (argc != 4) { print_usage(argv[0]); return 2; }
            encode_replay(argv[2], argv[3]);
            return 0;
        }
        else if (cmd == "validate") {
            if (argc != 3) { print_usage(argv[0]); return 2; }
            validate_binary(argv[2]);
            return 0;
        }
        else if (cmd == "train") {
            if (argc < 4) { print_usage(argv[0]); return 2; }
            int gens = std::atoi(argv[2]);
            int steps = std::atoi(argv[3]);
            std::string replay = (argc > 4) ? argv[4] : "";
            if (gens < 1 || gens > kGenerations || steps < 1 || steps > proto::kSteps) {
                std::fprintf(stderr, "invalid range: gens 1..%d, steps 1..%d\n",
                            kGenerations, proto::kSteps);
                return 2;
            }
            run_training(gens, steps, replay);
            return 0;
        }
        else if (cmd == "benchmark") {
            if (argc != 4) { print_usage(argv[0]); return 2; }
            int gens = std::atoi(argv[2]);
            int steps = std::atoi(argv[3]);
            run_training(gens, steps, "");
            return 0;
        }
        else {
            print_usage(argv[0]);
            return 2;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
}
// ============================================================================
// CATFISH (PREDATOR) ARCHITECTURE
     // ============================================================================
// The catfish does not farm. It is a market-maker that:
//   1. Observes aggregate market state (prices, inventory, time)
//   2. Observes predicted farmer behavior (from a compact "plasmid" encoding)
//   3. Chooses when to buy/sell/hold to maximize profit
//
// Plasmids: Compressed behavioral fingerprints harvested from Kaggle submissions.
//   Each plasmid is a small vector (e.g., 8-16 dims) that encodes a submission's
//   typical planting/selling schedule. The catfish pools these to predict supply.
// ============================================================================

constexpr int kCatfishPlasmidDim = 16;      // Compressed behavior fingerprint
constexpr int kCatfishMaxPlasmids = 64;     // Pool size
constexpr int kCatfishGlobalFeatures = 20;  // Market state
constexpr int kCatfishHidden = 64;
constexpr int kCatfishActions = 5;          // BUY_LOW, SELL_HIGH, HOLD, SHORT, HEDGE

struct CatfishWeights {
    // Plasmid encoder: transforms raw behavioral stats -> plasmid vector
    float plasmid_encoder[8 * kCatfishPlasmidDim];  // 8 raw stats -> 16-dim plasmid
    
    // Deep Sets pooling: sum of plasmid vectors (permutation invariant)
    // Main brain: global_state + pooled_plasmid -> actions
    float input[(kCatfishGlobalFeatures + kCatfishPlasmidDim) * kCatfishHidden];
    float output[kCatfishHidden * kCatfishActions];
};

// Raw stats extracted from a Kaggle submission replay:
// [0] avg_plants_per_day, [1] avg_sells_per_day, [2] crop_diversity,
// [3] water_reliability, [4] shop_unlock_speed, [5] money_velocity,
// [6] risk_aversion, [7] quadrant_expansion_rate
__device__ void encode_plasmid(const float* raw_stats, const float* weights, float* plasmid) {
    for (int i = 0; i < kCatfishPlasmidDim; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < 8; ++j) {
            sum += raw_stats[j] * weights[j * kCatfishPlasmidDim + i];
        }
        plasmid[i] = fmaxf(0.0f, sum);  // ReLU
    }
}

__device__ void pool_plasmids(const float* plasmids, int count, float* pooled) {
    // Deep Sets: sum pool, then normalize
    for (int i = 0; i < kCatfishPlasmidDim; ++i) pooled[i] = 0.0f;
    for (int p = 0; p < count; ++p) {
        for (int i = 0; i < kCatfishPlasmidDim; ++i) {
            pooled[i] += plasmids[p * kCatfishPlasmidDim + i];
        }
    }
    float inv_count = 1.0f / fmaxf(static_cast<float>(count), 1.0f);
    for (int i = 0; i < kCatfishPlasmidDim; ++i) pooled[i] *= inv_count;
}

// Catfish observes:
// [0-8]  market_prices (normalized)
// [9]    day_progress
// [10]   hour_progress
// [11]   total_market_inventory
// [12]   price_volatility
// [13]   predicted_wheat_supply (from plasmids)
// [14]   predicted_tomato_supply
// [15]   farmer_aggression_index
// [16-19] reserved
__device__ void catfish_make_features(const SimulatorState* market_state,
                                      const float* pooled_plasmids,
                                      float* features) {
    for (int i = 0; i < kCatfishGlobalFeatures; ++i) features[i] = 0.0f;
    
    for (int p = 0; p < proto::kProducts; ++p) {
        features[p] = market_state->hot_econ.market_prices[p] / 500.0f;
    }
    features[9] = static_cast<float>(market_state->hot_econ.day) / proto::kDays;
    features[10] = static_cast<float>(market_state->hot_econ.hour) / proto::kHoursPerDay;
    
    int total_inv = 0;
    for (int p = 0; p < proto::kProducts; ++p) total_inv += market_state->hot_econ.market_inventory[p];
    features[11] = static_cast<float>(total_inv) / 100000.0f;
    
    // Plasmid-derived predictions appended
    for (int i = 0; i < kCatfishPlasmidDim; ++i) {
        features[kCatfishGlobalFeatures - kCatfishPlasmidDim + i] = pooled_plasmids[i];
    }
}


