#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <vector>

constexpr int kPopulation = 4096;
constexpr int kGenerations = 100;
constexpr int kEpisodeSteps = 720;
constexpr int kBoardTiles = 100;
constexpr int kMaskWords = 4;
constexpr int kProducts = 9;
constexpr int kShops = 8;
constexpr int kActions = 37;
constexpr int kFeatures = 32;
constexpr int kHidden = 64;
constexpr int kStartingMoney = 3000;
constexpr int kMaxPlantableCrops = 5;
constexpr int kFirstBuySeedAction = 1;
constexpr int kFirstSellAction = 11;
constexpr int kFirstMoveAction = 19;
constexpr int kFirstPlantAction = 23;
constexpr int kWaterAction = 28;
constexpr int kHarvestAction = 29;
constexpr int kFirstShopAction = 30;
constexpr int kReplaySteps = 720;
constexpr int kReplayPlayers = 2;
constexpr int kReplayMaxHands = 16;
constexpr int kReplayMarketOrders = 10;
constexpr uint8_t kReplayTileWeed = 3;
constexpr uint8_t kReplayTilePlant = 2;

enum Product : int { kWheat = 0, kCarrot, kTomato, kStrawberry, kMelon,
                     kEgg, kMilk, kWool, kFertilizer };
enum Action : int {
    kPass = 0, kBuyWheatSeed, kBuyCarrotSeed, kBuyTomatoSeed,
    kBuyStrawberrySeed, kBuyMelonSeed, kUnusedBuy6, kUnusedBuy7,
    kUnusedBuy8, kUnusedBuy9, kUnusedBuy10, kSellWheat, kSellCarrot,
    kSellTomato, kSellStrawberry, kSellMelon, kUnused16, kUnused17,
    kUnused18, kNorth, kSouth, kEast, kWest, kPlantWheat, kPlantCarrot,
    kPlantTomato, kPlantStrawberry, kPlantMelon, kWater, kHarvest,
    kBakery, kPizzaShop, kBrunchSpot, kYarnStore, kIceCreamShop,
    kPetCafe, kSmoothieShop, kFarmersMarket
};
constexpr float kMoveBreadcrumb = 0.0001f;
constexpr float kBuySeedBreadcrumb = 0.09f;
constexpr float kPlantBreadcrumb = 0.25f;
constexpr float kWaterBreadcrumb = 0.5f;
constexpr float kHarvestBreadcrumb = 2.0f;
constexpr float kWeedBreadcrumb = -0.35f;

struct BreadcrumbWeights {
    float input[kFeatures * kHidden];
    float output[kHidden * kActions];
};

#pragma pack(push, 1)
struct ReplayTile {
    uint8_t kind, item, flags;
    int16_t planted_day, yield, lifetime, care, extra;
};
struct ReplayPlayer {
    float money;
    int16_t x, y;
    uint8_t unlocked_quadrants, hand_count;
    uint16_t hires_today;
    ReplayTile tiles[kBoardTiles];
};
struct ReplayAction {
    uint8_t operation, item, animal, token_count;
    int16_t quantity, x, y;
};
struct ReplayFrame {
    uint32_t step;
    uint16_t day, hour;
    uint8_t active_player;
    ReplayPlayer players[kReplayPlayers];
    int16_t shed[kProducts], seeds[kMaxPlantableCrops];
    int16_t hand_inventory[kReplayMaxHands][kProducts];
    int64_t market_inventory[kProducts];
    int16_t prices[kProducts];
    uint8_t shops[kShops];
    ReplayAction farmer_actions[kReplayPlayers];
    ReplayAction hand_actions[kReplayPlayers][kReplayMaxHands];
    ReplayAction market_actions[kReplayPlayers][kReplayMarketOrders];
    uint8_t hand_action_count[kReplayPlayers], market_action_count[kReplayPlayers];
};
#pragma pack(pop)
static_assert(sizeof(ReplayFrame) == 3591, "replay frame layout changed; update the CUDA replay handler");

struct HotState {
    uint32_t occupied[kMaskWords];
    uint32_t watered[kMaskWords];
    uint32_t weed[kMaskWords];
    int16_t yield[kBoardTiles];
    int8_t crop[kBoardTiles];
    int8_t unwatered_days[kBoardTiles];
    int32_t market_inventory[kProducts];
    int16_t market_prices[kProducts];
    int16_t shed[kProducts];
    int16_t seeds[kMaxPlantableCrops];
    uint8_t shops[kShops];
    float money;
    float revenue;
    float breadcrumb;
    int16_t farmer_x;
    int16_t farmer_y;
    int16_t day;
    int8_t hour;
};

__device__ __forceinline__ bool bit(const uint32_t* plane, int tile);
__device__ __forceinline__ void set_bit(uint32_t* plane, int tile);
__device__ __forceinline__ void clear_bit(uint32_t* plane, int tile);
__device__ void reset_state(HotState& state);

__device__ void load_replay_state(const ReplayFrame& frame, int player, HotState& state) {
    reset_state(state);
    state.money = frame.players[player].money;
    state.farmer_x = frame.players[player].x;
    state.farmer_y = frame.players[player].y;
    state.day = frame.day;
    state.hour = static_cast<int8_t>(frame.hour);
    for (int product = 0; product < kProducts; ++product) {
        state.market_inventory[product] = static_cast<int32_t>(frame.market_inventory[product]);
        state.market_prices[product] = frame.prices[product];
        state.shed[product] = frame.shed[product];
    }
    for (int crop = 0; crop < kMaxPlantableCrops; ++crop) state.seeds[crop] = frame.seeds[crop];
    for (int shop = 0; shop < kShops; ++shop) state.shops[shop] = frame.shops[shop];
    for (int tile = 0; tile < kBoardTiles; ++tile) {
        const ReplayTile& source = frame.players[player].tiles[tile];
        if (source.kind == kReplayTilePlant) {
            set_bit(state.occupied, tile);
            state.crop[tile] = static_cast<int8_t>(source.item - 1);
            state.yield[tile] = source.yield;
            if (source.flags & 1) set_bit(state.watered, tile);
        } else if (source.kind == kReplayTileWeed) {
            set_bit(state.weed, tile);
        }
    }
}

__device__ __forceinline__ bool bit(const uint32_t* plane, int tile) {
    return (plane[tile >> 5] & (1u << (tile & 31))) != 0;
}

__device__ __forceinline__ void set_bit(uint32_t* plane, int tile) {
    plane[tile >> 5] |= 1u << (tile & 31);
}

__device__ __forceinline__ void clear_bit(uint32_t* plane, int tile) {
    plane[tile >> 5] &= ~(1u << (tile & 31));
}

__device__ float market_price(int product, int inventory) {
    constexpr float base[kProducts] = {25.f, 35.f, 60.f, 120.f, 250.f, 50.f, 160.f, 200.f, 100.f};
    constexpr float throughput[kProducts] = {400.f, 450.f, 200.f, 100.f, 300.f, 332.f, 122.f, 105.f, 200.f};
    float displacement = (10000.f - static_cast<float>(inventory)) / throughput[product];
    return fmaxf(1.f, roundf(base[product] * (1.f + 0.2f * displacement)));
}

__device__ void reset_state(HotState& state) {
    for (int word = 0; word < kMaskWords; ++word) {
        state.occupied[word] = 0;
        state.watered[word] = 0;
        state.weed[word] = 0;
    }
    for (int tile = 0; tile < kBoardTiles; ++tile) {
        state.yield[tile] = 0;
        state.crop[tile] = 0;
        state.unwatered_days[tile] = 0;
    }
    for (int product = 0; product < kProducts; ++product) {
        state.market_inventory[product] = 10000;
        state.market_prices[product] = static_cast<int16_t>(market_price(product, 10000));
        state.shed[product] = 0;
    }
    for (int crop = 0; crop < 5; ++crop) state.seeds[crop] = 0;
    for (int shop = 0; shop < kShops; ++shop) state.shops[shop] = shop == 0;
    state.money = static_cast<float>(kStartingMoney);
    state.revenue = 0.f;
    state.breadcrumb = 0.f;
    state.farmer_x = 4;
    state.farmer_y = 4;
    state.day = 0;
    state.hour = 0;
}

__device__ void make_features(const HotState& state, float* features) {
    for (int feature = 0; feature < kFeatures; ++feature) features[feature] = 0.f;
    features[0] = state.money / 10000.f;
    features[1] = state.revenue / 10000.f;
    features[2] = state.day / 30.f;
    features[3] = state.hour / 24.f;
    features[4] = state.farmer_x / 9.f;
    features[5] = state.farmer_y / 9.f;
    features[6] = __popc(state.occupied[0]) + __popc(state.occupied[1]) +
                  __popc(state.occupied[2]) + __popc(state.occupied[3]);
    features[6] /= static_cast<float>(kBoardTiles);
    for (int product = 0; product < kProducts; ++product) {
        features[7 + product] = state.market_prices[product] / 500.f;
        features[16 + product] = state.shed[product] / 100.f;
    }
    for (int crop = 0; crop < kMaxPlantableCrops; ++crop) features[25 + crop] = state.seeds[crop] / 20.f;
    features[30] = state.revenue > 0.f ? 1.f : 0.f;
    features[31] = state.money >= 1000.f ? 1.f : 0.f;
}

__device__ int choose_action(const BreadcrumbWeights& weights, const HotState& state) {
    float features[kFeatures];
    float hidden[kHidden] = {};
    float logits[kActions] = {};
    make_features(state, features);
    for (int hidden_unit = 0; hidden_unit < kHidden; ++hidden_unit) {
        for (int feature = 0; feature < kFeatures; ++feature)
            hidden[hidden_unit] += features[feature] * weights.input[feature * kHidden + hidden_unit];
        hidden[hidden_unit] = fmaxf(0.f, hidden[hidden_unit]);
    }
    for (int action = 0; action < kActions; ++action)
        for (int hidden_unit = 0; hidden_unit < kHidden; ++hidden_unit)
            logits[action] += hidden[hidden_unit] * weights.output[hidden_unit * kActions + action];

    bool legal[kActions] = {};
    legal[0] = true;
    constexpr float seed_cost[kMaxPlantableCrops] = {10.f, 20.f, 50.f, 80.f, 100.f};
    for (int crop = 0; crop < kMaxPlantableCrops; ++crop)
        legal[kFirstBuySeedAction + crop] = state.money >= seed_cost[crop];
    for (int product = kWheat; product <= kMelon; ++product)
        legal[kFirstSellAction + product] = state.shed[product] > 0;
    legal[kFirstMoveAction + 0] = state.farmer_y > 0;
    legal[kFirstMoveAction + 1] = state.farmer_y < kBoardTiles / 10 - 1;
    legal[kFirstMoveAction + 2] = state.farmer_x < kBoardTiles / 10 - 1;
    legal[kFirstMoveAction + 3] = state.farmer_x > 0;
    int tile = state.farmer_y * 10 + state.farmer_x;
    for (int crop = 0; crop < kMaxPlantableCrops; ++crop)
        legal[kFirstPlantAction + crop] = state.seeds[crop] > 0 && !bit(state.occupied, tile);
    legal[kWaterAction] = bit(state.occupied, tile) && !bit(state.watered, tile);
    legal[kHarvestAction] = bit(state.occupied, tile) && state.yield[tile] >= 3;
    for (int shop = 0; shop < kShops; ++shop)
        legal[kFirstShopAction + shop] = state.shops[shop] && state.money >= 25.f;

    int selected = 0;
    float best = -1.0e30f;
    for (int action = 0; action < kActions; ++action) {
        if (legal[action] && logits[action] > best) {
            best = logits[action];
            selected = action;
        }
    }
    return selected;
}

__device__ void execute_action(HotState& state, int action) {
    int tile = state.farmer_y * 10 + state.farmer_x;
    switch (action) {
        case 1: state.money -= 10.f; state.seeds[0]++; state.breadcrumb += kBuySeedBreadcrumb; break;
        case 2: state.money -= 20.f; state.seeds[1]++; state.breadcrumb += kBuySeedBreadcrumb; break;
        case 3: state.money -= 50.f; state.seeds[2]++; state.breadcrumb += kBuySeedBreadcrumb; break;
        case 4: state.money -= 80.f; state.seeds[3]++; state.breadcrumb += kBuySeedBreadcrumb; break;
        case 5: state.money -= 100.f; state.seeds[4]++; state.breadcrumb += kBuySeedBreadcrumb; break;
        case 11: {
            int quantity = state.shed[0];
            float proceeds = quantity * state.market_prices[0];
            state.money += proceeds;
            state.revenue += proceeds;
            state.market_inventory[0] += quantity;
            state.shed[0] = 0;
            break;
        }
        case 12: case 13: case 14: case 15: {
            int product = action - 11;
            int quantity = state.shed[product];
            float proceeds = quantity * state.market_prices[product];
            state.money += proceeds;
            state.revenue += proceeds;
            state.market_inventory[product] += quantity;
            state.shed[product] = 0;
            break;
        }
        case 19: state.farmer_y--; state.breadcrumb += kMoveBreadcrumb; break;
        case 20: state.farmer_y++; state.breadcrumb += kMoveBreadcrumb; break;
        case 21: state.farmer_x++; state.breadcrumb += kMoveBreadcrumb; break;
        case 22: state.farmer_x--; state.breadcrumb += kMoveBreadcrumb; break;
        case 23: case 24: case 25: case 26: case 27: {
            int crop = action - 23;
            state.seeds[crop]--;
            set_bit(state.occupied, tile);
            state.crop[tile] = static_cast<int8_t>(crop);
            state.yield[tile] = 1;
            state.unwatered_days[tile] = 0;
            state.breadcrumb += kPlantBreadcrumb;
            break;
        }
        case 28:
            if (bit(state.occupied, tile) && !bit(state.weed, tile) && !bit(state.watered, tile)) {
                set_bit(state.watered, tile);
                state.unwatered_days[tile] = 0;
                state.breadcrumb += kWaterBreadcrumb;
            }
            break;
        case 29:
            state.shed[state.crop[tile]] += state.yield[tile];
            state.yield[tile] = 0;
            state.crop[tile] = 0;
            clear_bit(state.occupied, tile);
            clear_bit(state.watered, tile);
            state.unwatered_days[tile] = 0;
            state.breadcrumb += kHarvestBreadcrumb;
            break;
        case 30: case 31: case 32: case 33: case 34: case 35: case 36: {
            int shop = action - 30;
            constexpr float shop_cost[kShops] = {25.f, 40.f, 55.f, 70.f, 85.f, 100.f, 125.f, 150.f};
            if (state.shops[shop] && state.money >= shop_cost[shop]) {
                state.money -= shop_cost[shop];
                state.revenue += shop_cost[shop] * 1.1f;
                state.money += shop_cost[shop] * 1.1f;
                state.breadcrumb += 0.05f;
                if (shop + 1 < kShops) state.shops[shop + 1] = 1;
            }
            break;
        }
        default: break;
    }
}

__device__ int official_operation(int action) {
    if (action == kPass) return 0;
    if (action >= kFirstMoveAction && action < kFirstMoveAction + 4) return 1 + action - kFirstMoveAction;
    if (action >= kFirstPlantAction && action < kFirstPlantAction + kMaxPlantableCrops) return 6;
    if (action == kWaterAction) return 7;
    if (action == kHarvestAction) return 8;
    if (action >= kFirstBuySeedAction && action < kFirstBuySeedAction + kMaxPlantableCrops) return 17;
    if (action >= kFirstSellAction && action < kFirstSellAction + kMaxPlantableCrops) return 20;
    return -1;
}

__device__ float farming_fitness(const HotState& state) {
    constexpr float starting_money = 3000.f;
    constexpr float allowed_loss = 1000.f;
    float excess_loss = fmaxf(0.f, starting_money - state.money - allowed_loss);
    float loss_units = excess_loss / 50.f;
    float revenue_exponent = 2.f / (1.f + loss_units);
    float revenue_term = powf(fmaxf(state.revenue / 50.f, 0.f), revenue_exponent);
    float loss_penalty = excess_loss;
    return state.money + revenue_term - loss_penalty;
}

__global__ void evaluate(const BreadcrumbWeights* population, const ReplayFrame* replay,
                           float* fitness, int episode_steps, bool use_replay) {
     int agent = blockIdx.x * blockDim.x + threadIdx.x;
     if (agent >= kPopulation) return;
     HotState state;
     float action_matches = 0.f;
     float action_count = 0.f;
     if (use_replay) load_replay_state(replay[0], 0, state);
     else reset_state(state);
     const BreadcrumbWeights& weights = population[agent];
     for (int step = 0; step < episode_steps; ++step) {
         if (use_replay) {
             const ReplayFrame& frame = replay[step % kReplaySteps];
             for (int player = 0; player < kReplayPlayers; ++player) {
                 load_replay_state(frame, player, state);
                 int predicted = choose_action(weights, state);
                 int expected = static_cast<int>(frame.farmer_actions[player].operation);
                 if (official_operation(predicted) == expected) action_matches += 1.f;
                 action_count += 1.f;
             }
             continue;
         }
         else {
             state.day = step / 24;
             state.hour = step % 24;
         }
         if (state.money < 0.f) break;
         execute_action(state, choose_action(weights, state));
         if (state.hour == 23) {
             for (int tile = 0; tile < kBoardTiles; ++tile) {
                 if (!bit(state.occupied, tile) || bit(state.weed, tile)) continue;
                 if (!bit(state.watered, tile)) {
                     state.unwatered_days[tile]++;
                     if (state.unwatered_days[tile] >= 2) {
                         set_bit(state.weed, tile);
                         state.breadcrumb += kWeedBreadcrumb;
                     }
                 } else if (state.yield[tile] < 3) {
                     state.yield[tile]++;
                 }
             }
             for (int word = 0; word < kMaskWords; ++word) state.watered[word] = 0;
         }
     }
     if (use_replay) fitness[agent] = 100000.f * (action_matches / fmaxf(action_count, 1.f));
     else fitness[agent] = farming_fitness(state) + state.breadcrumb;
}

__global__ void mutate(BreadcrumbWeights* population, const int* elite_indices, unsigned int seed) {
    int agent = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent >= kPopulation || agent < 128) return;
    curandState rng;
    curand_init(seed, agent, 0, &rng);
    int parent = elite_indices[curand(&rng) % 128];
    BreadcrumbWeights& child = population[agent];
    const BreadcrumbWeights& source = population[parent];
    for (int index = 0; index < kFeatures * kHidden; ++index)
        child.input[index] = source.input[index] + (curand_uniform(&rng) < .08f ? curand_normal(&rng) * .05f : 0.f);
    for (int index = 0; index < kHidden * kActions; ++index)
        child.output[index] = source.output[index] + (curand_uniform(&rng) < .08f ? curand_normal(&rng) * .05f : 0.f);
}

int main(int argc, char** argv) {
    const auto training_start = std::chrono::steady_clock::now();
    const int generations = argc > 1 ? std::atoi(argv[1]) : kGenerations;
    const int episode_steps = argc > 2 ? std::atoi(argv[2]) : kEpisodeSteps;
    if (generations < 1 || episode_steps < 1 || generations > kGenerations || episode_steps > kEpisodeSteps) {
        std::fprintf(stderr, "invalid limits: generations 1..%d, episode steps 1..%d\n", kGenerations, kEpisodeSteps);
        return 2;
    }
    const bool use_replay = argc > 3;
    std::vector<ReplayFrame> host_replay;
    if (use_replay) {
        std::ifstream replay_file(argv[3], std::ios::binary | std::ios::ate);
        if (!replay_file) {
            std::fprintf(stderr, "cannot open replay dataset: %s\n", argv[3]);
            return 2;
        }
        const std::streamoff expected_bytes = static_cast<std::streamoff>(kReplaySteps) * sizeof(ReplayFrame);
        if (replay_file.tellg() != expected_bytes) {
            std::fprintf(stderr, "replay size mismatch: expected %lld bytes\n", static_cast<long long>(expected_bytes));
            return 2;
        }
        host_replay.resize(kReplaySteps);
        replay_file.seekg(0);
        replay_file.read(reinterpret_cast<char*>(host_replay.data()), expected_bytes);
        if (!replay_file) {
            std::fprintf(stderr, "truncated replay dataset: %s\n", argv[3]);
            return 2;
        }
    }
    BreadcrumbWeights* host_population = new BreadcrumbWeights[kPopulation]();
    float* host_fitness = new float[kPopulation];
    int host_elites[128];
    for (int agent = 0; agent < kPopulation; ++agent) {
        for (float& weight : host_population[agent].input) weight = (static_cast<float>(rand()) / RAND_MAX - .5f) * .1f;
        for (float& weight : host_population[agent].output) weight = (static_cast<float>(rand()) / RAND_MAX - .5f) * .1f;
    }
    BreadcrumbWeights* device_population = nullptr;
    ReplayFrame* device_replay = nullptr;
    float* device_fitness = nullptr;
    cudaError_t status = cudaMalloc(&device_population, sizeof(BreadcrumbWeights) * kPopulation);
    if (status == cudaSuccess) status = cudaMalloc(&device_fitness, sizeof(float) * kPopulation);
    if (status == cudaSuccess) status = cudaMemcpy(device_population, host_population, sizeof(BreadcrumbWeights) * kPopulation, cudaMemcpyHostToDevice);
    if (status == cudaSuccess && use_replay) status = cudaMalloc(&device_replay, sizeof(ReplayFrame) * kReplaySteps);
    if (status == cudaSuccess && use_replay) status = cudaMemcpy(device_replay, host_replay.data(), sizeof(ReplayFrame) * kReplaySteps, cudaMemcpyHostToDevice);
    int threads = 128;
    int blocks = (kPopulation + threads - 1) / threads;
    for (int generation = 0; status == cudaSuccess && generation < generations; ++generation) {
        evaluate<<<blocks, threads>>>(device_population, device_replay, device_fitness, episode_steps, use_replay);
        status = cudaGetLastError();
        if (status != cudaSuccess) break;
        status = cudaDeviceSynchronize();
        if (status != cudaSuccess) break;
        status = cudaMemcpy(host_fitness, device_fitness, sizeof(float) * kPopulation, cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) break;
        for (int agent = 0; agent < kPopulation; ++agent) {
            if (!std::isfinite(host_fitness[agent])) {
                std::fprintf(stderr, "invalid fitness at generation %d agent %d\n", generation, agent);
                status = cudaErrorInvalidValue;
                break;
            }
        }
        if (status != cudaSuccess) break;
        for (int rank = 0; rank < 128; ++rank) {
            host_elites[rank] = 0;
            for (int agent = 1; agent < kPopulation; ++agent)
                if (host_fitness[agent] > host_fitness[host_elites[rank]]) {
                    bool used = false;
                    for (int prior = 0; prior < rank; ++prior) if (host_elites[prior] == agent) used = true;
                    if (!used) host_elites[rank] = agent;
                }
        }
        if (generation % 10 == 0) std::printf("generation=%d best_fitness=%.1f\n", generation, host_fitness[host_elites[0]]);
        int* device_elites = nullptr;
        cudaMalloc(&device_elites, sizeof(host_elites));
        cudaMemcpy(device_elites, host_elites, sizeof(host_elites), cudaMemcpyHostToDevice);
        mutate<<<blocks, threads>>>(device_population, device_elites, 42u + generation);
        cudaFree(device_elites);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_population, device_population,
                            sizeof(BreadcrumbWeights) * kPopulation,
                            cudaMemcpyDeviceToHost);
    }
    if (status == cudaSuccess) {
        int best_agent = 0;
        for (int agent = 1; agent < kPopulation; ++agent)
            if (host_fitness[agent] > host_fitness[best_agent]) best_agent = agent;
        char versioned_name[128];
        std::snprintf(versioned_name, sizeof(versioned_name),
                  "kaggriculture_breadcrumb_weights_gen_%03d.bin", generations - 1);
        FILE* versioned = std::fopen(versioned_name, "wb");
        FILE* latest = std::fopen("kaggriculture_breadcrumb_weights_latest.bin", "wb");
        if (!versioned || !latest) {
            std::fprintf(stderr, "weight export failed: cannot open output files\n");
            status = cudaErrorUnknown;
        } else {
            const BreadcrumbWeights& elite = host_population[best_agent];
            const std::size_t weight_bytes = sizeof(BreadcrumbWeights);
            if (std::fwrite(&elite, 1, weight_bytes, versioned) != weight_bytes ||
                std::fwrite(&elite, 1, weight_bytes, latest) != weight_bytes) {
                std::fprintf(stderr, "weight export failed: short write\n");
                status = cudaErrorUnknown;
            }
        }
        if (versioned) std::fclose(versioned);
        if (latest) std::fclose(latest);
        const auto training_end = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(training_end - training_start).count();
        const double agent_steps = static_cast<double>(kPopulation) * generations * episode_steps;
        std::printf("weights_saved=%s and kaggriculture_breadcrumb_weights_latest.bin\n", versioned_name);
        std::printf("elapsed_seconds=%.3f agent_steps_per_second=%.0f\n", seconds, agent_steps / seconds);
    }
    if (status != cudaSuccess) std::fprintf(stderr, "breadcrumb CUDA failure: %s\n", cudaGetErrorString(status));
    cudaFree(device_fitness);
    cudaFree(device_replay);
    cudaFree(device_population);
    delete[] host_fitness;
    delete[] host_population;
    return status == cudaSuccess ? 0 : 1;
}
