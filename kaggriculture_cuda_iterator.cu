#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace kaggriculture {
constexpr int kBoardSize = 10;
constexpr int kBoardTiles = kBoardSize * kBoardSize;
constexpr int kPlayers = 2;
constexpr int kSteps = 720;
constexpr int kProducts = 9;
constexpr int kCrops = 5;
constexpr int kShops = 8;
constexpr int kMaxHands = 16;
constexpr int kMarketOrders = 10;
constexpr uint8_t kEmptyTile = 0;
constexpr uint8_t kLockedTile = 1;
constexpr uint8_t kWeedTile = 2;
constexpr uint8_t kPlantTile = 3;
constexpr uint8_t kCoopTile = 4;
constexpr uint8_t kPastureTile = 5;
constexpr uint8_t kUnknownAction = 23;

enum Action : uint8_t {
    Pass, North, South, East, West, Pickup, Plant, Water, Harvest, Fertilize,
    BuildCoop, BuildPasture, Dig, Place, Feed, CollectFertilizer, Care,
    BuySeed, BuyProduct, BuyAnimal, Sell, Hire, BuyLand, Drop, UnknownAction
};

#pragma pack(push, 1)
struct Tile {
    uint8_t kind, item, flags;
    int16_t planted_day, yield, lifetime, care, extra;
};
struct Player {
    float money;
    int16_t x, y;
    uint8_t unlocked_quadrants, hand_count;
    uint16_t hires_today;
    Tile tiles[kBoardTiles];
};
struct EncodedAction {
    uint8_t operation, item, animal, token_count;
    int16_t quantity, x, y;
};
struct Frame {
    uint32_t step;
    uint16_t day, hour;
    uint8_t active_player;
    Player players[kPlayers];
    int16_t shed[kProducts], seeds[kCrops];
    int16_t hand_inventory[kMaxHands][kProducts];
    int64_t market_inventory[kProducts];
    int16_t prices[kProducts];
    uint8_t shops[kShops];
    EncodedAction farmer_actions[kPlayers];
    EncodedAction hand_actions[kPlayers][kMaxHands];
    EncodedAction market_actions[kPlayers][kMarketOrders];
    uint8_t hand_action_count[kPlayers], market_action_count[kPlayers];
};
#pragma pack(pop)
static_assert(sizeof(Frame) == 3591, "binary frame contract changed");

struct GameState {
    uint16_t day, hour;
    Player players[kPlayers];
    int16_t shed[kProducts], seeds[kCrops];
    int16_t hand_inventory[kMaxHands][kProducts];
    int64_t market_inventory[kProducts];
    int16_t prices[kProducts];
    uint8_t shops[kShops];
};

struct CompareResult {
    int mismatch_count;
    int first_code;
};

__device__ __host__ static bool same_action(const EncodedAction& left, const EncodedAction& right) {
    return left.operation == right.operation && left.item == right.item &&
           left.animal == right.animal && left.token_count == right.token_count &&
           left.quantity == right.quantity && left.x == right.x && left.y == right.y;
}

__device__ void record_mismatch(CompareResult& result, int code) {
    atomicAdd(&result.mismatch_count, 1);
    atomicCAS(&result.first_code, 0, code);
}

__device__ void compare_player(const Player& expected, const Player& actual, int player, CompareResult& result) {
    int base = (player + 1) * 1000000;
    if (expected.money != actual.money) record_mismatch(result, base + 1);
    if (expected.x != actual.x) record_mismatch(result, base + 2);
    if (expected.y != actual.y) record_mismatch(result, base + 3);
    if (expected.unlocked_quadrants != actual.unlocked_quadrants) record_mismatch(result, base + 4);
    if (expected.hand_count != actual.hand_count) record_mismatch(result, base + 5);
    if (expected.hires_today != actual.hires_today) record_mismatch(result, base + 6);
    for (int tile = 0; tile < kBoardTiles; ++tile) {
        const Tile& left = expected.tiles[tile];
        const Tile& right = actual.tiles[tile];
        if (left.kind != right.kind || left.item != right.item || left.flags != right.flags ||
            left.planted_day != right.planted_day || left.yield != right.yield ||
            left.lifetime != right.lifetime || left.care != right.care || left.extra != right.extra) {
            record_mismatch(result, base + 100 + tile);
        }
    }
}

__global__ void compare_frame(const Frame* expected, const GameState* actual, int* result) {
    int player = blockIdx.x * blockDim.x + threadIdx.x;
    if (player >= kPlayers) return;
    CompareResult& comparison = *reinterpret_cast<CompareResult*>(result);
    compare_player(expected->players[player], actual->players[player], player, comparison);
    for (int product = 0; product < kProducts; ++product) {
        if (expected->shed[product] != actual->shed[product]) record_mismatch(comparison, 3000000 + product);
        if (expected->seeds[product < kCrops ? product : 0] != actual->seeds[product < kCrops ? product : 0] && product < kCrops) record_mismatch(comparison, 3010000 + product);
        if (expected->market_inventory[product] != actual->market_inventory[product]) record_mismatch(comparison, 3020000 + product);
        if (expected->prices[product] != actual->prices[product]) record_mismatch(comparison, 3030000 + product);
    }
    for (int shop = 0; shop < kShops; ++shop)
        if (expected->shops[shop] != actual->shops[shop]) record_mismatch(comparison, 3040000 + shop);
}

static void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

static void read_exact(std::ifstream& input, void* destination, std::size_t bytes, int step) {
    input.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes));
    if (!input || input.gcount() != static_cast<std::streamsize>(bytes))
        throw std::runtime_error("truncated frame at step " + std::to_string(step));
}

static void initialize_blank(GameState& state, const Frame& first) {
    state = {};
    state.day = first.day;
    state.hour = first.hour;
    for (int player = 0; player < kPlayers; ++player) {
        state.players[player] = first.players[player];
        state.players[player].money = 3000.0f;
    }
    for (int product = 0; product < kProducts; ++product) {
        state.market_inventory[product] = 10000;
        state.prices[product] = first.prices[product];
    }
}

static void apply_official_actions(GameState& state, const Frame& frame, int step) {
    for (int player = 0; player < kPlayers; ++player) {
        const EncodedAction& action = frame.farmer_actions[player];
        Player& farm = state.players[player];
        switch (action.operation) {
        case Pass:
            break;
        case North:
            if (farm.y <= 0) throw std::runtime_error("illegal NORTH at step " + std::to_string(step));
            --farm.y; break;
        case South:
            if (farm.y >= kBoardSize - 1) throw std::runtime_error("illegal SOUTH at step " + std::to_string(step));
            ++farm.y; break;
        case East:
            if (farm.x >= kBoardSize - 1) throw std::runtime_error("illegal EAST at step " + std::to_string(step));
            ++farm.x; break;
        case West:
            if (farm.x <= 0) throw std::runtime_error("illegal WEST at step " + std::to_string(step));
            --farm.x; break;
        case Harvest: {
            int tile = farm.y * kBoardSize + farm.x;
            if (farm.tiles[tile].kind != kPlantTile || farm.tiles[tile].yield <= 0)
                throw std::runtime_error("illegal HARVEST at step " + std::to_string(step));
            farm.tiles[tile].kind = kEmptyTile;
            farm.tiles[tile].yield = 0;
            break;
        }
        case Water: {
            int tile = farm.y * kBoardSize + farm.x;
            if (farm.tiles[tile].kind != kPlantTile) throw std::runtime_error("illegal WATER at step " + std::to_string(step));
            farm.tiles[tile].flags |= 1;
            break;
        }
        default:
            throw std::runtime_error("unsupported official transition at step " + std::to_string(step) +
                                     ": operation=" + std::to_string(action.operation));
        }
    }
    state.day = frame.hour == 23 ? frame.day + 1 : frame.day;
    state.hour = frame.hour == 23 ? 0 : frame.hour + 1;
}
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: kaggriculture_cuda_iterator <replay.kagrbin> <turn|all>\n");
        return 2;
    }
    try {
        const bool all_turns = std::string(argv[2]) == "all";
        const int target_turn = all_turns ? kaggriculture::kSteps - 1 : std::atoi(argv[2]);
        if (target_turn < 0 || target_turn >= kaggriculture::kSteps) throw std::runtime_error("turn must be in [0, 719]");
        std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error("cannot open replay binary");
        const std::streamoff expected_bytes = static_cast<std::streamoff>(kaggriculture::kSteps) * sizeof(kaggriculture::Frame);
        if (input.tellg() != expected_bytes) throw std::runtime_error("replay binary size does not match 720 frames");
        input.seekg(0);

        std::vector<kaggriculture::Frame> frames(kaggriculture::kSteps);
        kaggriculture::GameState host_state{};
        for (int step = 0; step < kaggriculture::kSteps; ++step)
            read_exact(input, &frames[step], sizeof(frames[step]), step);
        kaggriculture::initialize_blank(host_state, frames[0]);
        for (int step = 0; step < kaggriculture::kSteps; ++step) {
            const kaggriculture::Frame& expected = frames[step];
            if (expected.step != static_cast<uint32_t>(step) || expected.day != step / 24 || expected.hour != step % 24)
                throw std::runtime_error("frame index/clock mismatch at step " + std::to_string(step));
        }

        int mismatch_turns = 0;
        for (int step = 0; step <= target_turn; ++step) {
            const kaggriculture::Frame& expected = frames[step];
            kaggriculture::Frame* device_expected = nullptr;
            kaggriculture::GameState* device_actual = nullptr;
            int* device_result = nullptr;
            kaggriculture::CompareResult host_result{0, 0};
            kaggriculture::cuda_check(cudaMalloc(&device_expected, sizeof(expected)), "cudaMalloc expected");
            kaggriculture::cuda_check(cudaMalloc(&device_actual, sizeof(host_state)), "cudaMalloc actual");
            kaggriculture::cuda_check(cudaMalloc(&device_result, sizeof(host_result)), "cudaMalloc result");
            kaggriculture::cuda_check(cudaMemcpy(device_expected, &expected, sizeof(expected), cudaMemcpyHostToDevice), "copy expected");
            kaggriculture::cuda_check(cudaMemcpy(device_actual, &host_state, sizeof(host_state), cudaMemcpyHostToDevice), "copy actual");
            kaggriculture::cuda_check(cudaMemcpy(device_result, &host_result, sizeof(host_result), cudaMemcpyHostToDevice), "clear result");
            kaggriculture::compare_frame<<<1, kaggriculture::kPlayers>>>(device_expected, device_actual, device_result);
            kaggriculture::cuda_check(cudaGetLastError(), "compare_frame launch");
            kaggriculture::cuda_check(cudaDeviceSynchronize(), "compare_frame synchronize");
            kaggriculture::cuda_check(cudaMemcpy(&host_result, device_result, sizeof(host_result), cudaMemcpyDeviceToHost), "read result");
            cudaFree(device_result); cudaFree(device_actual); cudaFree(device_expected);
            if (host_result.mismatch_count != 0) {
                ++mismatch_turns;
                std::printf("MISMATCH turn=%d count=%d first_code=%d\n", step, host_result.mismatch_count, host_result.first_code);
                if (!all_turns) return 1;
            }
            if (step < target_turn) {
                try { kaggriculture::apply_official_actions(host_state, frames[step], step); }
                catch (const std::exception& error) { if (!all_turns) throw; std::printf("UNSUPPORTED turn=%d %s\n", step, error.what()); kaggriculture::initialize_blank(host_state, frames[step + 1]); }
            }
        }
        if (all_turns) { std::printf("ITERATOR REPORT turns=%d mismatch_turns=%d\n", target_turn + 1, mismatch_turns); return mismatch_turns == 0 ? 0 : 1; }
        std::printf("ITERATOR PASS turn=%d advanced_turns=%d players=%d\n", target_turn, target_turn, kaggriculture::kPlayers);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "PARITY FAIL: %s\n", error.what());
        return 1;
    }
}
