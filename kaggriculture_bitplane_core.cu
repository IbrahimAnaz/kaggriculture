#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

constexpr int BOARD_SIZE = 10;
constexpr int TILE_COUNT = BOARD_SIZE * BOARD_SIZE;
constexpr int WORD_COUNT = (TILE_COUNT + 31) / 32;
constexpr int CROP_COUNT = 5;
constexpr int ANIMAL_COUNT = 3;
constexpr int PRODUCT_COUNT = 9;
constexpr int SHOP_COUNT = 8;

struct HotFarmState {
    uint32_t occupied[WORD_COUNT];
    uint32_t plant[WORD_COUNT];
    uint32_t structure[WORD_COUNT];
    uint32_t watered_today[WORD_COUNT];
    uint32_t fed_today[WORD_COUNT];
    uint32_t cared_today[WORD_COUNT];
    int16_t yield_units[TILE_COUNT];
};

struct ColdFarmState {
    uint32_t locked[WORD_COUNT];
    uint32_t weed[WORD_COUNT];
    uint32_t crop[CROP_COUNT][WORD_COUNT];
    uint32_t animal[ANIMAL_COUNT][WORD_COUNT];
    int16_t planted_day[TILE_COUNT];
    int16_t fertilized_until_day[TILE_COUNT];
    int8_t consecutive_unwatered[TILE_COUNT];
    int8_t consecutive_unfed[TILE_COUNT];
    int8_t pending_care_bonus[TILE_COUNT];
};

struct FarmBitplanes {
    HotFarmState hot;
    ColdFarmState cold;
};

struct HotEconomyState {
    int32_t market_inventory[PRODUCT_COUNT];
    int16_t market_prices[PRODUCT_COUNT];
    int16_t shed_inventory[PRODUCT_COUNT];
    int16_t seeds[CROP_COUNT];
    int16_t farmer_x;
    int16_t farmer_y;
    int16_t day;
    int8_t hour;
    int8_t hands_today;
};

struct ColdEconomyState {
    uint8_t unlocked_shop_counts[SHOP_COUNT];
    uint8_t unlocked_quadrants;
    uint8_t hires_today;
    uint8_t reserved;
    int32_t money_cents;
};

struct SimulatorState {
    FarmBitplanes farm;
    HotEconomyState hot_economy;
    ColdEconomyState cold_economy;
};

__device__ __forceinline__ void set_bit(uint32_t* plane, int tile) {
    atomicOr(&plane[tile >> 5], 1u << (tile & 31));
}

__device__ __forceinline__ void clear_bit(uint32_t* plane, int tile) {
    atomicAnd(&plane[tile >> 5], ~(1u << (tile & 31)));
}

__device__ __forceinline__ bool has_bit(const uint32_t* plane, int tile) {
    return (plane[tile >> 5] & (1u << (tile & 31))) != 0;
}

__device__ void clear_tile(FarmBitplanes* farm, int tile) {
    clear_bit(farm->hot.occupied, tile);
    clear_bit(farm->cold.locked, tile);
    clear_bit(farm->cold.weed, tile);
    clear_bit(farm->hot.plant, tile);
    clear_bit(farm->hot.structure, tile);
    clear_bit(farm->hot.watered_today, tile);
    clear_bit(farm->hot.fed_today, tile);
    clear_bit(farm->hot.cared_today, tile);
    for (int crop = 0; crop < CROP_COUNT; ++crop) clear_bit(farm->cold.crop[crop], tile);
    for (int animal = 0; animal < ANIMAL_COUNT; ++animal) clear_bit(farm->cold.animal[animal], tile);
    farm->cold.planted_day[tile] = -1;
    farm->hot.yield_units[tile] = 0;
    farm->cold.fertilized_until_day[tile] = -1;
    farm->cold.consecutive_unwatered[tile] = 0;
    farm->cold.consecutive_unfed[tile] = 0;
    farm->cold.pending_care_bonus[tile] = 0;
}

__global__ void bitplane_smoke_test(SimulatorState* state, int* result) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    FarmBitplanes* farm = &state->farm;
    for (int tile = 0; tile < TILE_COUNT; ++tile) clear_tile(farm, tile);

    state->hot_economy.market_inventory[0] = 10000;
    state->hot_economy.market_prices[0] = 25;
    state->hot_economy.shed_inventory[0] = 3;
    state->hot_economy.seeds[0] = 2;
    state->hot_economy.farmer_x = 4;
    state->hot_economy.farmer_y = 4;
    state->cold_economy.unlocked_shop_counts[0] = 1;
    state->cold_economy.unlocked_quadrants = 1;
    state->cold_economy.money_cents = 300000;

    const int wheat = 14;
    const int pasture = 77;
    set_bit(farm->hot.occupied, wheat);
    set_bit(farm->hot.plant, wheat);
    set_bit(farm->cold.crop[0], wheat);
    set_bit(farm->hot.watered_today, wheat);
    farm->cold.planted_day[wheat] = 3;
    farm->hot.yield_units[wheat] = 4;

    set_bit(farm->hot.occupied, pasture);
    set_bit(farm->hot.structure, pasture);
    set_bit(farm->cold.animal[1], pasture);
    set_bit(farm->hot.fed_today, pasture);
    set_bit(farm->hot.cared_today, pasture);
    farm->hot.yield_units[pasture] = 2;

    result[0] = has_bit(farm->hot.plant, wheat) && has_bit(farm->cold.crop[0], wheat) &&
                has_bit(farm->hot.watered_today, wheat) && farm->hot.yield_units[wheat] == 4;
    result[1] = has_bit(farm->hot.structure, pasture) && has_bit(farm->cold.animal[1], pasture) &&
                has_bit(farm->hot.fed_today, pasture) && has_bit(farm->hot.cared_today, pasture);
    clear_tile(farm, wheat);
    result[2] = !has_bit(farm->hot.occupied, wheat) && !has_bit(farm->hot.plant, wheat) &&
                farm->hot.yield_units[wheat] == 0;
    result[3] = state->hot_economy.market_inventory[0] == 10000 &&
                state->hot_economy.market_prices[0] == 25 &&
                state->hot_economy.shed_inventory[0] == 3 &&
                state->hot_economy.seeds[0] == 2;
    result[4] = state->cold_economy.unlocked_shop_counts[0] == 1 &&
                state->cold_economy.unlocked_quadrants == 1 &&
                state->cold_economy.money_cents == 300000;
}

int main() {
    SimulatorState* device_state = nullptr;
    int* device_result = nullptr;
    int host_result[5] = {};
    cudaError_t status = cudaMalloc(&device_state, sizeof(SimulatorState));
    if (status != cudaSuccess) { std::fprintf(stderr, "cudaMalloc farm failed: %s\n", cudaGetErrorString(status)); return 1; }
    status = cudaMalloc(&device_result, sizeof(host_result));
    if (status != cudaSuccess) { std::fprintf(stderr, "cudaMalloc result failed: %s\n", cudaGetErrorString(status)); return 1; }
    status = cudaMemset(device_state, 0, sizeof(SimulatorState));
    if (status == cudaSuccess) bitplane_smoke_test<<<1, 1>>>(device_state, device_result);
    if (status == cudaSuccess) status = cudaGetLastError();
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    if (status == cudaSuccess) status = cudaMemcpy(host_result, device_result, sizeof(host_result), cudaMemcpyDeviceToHost);
    cudaFree(device_result);
    cudaFree(device_state);
    if (status != cudaSuccess) { std::fprintf(stderr, "bitplane test failed: %s\n", cudaGetErrorString(status)); return 1; }
    if (host_result[0] != 1 || host_result[1] != 1 || host_result[2] != 1 ||
        host_result[3] != 1 || host_result[4] != 1) { std::fprintf(stderr, "bitplane assertions failed\n"); return 1; }
    std::printf("bitplane smoke test passed: %d tiles, %d words, simulator state %zu bytes\n", TILE_COUNT, WORD_COUNT, sizeof(SimulatorState));
    return 0;
}
