#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#pragma pack(push, 1)
struct Header { char magic[8]; uint16_t version, board, players, max_hands; uint32_t frames; };
struct Frame { uint32_t step; uint16_t day, hour; uint8_t player; };
struct Farm { float money; int16_t farmer_x, farmer_y; uint8_t unlocked, hand_count; uint16_t hires_today; };
struct Tile { uint8_t kind, crop; uint16_t flags; int16_t planted_day, yield_units, max_lifespan_or_unfed, fertilized_or_care, extra; };
#pragma pack(pop)

constexpr int kProducts = 9;
constexpr int kCrops = 5;
constexpr int kBoard = 10;
constexpr int kMaxHands = 16;

class Reader {
public:
    explicit Reader(const char* path) : input(path, std::ios::binary) {
        if (!input) throw std::runtime_error("cannot open state file");
        read(header);
        if (std::string(header.magic, 8) != "KAGRST01" || header.version != 1 ||
            header.board != kBoard || header.players != 2 || header.max_hands != kMaxHands)
            throw std::runtime_error("unsupported Kaggriculture flat-state header");
    }

    void read_frame() {
        Frame frame{};
        read(frame);
        Farm farms[2]{};
        for (auto& farm : farms) {
            read(farm);
            input.ignore(kMaxHands * 2);
            for (int i = 0; i < kBoard * kBoard; ++i) { Tile tile{}; read(tile); }
        }
        int16_t shed[kProducts]{}; int16_t seeds[kCrops]{}; int16_t inventory[kMaxHands][kProducts]{};
        int64_t market_inventory[kProducts]{}; int16_t prices[kProducts]{}; uint8_t shops[8]{};
        read_bytes(shed, sizeof shed); read_bytes(seeds, sizeof seeds);
        read_bytes(inventory, sizeof inventory); read_bytes(market_inventory, sizeof market_inventory);
        read_bytes(prices, sizeof prices); read_bytes(shops, sizeof shops);
        if (frame.step == 0) {
            std::cout << "step=" << frame.step << " day=" << frame.day << " hour=" << frame.hour
                      << " player=" << static_cast<int>(frame.player)
                      << " money=" << farms[frame.player].money
                      << " wheat_price=" << prices[0] << '\n';
        }
    }

    void validate() {
        for (uint32_t i = 0; i < header.frames; ++i) read_frame();
        if (!input) throw std::runtime_error("truncated Kaggriculture flat-state stream");
        std::cout << "validated " << header.frames << " frames\n";
    }

private:
    template <typename T> void read(T& value) { read_bytes(&value, sizeof value); }
    void read_bytes(void* value, std::size_t size) { input.read(static_cast<char*>(value), size); }
    std::ifstream input;
    Header header{};
};

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: kaggriculture_flat_state <file>\n"; return 2; }
    try { Reader reader(argv[1]); reader.validate(); return 0; }
    catch (const std::exception& error) { std::cerr << "error: " << error.what() << '\n'; return 1; }
}
