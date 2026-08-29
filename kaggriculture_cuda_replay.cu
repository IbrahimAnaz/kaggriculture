#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace protocol {
constexpr int Board = 10;
constexpr int Players = 2;
constexpr int Steps = 720;
constexpr int Products = 9;
constexpr int Crops = 5;
constexpr int Shops = 8;
constexpr int MaxHands = 16;
constexpr int MaxActionTokens = 8;
constexpr char ReplayName[] = "kaggriculture";
constexpr char ReplayModule[] = "1.32.7";

enum Product : uint8_t { Wheat, Carrot, Tomato, Strawberry, Melon, Egg, Milk, Wool, Fertilizer };
enum Action : uint8_t {
    Pass, North, South, East, West, Pickup, Plant, Water, Harvest, Fertilize,
    BuildCoop, BuildPasture, Dig, Place, Feed, CollectFertilizer, Care,
    BuySeed, BuyProduct, BuyAnimal, Sell, Hire, BuyLand, Drop, UnknownAction
};

enum TileKind : uint8_t { Empty, Locked, Weed, TilePlant, Coop, Pasture };
enum Animal : uint8_t { NoAnimal, Goose, Cow, Sheep };

static const char* const ProductNames[Products] = {
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"
};
static const char* const ShopNames[Shops] = {
    "BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "ICE_CREAM_SHOP",
    "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET"
};
}

class JsonError : public std::runtime_error {
public:
    explicit JsonError(const std::string& message) : std::runtime_error(message) {}
};

struct JsonValue {
    enum Type : uint8_t { Null, Boolean, Number, String, Array, Object } type = Null;
    double number = 0.0;
    bool boolean = false;
    std::string string;
    std::vector<JsonValue> array;
    std::unordered_map<std::string, JsonValue> object;

    const JsonValue& get(const char* key) const {
        if (type != Object) throw JsonError("expected object while reading " + std::string(key));
        auto found = object.find(key);
        if (found == object.end()) throw JsonError("missing JSON key: " + std::string(key));
        return found->second;
    }
    const JsonValue* find(const char* key) const {
        if (type != Object) return nullptr;
        auto found = object.find(key);
        return found == object.end() ? nullptr : &found->second;
    }
    bool empty() const { return type == Array ? array.empty() : object.empty(); }
    const JsonValue& operator[](size_t index) const { return array.at(index); }
    int integer(const char* key) const {
        const JsonValue& value = get(key);
        if (value.type != Number || value.number != static_cast<int64_t>(value.number))
            throw JsonError("expected integer: " + std::string(key));
        return static_cast<int>(value.number);
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& input) : text(input) {}
    JsonValue parse() {
        skip();
        JsonValue result = value();
        skip();
        if (position != text.size()) error("trailing JSON data");
        return result;
    }
private:
    const std::string& text;
    size_t position = 0;
    [[noreturn]] void error(const std::string& message) const {
        throw JsonError("JSON byte " + std::to_string(position) + ": " + message);
    }
    void skip() { while (position < text.size() && (text[position] == ' ' || text[position] == '\n' || text[position] == '\r' || text[position] == '\t')) ++position; }
    void expect(char character) { if (position >= text.size() || text[position++] != character) error(std::string("expected '") + character + "'"); }
    JsonValue value() {
        skip();
        if (position >= text.size()) error("unexpected end");
        char character = text[position];
        if (character == '{') return object();
        if (character == '[') return array();
        if (character == '"') return JsonValue{JsonValue::String, 0.0, false, string(), {}, {}};
        if (character == 't') return literal("true", JsonValue::Boolean, true);
        if (character == 'f') return literal("false", JsonValue::Boolean, false);
        if (character == 'n') return literal("null", JsonValue::Null, false);
        if (character == '-' || (character >= '0' && character <= '9')) return number();
        error("invalid value");
    }
    JsonValue literal(const char* word, JsonValue::Type type, bool boolean) {
        size_t length = std::strlen(word);
        if (text.compare(position, length, word) != 0) error("invalid literal");
        position += length;
        JsonValue result; result.type = type; result.boolean = boolean; return result;
    }
    JsonValue number() {
        size_t start = position;
        if (text[position] == '-') ++position;
        while (position < text.size() && text[position] >= '0' && text[position] <= '9') ++position;
        if (position < text.size() && text[position] == '.') { ++position; while (position < text.size() && text[position] >= '0' && text[position] <= '9') ++position; }
        if (position < text.size() && (text[position] == 'e' || text[position] == 'E')) { ++position; if (text[position] == '+' || text[position] == '-') ++position; while (position < text.size() && text[position] >= '0' && text[position] <= '9') ++position; }
        JsonValue result; result.type = JsonValue::Number; result.number = std::stod(text.substr(start, position - start)); return result;
    }
    std::string string() {
        expect('"'); std::string result;
        while (position < text.size()) {
            char character = text[position++];
            if (character == '"') return result;
            if (character == '\\') {
                if (position >= text.size()) error("unterminated escape");
                char escaped = text[position++];
                const char* escapes = "\"\\/bfnrt"; const char* replacements = "\"\\/\b\f\n\r\t";
                const char* found = std::strchr(escapes, escaped);
                if (!found) error("unsupported escape; use ASCII JSON");
                result += replacements[found - escapes];
            } else { if (static_cast<unsigned char>(character) < 0x20) error("control character in string"); result += character; }
        }
        error("unterminated string");
    }
    JsonValue array() {
        JsonValue result; result.type = JsonValue::Array; expect('['); skip();
        if (position < text.size() && text[position] == ']') { ++position; return result; }
        for (;;) { result.array.push_back(value()); skip(); if (position < text.size() && text[position] == ']') { ++position; return result; } expect(','); }
    }
    JsonValue object() {
        JsonValue result; result.type = JsonValue::Object; expect('{'); skip();
        if (position < text.size() && text[position] == '}') { ++position; return result; }
        for (;;) { skip(); if (position >= text.size() || text[position] != '"') error("object key must be string"); std::string key = string(); skip(); expect(':'); result.object.emplace(std::move(key), value()); skip(); if (position < text.size() && text[position] == '}') { ++position; return result; } expect(','); }
    }
};

#pragma pack(push, 1)
struct GpuTile { uint8_t kind, item, flags; int16_t planted_day, yield, lifetime, care, extra; };
struct GpuPlayer { float money; int16_t x, y; uint8_t unlocked_quadrants, hand_count; uint16_t hires_today; GpuTile tiles[protocol::Board * protocol::Board]; };
struct EncodedAction { uint8_t operation, item, animal, token_count; int16_t quantity, x, y; };
struct GpuFrame {
    uint32_t step; uint16_t day, hour; uint8_t active_player;
    GpuPlayer players[protocol::Players];
    int16_t shed[protocol::Players][protocol::Products], seeds[protocol::Players][protocol::Crops];
    int16_t hand_inventory[protocol::Players][protocol::MaxHands][protocol::Products];
    int64_t market_inventory[protocol::Products]; int16_t prices[protocol::Products]; uint8_t shops[protocol::Shops];
    EncodedAction farmer_actions[protocol::Players];
    EncodedAction hand_actions[protocol::Players][protocol::MaxHands];
    EncodedAction market_actions[protocol::Players][10];
    uint8_t hand_action_count[protocol::Players], market_action_count[protocol::Players];
};
#pragma pack(pop)

static int enum_product(const std::string& name) {
    for (int i = 0; i < protocol::Products; ++i) if (name == protocol::ProductNames[i]) return i;
    throw JsonError("unknown product/item: " + name);
}
static int enum_shop(const std::string& name) {
    for (int i = 0; i < protocol::Shops; ++i) if (name == protocol::ShopNames[i]) return i;
    throw JsonError("unknown shop: " + name);
}
static protocol::Action enum_action(const std::string& name) {
    if (name == "PASS") return protocol::Pass;
    if (name == "NORTH") return protocol::North;
    if (name == "SOUTH") return protocol::South;
    if (name == "EAST") return protocol::East;
    if (name == "WEST") return protocol::West;
    if (name == "PICKUP") return protocol::Pickup;
    if (name == "PLANT") return protocol::Plant;
    if (name == "WATER") return protocol::Water;
    if (name == "HARVEST") return protocol::Harvest;
    if (name == "FERTILIZE") return protocol::Fertilize;
    if (name == "BUILD_COOP") return protocol::BuildCoop;
    if (name == "BUILD_PASTURE") return protocol::BuildPasture;
    if (name == "DIG") return protocol::Dig;
    if (name == "PLACE") return protocol::Place;
    if (name == "FEED") return protocol::Feed;
    if (name == "COLLECT_FERTILIZER") return protocol::CollectFertilizer;
    if (name == "CARE") return protocol::Care;
    if (name == "BUY_SEED") return protocol::BuySeed;
    if (name == "BUY_PRODUCT") return protocol::BuyProduct;
    if (name == "BUY_ANIMAL") return protocol::BuyAnimal;
    if (name == "SELL") return protocol::Sell;
    if (name == "HIRE") return protocol::Hire;
    if (name == "BUY_LAND") return protocol::BuyLand;
    if (name == "DROP") return protocol::Drop;
    return protocol::UnknownAction;
}

static int enum_animal(const std::string& name) {
    if (name == "GOOSE") return protocol::Goose;
    if (name == "COW") return protocol::Cow;
    if (name == "SHEEP") return protocol::Sheep;
    throw JsonError("unknown animal: " + name);
}

static EncodedAction parse_action(const JsonValue& value, int step, int player) {
    if (value.type != JsonValue::Array || value.empty() || value[0].type != JsonValue::String)
        throw JsonError("step " + std::to_string(step) + " player " + std::to_string(player) + ": malformed action");
    if (value.array.size() > protocol::MaxActionTokens)
        throw JsonError("step " + std::to_string(step) + ": action has too many tokens");
    EncodedAction action{}; action.operation = enum_action(value[0].string); action.token_count = static_cast<uint8_t>(value.array.size());
    if (action.operation == protocol::UnknownAction) throw JsonError("unknown action: " + value[0].string);
    for (size_t token = 1; token < value.array.size(); ++token) {
        const JsonValue& argument = value[token];
        if (argument.type == JsonValue::String) {
            bool mapped = false;
            for (int product = 0; product < protocol::Products; ++product) if (argument.string == protocol::ProductNames[product]) { action.item = static_cast<uint8_t>(product); mapped = true; break; }
            if (!mapped) {
                try { action.animal = static_cast<uint8_t>(enum_animal(argument.string)); mapped = true; } catch (const JsonError&) {}
            }
            if (!mapped && argument.string != "") throw JsonError("unknown action argument: " + argument.string);
        } else if (argument.type == JsonValue::Number) {
            if (argument.number != static_cast<int64_t>(argument.number)) throw JsonError("non-integral action argument");
            int value_number = static_cast<int>(argument.number);
            if (token == 1) action.quantity = static_cast<int16_t>(value_number);
            else if (token == 2) action.x = static_cast<int16_t>(value_number);
            else if (token == 3) action.y = static_cast<int16_t>(value_number);
            else throw JsonError("too many numeric action arguments");
        } else throw JsonError("action argument is neither string nor integer");
    }
    return action;
}

static GpuTile parse_tile(const JsonValue& value, int step, int tile_index) {
    GpuTile tile{};
    if (value.type == JsonValue::Null) return tile;
    if (value.type == JsonValue::String) { if (value.string != "LOCKED") throw JsonError("step " + std::to_string(step) + " tile " + std::to_string(tile_index) + ": unknown tile string"); tile.kind = protocol::Locked; return tile; }
    const std::string kind = value.get("kind").string;
    if (kind == "WEED") tile.kind = protocol::Weed;
    else if (kind == "PLANT") { tile.kind = protocol::TilePlant; tile.item = static_cast<uint8_t>(enum_product(value.get("crop").string)); tile.flags = value.find("watered_today") && value.get("watered_today").boolean ? 1 : 0; }
    else if (kind == "COOP") tile.kind = protocol::Coop;
    else if (kind == "PASTURE") tile.kind = protocol::Pasture;
    else throw JsonError("step " + std::to_string(step) + ": unknown tile kind " + kind);
    return tile;
}

static void parse_player(const JsonValue& farm, GpuPlayer& output, int step, int player) {
    const JsonValue& farmer = farm.get("farmer");
    if (farmer.type != JsonValue::Array || farmer.array.size() != 2) throw JsonError("step " + std::to_string(step) + " player " + std::to_string(player) + ": invalid farmer coordinate");
    output.money = static_cast<float>(farm.get("money").number); output.x = static_cast<int16_t>(farmer.array[0].number); output.y = static_cast<int16_t>(farmer.array[1].number);
    if (output.x < 0 || output.x >= protocol::Board || output.y < 0 || output.y >= protocol::Board) throw JsonError("farmer coordinate out of bounds");
    const JsonValue& tiles = farm.get("tiles");
    if (tiles.type != JsonValue::Array || tiles.array.size() != protocol::Board) throw JsonError("farm is not 10 rows");
    for (int y = 0; y < protocol::Board; ++y) { if (tiles.array[y].type != JsonValue::Array || tiles.array[y].array.size() != protocol::Board) throw JsonError("farm row is not 10 columns"); for (int x = 0; x < protocol::Board; ++x) output.tiles[y * protocol::Board + x] = parse_tile(tiles.array[y].array[x], step, y * protocol::Board + x); }
    const JsonValue& quadrants = farm.get("unlocked_quadrants"); output.unlocked_quadrants = static_cast<uint8_t>(quadrants.array.size());
    const JsonValue& hands = farm.get("hands"); if (hands.type != JsonValue::Array || hands.array.size() > protocol::MaxHands) throw JsonError("too many farmhands"); output.hand_count = static_cast<uint8_t>(hands.array.size()); output.hires_today = static_cast<uint16_t>(farm.get("hires_today").number);
}

static GpuFrame parse_frame(const JsonValue& step_value, int step) {
    if (step_value.type != JsonValue::Array || step_value.array.size() != protocol::Players) throw JsonError("step " + std::to_string(step) + ": expected two player records");
    GpuFrame frame{}; frame.step = step;
    const JsonValue* observation = nullptr;
    for (int player = 0; player < protocol::Players; ++player) {
        const JsonValue& record = step_value.array[player];
        const JsonValue& action = record.get("action"); observation = &record.get("observation");
        frame.farmer_actions[player] = parse_action(action.get("farmer"), step, player);
        const JsonValue& hand_actions = action.get("hands");
        if (hand_actions.type != JsonValue::Array || hand_actions.array.size() > protocol::MaxHands) throw JsonError("too many hand actions");
        frame.hand_action_count[player] = static_cast<uint8_t>(hand_actions.array.size());
        for (size_t hand = 0; hand < hand_actions.array.size(); ++hand) frame.hand_actions[player][hand] = parse_action(hand_actions[hand], step, player);
        const JsonValue& market_actions = action.get("market");
        if (market_actions.type != JsonValue::Array || market_actions.array.size() > 10) throw JsonError("too many market actions");
        frame.market_action_count[player] = static_cast<uint8_t>(market_actions.array.size());
        for (size_t order = 0; order < market_actions.array.size(); ++order) frame.market_actions[player][order] = parse_action(market_actions[order], step, player);
        parse_player(observation->get("farms").array[player], frame.players[player], step, player);
        const JsonValue& private_state = observation->get("private");
        const JsonValue& shed = private_state.get("shed");
        const JsonValue& seeds = private_state.get("seeds");
        for (int product = 0; product < protocol::Products; ++product) {
            const JsonValue* value = shed.find(protocol::ProductNames[product]);
            frame.shed[player][product] = value ? static_cast<int16_t>(value->number) : 0;
            if (product < protocol::Crops) {
                value = seeds.find(protocol::ProductNames[product]);
                frame.seeds[player][product] = value ? static_cast<int16_t>(value->number) : 0;
            }
        }
        if (player == 0) { frame.day = static_cast<uint16_t>(observation->get("day").number); frame.hour = static_cast<uint16_t>(observation->get("hour").number); }
        else if (frame.day != observation->get("day").number || frame.hour != observation->get("hour").number) throw JsonError("players disagree on clock");
    }
    if (frame.day != step / 24 || frame.hour != step % 24) throw JsonError("replay clock mismatch at step " + std::to_string(step));
    const JsonValue& observation_root = *observation;
    const JsonValue& market = observation_root.get("market");
    for (int product = 0; product < protocol::Products; ++product) { frame.market_inventory[product] = static_cast<int64_t>(market.get("inventory").get(protocol::ProductNames[product]).number); frame.prices[product] = static_cast<int16_t>(market.get("prices").get(protocol::ProductNames[product]).number); }
    const JsonValue& shops = observation_root.get("town").get("unlocked_shops"); for (const JsonValue& shop : shops.array) frame.shops[enum_shop(shop.string)] = 1;
    return frame;
}

__global__ void validate_frames(const GpuFrame* frames, int count, int* first_error) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; if (index >= count) return; const GpuFrame& frame = frames[index];
    if (frame.step != static_cast<uint32_t>(index) || frame.day != index / 24 || frame.hour != index % 24) atomicMin(first_error, index + 1);
    for (int player = 0; player < protocol::Players; ++player) { const GpuPlayer& farm = frame.players[player]; if (farm.x < 0 || farm.x >= protocol::Board || farm.y < 0 || farm.y >= protocol::Board || farm.hand_count > protocol::MaxHands || farm.unlocked_quadrants > 4 || frame.farmer_actions[player].operation >= protocol::UnknownAction || frame.hand_action_count[player] > protocol::MaxHands || frame.market_action_count[player] > 10) atomicMin(first_error, index + 1); }
}

static void cuda_check(cudaError_t error, const char* operation) { if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error)); }

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: kaggriculture_cuda_replay <replay.json> <output.kagrbin>\n"); return 2; }
    try {
        std::ifstream input(argv[1], std::ios::binary); if (!input) throw std::runtime_error("cannot open replay JSON");
        std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>()); JsonValue root = JsonParser(text).parse();
        if (root.get("name").string != protocol::ReplayName || root.get("module_version").string != protocol::ReplayModule) throw JsonError("unexpected Kaggriculture replay identity");
        const JsonValue& steps = root.get("steps"); if (steps.type != JsonValue::Array || steps.array.size() != protocol::Steps) throw JsonError("expected exactly 720 replay steps");
        std::vector<GpuFrame> frames; frames.reserve(protocol::Steps); for (int step = 0; step < protocol::Steps; ++step) frames.push_back(parse_frame(steps.array[step], step));
        std::ofstream output(argv[2], std::ios::binary); if (!output) throw std::runtime_error("cannot open output"); output.write(reinterpret_cast<const char*>(frames.data()), static_cast<std::streamsize>(frames.size() * sizeof(GpuFrame))); if (!output) throw std::runtime_error("short binary write");
        GpuFrame* device_frames = nullptr; int* device_error = nullptr; int host_error = std::numeric_limits<int>::max();
        cuda_check(cudaMalloc(&device_frames, frames.size() * sizeof(GpuFrame)), "cudaMalloc frames"); cuda_check(cudaMalloc(&device_error, sizeof(int)), "cudaMalloc error");
        cuda_check(cudaMemcpy(device_frames, frames.data(), frames.size() * sizeof(GpuFrame), cudaMemcpyHostToDevice), "cudaMemcpy frames"); cuda_check(cudaMemcpy(device_error, &host_error, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy error");
        validate_frames<<<(protocol::Steps + 127) / 128, 128>>>(device_frames, protocol::Steps, device_error); cuda_check(cudaGetLastError(), "validate_frames launch"); cuda_check(cudaDeviceSynchronize(), "validate_frames synchronize"); cuda_check(cudaMemcpy(&host_error, device_error, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy result");
        cudaFree(device_error); cudaFree(device_frames); if (host_error != std::numeric_limits<int>::max()) throw std::runtime_error("CUDA validation failed at step " + std::to_string(host_error - 1));
        std::printf("CUDA_REPLAY_VALIDATION_PASS steps=%d frame_bytes=%zu output=%s\n", protocol::Steps, sizeof(GpuFrame), argv[2]); return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "CUDA_REPLAY_VALIDATION_FAIL: %s\n", error.what()); return 1; }
}
