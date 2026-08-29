# NVIDIA CUDA Optimizer Handoff

## Objective
Optimize the CUDA Kaggriculture simulator and training loop for an NVIDIA GeForce MX570 (compute capability 8.6, 2 GiB VRAM). Preserve simulator correctness first; performance changes must not alter legal actions, state transitions, rewards, or replay parity.

## Active source
- `kaggriculture_breadcrumb.cu`
- Build target: `kaggriculture_breadcrumb_sm86.exe`
- Compile from a VS 2022 developer environment:

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
nvcc -O3 -arch=sm_86 kaggriculture_breadcrumb.cu -Xcompiler /W4 -lcurand -o kaggriculture_breadcrumb_sm86.exe
kaggriculture_breadcrumb_sm86.exe 1 24
```

## Current implementation
- Population: 4096 agents
- Episode maximum: 720 steps
- Policy: 32 FP32 features -> 64 ReLU hidden units -> 37 action logits
- Hard legal-action selection before execution
- GPU evaluation and mutation with cuRAND
- Bit-packed occupancy, watered, and weed planes
- Symbolic enums/constants for products and action IDs
- Current simplified state includes money, revenue, crops, seeds, shed, market prices, shops, farmer position, day/hour
- Crop lifecycle: plant -> water across days -> growth -> harvest
- Shop actions and progression are currently simplified

## Ground truth
`replay.json` is the Kaggriculture replay downloaded for competition replay `101491463`. Verify its identity before using it. It should contain 720 steps, two player records per step, and observations with:

- Both farms and 10x10 tiles
- Farmer coordinates
- Farmhands
- Private shed, seeds, and inventories
- Market inventory and prices
- Unlocked town shops
- Day/hour
- Recorded farmer, hand, and market actions

Official action contract includes:
`NORTH`, `SOUTH`, `EAST`, `WEST`, `PASS`, `PICKUP`, `PLANT`, `WATER`, `HARVEST`, `FERTILIZE`, `BUILD_COOP`, `BUILD_PASTURE`, `DIG`, `PLACE`, `FEED`, `COLLECT_FERTILIZER`, `CARE`, `BUY_SEED`, `BUY_PRODUCT`, `BUY_ANIMAL`, `SELL`, `HIRE`, `BUY_LAND`.

## Required next work
1. Build a CUDA-native fixed-width replay loader/validator. Do not use Python in the training path.
2. Represent both players, all official fields, actions, and observations.
3. Replay recorded actions against the simulator and compare to the next observation.
4. Fail loudly at the first mismatch: step, player, action, field, expected, actual, and state hashes.
5. Only optimize kernels after parity tests identify the exact supported state contract.
6. Add CUDA error checks after every launch, synchronization, allocation, and copy.
7. Keep hot state in coalesced SoA or compact structures; benchmark register pressure and occupancy on `sm_86`.
8. Do not replace deterministic legality with learned behavior.
9. Do not start long evolution runs until replay parity is established.

## Silent-failure rules
- Unknown action or item: hard error.
- Missing/truncated replay field: hard error.
- Invalid coordinates, negative inventory, NaN, or infinity: hard error.
- Player/day/hour disagreement: hard error.
- State hash mismatch: stop immediately.
- Kernel launch or `cudaDeviceSynchronize` error: stop immediately.
- Never clamp, ignore, or silently normalize mismatches.

## Validation baseline
The current source compiled and ran successfully with:

```text
architecture: sm_86
24-step smoke test: passed
recent throughput: approximately 78,556 agent-steps/sec
```

A performance change is accepted only if it preserves the baseline behavior on a deterministic seed and passes the replay differential validator.

## Engineering constraints
- Use symbolic constants/enums rather than hard-coded action/product strings or numeric IDs.
- Keep official replay data immutable.
- Preserve versioned weight export and verify the copied elite is actually the evaluated elite.
- Avoid adding farmhands/plasmids until the base farmer and simulator are faithful.
- Do not expose or hard-code API keys.
