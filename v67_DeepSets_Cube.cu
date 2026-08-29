
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <chrono>

#define POP_SIZE 4096
#define NUM_GEN 1000

#define OBS_GLOBAL 17
#define OBS_CROP 5
#define PLASMID_HIDDEN 16
#define CMD_INPUT (OBS_GLOBAL + PLASMID_HIDDEN)
#define CMD_HIDDEN 64
#define LATENT_DIM 8
#define MAX_UNITS 16
#define NUM_MICRO 40

#define TOTAL_W_PLASMID (OBS_CROP * PLASMID_HIDDEN)
#define TOTAL_W1_CMD (CMD_INPUT * CMD_HIDDEN)
#define TOTAL_W2_CMD (CMD_HIDDEN * MAX_UNITS * LATENT_DIM)
#define TOTAL_W1_LIMB ((LATENT_DIM + 2) * NUM_MICRO)

__device__ unsigned int fast_rand(unsigned int& seed) {
    seed = (214013 * seed + 2531011);
    return (seed >> 16) & 0x7FFF;
}

__global__ void evaluate_kernel(
    float* d_w_plasmid, float* d_w1_cmd, float* d_w2_cmd, float* d_w1_limb,
    double* d_fitness, int* d_sorted_idx, int gen
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= POP_SIZE) return;
    
    float s_money = 3000.0f;
    int s_x = 6, s_y = 6, s_seeds_t1 = 0, s_seeds_t2 = 0;
    int shed_t1 = 0, shed_t2 = 0;
    float total_revenue = 0.0f;
    
    uint8_t crop_x[49];
    uint8_t crop_y[49];
    uint8_t crop_age[49] = {0};
    uint8_t crop_unwatered[49] = {0};
    bool crop_watered[49] = {false};
    uint8_t crop_type[49] = {0};
    bool crop_active[49] = {false};
    int num_active_crops = 0;
    
    // BACKWARD INDUCTION CURRICULUM
    if (false) {
        shed_t1 = 300;
    } else if (false) {
        for(int y=0; y<7; y++) {
            for(int x=0; x<7; x++) {
                int id = y * 7 + x;
                crop_active[id] = true;
                crop_x[id] = x;
                crop_y[id] = y;
                num_active_crops++;
            }
        }
    } else if (false) {
        s_seeds_t1 = 300;
    }
    
    float h_mem[8] = {0}; 
    float alpha = 0.0f;
    if (gen >= 500) {
        alpha = 0.01f + 0.99f * fminf(1.0f, (float)(gen - 500) / 500.0f);
    }
    
    for(int step = 0; step < 720; step++) {
        float price_t1 = 10.0f + 5.0f * sinf(step * 0.05f);
        float price_t2 = 100.0f + 200.0f * sinf(step * 0.05f);
        
        if (s_money < 1000.0f) break; // Margin Call (-2000 profit allowed)
        
        bool turn_ended = false;
        for(int sub_step = 0; sub_step < 10; sub_step++) {
            if (turn_ended) break;
            
            // 1. DEEP SETS PLASMID POOLING
            float pool_latent[PLASMID_HIDDEN] = {0};
            for(int c=0; c<49; c++) {
                if (!crop_active[c]) continue;
                float c_vec[OBS_CROP];
                c_vec[0] = crop_x[c] / 10.0f;
                c_vec[1] = crop_y[c] / 10.0f;
                c_vec[2] = crop_age[c] / 10.0f;
                c_vec[3] = crop_watered[c] ? 1.0f : 0.0f;
                c_vec[4] = crop_type[c] / 2.0f;
                
                for(int i=0; i<PLASMID_HIDDEN; i++) {
                    float val = 0;
                    for(int j=0; j<OBS_CROP; j++) {
                        val += c_vec[j] * d_w_plasmid[(j * PLASMID_HIDDEN + i) * POP_SIZE + idx];
                    }
                    pool_latent[i] += fmaxf(0.0f, val); // ReLU sum pooling
                }
            }
            
            // 2. MAIN BRAIN INPUT
            float vec[CMD_INPUT] = {0};
            vec[0] = s_money / 10000.0f;
            vec[1] = (step / 24) / 30.0f;
            vec[2] = price_t1 / 100.0f;
            vec[14] = price_t2 / 1000.0f;
            vec[3] = shed_t1 / 100.0f;
            vec[4] = shed_t2 / 10.0f;
            vec[5] = s_x / 10.0f;
            vec[6] = s_y / 10.0f;
            vec[7] = s_seeds_t1 / 100.0f;
            vec[8] = s_seeds_t2 / 10.0f;
            for(int i=0; i<8; i++) vec[9+i] = h_mem[i] * alpha;
            for(int i=0; i<PLASMID_HIDDEN; i++) vec[OBS_GLOBAL + i] = pool_latent[i] / 49.0f; // Normalize by max crops
            
            // 3. MLP FORWARD PASS
            float h_cmd[CMD_HIDDEN] = {0};
            for(int j=0; j<CMD_INPUT; j++) {
                if (vec[j] == 0) continue;
                for(int i=0; i<CMD_HIDDEN; i++) {
                    h_cmd[i] += vec[j] * d_w1_cmd[(j * CMD_HIDDEN + i) * POP_SIZE + idx];
                }
            }
            for(int i=0; i<CMD_HIDDEN; i++) h_cmd[i] = fmaxf(0.0f, h_cmd[i]);
            
            float z_unit[LATENT_DIM] = {0};
            for(int j=0; j<CMD_HIDDEN; j++) {
                if (h_cmd[j] == 0) continue;
                for(int i=0; i<LATENT_DIM; i++) {
                    z_unit[i] += h_cmd[j] * d_w2_cmd[(j * MAX_UNITS * LATENT_DIM + 0 * LATENT_DIM + i) * POP_SIZE + idx];
                }
            }
            for(int i=0; i<LATENT_DIM; i++) z_unit[i] = fmaxf(0.0f, z_unit[i]);
            
            float out_L3[NUM_MICRO] = {0};
            for(int j=0; j<LATENT_DIM; j++) {
                if (z_unit[j] == 0) continue;
                for(int i=0; i<NUM_MICRO; i++) {
                    out_L3[i] += z_unit[j] * d_w1_limb[(j * NUM_MICRO + i) * POP_SIZE + idx];
                }
            }
            
            for(int i=0; i<8; i++) h_mem[i] = out_L3[20+i]; 
            
            // 4. ACTION MASKING
            bool m[NUM_MICRO] = {false};
            m[0] = true; 
            
            if (s_y > 0) m[1] = true;
            if (s_y < 14) m[2] = true;
            if (s_x < 14) m[3] = true;
            if (s_x > 0) m[4] = true;
            
            int active_crop_idx = -1;
            for(int c=0; c<49; c++) {
                if (crop_active[c] && crop_x[c] == s_x && crop_y[c] == s_y) {
                    active_crop_idx = c;
                    break;
                }
            }
            

            if (active_crop_idx != -1) {
                if (crop_age[active_crop_idx] >= 5) m[8] = true; // HARVEST (5-day growth timer)
                if (!crop_watered[active_crop_idx]) m[18] = true; // WATER
            }
 else if (s_x < 7 && s_y < 7) {
                 // PLANT
            }
            
            if (s_money >= 10.0f) m[11] = true; 
            if (s_money >= 100.0f) m[12] = true; 
             // SELL ALL
            
            int best_act = 0;
            float best_val = -99999.0f;
            for(int i=0; i<NUM_MICRO; i++) {
                if (m[i] && out_L3[i] > best_val) {
                    best_val = out_L3[i];
                    best_act = i;
                }
            }
            
            // 5. EXECUTE ACTION
            if (best_act == 0) { turn_ended = true; }
            else if (best_act == 1) { s_y--; turn_ended = true; }
            else if (best_act == 2) { s_y++; turn_ended = true; }
            else if (best_act == 3) { s_x++; turn_ended = true; }
            else if (best_act == 4) { s_x--; turn_ended = true; }
            else if (best_act == 8) { // HARVEST
                if (crop_type[active_crop_idx] == 1) shed_t1++;
                else shed_t2++;
                crop_active[active_crop_idx] = false;
                num_active_crops--;
                turn_ended = true;
            }
            else if (best_act == 16 || best_act == 19) { // PLANT
                int empty_slot = -1;
                for(int c=0; c<49; c++) if (!crop_active[c]) { empty_slot = c; break; }
                if (empty_slot != -1) {
                    crop_active[empty_slot] = true;
                    crop_x[empty_slot] = s_x;
                    crop_y[empty_slot] = s_y;
                    crop_age[empty_slot] = 0;
                    crop_watered[empty_slot] = true;
                    crop_unwatered[empty_slot] = 0;
                    crop_type[empty_slot] = (best_act == 16) ? 1 : 2;
                    if (best_act == 16) s_seeds_t1--; else s_seeds_t2--;
                    num_active_crops++;
                }
                turn_ended = true;
            }
            else if (best_act == 11) { s_money -= 10.0f; s_seeds_t1 += 1; }
            else if (best_act == 12) { s_money -= 100.0f; s_seeds_t1 += 10; }
            else if (best_act == 14) { s_money -= 100.0f; s_seeds_t2 += 1; }
            else if (best_act == 15) { s_money -= 1000.0f; s_seeds_t2 += 10; }
            else if (best_act == 17) { // SELL
                float rev = (shed_t1 * price_t1) + (shed_t2 * price_t2);
                s_money += rev;
                total_revenue += rev;
                shed_t1 = 0; shed_t2 = 0;
            }
        }


        // 6. END OF DAY BIOLOGICAL TIMERS
        if (step % 24 == 23) {
            for(int c=0; c<49; c++) {
                if (!crop_active[c]) continue;
                if (!crop_watered[c]) {
                    crop_unwatered[c]++;
                    if (crop_unwatered[c] >= 2) {
                        crop_active[c] = false; // DIED
                        num_active_crops--;
                        continue;
                    }
                } else {
                    crop_age[c]++;
                }
                crop_watered[c] = false; // Reset for tomorrow
            }
        }

    }
    
    d_fitness[idx] = (double)s_money + ((double)total_revenue * (double)total_revenue);
}

__global__ void reorder_elites_kernel(
    float* src_wp, float* dest_wp,
    float* src_w1, float* dest_w1, 
    float* src_w2, float* dest_w2,
    float* src_l, float* dest_l,
    int* sorted_idx, int total_wp, int total_w1, int total_w2, int total_l
) {
    int param_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int agent_idx = blockIdx.y; 
    if (agent_idx >= POP_SIZE) return;
    int src_agent = sorted_idx[agent_idx];
    
    if (param_idx < total_wp) dest_wp[param_idx * POP_SIZE + agent_idx] = src_wp[param_idx * POP_SIZE + src_agent];
    if (param_idx < total_w1) dest_w1[param_idx * POP_SIZE + agent_idx] = src_w1[param_idx * POP_SIZE + src_agent];
    if (param_idx < total_w2) dest_w2[param_idx * POP_SIZE + agent_idx] = src_w2[param_idx * POP_SIZE + src_agent];
    if (param_idx < total_l) dest_l[param_idx * POP_SIZE + agent_idx] = src_l[param_idx * POP_SIZE + src_agent];
}

__global__ void mutate_kernel(
    float* d_wp, float* d_w1, float* d_w2, float* d_l,
    unsigned int seed
) {
    int param_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int agent_idx = blockIdx.y; 
    if (agent_idx >= POP_SIZE || agent_idx < 128) return; // Keep top 128 pure
    
    unsigned int rseed = seed + agent_idx * 1337 + param_idx;
    float mutation_rate = 0.2f;
    float mutation_scale = 0.1f;
    
    auto rand_float = [&]() {
        fast_rand(rseed);
        fast_rand(rseed);
        return ((float)(fast_rand(rseed) % 10000) / 5000.0f) - 1.0f;
    };
    
    if (param_idx < TOTAL_W_PLASMID && (fast_rand(rseed) % 100) < (mutation_rate * 100)) d_wp[param_idx * POP_SIZE + agent_idx] += rand_float() * mutation_scale;
    if (param_idx < TOTAL_W1_CMD && (fast_rand(rseed) % 100) < (mutation_rate * 100)) d_w1[param_idx * POP_SIZE + agent_idx] += rand_float() * mutation_scale;
    if (param_idx < TOTAL_W2_CMD && (fast_rand(rseed) % 100) < (mutation_rate * 100)) d_w2[param_idx * POP_SIZE + agent_idx] += rand_float() * mutation_scale;
    if (param_idx < TOTAL_W1_LIMB && (fast_rand(rseed) % 100) < (mutation_rate * 100)) d_l[param_idx * POP_SIZE + agent_idx] += rand_float() * mutation_scale;
}

int main() {
    float *d_w_plasmid, *d_w1_cmd, *d_w2_cmd, *d_w1_limb;
    double *d_fitness;
    int *d_sorted_idx;
    
    cudaMalloc(&d_w_plasmid, POP_SIZE * TOTAL_W_PLASMID * sizeof(float));
    cudaMalloc(&d_w1_cmd, POP_SIZE * TOTAL_W1_CMD * sizeof(float));
    cudaMalloc(&d_w2_cmd, POP_SIZE * TOTAL_W2_CMD * sizeof(float));
    cudaMalloc(&d_w1_limb, POP_SIZE * TOTAL_W1_LIMB * sizeof(float));
    cudaMalloc(&d_fitness, POP_SIZE * sizeof(double));
    cudaMalloc(&d_sorted_idx, POP_SIZE * sizeof(int));
    
    float *h_w_plasmid = new float[POP_SIZE * TOTAL_W_PLASMID]();
    float *h_w1_cmd = new float[POP_SIZE * TOTAL_W1_CMD]();
    float *h_w2_cmd = new float[POP_SIZE * TOTAL_W2_CMD]();
    float *h_w1_limb = new float[POP_SIZE * TOTAL_W1_LIMB]();
    double *h_fitness = new double[POP_SIZE]();
    int *h_sorted_idx = new int[POP_SIZE]();
    
    for(int i=0; i<POP_SIZE * TOTAL_W_PLASMID; i++) h_w_plasmid[i] = ((rand() % 1000)/500.0f - 1.0f)*0.1f;
    for(int i=0; i<POP_SIZE * TOTAL_W1_CMD; i++) h_w1_cmd[i] = ((rand() % 1000)/500.0f - 1.0f)*0.1f;
    for(int i=0; i<POP_SIZE * TOTAL_W2_CMD; i++) h_w2_cmd[i] = ((rand() % 1000)/500.0f - 1.0f)*0.1f;
    for(int i=0; i<POP_SIZE * TOTAL_W1_LIMB; i++) h_w1_limb[i] = ((rand() % 1000)/500.0f - 1.0f)*0.1f;
    
    cudaMemcpy(d_w_plasmid, h_w_plasmid, POP_SIZE * TOTAL_W_PLASMID * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w1_cmd, h_w1_cmd, POP_SIZE * TOTAL_W1_CMD * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2_cmd, h_w2_cmd, POP_SIZE * TOTAL_W2_CMD * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w1_limb, h_w1_limb, POP_SIZE * TOTAL_W1_LIMB * sizeof(float), cudaMemcpyHostToDevice);
    
    float *d_wp_buf, *d_w1_buf, *d_w2_buf, *d_l_buf;
    cudaMalloc(&d_wp_buf, POP_SIZE * TOTAL_W_PLASMID * sizeof(float));
    cudaMalloc(&d_w1_buf, POP_SIZE * TOTAL_W1_CMD * sizeof(float));
    cudaMalloc(&d_w2_buf, POP_SIZE * TOTAL_W2_CMD * sizeof(float));
    cudaMalloc(&d_l_buf, POP_SIZE * TOTAL_W1_LIMB * sizeof(float));
    
    dim3 block(256);
    dim3 grid((POP_SIZE + block.x - 1) / block.x);
    
    for (int gen = 0; gen < NUM_GEN; gen++) {
        evaluate_kernel<<<grid, block>>>(d_w_plasmid, d_w1_cmd, d_w2_cmd, d_w1_limb, d_fitness, d_sorted_idx, gen);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_fitness, d_fitness, POP_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
        
        for(int i=0; i<POP_SIZE; i++) h_sorted_idx[i] = i;
        std::sort(h_sorted_idx, h_sorted_idx + POP_SIZE, [&](int a, int b) {
            return h_fitness[a] > h_fitness[b];
        });
        
        if (gen % 10 == 0) {
            printf("Gen %d | Fitness: %.1f\n", gen, h_fitness[h_sorted_idx[0]]);
            fflush(stdout);
        }
        
        cudaMemcpy(d_sorted_idx, h_sorted_idx, POP_SIZE * sizeof(int), cudaMemcpyHostToDevice);
        
        int max_params = std::max(TOTAL_W_PLASMID, std::max(TOTAL_W1_CMD, std::max(TOTAL_W2_CMD, TOTAL_W1_LIMB)));
        dim3 copy_block(256);
        dim3 copy_grid((max_params + copy_block.x - 1)/copy_block.x, POP_SIZE);
        
        reorder_elites_kernel<<<copy_grid, copy_block>>>(
            d_w_plasmid, d_wp_buf, d_w1_cmd, d_w1_buf, d_w2_cmd, d_w2_buf, d_w1_limb, d_l_buf,
            d_sorted_idx, TOTAL_W_PLASMID, TOTAL_W1_CMD, TOTAL_W2_CMD, TOTAL_W1_LIMB
        );
        cudaDeviceSynchronize();
        
        std::swap(d_w_plasmid, d_wp_buf);
        std::swap(d_w1_cmd, d_w1_buf);
        std::swap(d_w2_cmd, d_w2_buf);
        std::swap(d_w1_limb, d_l_buf);
        
        
        mutate_kernel<<<copy_grid, copy_block>>>(d_w_plasmid, d_w1_cmd, d_w2_cmd, d_w1_limb, rand());
        cudaDeviceSynchronize();
    }
    
    // SAVE THE ELITE AGENT
    FILE* fp = fopen("v67_elite_weights.bin", "wb");
    if (fp) {
        float* elite_wp = new float[TOTAL_W_PLASMID];
        float* elite_w1 = new float[TOTAL_W1_CMD];
        float* elite_w2 = new float[TOTAL_W2_CMD];
        float* elite_l = new float[TOTAL_W1_LIMB];
        
        for(int i=0; i<TOTAL_W_PLASMID; i++) elite_wp[i] = h_w_plasmid[i * POP_SIZE + h_sorted_idx[0]];
        for(int i=0; i<TOTAL_W1_CMD; i++) elite_w1[i] = h_w1_cmd[i * POP_SIZE + h_sorted_idx[0]];
        for(int i=0; i<TOTAL_W2_CMD; i++) elite_w2[i] = h_w2_cmd[i * POP_SIZE + h_sorted_idx[0]];
        for(int i=0; i<TOTAL_W1_LIMB; i++) elite_l[i] = h_w1_limb[i * POP_SIZE + h_sorted_idx[0]];
        
        fwrite(elite_wp, sizeof(float), TOTAL_W_PLASMID, fp);
        fwrite(elite_w1, sizeof(float), TOTAL_W1_CMD, fp);
        fwrite(elite_w2, sizeof(float), TOTAL_W2_CMD, fp);
        fwrite(elite_l, sizeof(float), TOTAL_W1_LIMB, fp);
        fclose(fp);
        printf("Elite weights saved to v67_elite_weights.bin!\n");
    }
    
    return 0;

}
