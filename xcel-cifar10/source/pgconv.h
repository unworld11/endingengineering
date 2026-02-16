#ifndef PGCONV_H
#define PGCONV_H

#include "typedefs.h"
#include "dimension_def.h"
#include "gate_mask.h"   // 🔴 Adaptive PG gate mask
#include <iostream>

using namespace std;

// Forward declaration of global gate index (defined in bnn_tiled.cc)
extern int gate_idx;
extern int enabled_gates_count;

// ------------------------------------------------------------------
// Popcount helper constants
// ------------------------------------------------------------------
const uint64 m1 = 6148914691236517205;
const uint64 m2 = 3689348814741910323;
const uint64 m4 = 1085102592571150095;

// ------------------------------------------------------------------
// 64-bit XNOR + popcount engine
// ------------------------------------------------------------------
inline uint8 compute_engine_64(uint64 b, uint64 w)
{
#pragma HLS latency max=1
    uint64 x = b ^ w;

    x -= (x >> 1) & m1;
    x = (x & m2) + ((x >> 2) & m2);
    x = (x + (x >> 4)) & m4;
    x += x >>  8;
    x += x >> 16;
    x += x >> 32;
    return (x & 0x7f);
}

/*
 * Binary convolutional layer
 * NOTE:
 * - MSB path always executes when switch_on = 1
 * - LSB path executes only when switch_on = 1
 */
void binary_conv3x3_tile(
        uint64 msb_inputs[WIDTH][WIDTH],
        const uint64 weights[OUT_CHANNEL_PARALLELISM][3][3],
        int16 msb_outputs[CHANNEL_OUT_T][WIDTH][WIDTH],

        int16 comparator[CHANNEL_OUT_T][WIDTH][WIDTH],
        const FIX_WT threshold[OUT_CHANNEL_PARALLELISM],
        bool switch_on[OUT_CHANNEL_PARALLELISM],

        int c_in,
        int in_channels,
        int H_fmap_out
)
{
#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=weights complete dim=3
#pragma HLS ARRAY_PARTITION variable=msb_outputs complete dim=1
#pragma HLS ARRAY_PARTITION variable=comparator complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshold complete dim=1
#pragma HLS ARRAY_PARTITION variable=switch_on complete dim=1

    const FIX_WT msb_scale = 2.0 / 3.0;

    uint64 msb_line_buffer[2][WIDTH] = {0};
    uint64 msb_window_buffer[3][3] = {0};
#pragma HLS ARRAY_PARTITION variable=msb_line_buffer complete dim=0
#pragma HLS ARRAY_PARTITION variable=msb_window_buffer complete dim=0

    int16 msb_partial_out_feature[OUT_CHANNEL_PARALLELISM] = {0};
#pragma HLS ARRAY_PARTITION variable=msb_partial_out_feature complete dim=1

    for (int row = 0; row < H_fmap_out + 1; row++) {
        for (int col = 0; col < H_fmap_out + 1; col++) {
#pragma HLS PIPELINE

            // Update window buffers
            for (int i = 0; i < 3; i++) {
                msb_window_buffer[i][0] = msb_window_buffer[i][1];
                msb_window_buffer[i][1] = msb_window_buffer[i][2];
            }

            msb_window_buffer[0][2] = (msb_line_buffer[0][col]);
            msb_window_buffer[1][2] = (msb_line_buffer[0][col] = msb_line_buffer[1][col]);
            msb_window_buffer[2][2] = (msb_line_buffer[1][col] = msb_inputs[row][col]);

            // Load partial sums
            for (int channel_pt = 0; channel_pt < OUT_CHANNEL_PARALLELISM; channel_pt++) {
                if (c_in > 0)
                    msb_partial_out_feature[channel_pt] = msb_outputs[channel_pt][row][col];
                else
                    msb_partial_out_feature[channel_pt] = 0;
            }

            // Compute convolution
            for (int channel_pt = 0; channel_pt < OUT_CHANNEL_PARALLELISM; channel_pt++) {
                int16 msb_accumulation = 0;

                if (switch_on[channel_pt] || (msb_scale * comparator[channel_pt][row][col] > threshold[channel_pt])) {
                    for (int k_row = 0; k_row < 3; k_row++) {
                        for (int k_col = 0; k_col < 3; k_col++) {
                            int row_idx_pad = row - 2 + k_row;
                            int col_idx_pad = col - 2 + k_col;
                            if (row_idx_pad >= 0 && row_idx_pad < H_fmap_out &&
                                col_idx_pad >= 0 && col_idx_pad < H_fmap_out) {

                                uint64 msb_a = msb_window_buffer[k_row][k_col];
                                uint64 w = weights[channel_pt][k_row][k_col];
                                msb_accumulation += in_channels -
                                                   2 * compute_engine_64(msb_a, w);
                            }
                        }
                    }
                }

                msb_partial_out_feature[channel_pt] += msb_accumulation;
                
#ifndef __SYNTHESIS__
                if (row == 0 && col == 0 && channel_pt == 0) {
                     //cout << "Intermediate accumulation (row=0, col=0, ch=0): " << msb_accumulation
                     //     << " Partial sum: " << msb_partial_out_feature[channel_pt] << endl;
                }
#endif
            }

            // Write back
            for (int channel_pt = 0; channel_pt < OUT_CHANNEL_PARALLELISM; channel_pt++) {
                msb_outputs[channel_pt][row][col] = msb_partial_out_feature[channel_pt];
            }
        }
    }
}

// ------------------------------------------------------------------
// Precision-Gated (Adaptive PG) fractional convolution
// ------------------------------------------------------------------
inline void pg_conv3x3_tile(
        uint64 msb_inputs[WIDTH][WIDTH],
        uint64 lsb_inputs[WIDTH][WIDTH],
        const uint64 weights[OUT_CHANNEL_PARALLELISM][3][3],
        int16 msb_outputs[CHANNEL_OUT_T][WIDTH][WIDTH],
        int16 lsb_outputs[CHANNEL_OUT_T][WIDTH][WIDTH],
        const FIX_WT threshold[OUT_CHANNEL_PARALLELISM],

        int c_in,
        int in_channels,
        int H_fmap_out,
        bool use_gate_mask
)
{
#pragma HLS INLINE
#pragma HLS ALLOCATION instances=binary_conv3x3_tile limit=1 function

    bool switch_on[OUT_CHANNEL_PARALLELISM];
#pragma HLS ARRAY_PARTITION variable=switch_on complete dim=1

    // --------------------------------------------------
    // Phase 1: MSB path (ALWAYS ON)
    // --------------------------------------------------
    for (int i = 0; i < OUT_CHANNEL_PARALLELISM; i++) {
#pragma HLS UNROLL
        switch_on[i] = 1;
    }

    binary_conv3x3_tile(
        msb_inputs, weights, msb_outputs,
        lsb_outputs,    // comparator (unused here)
        threshold,
        switch_on,
        c_in, in_channels, H_fmap_out
    );

    // --------------------------------------------------
    // Phase 2: LSB path (Adaptive PG)
    // --------------------------------------------------
    for (int i = 0; i < OUT_CHANNEL_PARALLELISM; i++) {
#pragma HLS UNROLL
        bool pg_enable = false;

        if (use_gate_mask) {
            // Task A: Gating Ablation Logic
            if (INFERENCE_MODE == MODE_BINARY) {
                pg_enable = false;
            } else if (INFERENCE_MODE == MODE_FRACTIONAL) {
                 pg_enable = true;
            } else { // MODE_ADAPTIVE
                pg_enable = (gate_mask[gate_idx % 672] > 0.5f);
            }
            
            // Increment gate index only for adaptive/fractional logic flow consistency
            // In original code: gate_idx++ was called per channel when use_gate_mask is true.
            // We maintain this behavior to keep the sequence aligned even if we ignore the value in non-Adaptive modes?
            // User requirement: "Do not change convolution logic". 
            // However, if we are in BINARY mode, we effectively ignore the mask. 
            // BUT, if we want to toggle modes without "retraining" or "refactoring", we should probably 
            // keep the gate_idx increment to keep the "virtual" pointers aligned if we ever switch back dynamically?
            // Actually, for "All gates = 0" or "All gates = 1", the mask values don't matter.
            // But let's keep the logging for debug.
            
#ifndef __SYNTHESIS__
            // Task B: Gate Sparsity Instrumentation
            layer_stats[current_layer_id].total_gates++;
            if (pg_enable) {
                layer_stats[current_layer_id].active_gates++;
            }
            
             // Debug print (optional, guarding to reduce noise if needed)
            // std::cout << "gate[" << gate_idx << "] = " << gate_mask[gate_idx] << " -> " << pg_enable << std::endl;
#endif

            gate_idx++;
            if (pg_enable) enabled_gates_count++;
            
            switch_on[i] = pg_enable ? 1 : 0;
            
        } else {
            switch_on[i] = 1; // Always ON for Conv1 (first layer usually plain binary or fixed full)
        }
    }
    
#ifndef __SYNTHESIS__
    // Task C: BMAC Estimation
    // Each call to binary_conv3x3_tile computes (H_fmap_out * H_fmap_out) * 9 (3x3) * OUT_CHANNEL_PARALLELISM * in_channels ops?
    // Actually, let's look at binary_conv3x3_tile. 
    // It loops row, col (H_fmap_out+1?). The convolution itself is 3x3.
    // Each active channel_pt in switch_on contributes to ops.
    // Ops per switch_on[i]: (H_fmap_out * H_fmap_out) * 9 * 1 (binary op is XNOR+popcount, counting as 1 BMAC per bit or per kernel?). 
    // Standard BMAC usually means 1 Binary Multiply-Accumulate. 
    // Here we have input (in_channels bits) x weight (in_channels bits). 
    // Each convolution window op is `in_channels` wide.
    // So 1 conv window = `in_channels` binary operations (XNORs) + accumulation.
    // Total ops = pixels * kernel_size * channels.
    
    // We will approximate: 
    // 1 tile call covers 1 tile of outputs.
    // It processes `in_channels` input channels.
    // It produces `OUT_CHANNEL_PARALLELISM` output channels.
    
    // MSB Phase (always on):
    unsigned long long ops_per_channel = (unsigned long long)H_fmap_out * H_fmap_out * 3 * 3 * in_channels;
    
    // LayerStats accumulation
    layer_stats[current_layer_id].msb_bmacs += (ops_per_channel * OUT_CHANNEL_PARALLELISM);
    
    // LSB Phase (depends on switch_on):
    // We count how many in switch_on are true.
    int active_lsb_channels = 0;
    for(int i=0; i<OUT_CHANNEL_PARALLELISM; i++) {
        if(switch_on[i]) active_lsb_channels++;
    }
    layer_stats[current_layer_id].lsb_bmacs += (ops_per_channel * active_lsb_channels);
#endif

    binary_conv3x3_tile(
        lsb_inputs, weights, lsb_outputs,
        msb_outputs,    // comparator (unused when switch_on=1)
        threshold,      // threshold (unused when switch_on=1)
        switch_on,
        c_in, in_channels, H_fmap_out
    );
}

#endif // PGCONV_H
