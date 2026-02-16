#include "bnn.h"
#include "layer.h"

using namespace std;

#ifndef __SYNTHESIS__
#include <iostream>
#endif


// --------------------------------------------------
// Global gate pointer for Adaptive PG
// --------------------------------------------------
// --------------------------------------------------
// Global gate pointer for Adaptive PG
// --------------------------------------------------
int gate_idx = 0;
int enabled_gates_count = 0;

#ifndef __SYNTHESIS__
LayerStats layer_stats[20];
int current_layer_id = 0;
int inference_mode_sw = 2; // Default to Adaptive

void init_layer_stats() {
    const char* names[] = {
        "Conv1",
        "L1_0_PG1", "L1_0_PG2", "L1_1_PG1", "L1_1_PG2", "L1_2_PG1", "L1_2_PG2",
        "L2_0_PG1", "L2_0_PG2", "L2_1_PG1", "L2_1_PG2", "L2_2_PG1", "L2_2_PG2",
        "L3_0_PG1", "L3_0_PG2", "L3_1_PG1", "L3_1_PG2", "L3_2_PG1", "L3_2_PG2"
    };
    for(int i=0; i<19; i++) {
        layer_stats[i].name = names[i];
        layer_stats[i].total_gates = 0;
        layer_stats[i].active_gates = 0;
        layer_stats[i].msb_bmacs = 0;
        layer_stats[i].lsb_bmacs = 0;
    }
}
#endif

//--------------------
//  Top Function 
//--------------------
void FracNet_T(
        uint64 image[3][32][32],
        float output[10]
)
{
    // 🔴 REQUIRED: reset gate index ONCE per inference
    // 🔴 REQUIRED: reset gate index ONCE per inference
    gate_idx = 0;
    enabled_gates_count = 0;
    
#ifndef __SYNTHESIS__
    init_layer_stats();
#endif

#ifndef __SYNTHESIS__
    cout << "Hardware: Starting FracNet_T" << endl;
#endif


#pragma HLS INTERFACE m_axi depth=3072 port=image offset=slave bundle=IMG
#pragma HLS INTERFACE m_axi depth=10 port=output offset=slave bundle=RESULT
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

#pragma HLS ALLOCATION instances=binary_conv3x3_tile limit=1 function
#pragma HLS ALLOCATION instances=bn_relu_shortcut limit=1 function
#pragma HLS ALLOCATION instances=quant_and_pack limit=1 function

    uint64 msb_fmap[3][WIDTH][WIDTH];
#pragma HLS ARRAY_PARTITION variable=msb_fmap complete dim=1
#pragma HLS ARRAY_PARTITION variable=lsb_fmap complete dim=1

    FIX_FM_acc out_buf_0[CHANNEL_OUT/CHANNEL_OUT_T][CHANNEL_OUT_T][WIDTH][WIDTH];
    int16 out_buf_t0[CHANNEL_OUT_T][WIDTH][WIDTH];
    int16 out_buf_t1[CHANNEL_OUT_T][WIDTH][WIDTH];
#pragma HLS ARRAY_PARTITION variable=out_buf_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_buf_0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=out_buf_t0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_buf_t1 complete dim=1

    /* Initialize the buffers to 0 */
    global_buffer_init_0:
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
#pragma HLS PIPELINE
            for (int k = 0; k < 3; k++) {
                msb_fmap[k][i][j] = 0;
            }
            for (int c = 0; c < CHANNEL_OUT/CHANNEL_OUT_T; c++) {
                for (int k = 0; k < CHANNEL_OUT_T; k++) {
                    out_buf_0[c][k][i][j] = 0;
                }
            }
            for (int k = 0; k < CHANNEL_OUT_T; k++) {
                out_buf_t0[k][i][j] = 0;
                out_buf_t1[k][i][j] = 0;
            }
        }
    }

	int H_fmap_in, H_fmap_out, in_channels, in_channels_after_pack; 
    int out_channels, out_channel_start, stride, conv_weight_ptr;



	////////////////////////////////////////////////
	//////////// GET IMAGE /////////////////////////
	////////////////////////////////////////////////

	LOOP_GetImg:
	for (int c = 0; c < 3; c ++) {
		for (int row = 0; row < 32; row ++) {
			for (int col = 0; col < 32; col ++) {
				//#pragma HLS PIPELINE
				msb_fmap[c][row][col] = image[c][row][col];
			}
		}
	}

	////////////////////////////////////////////////
	//////////// CONV 1 Binary /////////////////////
	////////////////////////////////////////////////

	in_channels = 32;
	in_channels_after_pack = 3;
	out_channels = 16;
	H_fmap_out = 32;
	conv_weight_ptr = 0;

#ifndef __SYNTHESIS__
    current_layer_id = 0; // Conv1
#endif
    LOOP_Conv1:
#ifndef __SYNTHESIS__
    cout << "Hardware: Executing LOOP_Conv1" << endl;
#endif

    for (int c_out = 0; c_out < out_channels/OUT_CHANNEL_PARALLELISM; c_out ++){
        int c_in = 0;
        pg_conv3x3_tile(
                msb_fmap[c_in], msb_fmap[(c_in+1)%CHANNEL_IN], conv_weight_all[conv_weight_ptr],
                out_buf_t0, out_buf_t1,
                layer1_0_conv1_threshold_fix[c_out],
                c_in, in_channels, H_fmap_out, true
        );
        conv_weight_ptr += 1;
        c_in = 1;
        pg_conv3x3_tile(
                msb_fmap[c_in], msb_fmap[(c_in+1)%CHANNEL_IN], conv_weight_all[conv_weight_ptr],
                out_buf_t0, out_buf_t1,
                layer1_0_conv1_threshold_fix[c_out],
                c_in, in_channels, H_fmap_out, true
        );
        conv_weight_ptr += 1;
        c_in = 2;
        pg_conv3x3_tile(
                msb_fmap[c_in], msb_fmap[(c_in+1)%CHANNEL_IN], conv_weight_all[conv_weight_ptr],
                out_buf_t0, out_buf_t1,
                layer1_0_conv1_threshold_fix[c_out],
                c_in, in_channels, H_fmap_out, true
        );
        conv_weight_ptr += 1;

        bn1(
                out_buf_0, out_buf_t0,
                bn1_weight_fix[c_out], bn1_bias_fix[c_out],
                stride, c_out, H_fmap_out
        );
    }


	////////////////////////////////////////////////
	//////////// LAYER 1 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 32;
	H_fmap_out = 32;
	in_channels = 16;
	in_channels_after_pack = 1;
	out_channels = 16;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer1_0 PG1 /////////////////////
	quant_and_pack(out_buf_0, msb_fmap, H_fmap_in, in_channels);
#ifndef __SYNTHESIS__
    current_layer_id = 1; // L1_0_PG1
#endif
	LOOP_layer1_0_PGConv1:
#ifndef __SYNTHESIS__
    cout << "Hardware: Executing Layer 1" << endl;
#endif

	for (int c_out = 0; c_out < out_channels/OUT_CHANNEL_PARALLELISM; c_out ++) {
		int c_in = 0;
		pg_conv3x3_tile(
				msb_fmap[c_in], msb_fmap[(c_in+1)%CHANNEL_IN], conv_weight_all[conv_weight_ptr],
				out_buf_t0, out_buf_t1,
                layer1_0_conv1_threshold_fix[c_out],
				c_in, in_channels, H_fmap_out, true
		);
		conv_weight_ptr += 1;
		bn_relu_shortcut(
				out_buf_0, out_buf_t0, out_buf_t1,

				layer1_0_bn1_weight_fix[c_out],
				layer1_0_bn3_weight_fix[c_out],
				layer1_0_bn1_bias_fix[c_out],
				layer1_0_bn3_bias_fix[c_out],
				layer1_0_rprelu1_shift_x_bias_fix[c_out],
				layer1_0_rprelu1_shift_y_bias_fix[c_out],
				layer1_0_rprelu1_prelu_weight_fix[c_out],

				stride, c_out, H_fmap_out, out_channels
		);
	}

	////////////////////////////////////////////////
	//////////// layer1_0 PG2 /////////////////////
	quant_and_pack(out_buf_0, msb_fmap, H_fmap_in, in_channels);
#ifndef __SYNTHESIS__
    current_layer_id = 2; // L1_0_PG2
#endif
	LOOP_layer1_0_PGConv2:
	for (int c_out = 0; c_out < out_channels/OUT_CHANNEL_PARALLELISM; c_out ++) {
		int c_in = 0;
		pg_conv3x3_tile(
				msb_fmap[c_in], msb_fmap[(c_in+1)%CHANNEL_IN], conv_weight_all[conv_weight_ptr],
				out_buf_t0, out_buf_t1,
				layer1_0_conv2_threshold_fix[c_out],
				c_in, in_channels, H_fmap_out, true
		);
		conv_weight_ptr += 1;
		bn_relu_shortcut(
				out_buf_0, out_buf_t0, out_buf_t1,

				layer1_0_bn2_weight_fix[c_out],
				layer1_0_bn4_weight_fix[c_out],
				layer1_0_bn2_bias_fix[c_out],
				layer1_0_bn4_bias_fix[c_out],
				layer1_0_rprelu2_shift_x_bias_fix[c_out],
				layer1_0_rprelu2_shift_y_bias_fix[c_out],
				layer1_0_rprelu2_prelu_weight_fix[c_out],

				stride, c_out, H_fmap_out, out_channels
		);
	}

	////////////////////////////////////////////////
	//////////// layer1_1 PG1 /////////////////////
	quant_and_pack(out_buf_0, msb_fmap, H_fmap_in, in_channels);
#ifndef __SYNTHESIS__
    current_layer_id = 3; // L1_1_PG1
#endif
	LOOP_layer1_1_PGConv1:
	for (int c_out = 0; c_out < out_channels/OUT_CHANNEL_PARALLELISM; c_out ++) {
		int c_in = 0;
		pg_conv3x3_tile(
				msb_fmap[c_in], msb_fmap[(c_in+1)%CHANNEL_IN], conv_weight_all[conv_weight_ptr],
				out_buf_t0, out_buf_t1,
				layer1_1_conv1_threshold_fix[c_out],
				c_in, in_channels, H_fmap_out, true
		);
		conv_weight_ptr += 1;
		bn_relu_shortcut(
				out_buf_0, out_buf_t0, out_buf_t1,

				layer1_1_bn1_weight_fix[c_out],
				layer1_1_bn3_weight_fix[c_out],
				layer1_1_bn1_bias_fix[c_out],
				layer1_1_bn3_bias_fix[c_out],
				layer1_1_rprelu1_shift_x_bias_fix[c_out],
				layer1_1_rprelu1_shift_y_bias_fix[c_out],
				layer1_1_rprelu1_prelu_weight_fix[c_out],

				stride, c_out, H_fmap_out, out_channels
		);
	}

	////////////////////////////////////////////////
	//////////// layer1_1 PG2 /////////////////////
	quant_and_pack(out_buf_0, msb_fmap, H_fmap_in, in_channels);
#ifndef __SYNTHESIS__
    current_layer_id = 4; // L1_1_PG2
#endif
	LOOP_layer1_1_PGConv2:
	for (int c_out = 0; c_out < out_channels/OUT_CHANNEL_PARALLELISM; c_out ++) {
		int c_in = 0;
		pg_conv3x3_tile(
				msb_fmap[c_in], msb_fmap[(c_in+1)%CHANNEL_IN], conv_weight_all[conv_weight_ptr],
				out_buf_t0, out_buf_t1,
				layer1_1_conv2_threshold_fix[c_out],
				c_in, in_channels, H_fmap_out, true
		);
		conv_weight_ptr += 1;
		bn_relu_shortcut(
				out_buf_0, out_buf_t0, out_buf_t1,

				layer1_1_bn2_weight_fix[c_out],
				layer1_1_bn4_weight_fix[c_out],
				layer1_1_bn2_bias_fix[c_out],
				layer1_1_bn4_bias_fix[c_out],
				layer1_1_rprelu2_shift_x_bias_fix[c_out],
				layer1_1_rprelu2_shift_y_bias_fix[c_out],
				layer1_1_rprelu2_prelu_weight_fix[c_out],

				stride, c_out, H_fmap_out, out_channels
		);
	}

	////////////////////////////////////////////////
	//////////// layer1_2 PG1 /////////////////////
	quant_and_pack(out_buf_0, msb_fmap, H_fmap_in, in_channels);
#ifndef __SYNTHESIS__
    current_layer_id = 5; // L1_2_PG1
#endif
	LOOP_layer1_2_PGConv1:
	for (int c_out = 0; c_out < out_channels/OUT_CHANNEL_PARALLELISM; c_out ++) {
		int c_in = 0;
		pg_conv3x3_tile(
				msb_fmap[c_in], msb_fmap[(c_in+1)%CHANNEL_IN], conv_weight_all[conv_weight_ptr],
				out_buf_t0, out_buf_t1,
				layer1_2_conv1_threshold_fix[c_out],
				c_in, in_channels, H_fmap_out, true
		);
		conv_weight_ptr += 1;
		bn_relu_shortcut(
				out_buf_0, out_buf_t0, out_buf_t1,

				layer1_2_bn1_weight_fix[c_out],
				layer1_2_bn3_weight_fix[c_out],
				layer1_2_bn1_bias_fix[c_out],
				layer1_2_bn3_bias_fix[c_out],
				layer1_2_rprelu1_shift_x_bias_fix[c_out],
				layer1_2_rprelu1_shift_y_bias_fix[c_out],
				layer1_2_rprelu1_prelu_weight_fix[c_out],

				stride, c_out, H_fmap_out, out_channels
		);
	}

	////////////////////////////////////////////////
	//////////// layer1_2 PG2 /////////////////////
	quant_and_pack(out_buf_0, msb_fmap, H_fmap_in, in_channels);
#ifndef __SYNTHESIS__
    current_layer_id = 6; // L1_2_PG2
#endif
	LOOP_layer1_2_PGConv2:
	for (int c_out = 0; c_out < out_channels/OUT_CHANNEL_PARALLELISM; c_out ++) {
		int c_in = 0;
		pg_conv3x3_tile(
				msb_fmap[c_in], msb_fmap[(c_in+1)%CHANNEL_IN], conv_weight_all[conv_weight_ptr],
				out_buf_t0, out_buf_t1,
				layer1_2_conv2_threshold_fix[c_out],
				c_in, in_channels, H_fmap_out, true
		);
		conv_weight_ptr += 1;
		bn_relu_shortcut(
				out_buf_0, out_buf_t0, out_buf_t1,

				layer1_2_bn2_weight_fix[c_out],
				layer1_2_bn4_weight_fix[c_out],
				layer1_2_bn2_bias_fix[c_out],
				layer1_2_bn4_bias_fix[c_out],
				layer1_2_rprelu2_shift_x_bias_fix[c_out],
				layer1_2_rprelu2_shift_y_bias_fix[c_out],
				layer1_2_rprelu2_prelu_weight_fix[c_out],

				stride, c_out, H_fmap_out, out_channels
		);
	}

	////////////////////////////////////////////////
	//////////// LAYER 2 Downsample ////////////////
	////////////////////////////////////////////////

	H_fmap_in = 32;
	H_fmap_out = 16;
	in_channels = 16;
	in_channels_after_pack = 1;
	out_channels = 32;
	stride = 2;

	////////////////////////////////////////////////
	//////////// layer2_0 PG1 /////////////////////
	quant_and_pack(out_buf_0, msb_fmap, H_fmap_in, in_channels);
	avgpool_concat(out_buf_0, H_fmap_out, in_channels);
#ifndef __SYNTHESIS__
    current_layer_id = 7; // L2_0_PG1
#endif
	LOOP_layer2_0_PGConv1:
#ifndef __SYNTHESIS__
    cout << "Hardware: Executing Layer 2" << endl;
#endif

	for (int c_out = 0; c_out < out_channels/OUT_CHANNEL_PARALLELISM; c_out ++) {
		int c_in = 0;
		pg_conv3x3_tile(
				msb_fmap[c_in], msb_fmap[(c_in+1)%CHANNEL_IN], conv_weight_all[conv_weight_ptr],
				out_buf_t0, out_buf_t1,
				layer2_0_conv1_threshold_fix[c_out],
				c_in, in_channels, H_fmap_in, true
		);
		conv_weight_ptr += 1;
		bn_relu_shortcut(
				out_buf_0, out_buf_t0, out_buf_t1,

				layer2_0_bn1_weight_fix[c_out],
				layer2_0_bn3_weight_fix[c_out],
				layer2_0_bn1_bias_fix[c_out],
				layer2_0_bn3_bias_fix[c_out],
				layer2_0_rprelu1_shift_x_bias_fix[c_out],
				layer2_0_rprelu1_shift_y_bias_fix[c_out],
				layer2_0_rprelu1_prelu_weight_fix[c_out],

				stride, c_out, H_fmap_out, out_channels
		);
	}

	////////////////////////////////////////////////
	//////////// LAYER 2 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 16;
	H_fmap_out = 16;
	in_channels = 32;
	in_channels_after_pack = 1;
	out_channels = 32;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer2_0 PG2 /////////////////////
	quant_and_pack(out_buf_0, msb_fmap, H_fmap_in, in_channels);
#ifndef __SYNTHESIS__
    current_layer_id = 8; // L2_0_PG2
#endif
	LOOP_layer2_0_PGConv2:
	for (int c_out = 0; c_out < out_channels/OUT_CHANNEL_PARALLELISM; c_out ++) {
		int c_in = 0;
		pg_conv3x3_tile(
				msb_fmap[c_in], msb_fmap[(c_in+1)%CHANNEL_IN], conv_weight_all[conv_weight_ptr],
				out_buf_t0, out_buf_t1,
				layer2_0_conv2_threshold_fix[c_out],
				c_in, in_channels, H_fmap_out, true
		);
		conv_weight_ptr += 1;
		bn_relu_shortcut(
				out_buf_0, out_buf_t0, out_buf_t1,

				layer2_0_bn2_weight_fix[c_out],
				layer2_0_bn4_weight_fix[c_out],
				layer2_0_bn2_bias_fix[c_out],
				layer2_0_bn4_bias_fix[c_out],
				layer2_0_rprelu2_shift_x_bias_fix[c_out],
				layer2_0_rprelu2_shift_y_bias_fix[c_out],
				layer2_0_rprelu2_prelu_weight_fix[c_out],

				stride, c_out, H_fmap_out, out_channels
		);
	}

	////////////////////////////////////////////////
	//////////// layer2_1 PG1 /////////////////////
	quant_and_pack(out_buf_0, msb_fmap, H_fmap_in, in_channels);
#ifndef __SYNTHESIS__
    current_layer_id = 9; // L2_1_PG1
#endif
	LOOP_layer2_1_PGConv1:
	for (int c_out = 0; c_out < out_channels/OUT_CHANNEL_PARALLELISM; c_out ++) {
		int c_in = 0;
		pg_conv3x3_tile(
				msb_fmap[c_in], msb_fmap[(c_in+1)%CHANNEL_IN], conv_weight_all[conv_weight_ptr],
				out_buf_t0, out_buf_t1,
				layer2_1_conv1_threshold_fix[c_out],
				c_in, in_channels, H_fmap_out, true
		);
		conv_weight_ptr += 1;
		bn_relu_shortcut(
				out_buf_0, out_buf_t0, out_buf_t1,

				layer2_1_bn1_weight_fix[c_out],
				layer2_1_bn3_weight_fix[c_out],
				layer2_1_bn1_bias_fix[c_out],
				layer2_1_bn3_bias_fix[c_out],
				layer2_1_rprelu1_shift_x_bias_fix[c_out],
				layer2_1_rprelu1_shift_y_bias_fix[c_out],
				layer2_1_rprelu1_prelu_weight_fix[c_out],


				stride, c_out, H_fmap_out, out_channels
		);
	}

	////////////////////////////////////////////////
	//////////// layer2_1 PG2 /////////////////////
	quant_and_pack(out_buf_0, msb_fmap, H_fmap_in, in_channels);
#ifndef __SYNTHESIS__
    current_layer_id = 10; // L2_1_PG2
#endif
	LOOP_layer2_1_PGConv2:
	for (int c_out = 0; c_out < out_channels/OUT_CHANNEL_PARALLELISM; c_out ++) {
		int c_in = 0;
		pg_conv3x3_tile(
				msb_fmap[c_in], msb_fmap[(c_in+1)%CHANNEL_IN], conv_weight_all[conv_weight_ptr],
				out_buf_t0, out_buf_t1,
				layer2_1_conv2_threshold_fix[c_out],
				c_in, in_channels, H_fmap_out, true
		);
		conv_weight_ptr += 1;
		bn_relu_shortcut(
				out_buf_0, out_buf_t0, out_buf_t1,

				layer2_1_bn2_weight_fix[c_out],
				layer2_1_bn4_weight_fix[c_out],
				layer2_1_bn2_bias_fix[c_out],
				layer2_1_bn4_bias_fix[c_out],
				layer2_1_rprelu2_shift_x_bias_fix[c_out],
				layer2_1_rprelu2_shift_y_bias_fix[c_out],
				layer2_1_rprelu2_prelu_weight_fix[c_out],

				stride, c_out, H_fmap_out, out_channels
		);
	}

	////////////////////////////////////////////////
	//////////// layer2_2 PG1 /////////////////////
	quant_and_pack(out_buf_0, msb_fmap, H_fmap_in, in_channels);
#ifndef __SYNTHESIS__
    current_layer_id = 11; // L2_2_PG1
#endif
	LOOP_layer2_2_PGConv1:
	for (int c_out = 0; c_out < out_channels/OUT_CHANNEL_PARALLELISM; c_out ++) {
		int c_in = 0;
		pg_conv3x3_tile(
				msb_fmap[c_in], msb_fmap[(c_in+1)%CHANNEL_IN], conv_weight_all[conv_weight_ptr],
				out_buf_t0, out_buf_t1,
				layer2_2_conv1_threshold_fix[c_out],
				c_in, in_channels, H_fmap_out, true
		);
		conv_weight_ptr += 1;
		bn_relu_shortcut(
				out_buf_0, out_buf_t0, out_buf_t1,

				layer2_2_bn1_weight_fix[c_out],
				layer2_2_bn3_weight_fix[c_out],
				layer2_2_bn1_bias_fix[c_out],
				layer2_2_bn3_bias_fix[c_out],
				layer2_2_rprelu1_shift_x_bias_fix[c_out],
				layer2_2_rprelu1_shift_y_bias_fix[c_out],
				layer2_2_rprelu1_prelu_weight_fix[c_out],


				stride, c_out, H_fmap_out, out_channels
		);
	}

	////////////////////////////////////////////////
	//////////// layer2_2 PG2 /////////////////////
	quant_and_pack(out_buf_0, msb_fmap, H_fmap_in, in_channels);
#ifndef __SYNTHESIS__
    current_layer_id = 12; // L2_2_PG2
#endif
	LOOP_layer2_2_PGConv2:
	for (int c_out = 0; c_out < out_channels/OUT_CHANNEL_PARALLELISM; c_out ++) {
		int c_in = 0;
		pg_conv3x3_tile(
				msb_fmap[c_in], msb_fmap[(c_in+1)%CHANNEL_IN], conv_weight_all[conv_weight_ptr],
				out_buf_t0, out_buf_t1,
				layer2_2_conv2_threshold_fix[c_out],
				c_in, in_channels, H_fmap_out, true
		);
		conv_weight_ptr += 1;
		bn_relu_shortcut(
				out_buf_0, out_buf_t0, out_buf_t1,

				layer2_2_bn2_weight_fix[c_out],
				layer2_2_bn4_weight_fix[c_out],
				layer2_2_bn2_bias_fix[c_out],
				layer2_2_bn4_bias_fix[c_out],
				layer2_2_rprelu2_shift_x_bias_fix[c_out],
				layer2_2_rprelu2_shift_y_bias_fix[c_out],
				layer2_2_rprelu2_prelu_weight_fix[c_out],

				stride, c_out, H_fmap_out, out_channels
		);
	}

	////////////////////////////////////////////////
	//////////// LAYER 3 Downsample ////////////////
	////////////////////////////////////////////////

	H_fmap_in = 16;
	H_fmap_out = 8;
	in_channels = 32;
	in_channels_after_pack = 1;
	out_channels = 64;
	stride = 2;

	////////////////////////////////////////////////
	//////////// layer3_0 PG1 /////////////////////
	quant_and_pack(out_buf_0, msb_fmap, H_fmap_in, in_channels);
	avgpool_concat(out_buf_0, H_fmap_out, in_channels);
#ifndef __SYNTHESIS__
    current_layer_id = 13; // L3_0_PG1
#endif
	LOOP_layer3_0_PGConv1:
#ifndef __SYNTHESIS__
    cout << "Hardware: Executing Layer 3" << endl;
#endif

	for (int c_out = 0; c_out < out_channels/OUT_CHANNEL_PARALLELISM; c_out ++) {
		int c_in = 0;
		pg_conv3x3_tile(
				msb_fmap[c_in], msb_fmap[(c_in+1)%CHANNEL_IN], conv_weight_all[conv_weight_ptr],
				out_buf_t0, out_buf_t1,
				layer3_0_conv1_threshold_fix[c_out],
				c_in, in_channels, H_fmap_in, true
		);
		conv_weight_ptr += 1;
		bn_relu_shortcut(
				out_buf_0, out_buf_t0, out_buf_t1,

				layer3_0_bn1_weight_fix[c_out],
				layer3_0_bn3_weight_fix[c_out],
				layer3_0_bn1_bias_fix[c_out],
				layer3_0_bn3_bias_fix[c_out],
				layer3_0_rprelu1_shift_x_bias_fix[c_out],
				layer3_0_rprelu1_shift_y_bias_fix[c_out],
				layer3_0_rprelu1_prelu_weight_fix[c_out],

				stride, c_out, H_fmap_out, out_channels
		);
	}

	////////////////////////////////////////////////
	//////////// LAYER 3 ///////////////////////////
	////////////////////////////////////////////////

	H_fmap_in = 8;
	H_fmap_out = 8;
	in_channels = 64;
	in_channels_after_pack = 1;
	out_channels = 64;
	stride = 1;

	////////////////////////////////////////////////
	//////////// layer3_0 PG2 /////////////////////
	quant_and_pack(out_buf_0, msb_fmap, H_fmap_in, in_channels);
#ifndef __SYNTHESIS__
    current_layer_id = 14; // L3_0_PG2
#endif
	LOOP_layer3_0_PGConv2:
	for (int c_out = 0; c_out < out_channels/OUT_CHANNEL_PARALLELISM; c_out ++) {
		int c_in = 0;
		pg_conv3x3_tile(
				msb_fmap[c_in], msb_fmap[(c_in+1)%CHANNEL_IN], conv_weight_all[conv_weight_ptr],
				out_buf_t0, out_buf_t1,
				layer3_0_conv2_threshold_fix[c_out],
				c_in, in_channels, H_fmap_out, true
		);
		conv_weight_ptr += 1;
		bn_relu_shortcut(
				out_buf_0, out_buf_t0, out_buf_t1,

				layer3_0_bn2_weight_fix[c_out],
				layer3_0_bn4_weight_fix[c_out],
				layer3_0_bn2_bias_fix[c_out],
				layer3_0_bn4_bias_fix[c_out],
				layer3_0_rprelu2_shift_x_bias_fix[c_out],
				layer3_0_rprelu2_shift_y_bias_fix[c_out],
				layer3_0_rprelu2_prelu_weight_fix[c_out],

				stride, c_out, H_fmap_out, out_channels
		);
	}

	////////////////////////////////////////////////
	//////////// layer3_1 PG1 /////////////////////
	quant_and_pack(out_buf_0, msb_fmap, H_fmap_in, in_channels);
#ifndef __SYNTHESIS__
    current_layer_id = 15; // L3_1_PG1
#endif
	LOOP_layer3_1_PGConv1:
	for (int c_out = 0; c_out < out_channels/OUT_CHANNEL_PARALLELISM; c_out ++) {
		int c_in = 0;
		pg_conv3x3_tile(
				msb_fmap[c_in], msb_fmap[(c_in+1)%CHANNEL_IN], conv_weight_all[conv_weight_ptr],
				out_buf_t0, out_buf_t1,
				layer3_1_conv1_threshold_fix[c_out],
				c_in, in_channels, H_fmap_out, true
		);
		conv_weight_ptr += 1;
		bn_relu_shortcut(
				out_buf_0, out_buf_t0, out_buf_t1,

				layer3_1_bn1_weight_fix[c_out],
				layer3_1_bn3_weight_fix[c_out],
				layer3_1_bn1_bias_fix[c_out],
				layer3_1_bn3_bias_fix[c_out],
				layer3_1_rprelu1_shift_x_bias_fix[c_out],
				layer3_1_rprelu1_shift_y_bias_fix[c_out],
				layer3_1_rprelu1_prelu_weight_fix[c_out],


				stride, c_out, H_fmap_out, out_channels
		);
	}

	////////////////////////////////////////////////
	//////////// layer3_1 PG2 /////////////////////
	quant_and_pack(out_buf_0, msb_fmap, H_fmap_in, in_channels);
#ifndef __SYNTHESIS__
    current_layer_id = 16; // L3_1_PG2
#endif
	LOOP_layer3_1_PGConv2:
	for (int c_out = 0; c_out < out_channels/OUT_CHANNEL_PARALLELISM; c_out ++) {
		int c_in = 0;
		pg_conv3x3_tile(
				msb_fmap[c_in], msb_fmap[(c_in+1)%CHANNEL_IN], conv_weight_all[conv_weight_ptr],
				out_buf_t0, out_buf_t1,
				layer3_1_conv2_threshold_fix[c_out],
				c_in, in_channels, H_fmap_out, true
		);
		conv_weight_ptr += 1;
		bn_relu_shortcut(
				out_buf_0, out_buf_t0, out_buf_t1,

				layer3_1_bn2_weight_fix[c_out],
				layer3_1_bn4_weight_fix[c_out],
				layer3_1_bn2_bias_fix[c_out],
				layer3_1_bn4_bias_fix[c_out],
				layer3_1_rprelu2_shift_x_bias_fix[c_out],
				layer3_1_rprelu2_shift_y_bias_fix[c_out],
				layer3_1_rprelu2_prelu_weight_fix[c_out],

				stride, c_out, H_fmap_out, out_channels
		);
	}

	////////////////////////////////////////////////
	//////////// layer3_2 PG1 /////////////////////
	quant_and_pack(out_buf_0, msb_fmap, H_fmap_in, in_channels);
#ifndef __SYNTHESIS__
    current_layer_id = 17; // L3_2_PG1
#endif
	LOOP_layer3_2_PGConv1:
	for (int c_out = 0; c_out < out_channels/OUT_CHANNEL_PARALLELISM; c_out ++) {
		int c_in = 0;
		pg_conv3x3_tile(
				msb_fmap[c_in], msb_fmap[(c_in+1)%CHANNEL_IN], conv_weight_all[conv_weight_ptr],
				out_buf_t0, out_buf_t1,
				layer3_2_conv1_threshold_fix[c_out],
				c_in, in_channels, H_fmap_out, true
		);
		conv_weight_ptr += 1;
		bn_relu_shortcut(
				out_buf_0, out_buf_t0, out_buf_t1,

				layer3_2_bn1_weight_fix[c_out],
				layer3_2_bn3_weight_fix[c_out],
				layer3_2_bn1_bias_fix[c_out],
				layer3_2_bn3_bias_fix[c_out],
				layer3_2_rprelu1_shift_x_bias_fix[c_out],
				layer3_2_rprelu1_shift_y_bias_fix[c_out],
				layer3_2_rprelu1_prelu_weight_fix[c_out],


				stride, c_out, H_fmap_out, out_channels
		);
	}

	////////////////////////////////////////////////
	//////////// layer3_2 PG2 /////////////////////
	quant_and_pack(out_buf_0, msb_fmap, H_fmap_in, in_channels);
#ifndef __SYNTHESIS__
    current_layer_id = 18; // L3_2_PG2
#endif
	LOOP_layer3_2_PGConv2:
	for (int c_out = 0; c_out < out_channels/OUT_CHANNEL_PARALLELISM; c_out ++) {
		int c_in = 0;
		pg_conv3x3_tile(
				msb_fmap[c_in], msb_fmap[(c_in+1)%CHANNEL_IN], conv_weight_all[conv_weight_ptr],
				out_buf_t0, out_buf_t1,
				layer3_2_conv2_threshold_fix[c_out],
				c_in, in_channels, H_fmap_out, true
		);
		conv_weight_ptr += 1;
		bn_relu_shortcut(
				out_buf_0, out_buf_t0, out_buf_t1,

				layer3_2_bn2_weight_fix[c_out],
				layer3_2_bn4_weight_fix[c_out],
				layer3_2_bn2_bias_fix[c_out],
				layer3_2_bn4_bias_fix[c_out],
				layer3_2_rprelu2_shift_x_bias_fix[c_out],
				layer3_2_rprelu2_shift_y_bias_fix[c_out],
				layer3_2_rprelu2_prelu_weight_fix[c_out],

				stride, c_out, H_fmap_out, out_channels
		);
	}


    /* Initialize the buffers for pooling and FC layer  */
	FIX_32_10 pool_out_buf[64];
	FIX_32_10 linear_out_buf[10];
	pool_out_buf_init:
	for (int i = 0; i < 64; i ++){
		pool_out_buf[i] = 0;
	}

	linear_out_buf_init:
	for (int i = 0; i < 10; i ++){
		linear_out_buf[i] = 0;
	}


	avgpool_8x8(out_buf_0, pool_out_buf);
#ifndef __SYNTHESIS__
    cout << "Hardware: Executing AvgPool and FC" << endl;
#endif

	matmul(pool_out_buf, linear_weight_fix, linear_bias_fix, linear_out_buf);

    write_output:
	for(int i=0; i<10; i++){
		output[i] = linear_out_buf[i];
	}

#ifndef __SYNTHESIS__
    cout << "Total LSB masks used (enabled): " << enabled_gates_count << " / " << gate_idx << endl;
    
    // Task B & C Report
    cout << "\n========================================================" << endl;
    cout << "           Ada-FracBNN Evaluation Report" << endl;
    cout << "           Mode: " << INFERENCE_MODE << " (0:BIN, 1:FRAC, 2:ADAPTIVE)" << endl;
    cout << "========================================================" << endl;
    cout << "Layer Name | Total Gates | Active Gates | Sparsity(%) | MSB BMACs | LSB BMACs | Eff. Bits" << endl;
    cout << "---------------------------------------------------------------------------------------" << endl;
    
    unsigned long long grand_total_msb = 0;
    unsigned long long grand_total_lsb = 0;
    
    for(int i=0; i<19; i++) {
        double sparsity = 0;
        if(layer_stats[i].total_gates > 0)
            sparsity = 100.0 * (double)layer_stats[i].active_gates / (double)layer_stats[i].total_gates;
            
        double eff_bits = 1.0;
        if(layer_stats[i].msb_bmacs > 0)
            eff_bits = 1.0 + (double)layer_stats[i].lsb_bmacs / (double)layer_stats[i].msb_bmacs;
            
        printf("%-10s | %11llu | %12llu | %11.2f | %9llu | %9llu | %9.2f\n", 
            layer_stats[i].name, 
            layer_stats[i].total_gates, 
            layer_stats[i].active_gates, 
            sparsity,
            layer_stats[i].msb_bmacs,
            layer_stats[i].lsb_bmacs,
            eff_bits);
            
        grand_total_msb += layer_stats[i].msb_bmacs;
        grand_total_lsb += layer_stats[i].lsb_bmacs;
    }
    cout << "---------------------------------------------------------------------------------------" << endl;
    double total_eff_bits = 1.0 + (double)grand_total_lsb / (double)grand_total_msb;
    cout << "TOTAL EFFECTIVE PRECISION: " << total_eff_bits << " bits" << endl;
    cout << "========================================================" << endl;
#endif

}
