#include "FracNet_T.h"
#include "AESL_pkg.h"

using namespace std;

namespace ap_rtl {

void FracNet_T::thread_ap_clk_no_reset_() {
    if ( ap_rst_n_inv.read() == ap_const_logic_1) {
        ap_CS_fsm = ap_ST_fsm_state1;
    } else {
        ap_CS_fsm = ap_NS_fsm.read();
    }
    if ( ap_rst_n_inv.read() == ap_const_logic_1) {
        ap_enable_reg_pp0_iter0 = ap_const_logic_0;
    } else {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_pp0_stage0.read()) && 
             esl_seteq<1,1,1>(ap_block_pp0_stage0_subdone.read(), ap_const_boolean_0) && 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_condition_pp0_exit_iter0_state2.read()))) {
            ap_enable_reg_pp0_iter0 = ap_const_logic_0;
        } else if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state1.read()) && 
                    esl_seteq<1,1,1>(ap_start.read(), ap_const_logic_1))) {
            ap_enable_reg_pp0_iter0 = ap_const_logic_1;
        }
    }
    if ( ap_rst_n_inv.read() == ap_const_logic_1) {
        ap_enable_reg_pp0_iter1 = ap_const_logic_0;
    } else {
        if ((esl_seteq<1,1,1>(ap_block_pp0_stage0_subdone.read(), ap_const_boolean_0) && 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_condition_pp0_exit_iter0_state2.read()))) {
            ap_enable_reg_pp0_iter1 = (ap_condition_pp0_exit_iter0_state2.read() ^ ap_const_logic_1);
        } else if (esl_seteq<1,1,1>(ap_block_pp0_stage0_subdone.read(), ap_const_boolean_0)) {
            ap_enable_reg_pp0_iter1 = ap_enable_reg_pp0_iter0.read();
        } else if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state1.read()) && 
                    esl_seteq<1,1,1>(ap_start.read(), ap_const_logic_1))) {
            ap_enable_reg_pp0_iter1 = ap_const_logic_0;
        }
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state4.read())) {
        c33_0_reg_5625 = ap_const_lv2_0;
    } else if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state6.read()) && 
                esl_seteq<1,1,1>(icmp_ln73_fu_19305_p2.read(), ap_const_lv1_1))) {
        c33_0_reg_5625 = c_reg_20570.read();
    }
    if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state89.read()) && 
         esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1))) {
        c_out46_0_reg_5672 = c_out_reg_20637.read();
    } else if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state80.read()) && 
                esl_seteq<1,1,1>(grp_avgpool_concat_fu_18852_ap_done.read(), ap_const_logic_1))) {
        c_out46_0_reg_5672 = ap_const_lv2_0;
    }
    if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state99.read()))) {
        c_out48_0_reg_5696 = c_out_1_reg_20693.read();
    } else if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state90.read()) && 
                esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1))) {
        c_out48_0_reg_5696 = ap_const_lv2_0;
    }
    if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state109.read()))) {
        c_out50_0_reg_5719 = c_out_2_reg_20749.read();
    } else if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && 
                esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state100.read()))) {
        c_out50_0_reg_5719 = ap_const_lv2_0;
    }
    if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state119.read()))) {
        c_out52_0_reg_5743 = c_out_3_reg_20810.read();
    } else if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && 
                esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state110.read()))) {
        c_out52_0_reg_5743 = ap_const_lv2_0;
    }
    if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state129.read()))) {
        c_out54_0_reg_5767 = c_out_4_reg_20866.read();
    } else if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && 
                esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state120.read()))) {
        c_out54_0_reg_5767 = ap_const_lv2_0;
    }
    if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state139.read()))) {
        c_out56_0_reg_5791 = c_out_5_reg_20922.read();
    } else if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && 
                esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state130.read()))) {
        c_out56_0_reg_5791 = ap_const_lv2_0;
    }
    if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state151.read()))) {
        c_out58_0_reg_5815 = c_out_6_reg_20978.read();
    } else if ((esl_seteq<1,1,1>(grp_avgpool_concat_fu_18852_ap_done.read(), ap_const_logic_1) && 
                esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state142.read()))) {
        c_out58_0_reg_5815 = ap_const_lv3_0;
    }
    if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state161.read()))) {
        c_out60_0_reg_5838 = c_out_7_reg_21021.read();
    } else if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && 
                esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state152.read()))) {
        c_out60_0_reg_5838 = ap_const_lv3_0;
    }
    if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state171.read()))) {
        c_out62_0_reg_5862 = c_out_8_reg_21069.read();
    } else if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && 
                esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state162.read()))) {
        c_out62_0_reg_5862 = ap_const_lv3_0;
    }
    if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state181.read()))) {
        c_out64_0_reg_5886 = c_out_9_reg_21112.read();
    } else if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && 
                esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state172.read()))) {
        c_out64_0_reg_5886 = ap_const_lv3_0;
    }
    if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state191.read()))) {
        c_out66_0_reg_5910 = c_out_10_reg_21155.read();
    } else if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && 
                esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state182.read()))) {
        c_out66_0_reg_5910 = ap_const_lv3_0;
    }
    if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state201.read()))) {
        c_out68_0_reg_5934 = c_out_11_reg_21198.read();
    } else if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && 
                esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state192.read()))) {
        c_out68_0_reg_5934 = ap_const_lv3_0;
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state13.read())) {
        col_0_reg_5649 = ap_const_lv6_0;
    } else if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state15.read())) {
        col_0_reg_5649 = col_reg_20607.read();
    }
    if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state89.read()) && 
         esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1))) {
        gate_idx_0_reg_5660 = add_ln166_reg_20680.read();
    } else if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state80.read()) && 
                esl_seteq<1,1,1>(grp_avgpool_concat_fu_18852_ap_done.read(), ap_const_logic_1))) {
        gate_idx_0_reg_5660 = ap_const_lv4_9;
    }
    if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state191.read()))) {
        gate_idx_10_reg_5898 = conv_weight_ptr_8_reg_21185.read();
    } else if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && 
                esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state182.read()))) {
        gate_idx_10_reg_5898 = ap_const_lv6_25;
    }
    if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state201.read()))) {
        gate_idx_11_reg_5922 = conv_weight_ptr_9_reg_21228.read();
    } else if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && 
                esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state192.read()))) {
        gate_idx_11_reg_5922 = ap_const_lv6_29;
    }
    if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state99.read()))) {
        gate_idx_1_reg_5684 = add_ln166_1_reg_20736.read();
    } else if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state90.read()) && 
                esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1))) {
        gate_idx_1_reg_5684 = ap_const_lv4_B;
    }
    if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state109.read()))) {
        gate_idx_2_reg_5708 = conv_weight_ptr_reg_20764.read();
    } else if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && 
                esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state100.read()))) {
        gate_idx_2_reg_5708 = ap_const_lv3_5;
    }
    if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state119.read()))) {
        gate_idx_3_reg_5731 = conv_weight_ptr_1_reg_20853.read();
    } else if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && 
                esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state110.read()))) {
        gate_idx_3_reg_5731 = ap_const_lv5_F;
    }
    if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state129.read()))) {
        gate_idx_4_reg_5755 = conv_weight_ptr_2_reg_20909.read();
    } else if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && 
                esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state120.read()))) {
        gate_idx_4_reg_5755 = ap_const_lv5_11;
    }
    if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state139.read()))) {
        gate_idx_5_reg_5779 = conv_weight_ptr_3_reg_20965.read();
    } else if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && 
                esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state130.read()))) {
        gate_idx_5_reg_5779 = ap_const_lv5_13;
    }
    if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state151.read()))) {
        gate_idx_6_reg_5803 = conv_weight_ptr_4_reg_21008.read();
    } else if ((esl_seteq<1,1,1>(grp_avgpool_concat_fu_18852_ap_done.read(), ap_const_logic_1) && 
                esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state142.read()))) {
        gate_idx_6_reg_5803 = ap_const_lv5_15;
    }
    if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state161.read()))) {
        gate_idx_7_reg_5827 = conv_weight_ptr_5_reg_21036.read();
    } else if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && 
                esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state152.read()))) {
        gate_idx_7_reg_5827 = ap_const_lv4_9;
    }
    if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state171.read()))) {
        gate_idx_8_reg_5850 = conv_weight_ptr_6_reg_21099.read();
    } else if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && 
                esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state162.read()))) {
        gate_idx_8_reg_5850 = ap_const_lv6_1D;
    }
    if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state181.read()))) {
        gate_idx_9_reg_5874 = conv_weight_ptr_7_reg_21142.read();
    } else if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && 
                esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state172.read()))) {
        gate_idx_9_reg_5874 = ap_const_lv6_21;
    }
    if ( ap_rst_n_inv.read() == ap_const_logic_1) {
        grp_avgpool_8x8_fu_18926_ap_start_reg = ap_const_logic_0;
    } else {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state202.read()) && 
             esl_seteq<1,1,1>(ap_const_lv1_1, icmp_ln701_fu_20239_p2.read()))) {
            grp_avgpool_8x8_fu_18926_ap_start_reg = ap_const_logic_1;
        } else if (esl_seteq<1,1,1>(ap_const_logic_1, grp_avgpool_8x8_fu_18926_ap_ready.read())) {
            grp_avgpool_8x8_fu_18926_ap_start_reg = ap_const_logic_0;
        }
    }
    if ( ap_rst_n_inv.read() == ap_const_logic_1) {
        grp_avgpool_concat_fu_18852_ap_start_reg = ap_const_logic_0;
    } else {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state79.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state141.read()))) {
            grp_avgpool_concat_fu_18852_ap_start_reg = ap_const_logic_1;
        } else if (esl_seteq<1,1,1>(ap_const_logic_1, grp_avgpool_concat_fu_18852_ap_ready.read())) {
            grp_avgpool_concat_fu_18852_ap_start_reg = ap_const_logic_0;
        }
    }
    if ( ap_rst_n_inv.read() == ap_const_logic_1) {
        grp_binary_conv3x3_tile_fu_5968_ap_start_reg = ap_const_logic_0;
    } else {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state84.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state94.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state104.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state114.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state124.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state134.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state146.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state156.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state166.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state176.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state186.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state196.read()) || 
             (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state5.read()) && 
              esl_seteq<1,1,1>(icmp_ln72_fu_19281_p2.read(), ap_const_lv1_1)) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state17.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state19.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state21.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state23.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state25.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state31.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state33.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state39.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state41.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state47.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state49.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state55.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state57.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state63.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state65.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state71.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state73.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state86.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state96.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state106.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state116.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state126.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state136.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state148.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state158.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state168.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state178.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state188.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state198.read()))) {
            grp_binary_conv3x3_tile_fu_5968_ap_start_reg = ap_const_logic_1;
        } else if (esl_seteq<1,1,1>(ap_const_logic_1, grp_binary_conv3x3_tile_fu_5968_ap_ready.read())) {
            grp_binary_conv3x3_tile_fu_5968_ap_start_reg = ap_const_logic_0;
        }
    }
    if ( ap_rst_n_inv.read() == ap_const_logic_1) {
        grp_bn1_fu_18996_ap_start_reg = ap_const_logic_0;
    } else {
        if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state27.read())) {
            grp_bn1_fu_18996_ap_start_reg = ap_const_logic_1;
        } else if (esl_seteq<1,1,1>(ap_const_logic_1, grp_bn1_fu_18996_ap_ready.read())) {
            grp_bn1_fu_18996_ap_start_reg = ap_const_logic_0;
        }
    }
    if ( ap_rst_n_inv.read() == ap_const_logic_1) {
        grp_bn_relu_shortcut_fu_8106_ap_start_reg = ap_const_logic_0;
    } else {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state35.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state43.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state51.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state59.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state67.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state75.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state88.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state98.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state108.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state118.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state128.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state138.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state150.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state160.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state170.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state180.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state190.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state200.read()))) {
            grp_bn_relu_shortcut_fu_8106_ap_start_reg = ap_const_logic_1;
        } else if (esl_seteq<1,1,1>(ap_const_logic_1, grp_bn_relu_shortcut_fu_8106_ap_ready.read())) {
            grp_bn_relu_shortcut_fu_8106_ap_start_reg = ap_const_logic_0;
        }
    }
    if ( ap_rst_n_inv.read() == ap_const_logic_1) {
        grp_matmul_fu_19032_ap_start_reg = ap_const_logic_0;
    } else {
        if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state204.read())) {
            grp_matmul_fu_19032_ap_start_reg = ap_const_logic_1;
        } else if (esl_seteq<1,1,1>(ap_const_logic_1, grp_matmul_fu_19032_ap_ready.read())) {
            grp_matmul_fu_19032_ap_start_reg = ap_const_logic_0;
        }
    }
    if ( ap_rst_n_inv.read() == ap_const_logic_1) {
        grp_quant_and_pack_fu_7897_ap_start_reg = ap_const_logic_0;
    } else {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state29.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state37.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state45.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state53.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state61.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state69.read()) || 
             esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state77.read()) || 
             (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state81.read()) && 
              esl_seteq<1,1,1>(icmp_ln321_fu_19399_p2.read(), ap_const_lv1_1)) || 
             (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state91.read()) && 
              esl_seteq<1,1,1>(icmp_ln360_fu_19471_p2.read(), ap_const_lv1_1)) || 
             (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state101.read()) && 
              esl_seteq<1,1,1>(icmp_ln388_fu_19543_p2.read(), ap_const_lv1_1)) || 
             (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state111.read()) && 
              esl_seteq<1,1,1>(icmp_ln417_fu_19618_p2.read(), ap_const_lv1_1)) || 
             (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state121.read()) && 
              esl_seteq<1,1,1>(icmp_ln445_fu_19690_p2.read(), ap_const_lv1_1)) || 
             (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state131.read()) && 
              esl_seteq<1,1,1>(icmp_ln474_fu_19762_p2.read(), ap_const_lv1_1)) || 
             (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state143.read()) && 
              esl_seteq<1,1,1>(icmp_ln514_fu_19834_p2.read(), ap_const_lv1_1)) || 
             (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state153.read()) && 
              esl_seteq<1,1,1>(icmp_ln553_fu_19901_p2.read(), ap_const_lv1_1)) || 
             (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state163.read()) && 
              esl_seteq<1,1,1>(icmp_ln581_fu_19971_p2.read(), ap_const_lv1_1)) || 
             (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state173.read()) && 
              esl_seteq<1,1,1>(icmp_ln610_fu_20038_p2.read(), ap_const_lv1_1)) || 
             (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state183.read()) && 
              esl_seteq<1,1,1>(icmp_ln638_fu_20105_p2.read(), ap_const_lv1_1)))) {
            grp_quant_and_pack_fu_7897_ap_start_reg = ap_const_logic_1;
        } else if (esl_seteq<1,1,1>(ap_const_logic_1, grp_quant_and_pack_fu_7897_ap_ready.read())) {
            grp_quant_and_pack_fu_7897_ap_start_reg = ap_const_logic_0;
        }
    }
    if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state193.read()) && 
         esl_seteq<1,1,1>(icmp_ln667_fu_20172_p2.read(), ap_const_lv1_1))) {
        i73_0_reg_5946 = ap_const_lv4_0;
    } else if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state202.read()) && 
                esl_seteq<1,1,1>(ap_const_lv1_0, icmp_ln701_fu_20239_p2.read()))) {
        i73_0_reg_5946 = i_5_fu_20245_p2.read();
    }
    if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state212.read()) && 
         esl_seteq<1,1,1>(RESULT_WREADY.read(), ap_const_logic_1))) {
        i74_0_reg_5957 = i_6_reg_21249.read();
    } else if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state205.read()) && 
                !(esl_seteq<1,1,1>(ap_const_logic_0, RESULT_AWREADY.read()) || esl_seteq<1,1,1>(ap_const_logic_0, grp_matmul_fu_19032_ap_done.read())))) {
        i74_0_reg_5957 = ap_const_lv4_0;
    }
    if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_pp0_stage0.read()) && 
         esl_seteq<1,1,1>(ap_block_pp0_stage0_11001.read(), ap_const_boolean_0) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_enable_reg_pp0_iter1.read()) && 
         esl_seteq<1,1,1>(ap_const_lv1_0, icmp_ln44_reg_20541.read()))) {
        i_0_reg_5603 = select_ln48_1_reg_20555.read();
    } else if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state1.read()) && 
                esl_seteq<1,1,1>(ap_start.read(), ap_const_logic_1))) {
        i_0_reg_5603 = ap_const_lv6_0;
    }
    if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_pp0_stage0.read()) && 
         esl_seteq<1,1,1>(ap_block_pp0_stage0_11001.read(), ap_const_boolean_0) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_enable_reg_pp0_iter0.read()) && 
         esl_seteq<1,1,1>(ap_const_lv1_0, icmp_ln44_fu_19103_p2.read()))) {
        indvar_flatten_reg_5592 = add_ln44_fu_19109_p2.read();
    } else if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state1.read()) && 
                esl_seteq<1,1,1>(ap_start.read(), ap_const_logic_1))) {
        indvar_flatten_reg_5592 = ap_const_lv11_0;
    }
    if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_pp0_stage0.read()) && 
         esl_seteq<1,1,1>(ap_block_pp0_stage0_11001.read(), ap_const_boolean_0) && 
         esl_seteq<1,1,1>(ap_const_logic_1, ap_enable_reg_pp0_iter0.read()) && 
         esl_seteq<1,1,1>(ap_const_lv1_0, icmp_ln44_fu_19103_p2.read()))) {
        j_0_reg_5614 = j_fu_19143_p2.read();
    } else if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state1.read()) && 
                esl_seteq<1,1,1>(ap_start.read(), ap_const_logic_1))) {
        j_0_reg_5614 = ap_const_lv6_0;
    }
    if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state14.read()) && 
         !(esl_seteq<1,1,1>(icmp_ln74_fu_19371_p2.read(), ap_const_lv1_0) && esl_seteq<1,1,1>(ap_const_logic_0, IMG_RVALID.read())) && 
         esl_seteq<1,1,1>(icmp_ln74_fu_19371_p2.read(), ap_const_lv1_1))) {
        row_0_reg_5637 = row_reg_20583.read();
    } else if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state5.read()) && 
                esl_seteq<1,1,1>(ap_const_lv1_0, icmp_ln72_fu_19281_p2.read()))) {
        row_0_reg_5637 = ap_const_lv6_0;
    }
    if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state14.read()) && esl_seteq<1,1,1>(icmp_ln74_fu_19371_p2.read(), ap_const_lv1_0) && !(esl_seteq<1,1,1>(icmp_ln74_fu_19371_p2.read(), ap_const_lv1_0) && esl_seteq<1,1,1>(ap_const_logic_0, IMG_RVALID.read())))) {
        IMG_addr_read_reg_20627 = IMG_RDATA.read();
        msb_fmap_0_V_addr_1_reg_20612 =  (sc_lv<11>) (zext_ln321_12_fu_19392_p1.read());
        msb_fmap_1_V_addr_1_reg_20617 =  (sc_lv<11>) (zext_ln321_12_fu_19392_p1.read());
        msb_fmap_2_V_addr_1_reg_20622 =  (sc_lv<11>) (zext_ln321_12_fu_19392_p1.read());
    }
    if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state1.read()) && esl_seteq<1,1,1>(ap_start.read(), ap_const_logic_1))) {
        RESULT_addr_reg_20530 =  (sc_lv<32>) (empty_fu_19079_p1.read());
        empty_864_reg_20536 = empty_864_fu_19099_p1.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state94.read())) {
        add_ln166_1_reg_20736 = add_ln166_1_fu_19528_p2.read();
        tmp_340_reg_20731 = grp_fu_19059_p2.read();
        zext_ln154_2_reg_20713 = zext_ln154_2_fu_19518_p1.read();
        zext_ln154_3_reg_20718 = zext_ln154_3_fu_19523_p1.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state84.read())) {
        add_ln166_reg_20680 = add_ln166_fu_19456_p2.read();
        tmp_338_reg_20675 = grp_fu_19059_p2.read();
        zext_ln154_1_reg_20662 = zext_ln154_1_fu_19451_p1.read();
        zext_ln154_reg_20657 = zext_ln154_fu_19446_p1.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state13.read())) {
        add_ln321_5_reg_20599 = add_ln321_5_fu_19365_p2.read();
    }
    if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state6.read()) && esl_seteq<1,1,1>(ap_const_lv1_0, icmp_ln73_fu_19305_p2.read()))) {
        add_ln321_6_reg_20588 = add_ln321_6_fu_19338_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state183.read())) {
        c_out_10_reg_21155 = c_out_10_fu_20111_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state193.read())) {
        c_out_11_reg_21198 = c_out_11_fu_20178_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state91.read())) {
        c_out_1_reg_20693 = c_out_1_fu_19477_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state101.read())) {
        c_out_2_reg_20749 = c_out_2_fu_19549_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state111.read())) {
        c_out_3_reg_20810 = c_out_3_fu_19624_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state121.read())) {
        c_out_4_reg_20866 = c_out_4_fu_19696_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state131.read())) {
        c_out_5_reg_20922 = c_out_5_fu_19768_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state143.read())) {
        c_out_6_reg_20978 = c_out_6_fu_19840_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state153.read())) {
        c_out_7_reg_21021 = c_out_7_fu_19907_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state163.read())) {
        c_out_8_reg_21069 = c_out_8_fu_19977_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state173.read())) {
        c_out_9_reg_21112 = c_out_9_fu_20044_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state81.read())) {
        c_out_reg_20637 = c_out_fu_19405_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state5.read())) {
        c_reg_20570 = c_fu_19287_p2.read();
    }
    if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state14.read()) && !(esl_seteq<1,1,1>(icmp_ln74_fu_19371_p2.read(), ap_const_lv1_0) && esl_seteq<1,1,1>(ap_const_logic_0, IMG_RVALID.read())))) {
        col_reg_20607 = col_fu_19377_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state114.read())) {
        conv_weight_ptr_1_reg_20853 = conv_weight_ptr_1_fu_19675_p2.read();
        tmp_344_reg_20848 = grp_fu_19059_p2.read();
        zext_ln154_6_reg_20830 = zext_ln154_6_fu_19665_p1.read();
        zext_ln154_7_reg_20835 = zext_ln154_7_fu_19670_p1.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state124.read())) {
        conv_weight_ptr_2_reg_20909 = conv_weight_ptr_2_fu_19747_p2.read();
        tmp_346_reg_20904 = grp_fu_19059_p2.read();
        zext_ln154_8_reg_20886 = zext_ln154_8_fu_19737_p1.read();
        zext_ln154_9_reg_20891 = zext_ln154_9_fu_19742_p1.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state134.read())) {
        conv_weight_ptr_3_reg_20965 = conv_weight_ptr_3_fu_19819_p2.read();
        tmp_348_reg_20960 = grp_fu_19059_p2.read();
        zext_ln154_10_reg_20942 = zext_ln154_10_fu_19809_p1.read();
        zext_ln154_11_reg_20947 = zext_ln154_11_fu_19814_p1.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state146.read())) {
        conv_weight_ptr_4_reg_21008 = conv_weight_ptr_4_fu_19886_p2.read();
        tmp_350_reg_21003 = grp_fu_19059_p2.read();
        zext_ln154_12_reg_20998 = zext_ln154_12_fu_19881_p1.read();
    }
    if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state153.read()) && esl_seteq<1,1,1>(ap_const_lv1_0, icmp_ln553_fu_19901_p2.read()))) {
        conv_weight_ptr_5_reg_21036 = conv_weight_ptr_5_fu_19922_p2.read();
        sext_ln553_reg_21026 = sext_ln553_fu_19913_p1.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state166.read())) {
        conv_weight_ptr_6_reg_21099 = conv_weight_ptr_6_fu_20023_p2.read();
        tmp_354_reg_21094 = grp_fu_19059_p2.read();
        zext_ln154_14_reg_21089 = zext_ln154_14_fu_20018_p1.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state176.read())) {
        conv_weight_ptr_7_reg_21142 = conv_weight_ptr_7_fu_20090_p2.read();
        tmp_356_reg_21137 = grp_fu_19059_p2.read();
        zext_ln154_15_reg_21132 = zext_ln154_15_fu_20085_p1.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state186.read())) {
        conv_weight_ptr_8_reg_21185 = conv_weight_ptr_8_fu_20157_p2.read();
        tmp_358_reg_21180 = grp_fu_19059_p2.read();
        zext_ln154_16_reg_21175 = zext_ln154_16_fu_20152_p1.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state196.read())) {
        conv_weight_ptr_9_reg_21228 = conv_weight_ptr_9_fu_20224_p2.read();
        tmp_360_reg_21223 = grp_fu_19059_p2.read();
        zext_ln154_17_reg_21218 = zext_ln154_17_fu_20219_p1.read();
    }
    if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state101.read()) && esl_seteq<1,1,1>(ap_const_lv1_0, icmp_ln388_fu_19543_p2.read()))) {
        conv_weight_ptr_reg_20764 = conv_weight_ptr_fu_19564_p2.read();
        sext_ln388_reg_20754 = sext_ln388_fu_19555_p1.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state206.read())) {
        i_6_reg_21249 = i_6_fu_20262_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state133.read())) {
        icmp_ln165_10_reg_20932 = icmp_ln165_10_fu_19797_p2.read();
        icmp_ln165_11_reg_20937 = icmp_ln165_11_fu_19803_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state145.read())) {
        icmp_ln165_12_reg_20988 = icmp_ln165_12_fu_19869_p2.read();
        icmp_ln165_13_reg_20993 = icmp_ln165_13_fu_19875_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state155.read())) {
        icmp_ln165_14_reg_21041 = icmp_ln165_14_fu_19946_p2.read();
        icmp_ln165_15_reg_21046 = icmp_ln165_15_fu_19952_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state165.read())) {
        icmp_ln165_16_reg_21079 = icmp_ln165_16_fu_20006_p2.read();
        icmp_ln165_17_reg_21084 = icmp_ln165_17_fu_20012_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state175.read())) {
        icmp_ln165_18_reg_21122 = icmp_ln165_18_fu_20073_p2.read();
        icmp_ln165_19_reg_21127 = icmp_ln165_19_fu_20079_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state83.read())) {
        icmp_ln165_1_reg_20652 = icmp_ln165_1_fu_19440_p2.read();
        icmp_ln165_reg_20647 = icmp_ln165_fu_19434_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state185.read())) {
        icmp_ln165_20_reg_21165 = icmp_ln165_20_fu_20140_p2.read();
        icmp_ln165_21_reg_21170 = icmp_ln165_21_fu_20146_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state195.read())) {
        icmp_ln165_22_reg_21208 = icmp_ln165_22_fu_20207_p2.read();
        icmp_ln165_23_reg_21213 = icmp_ln165_23_fu_20213_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state93.read())) {
        icmp_ln165_2_reg_20703 = icmp_ln165_2_fu_19506_p2.read();
        icmp_ln165_3_reg_20708 = icmp_ln165_3_fu_19512_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state103.read())) {
        icmp_ln165_4_reg_20769 = icmp_ln165_4_fu_19588_p2.read();
        icmp_ln165_5_reg_20774 = icmp_ln165_5_fu_19594_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state113.read())) {
        icmp_ln165_6_reg_20820 = icmp_ln165_6_fu_19653_p2.read();
        icmp_ln165_7_reg_20825 = icmp_ln165_7_fu_19659_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state123.read())) {
        icmp_ln165_8_reg_20876 = icmp_ln165_8_fu_19725_p2.read();
        icmp_ln165_9_reg_20881 = icmp_ln165_9_fu_19731_p2.read();
    }
    if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_pp0_stage0.read()) && esl_seteq<1,1,1>(ap_block_pp0_stage0_11001.read(), ap_const_boolean_0))) {
        icmp_ln44_reg_20541 = icmp_ln44_fu_19103_p2.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state208.read())) {
        icmp_ln935_reg_21276 = icmp_ln935_fu_20287_p2.read();
        sub_ln944_reg_21289 = sub_ln944_fu_20315_p2.read();
        tmp_V_4_reg_21281 = tmp_V_4_fu_20292_p3.read();
        trunc_ln943_reg_21301 = trunc_ln943_fu_20325_p1.read();
        trunc_ln947_reg_21296 = trunc_ln947_fu_20321_p1.read();
    }
    if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state209.read()) && esl_seteq<1,1,1>(ap_const_lv1_0, icmp_ln935_reg_21276.read()))) {
        icmp_ln958_reg_21311 = icmp_ln958_fu_20423_p2.read();
        or_ln_reg_21306 = or_ln_fu_20415_p3.read();
    }
    if ((esl_seteq<1,1,1>(ap_const_lv1_0, icmp_ln935_reg_21276.read()) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state210.read()))) {
        m_reg_21316 = m_3_fu_20456_p2.read().range(31, 1);
        tmp_1237_reg_21321 = m_3_fu_20456_p2.read().range(25, 25);
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state207.read())) {
        p_Result_7_reg_21265 = linear_out_buf_V_q0.read().range(31, 31);
        tmp_V_3_reg_21259 = linear_out_buf_V_q0.read();
        tmp_V_reg_21271 = tmp_V_fu_20281_p2.read();
    }
    if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state187.read()))) {
        pg_enable_10_reg_21190 = pg_enable_10_fu_20167_p2.read();
    }
    if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state197.read()))) {
        pg_enable_11_reg_21233 = pg_enable_11_fu_20234_p2.read();
    }
    if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state95.read()))) {
        pg_enable_1_reg_20741 = pg_enable_1_fu_19538_p2.read();
    }
    if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state105.read()))) {
        pg_enable_2_reg_20802 = pg_enable_2_fu_19613_p2.read();
    }
    if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state115.read()))) {
        pg_enable_3_reg_20858 = pg_enable_3_fu_19685_p2.read();
    }
    if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state125.read()))) {
        pg_enable_4_reg_20914 = pg_enable_4_fu_19757_p2.read();
    }
    if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state135.read()))) {
        pg_enable_5_reg_20970 = pg_enable_5_fu_19829_p2.read();
    }
    if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state147.read()))) {
        pg_enable_6_reg_21013 = pg_enable_6_fu_19896_p2.read();
    }
    if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state157.read()))) {
        pg_enable_7_reg_21061 = pg_enable_7_fu_19966_p2.read();
    }
    if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state167.read()))) {
        pg_enable_8_reg_21104 = pg_enable_8_fu_20033_p2.read();
    }
    if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state177.read()))) {
        pg_enable_9_reg_21147 = pg_enable_9_fu_20100_p2.read();
    }
    if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state85.read()) && esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1))) {
        pg_enable_reg_20685 = pg_enable_fu_19466_p2.read();
    }
    if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state82.read()) || esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state92.read()) || esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state102.read()) || esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state112.read()) || esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state122.read()) || esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state132.read()) || esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state144.read()) || esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state154.read()) || esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state164.read()) || esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state174.read()) || esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state184.read()) || esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state194.read()))) {
        reg_19064 = gate_mask_q0.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state6.read())) {
        row_reg_20583 = row_fu_19311_p2.read();
    }
    if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_pp0_stage0.read()) && esl_seteq<1,1,1>(ap_block_pp0_stage0_11001.read(), ap_const_boolean_0) && esl_seteq<1,1,1>(ap_const_logic_1, ap_enable_reg_pp0_iter0.read()) && esl_seteq<1,1,1>(ap_const_lv1_0, icmp_ln44_fu_19103_p2.read()))) {
        select_ln48_1_reg_20555 = select_ln48_1_fu_19135_p3.read();
    }
    if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_pp0_stage0.read()) && esl_seteq<1,1,1>(ap_block_pp0_stage0_11001.read(), ap_const_boolean_0) && esl_seteq<1,1,1>(ap_const_lv1_0, icmp_ln44_fu_19103_p2.read()))) {
        select_ln48_reg_20550 = select_ln48_fu_19127_p3.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state211.read())) {
        select_ln935_reg_21326 = select_ln935_fu_20523_p3.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state104.read())) {
        tmp_342_reg_20797 = grp_fu_19059_p2.read();
        zext_ln154_4_reg_20779 = zext_ln154_4_fu_19600_p1.read();
        zext_ln154_5_reg_20784 = zext_ln154_5_fu_19604_p1.read();
    }
    if (esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state156.read())) {
        tmp_352_reg_21056 = grp_fu_19059_p2.read();
        zext_ln154_13_reg_21051 = zext_ln154_13_fu_19958_p1.read();
    }
    if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state5.read()) && esl_seteq<1,1,1>(ap_const_lv1_0, icmp_ln72_fu_19281_p2.read()))) {
        zext_ln73_reg_20575 = zext_ln73_fu_19301_p1.read();
    }
}

void FracNet_T::thread_ap_NS_fsm() {
    if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state1))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state1.read()) && esl_seteq<1,1,1>(ap_start.read(), ap_const_logic_1))) {
            ap_NS_fsm = ap_ST_fsm_pp0_stage0;
        } else {
            ap_NS_fsm = ap_ST_fsm_state1;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_pp0_stage0))
    {
        if (!(esl_seteq<1,1,1>(ap_const_logic_1, ap_enable_reg_pp0_iter0.read()) && esl_seteq<1,1,1>(ap_block_pp0_stage0_subdone.read(), ap_const_boolean_0) && esl_seteq<1,1,1>(icmp_ln44_fu_19103_p2.read(), ap_const_lv1_1))) {
            ap_NS_fsm = ap_ST_fsm_pp0_stage0;
        } else if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_enable_reg_pp0_iter0.read()) && esl_seteq<1,1,1>(ap_block_pp0_stage0_subdone.read(), ap_const_boolean_0) && esl_seteq<1,1,1>(icmp_ln44_fu_19103_p2.read(), ap_const_lv1_1))) {
            ap_NS_fsm = ap_ST_fsm_state4;
        } else {
            ap_NS_fsm = ap_ST_fsm_pp0_stage0;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state4))
    {
        ap_NS_fsm = ap_ST_fsm_state5;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state5))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state5.read()) && esl_seteq<1,1,1>(ap_const_lv1_0, icmp_ln72_fu_19281_p2.read()))) {
            ap_NS_fsm = ap_ST_fsm_state6;
        } else {
            ap_NS_fsm = ap_ST_fsm_state16;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state6))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state6.read()) && esl_seteq<1,1,1>(icmp_ln73_fu_19305_p2.read(), ap_const_lv1_1))) {
            ap_NS_fsm = ap_ST_fsm_state5;
        } else {
            ap_NS_fsm = ap_ST_fsm_state7;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state7))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state7.read()) && esl_seteq<1,1,1>(IMG_ARREADY.read(), ap_const_logic_1))) {
            ap_NS_fsm = ap_ST_fsm_state8;
        } else {
            ap_NS_fsm = ap_ST_fsm_state7;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state8))
    {
        ap_NS_fsm = ap_ST_fsm_state9;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state9))
    {
        ap_NS_fsm = ap_ST_fsm_state10;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state10))
    {
        ap_NS_fsm = ap_ST_fsm_state11;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state11))
    {
        ap_NS_fsm = ap_ST_fsm_state12;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state12))
    {
        ap_NS_fsm = ap_ST_fsm_state13;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state13))
    {
        ap_NS_fsm = ap_ST_fsm_state14;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state14))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state14.read()) && !(esl_seteq<1,1,1>(icmp_ln74_fu_19371_p2.read(), ap_const_lv1_0) && esl_seteq<1,1,1>(ap_const_logic_0, IMG_RVALID.read())) && esl_seteq<1,1,1>(icmp_ln74_fu_19371_p2.read(), ap_const_lv1_1))) {
            ap_NS_fsm = ap_ST_fsm_state6;
        } else if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state14.read()) && esl_seteq<1,1,1>(icmp_ln74_fu_19371_p2.read(), ap_const_lv1_0) && !(esl_seteq<1,1,1>(icmp_ln74_fu_19371_p2.read(), ap_const_lv1_0) && esl_seteq<1,1,1>(ap_const_logic_0, IMG_RVALID.read())))) {
            ap_NS_fsm = ap_ST_fsm_state15;
        } else {
            ap_NS_fsm = ap_ST_fsm_state14;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state15))
    {
        ap_NS_fsm = ap_ST_fsm_state14;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state16))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state16.read()))) {
            ap_NS_fsm = ap_ST_fsm_state17;
        } else {
            ap_NS_fsm = ap_ST_fsm_state16;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state17))
    {
        ap_NS_fsm = ap_ST_fsm_state18;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state18))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state18.read()))) {
            ap_NS_fsm = ap_ST_fsm_state19;
        } else {
            ap_NS_fsm = ap_ST_fsm_state18;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state19))
    {
        ap_NS_fsm = ap_ST_fsm_state20;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state20))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state20.read()))) {
            ap_NS_fsm = ap_ST_fsm_state21;
        } else {
            ap_NS_fsm = ap_ST_fsm_state20;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state21))
    {
        ap_NS_fsm = ap_ST_fsm_state22;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state22))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state22.read()))) {
            ap_NS_fsm = ap_ST_fsm_state23;
        } else {
            ap_NS_fsm = ap_ST_fsm_state22;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state23))
    {
        ap_NS_fsm = ap_ST_fsm_state24;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state24))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state24.read()))) {
            ap_NS_fsm = ap_ST_fsm_state25;
        } else {
            ap_NS_fsm = ap_ST_fsm_state24;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state25))
    {
        ap_NS_fsm = ap_ST_fsm_state26;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state26))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state26.read()))) {
            ap_NS_fsm = ap_ST_fsm_state27;
        } else {
            ap_NS_fsm = ap_ST_fsm_state26;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state27))
    {
        ap_NS_fsm = ap_ST_fsm_state28;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state28))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state28.read()) && esl_seteq<1,1,1>(grp_bn1_fu_18996_ap_done.read(), ap_const_logic_1))) {
            ap_NS_fsm = ap_ST_fsm_state29;
        } else {
            ap_NS_fsm = ap_ST_fsm_state28;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state29))
    {
        ap_NS_fsm = ap_ST_fsm_state30;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state30))
    {
        if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state30.read()))) {
            ap_NS_fsm = ap_ST_fsm_state31;
        } else {
            ap_NS_fsm = ap_ST_fsm_state30;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state31))
    {
        ap_NS_fsm = ap_ST_fsm_state32;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state32))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state32.read()))) {
            ap_NS_fsm = ap_ST_fsm_state33;
        } else {
            ap_NS_fsm = ap_ST_fsm_state32;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state33))
    {
        ap_NS_fsm = ap_ST_fsm_state34;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state34))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state34.read()))) {
            ap_NS_fsm = ap_ST_fsm_state35;
        } else {
            ap_NS_fsm = ap_ST_fsm_state34;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state35))
    {
        ap_NS_fsm = ap_ST_fsm_state36;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state36))
    {
        if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state36.read()))) {
            ap_NS_fsm = ap_ST_fsm_state37;
        } else {
            ap_NS_fsm = ap_ST_fsm_state36;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state37))
    {
        ap_NS_fsm = ap_ST_fsm_state38;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state38))
    {
        if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state38.read()))) {
            ap_NS_fsm = ap_ST_fsm_state39;
        } else {
            ap_NS_fsm = ap_ST_fsm_state38;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state39))
    {
        ap_NS_fsm = ap_ST_fsm_state40;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state40))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state40.read()))) {
            ap_NS_fsm = ap_ST_fsm_state41;
        } else {
            ap_NS_fsm = ap_ST_fsm_state40;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state41))
    {
        ap_NS_fsm = ap_ST_fsm_state42;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state42))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state42.read()))) {
            ap_NS_fsm = ap_ST_fsm_state43;
        } else {
            ap_NS_fsm = ap_ST_fsm_state42;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state43))
    {
        ap_NS_fsm = ap_ST_fsm_state44;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state44))
    {
        if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state44.read()))) {
            ap_NS_fsm = ap_ST_fsm_state45;
        } else {
            ap_NS_fsm = ap_ST_fsm_state44;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state45))
    {
        ap_NS_fsm = ap_ST_fsm_state46;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state46))
    {
        if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state46.read()))) {
            ap_NS_fsm = ap_ST_fsm_state47;
        } else {
            ap_NS_fsm = ap_ST_fsm_state46;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state47))
    {
        ap_NS_fsm = ap_ST_fsm_state48;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state48))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state48.read()))) {
            ap_NS_fsm = ap_ST_fsm_state49;
        } else {
            ap_NS_fsm = ap_ST_fsm_state48;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state49))
    {
        ap_NS_fsm = ap_ST_fsm_state50;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state50))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state50.read()))) {
            ap_NS_fsm = ap_ST_fsm_state51;
        } else {
            ap_NS_fsm = ap_ST_fsm_state50;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state51))
    {
        ap_NS_fsm = ap_ST_fsm_state52;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state52))
    {
        if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state52.read()))) {
            ap_NS_fsm = ap_ST_fsm_state53;
        } else {
            ap_NS_fsm = ap_ST_fsm_state52;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state53))
    {
        ap_NS_fsm = ap_ST_fsm_state54;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state54))
    {
        if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state54.read()))) {
            ap_NS_fsm = ap_ST_fsm_state55;
        } else {
            ap_NS_fsm = ap_ST_fsm_state54;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state55))
    {
        ap_NS_fsm = ap_ST_fsm_state56;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state56))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state56.read()))) {
            ap_NS_fsm = ap_ST_fsm_state57;
        } else {
            ap_NS_fsm = ap_ST_fsm_state56;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state57))
    {
        ap_NS_fsm = ap_ST_fsm_state58;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state58))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state58.read()))) {
            ap_NS_fsm = ap_ST_fsm_state59;
        } else {
            ap_NS_fsm = ap_ST_fsm_state58;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state59))
    {
        ap_NS_fsm = ap_ST_fsm_state60;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state60))
    {
        if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state60.read()))) {
            ap_NS_fsm = ap_ST_fsm_state61;
        } else {
            ap_NS_fsm = ap_ST_fsm_state60;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state61))
    {
        ap_NS_fsm = ap_ST_fsm_state62;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state62))
    {
        if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state62.read()))) {
            ap_NS_fsm = ap_ST_fsm_state63;
        } else {
            ap_NS_fsm = ap_ST_fsm_state62;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state63))
    {
        ap_NS_fsm = ap_ST_fsm_state64;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state64))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state64.read()))) {
            ap_NS_fsm = ap_ST_fsm_state65;
        } else {
            ap_NS_fsm = ap_ST_fsm_state64;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state65))
    {
        ap_NS_fsm = ap_ST_fsm_state66;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state66))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state66.read()))) {
            ap_NS_fsm = ap_ST_fsm_state67;
        } else {
            ap_NS_fsm = ap_ST_fsm_state66;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state67))
    {
        ap_NS_fsm = ap_ST_fsm_state68;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state68))
    {
        if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state68.read()))) {
            ap_NS_fsm = ap_ST_fsm_state69;
        } else {
            ap_NS_fsm = ap_ST_fsm_state68;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state69))
    {
        ap_NS_fsm = ap_ST_fsm_state70;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state70))
    {
        if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state70.read()))) {
            ap_NS_fsm = ap_ST_fsm_state71;
        } else {
            ap_NS_fsm = ap_ST_fsm_state70;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state71))
    {
        ap_NS_fsm = ap_ST_fsm_state72;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state72))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state72.read()))) {
            ap_NS_fsm = ap_ST_fsm_state73;
        } else {
            ap_NS_fsm = ap_ST_fsm_state72;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state73))
    {
        ap_NS_fsm = ap_ST_fsm_state74;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state74))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state74.read()))) {
            ap_NS_fsm = ap_ST_fsm_state75;
        } else {
            ap_NS_fsm = ap_ST_fsm_state74;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state75))
    {
        ap_NS_fsm = ap_ST_fsm_state76;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state76))
    {
        if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state76.read()))) {
            ap_NS_fsm = ap_ST_fsm_state77;
        } else {
            ap_NS_fsm = ap_ST_fsm_state76;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state77))
    {
        ap_NS_fsm = ap_ST_fsm_state78;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state78))
    {
        if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state78.read()))) {
            ap_NS_fsm = ap_ST_fsm_state79;
        } else {
            ap_NS_fsm = ap_ST_fsm_state78;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state79))
    {
        ap_NS_fsm = ap_ST_fsm_state80;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state80))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state80.read()) && esl_seteq<1,1,1>(grp_avgpool_concat_fu_18852_ap_done.read(), ap_const_logic_1))) {
            ap_NS_fsm = ap_ST_fsm_state81;
        } else {
            ap_NS_fsm = ap_ST_fsm_state80;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state81))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state81.read()) && esl_seteq<1,1,1>(icmp_ln321_fu_19399_p2.read(), ap_const_lv1_1))) {
            ap_NS_fsm = ap_ST_fsm_state90;
        } else {
            ap_NS_fsm = ap_ST_fsm_state82;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state82))
    {
        ap_NS_fsm = ap_ST_fsm_state83;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state83))
    {
        ap_NS_fsm = ap_ST_fsm_state84;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state84))
    {
        ap_NS_fsm = ap_ST_fsm_state85;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state85))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state85.read()) && esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1))) {
            ap_NS_fsm = ap_ST_fsm_state86;
        } else {
            ap_NS_fsm = ap_ST_fsm_state85;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state86))
    {
        ap_NS_fsm = ap_ST_fsm_state87;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state87))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state87.read()))) {
            ap_NS_fsm = ap_ST_fsm_state88;
        } else {
            ap_NS_fsm = ap_ST_fsm_state87;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state88))
    {
        ap_NS_fsm = ap_ST_fsm_state89;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state89))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state89.read()) && esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1))) {
            ap_NS_fsm = ap_ST_fsm_state81;
        } else {
            ap_NS_fsm = ap_ST_fsm_state89;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state90))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state90.read()) && esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1))) {
            ap_NS_fsm = ap_ST_fsm_state91;
        } else {
            ap_NS_fsm = ap_ST_fsm_state90;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state91))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state91.read()) && esl_seteq<1,1,1>(icmp_ln360_fu_19471_p2.read(), ap_const_lv1_1))) {
            ap_NS_fsm = ap_ST_fsm_state100;
        } else {
            ap_NS_fsm = ap_ST_fsm_state92;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state92))
    {
        ap_NS_fsm = ap_ST_fsm_state93;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state93))
    {
        ap_NS_fsm = ap_ST_fsm_state94;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state94))
    {
        ap_NS_fsm = ap_ST_fsm_state95;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state95))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state95.read()))) {
            ap_NS_fsm = ap_ST_fsm_state96;
        } else {
            ap_NS_fsm = ap_ST_fsm_state95;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state96))
    {
        ap_NS_fsm = ap_ST_fsm_state97;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state97))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state97.read()))) {
            ap_NS_fsm = ap_ST_fsm_state98;
        } else {
            ap_NS_fsm = ap_ST_fsm_state97;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state98))
    {
        ap_NS_fsm = ap_ST_fsm_state99;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state99))
    {
        if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state99.read()))) {
            ap_NS_fsm = ap_ST_fsm_state91;
        } else {
            ap_NS_fsm = ap_ST_fsm_state99;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state100))
    {
        if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state100.read()))) {
            ap_NS_fsm = ap_ST_fsm_state101;
        } else {
            ap_NS_fsm = ap_ST_fsm_state100;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state101))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state101.read()) && esl_seteq<1,1,1>(icmp_ln388_fu_19543_p2.read(), ap_const_lv1_1))) {
            ap_NS_fsm = ap_ST_fsm_state110;
        } else {
            ap_NS_fsm = ap_ST_fsm_state102;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state102))
    {
        ap_NS_fsm = ap_ST_fsm_state103;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state103))
    {
        ap_NS_fsm = ap_ST_fsm_state104;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state104))
    {
        ap_NS_fsm = ap_ST_fsm_state105;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state105))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state105.read()))) {
            ap_NS_fsm = ap_ST_fsm_state106;
        } else {
            ap_NS_fsm = ap_ST_fsm_state105;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state106))
    {
        ap_NS_fsm = ap_ST_fsm_state107;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state107))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state107.read()))) {
            ap_NS_fsm = ap_ST_fsm_state108;
        } else {
            ap_NS_fsm = ap_ST_fsm_state107;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state108))
    {
        ap_NS_fsm = ap_ST_fsm_state109;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state109))
    {
        if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state109.read()))) {
            ap_NS_fsm = ap_ST_fsm_state101;
        } else {
            ap_NS_fsm = ap_ST_fsm_state109;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state110))
    {
        if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state110.read()))) {
            ap_NS_fsm = ap_ST_fsm_state111;
        } else {
            ap_NS_fsm = ap_ST_fsm_state110;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state111))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state111.read()) && esl_seteq<1,1,1>(icmp_ln417_fu_19618_p2.read(), ap_const_lv1_1))) {
            ap_NS_fsm = ap_ST_fsm_state120;
        } else {
            ap_NS_fsm = ap_ST_fsm_state112;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state112))
    {
        ap_NS_fsm = ap_ST_fsm_state113;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state113))
    {
        ap_NS_fsm = ap_ST_fsm_state114;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state114))
    {
        ap_NS_fsm = ap_ST_fsm_state115;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state115))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state115.read()))) {
            ap_NS_fsm = ap_ST_fsm_state116;
        } else {
            ap_NS_fsm = ap_ST_fsm_state115;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state116))
    {
        ap_NS_fsm = ap_ST_fsm_state117;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state117))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state117.read()))) {
            ap_NS_fsm = ap_ST_fsm_state118;
        } else {
            ap_NS_fsm = ap_ST_fsm_state117;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state118))
    {
        ap_NS_fsm = ap_ST_fsm_state119;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state119))
    {
        if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state119.read()))) {
            ap_NS_fsm = ap_ST_fsm_state111;
        } else {
            ap_NS_fsm = ap_ST_fsm_state119;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state120))
    {
        if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state120.read()))) {
            ap_NS_fsm = ap_ST_fsm_state121;
        } else {
            ap_NS_fsm = ap_ST_fsm_state120;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state121))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state121.read()) && esl_seteq<1,1,1>(icmp_ln445_fu_19690_p2.read(), ap_const_lv1_1))) {
            ap_NS_fsm = ap_ST_fsm_state130;
        } else {
            ap_NS_fsm = ap_ST_fsm_state122;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state122))
    {
        ap_NS_fsm = ap_ST_fsm_state123;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state123))
    {
        ap_NS_fsm = ap_ST_fsm_state124;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state124))
    {
        ap_NS_fsm = ap_ST_fsm_state125;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state125))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state125.read()))) {
            ap_NS_fsm = ap_ST_fsm_state126;
        } else {
            ap_NS_fsm = ap_ST_fsm_state125;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state126))
    {
        ap_NS_fsm = ap_ST_fsm_state127;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state127))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state127.read()))) {
            ap_NS_fsm = ap_ST_fsm_state128;
        } else {
            ap_NS_fsm = ap_ST_fsm_state127;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state128))
    {
        ap_NS_fsm = ap_ST_fsm_state129;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state129))
    {
        if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state129.read()))) {
            ap_NS_fsm = ap_ST_fsm_state121;
        } else {
            ap_NS_fsm = ap_ST_fsm_state129;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state130))
    {
        if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state130.read()))) {
            ap_NS_fsm = ap_ST_fsm_state131;
        } else {
            ap_NS_fsm = ap_ST_fsm_state130;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state131))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state131.read()) && esl_seteq<1,1,1>(icmp_ln474_fu_19762_p2.read(), ap_const_lv1_1))) {
            ap_NS_fsm = ap_ST_fsm_state140;
        } else {
            ap_NS_fsm = ap_ST_fsm_state132;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state132))
    {
        ap_NS_fsm = ap_ST_fsm_state133;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state133))
    {
        ap_NS_fsm = ap_ST_fsm_state134;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state134))
    {
        ap_NS_fsm = ap_ST_fsm_state135;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state135))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state135.read()))) {
            ap_NS_fsm = ap_ST_fsm_state136;
        } else {
            ap_NS_fsm = ap_ST_fsm_state135;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state136))
    {
        ap_NS_fsm = ap_ST_fsm_state137;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state137))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state137.read()))) {
            ap_NS_fsm = ap_ST_fsm_state138;
        } else {
            ap_NS_fsm = ap_ST_fsm_state137;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state138))
    {
        ap_NS_fsm = ap_ST_fsm_state139;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state139))
    {
        if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state139.read()))) {
            ap_NS_fsm = ap_ST_fsm_state131;
        } else {
            ap_NS_fsm = ap_ST_fsm_state139;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state140))
    {
        if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state140.read()))) {
            ap_NS_fsm = ap_ST_fsm_state141;
        } else {
            ap_NS_fsm = ap_ST_fsm_state140;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state141))
    {
        ap_NS_fsm = ap_ST_fsm_state142;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state142))
    {
        if ((esl_seteq<1,1,1>(grp_avgpool_concat_fu_18852_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state142.read()))) {
            ap_NS_fsm = ap_ST_fsm_state143;
        } else {
            ap_NS_fsm = ap_ST_fsm_state142;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state143))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state143.read()) && esl_seteq<1,1,1>(icmp_ln514_fu_19834_p2.read(), ap_const_lv1_1))) {
            ap_NS_fsm = ap_ST_fsm_state152;
        } else {
            ap_NS_fsm = ap_ST_fsm_state144;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state144))
    {
        ap_NS_fsm = ap_ST_fsm_state145;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state145))
    {
        ap_NS_fsm = ap_ST_fsm_state146;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state146))
    {
        ap_NS_fsm = ap_ST_fsm_state147;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state147))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state147.read()))) {
            ap_NS_fsm = ap_ST_fsm_state148;
        } else {
            ap_NS_fsm = ap_ST_fsm_state147;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state148))
    {
        ap_NS_fsm = ap_ST_fsm_state149;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state149))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state149.read()))) {
            ap_NS_fsm = ap_ST_fsm_state150;
        } else {
            ap_NS_fsm = ap_ST_fsm_state149;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state150))
    {
        ap_NS_fsm = ap_ST_fsm_state151;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state151))
    {
        if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state151.read()))) {
            ap_NS_fsm = ap_ST_fsm_state143;
        } else {
            ap_NS_fsm = ap_ST_fsm_state151;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state152))
    {
        if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state152.read()))) {
            ap_NS_fsm = ap_ST_fsm_state153;
        } else {
            ap_NS_fsm = ap_ST_fsm_state152;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state153))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state153.read()) && esl_seteq<1,1,1>(icmp_ln553_fu_19901_p2.read(), ap_const_lv1_1))) {
            ap_NS_fsm = ap_ST_fsm_state162;
        } else {
            ap_NS_fsm = ap_ST_fsm_state154;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state154))
    {
        ap_NS_fsm = ap_ST_fsm_state155;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state155))
    {
        ap_NS_fsm = ap_ST_fsm_state156;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state156))
    {
        ap_NS_fsm = ap_ST_fsm_state157;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state157))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state157.read()))) {
            ap_NS_fsm = ap_ST_fsm_state158;
        } else {
            ap_NS_fsm = ap_ST_fsm_state157;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state158))
    {
        ap_NS_fsm = ap_ST_fsm_state159;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state159))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state159.read()))) {
            ap_NS_fsm = ap_ST_fsm_state160;
        } else {
            ap_NS_fsm = ap_ST_fsm_state159;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state160))
    {
        ap_NS_fsm = ap_ST_fsm_state161;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state161))
    {
        if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state161.read()))) {
            ap_NS_fsm = ap_ST_fsm_state153;
        } else {
            ap_NS_fsm = ap_ST_fsm_state161;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state162))
    {
        if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state162.read()))) {
            ap_NS_fsm = ap_ST_fsm_state163;
        } else {
            ap_NS_fsm = ap_ST_fsm_state162;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state163))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state163.read()) && esl_seteq<1,1,1>(icmp_ln581_fu_19971_p2.read(), ap_const_lv1_1))) {
            ap_NS_fsm = ap_ST_fsm_state172;
        } else {
            ap_NS_fsm = ap_ST_fsm_state164;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state164))
    {
        ap_NS_fsm = ap_ST_fsm_state165;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state165))
    {
        ap_NS_fsm = ap_ST_fsm_state166;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state166))
    {
        ap_NS_fsm = ap_ST_fsm_state167;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state167))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state167.read()))) {
            ap_NS_fsm = ap_ST_fsm_state168;
        } else {
            ap_NS_fsm = ap_ST_fsm_state167;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state168))
    {
        ap_NS_fsm = ap_ST_fsm_state169;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state169))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state169.read()))) {
            ap_NS_fsm = ap_ST_fsm_state170;
        } else {
            ap_NS_fsm = ap_ST_fsm_state169;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state170))
    {
        ap_NS_fsm = ap_ST_fsm_state171;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state171))
    {
        if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state171.read()))) {
            ap_NS_fsm = ap_ST_fsm_state163;
        } else {
            ap_NS_fsm = ap_ST_fsm_state171;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state172))
    {
        if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state172.read()))) {
            ap_NS_fsm = ap_ST_fsm_state173;
        } else {
            ap_NS_fsm = ap_ST_fsm_state172;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state173))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state173.read()) && esl_seteq<1,1,1>(icmp_ln610_fu_20038_p2.read(), ap_const_lv1_1))) {
            ap_NS_fsm = ap_ST_fsm_state182;
        } else {
            ap_NS_fsm = ap_ST_fsm_state174;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state174))
    {
        ap_NS_fsm = ap_ST_fsm_state175;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state175))
    {
        ap_NS_fsm = ap_ST_fsm_state176;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state176))
    {
        ap_NS_fsm = ap_ST_fsm_state177;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state177))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state177.read()))) {
            ap_NS_fsm = ap_ST_fsm_state178;
        } else {
            ap_NS_fsm = ap_ST_fsm_state177;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state178))
    {
        ap_NS_fsm = ap_ST_fsm_state179;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state179))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state179.read()))) {
            ap_NS_fsm = ap_ST_fsm_state180;
        } else {
            ap_NS_fsm = ap_ST_fsm_state179;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state180))
    {
        ap_NS_fsm = ap_ST_fsm_state181;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state181))
    {
        if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state181.read()))) {
            ap_NS_fsm = ap_ST_fsm_state173;
        } else {
            ap_NS_fsm = ap_ST_fsm_state181;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state182))
    {
        if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state182.read()))) {
            ap_NS_fsm = ap_ST_fsm_state183;
        } else {
            ap_NS_fsm = ap_ST_fsm_state182;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state183))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state183.read()) && esl_seteq<1,1,1>(icmp_ln638_fu_20105_p2.read(), ap_const_lv1_1))) {
            ap_NS_fsm = ap_ST_fsm_state192;
        } else {
            ap_NS_fsm = ap_ST_fsm_state184;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state184))
    {
        ap_NS_fsm = ap_ST_fsm_state185;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state185))
    {
        ap_NS_fsm = ap_ST_fsm_state186;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state186))
    {
        ap_NS_fsm = ap_ST_fsm_state187;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state187))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state187.read()))) {
            ap_NS_fsm = ap_ST_fsm_state188;
        } else {
            ap_NS_fsm = ap_ST_fsm_state187;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state188))
    {
        ap_NS_fsm = ap_ST_fsm_state189;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state189))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state189.read()))) {
            ap_NS_fsm = ap_ST_fsm_state190;
        } else {
            ap_NS_fsm = ap_ST_fsm_state189;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state190))
    {
        ap_NS_fsm = ap_ST_fsm_state191;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state191))
    {
        if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state191.read()))) {
            ap_NS_fsm = ap_ST_fsm_state183;
        } else {
            ap_NS_fsm = ap_ST_fsm_state191;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state192))
    {
        if ((esl_seteq<1,1,1>(grp_quant_and_pack_fu_7897_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state192.read()))) {
            ap_NS_fsm = ap_ST_fsm_state193;
        } else {
            ap_NS_fsm = ap_ST_fsm_state192;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state193))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state193.read()) && esl_seteq<1,1,1>(icmp_ln667_fu_20172_p2.read(), ap_const_lv1_1))) {
            ap_NS_fsm = ap_ST_fsm_state202;
        } else {
            ap_NS_fsm = ap_ST_fsm_state194;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state194))
    {
        ap_NS_fsm = ap_ST_fsm_state195;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state195))
    {
        ap_NS_fsm = ap_ST_fsm_state196;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state196))
    {
        ap_NS_fsm = ap_ST_fsm_state197;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state197))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state197.read()))) {
            ap_NS_fsm = ap_ST_fsm_state198;
        } else {
            ap_NS_fsm = ap_ST_fsm_state197;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state198))
    {
        ap_NS_fsm = ap_ST_fsm_state199;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state199))
    {
        if ((esl_seteq<1,1,1>(grp_binary_conv3x3_tile_fu_5968_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state199.read()))) {
            ap_NS_fsm = ap_ST_fsm_state200;
        } else {
            ap_NS_fsm = ap_ST_fsm_state199;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state200))
    {
        ap_NS_fsm = ap_ST_fsm_state201;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state201))
    {
        if ((esl_seteq<1,1,1>(grp_bn_relu_shortcut_fu_8106_ap_done.read(), ap_const_logic_1) && esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state201.read()))) {
            ap_NS_fsm = ap_ST_fsm_state193;
        } else {
            ap_NS_fsm = ap_ST_fsm_state201;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state202))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state202.read()) && esl_seteq<1,1,1>(ap_const_lv1_0, icmp_ln701_fu_20239_p2.read()))) {
            ap_NS_fsm = ap_ST_fsm_state202;
        } else {
            ap_NS_fsm = ap_ST_fsm_state203;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state203))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state203.read()) && esl_seteq<1,1,1>(grp_avgpool_8x8_fu_18926_ap_done.read(), ap_const_logic_1))) {
            ap_NS_fsm = ap_ST_fsm_state204;
        } else {
            ap_NS_fsm = ap_ST_fsm_state203;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state204))
    {
        ap_NS_fsm = ap_ST_fsm_state205;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state205))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state205.read()) && !(esl_seteq<1,1,1>(ap_const_logic_0, RESULT_AWREADY.read()) || esl_seteq<1,1,1>(ap_const_logic_0, grp_matmul_fu_19032_ap_done.read())))) {
            ap_NS_fsm = ap_ST_fsm_state206;
        } else {
            ap_NS_fsm = ap_ST_fsm_state205;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state206))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state206.read()) && esl_seteq<1,1,1>(icmp_ln710_fu_20256_p2.read(), ap_const_lv1_1))) {
            ap_NS_fsm = ap_ST_fsm_state213;
        } else {
            ap_NS_fsm = ap_ST_fsm_state207;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state207))
    {
        ap_NS_fsm = ap_ST_fsm_state208;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state208))
    {
        ap_NS_fsm = ap_ST_fsm_state209;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state209))
    {
        ap_NS_fsm = ap_ST_fsm_state210;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state210))
    {
        ap_NS_fsm = ap_ST_fsm_state211;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state211))
    {
        ap_NS_fsm = ap_ST_fsm_state212;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state212))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state212.read()) && esl_seteq<1,1,1>(RESULT_WREADY.read(), ap_const_logic_1))) {
            ap_NS_fsm = ap_ST_fsm_state206;
        } else {
            ap_NS_fsm = ap_ST_fsm_state212;
        }
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state213))
    {
        ap_NS_fsm = ap_ST_fsm_state214;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state214))
    {
        ap_NS_fsm = ap_ST_fsm_state215;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state215))
    {
        ap_NS_fsm = ap_ST_fsm_state216;
    }
    else if (esl_seteq<1,215,215>(ap_CS_fsm.read(), ap_ST_fsm_state216))
    {
        if ((esl_seteq<1,1,1>(ap_const_logic_1, ap_CS_fsm_state216.read()) && esl_seteq<1,1,1>(RESULT_BVALID.read(), ap_const_logic_1))) {
            ap_NS_fsm = ap_ST_fsm_state1;
        } else {
            ap_NS_fsm = ap_ST_fsm_state216;
        }
    }
    else
    {
        ap_NS_fsm =  (sc_lv<215>) ("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX");
    }
}
}

