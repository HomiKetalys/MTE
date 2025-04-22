//
// Created by kyuliea on 2025/4/13.
//

#ifndef AVG_POOL2D_H
#define AVG_POOL2D_H
#include "mte_core.h"

void avg_pool2d(
    const int8_t* input, int32_t input_h, int32_t input_w,uint16_t input_ch,int32_t input_offset,
    int32_t pad_h_low,int32_t pad_h_high,int32_t pad_w_low,int32_t pad_w_high,
    int32_t kernel_h, int32_t kernel_w,int32_t stride_h,int32_t stride_w,
    int32_t act_min,int32_t act_max,
    int8_t *output,int32_t output_h, int32_t output_w);

#endif//AVG_POOL2D_H
