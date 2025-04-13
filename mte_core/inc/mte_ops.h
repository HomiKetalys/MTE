//
// Created by kyuliea on 2025/4/1.
//

#ifndef MTE_OPS_H
#define MTE_OPS_H

#include "mte_core.h"
#include "mte_custom_ops.h"

void conv2d_input_3_3x3_stride_2_2_dilate_1_1_s8(
    const int8_t *input, int32_t input_h, int32_t input_w,int32_t input_ch,int32_t input_offset,
    const int16_t *weight,int16_t *weight_cache,
    const int32_t *bias,int32_t *bias_cache,
    const int32_t *scale,int32_t *scale_cache,
    int8_t *input_buffer,
    int32_t act_min,int32_t act_max,
    int8_t *output,
    int32_t output_h,int32_t output_w,int32_t output_ch,int32_t output_offset
    );

void conv2d_1x1_s8(
    //input
    const int8_t *input,
    int32_t input_h,
    int32_t input_w,
    int32_t input_ch,
    int32_t input_offset,
    //kernel
    const int8_t *weight,
    int8_t *weight_cache,
    const int32_t *bias,
    int32_t *bias_cache,
    const int32_t *scale,
    int32_t *scale_cache,
    int8_t *buffer,
    int32_t stride_h,
    int32_t stride_w,
    //activation
    int32_t act_min,
    int32_t act_max,
    //output
    int8_t *output,
    int32_t output_h,
    int32_t output_w,
    int32_t output_ch,
    int32_t output_offset);

void dw_conv2d_3x3_stride_1_1_dilate_1_1_s8(
    //input
    const int8_t *input,
    int32_t input_h,
    int32_t input_w,
    int32_t input_ch,
    int32_t input_offset,
    int32_t pad_h_low,
    int32_t pad_h_high,
    int32_t pad_w_low,
    int32_t pad_w_high,
    //kernel
    const int8_t *weight,
    int8_t *weight_cache,
    const int32_t *bias,
    int32_t *bias_cache,
    const int32_t *scale,
    int32_t *scale_cache,
    int8_t *buffer,
    //activation
    int32_t act_min,
    int32_t act_max,
    //output
    int8_t *output,
    int32_t output_h,
    int32_t output_w,
    int32_t output_ch,
    int32_t output_offset);


void dw_conv2d_3x3_stride_2_2_dilate_1_1_s8(
    //input
    const int8_t *input,
    int32_t input_h,
    int32_t input_w,
    int32_t input_ch,
    int32_t input_offset,
    int32_t pad_h_low,
    int32_t pad_h_high,
    int32_t pad_w_low,
    int32_t pad_w_high,
    //kernel
    const int8_t *weight,
    int8_t *weight_cache,
    const int32_t *bias,
    int32_t *bias_cache,
    const int32_t *scale,
    int32_t *scale_cache,
    int8_t *buffer,
    //activation
    int32_t act_min,
    int32_t act_max,
    //output
    int8_t *output,
    int32_t output_h,
    int32_t output_w,
    int32_t output_ch,
    int32_t output_offset);


void dw_conv2d_5x5_stride_1_1_dilate_1_1_s8(
    //input
    const int8_t *input,
    int32_t input_h,
    int32_t input_w,
    int32_t input_ch,
    int32_t input_offset,
    int32_t pad_h_low,
    int32_t pad_h_high,
    int32_t pad_w_low,
    int32_t pad_w_high,
    //kernel
    const int8_t *weight,
    int8_t *weight_cache,
    const int32_t *bias,
    int32_t *bias_cache,
    const int32_t *scale,
    int32_t *scale_cache,
    int8_t *buffer,
    //activation
    int32_t act_min,
    int32_t act_max,
    //output
    int8_t *output,
    int32_t output_h,
    int32_t output_w,
    int32_t output_ch,
    int32_t output_offset);

void dw_conv2d_7x7_stride_1_1_dilate_1_1_s8(
    //input
    const int8_t *input,
    int32_t input_h,
    int32_t input_w,
    int32_t input_ch,
    int32_t input_offset,
    int32_t pad_h_low,
    int32_t pad_h_high,
    int32_t pad_w_low,
    int32_t pad_w_high,
    //kernel
    const int8_t *weight,
    int8_t *weight_cache,
    const int32_t *bias,
    int32_t *bias_cache,
    const int32_t *scale,
    int32_t *scale_cache,
    int8_t *buffer,
    //activation
    int32_t act_min,
    int32_t act_max,
    //output
    int8_t *output,
    int32_t output_h,
    int32_t output_w,
    int32_t output_ch,
    int32_t output_offset);

void add(const int8_t *input0, int32_t scale0,
         const int8_t *input1, int32_t scale1,
         int32_t ele_nums,
         int8_t *output, int32_t offset);

#define CREATE_ELE_FUNC_DECLARE(NAME, OUTPUT_TYPE, OUTPUT_BYTES) \
    void mte_##NAME(                                     \
        const int8_t *input, int32_t ele_nums,           \
        const OUTPUT_TYPE *map, OUTPUT_TYPE *map_cache,  \
        OUTPUT_TYPE *output);

CREATE_ELE_FUNC_DECLARE(tanh,int8_t,1)
CREATE_ELE_FUNC_DECLARE(sigmoid,int8_t,1)
CREATE_ELE_FUNC_DECLARE(quantize, int8_t, 1)
CREATE_ELE_FUNC_DECLARE(dequantize, float, 4)

void concat(const int32_t  *input_addrs,const int32_t *channels,int32_t addr_nums,int32_t concat_nums,int8_t *output);

void gather(const int8_t *input,int32_t gather_nums,int32_t gather_block_size,int32_t block_size,
            const int32_t *gather_idx,int32_t idx_nums,int8_t *output);

void max_pool2d(
    const int8_t* input, int32_t input_h, int32_t input_w,uint16_t input_ch,int32_t input_offset,
    int32_t pad_h_low,int32_t pad_h_high,int32_t pad_w_low,int32_t pad_w_high,
    int32_t kernel_h, int32_t kernel_w,int32_t stride_h,int32_t stride_w,
    int32_t act_min,int32_t act_max,
    int8_t *output,int32_t output_h, int32_t output_w);

void pack(const int8_t  *input_addrs,int32_t channels,int32_t pack_nums,int32_t repeat_nums,int8_t *output);

void transpose(
    const int8_t *input, int32_t ele_nums,const int32_t *perm, const int32_t *input_shape,const int32_t *output_shape,int32_t dim_nums,
    int32_t *idx_buffer,
    int8_t *output);

void mte_softmax(
    const int8_t *input,int32_t nums0,int32_t n,int32_t nums1,
    const float *map0,float *map0_cache,const float *map1,float *map1_cache,
    int8_t *output,int32_t output_offset
);

#endif//MTE_OPS_H
