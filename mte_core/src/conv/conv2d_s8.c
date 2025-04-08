//
// Created by kyuliea on 2024/11/3.
//
#include "mat_mul_kernels.h"
#include "mte_core.h"

void conv2d_1x1_s8(
    //input
    const int8_t *input,
    const int32_t input_h,
    const int32_t input_w,
    const int32_t input_ch,
    const int32_t input_offset,
    //kernel
    const int8_t *weight,
    int8_t *weight_cache,
    const int32_t *bias,
    int32_t *bias_cache,
    const int32_t *scale,
    int32_t *scale_cache,
    int8_t *buffer,
    const int32_t stride_h,
    const int32_t stride_w,
    //activation
    const int32_t act_min,
    const int32_t act_max,
    //output
    int8_t *output,
    const int32_t output_h,
    const int32_t output_w,
    const int32_t output_ch,
    const int32_t output_offset){

    int32_t i;

    const int32_t num_elements = output_h * output_w;
    const int32_t channel_div4 = (input_ch >> 2);
    const int32_t offset_i16x2 = __PKHBT(-input_offset, -input_offset, 16);

    const int32_t weight_byte_size = input_ch * output_ch;
    const int32_t bias_byte_size = 4 * output_ch;
    const int32_t scale_byte_size = bias_byte_size;
    if(weight_cache!=0) {
        memcpy(weight_cache, weight, weight_byte_size);
        weight=weight_cache;
    }
    if(bias_cache!=0) {
        memcpy(bias_cache, bias, bias_byte_size);
        bias=bias_cache;
    }
    if(scale_cache!=0) {
        memcpy(scale_cache, scale, scale_byte_size);
        scale=scale_cache;
    }

    int8_t *(*mat_mult_kernel_2col_func)(
        //input a
        const int8_t *input_a, const int32_t *bias, const int32_t *scale,
        //input b
        const int16_t *input_b, const int32_t input_ch,
        //output
        int8_t *output, const int32_t output_ch, const int32_t output_offset,
        //activation
        const int32_t act_min, const int32_t act_max)=NULL;
    if (act_min == -128 && act_max == 127) {
        if (input_ch % 8 == 0)
            mat_mult_kernel_2col_func = mat_mult_kernel_s8_s16_reordered_2col_8ch_ssat;
        else if (input_ch % 4 == 0)
            mat_mult_kernel_2col_func = mat_mult_kernel_s8_s16_reordered_2col_4ch_ssat;
    }
    else {
        if (input_ch % 8 == 0)
            mat_mult_kernel_2col_func = mat_mult_kernel_s8_s16_reordered_2col_8ch;
        else if (input_ch % 4 == 0)
            mat_mult_kernel_2col_func = mat_mult_kernel_s8_s16_reordered_2col_4ch;
    }

    const int32_t var0 = input_w * input_ch * stride_h;
    const int32_t var1 = input_ch * stride_w;
    int16_t *input_buffer=(int16_t *)buffer;
    for (i = 0; i < num_elements / 2; i++) {
        /* Fill buffer for partial im2col - two columns at a time */
        const int32_t _2i = 2 * i;
        const int8_t *src = &input[(_2i / output_w) * var0 + (_2i % output_w) * var1];
        int16_t *dst = input_buffer;
        uint32_t cnt = channel_div4;//two columns
        while (cnt > 0) {
            i8x4_to_2xi16x2_offset_reordered_ele_ia((int32_t **) &src, (int32_t **) &dst, offset_i16x2);
            cnt--;
        }
        src = &input[((_2i + 1) / output_w) * var0 + ((_2i + 1) % output_w) * var1];
        cnt = channel_div4;//two columns
        while (cnt > 0) {
            i8x4_to_2xi16x2_offset_reordered_ele_ia((int32_t **) &src, (int32_t **) &dst, offset_i16x2);
            cnt--;
        }
        output = mat_mult_kernel_2col_func(
            weight, bias, scale,
            input_buffer, input_ch,
            output, output_ch, output_offset,
            act_min, act_max);
    }

    /* check if there is an odd column left-over for computation */
    if (num_elements & 0x1) {
        const uint32_t _2i = num_elements & 0x1;
        const int8_t *src = &input[(_2i / output_w) * var0 + (_2i % output_w) * var1];
        int16_t *dst = input_buffer;
        uint32_t cnt = channel_div4;//two * num of 2col columns
        while (cnt > 0) {
            i8x4_to_2xi16x2_offset_reordered_ele_ia((int32_t **) &src, (int32_t **) &dst, offset_i16x2);
            cnt--;
        }
        int8_t *(*mat_mult_kernel_1col_func)(
            //input a
            const int8_t *input_a, const int32_t *bias, const int32_t *scale,
            //input b
            const int16_t *input_b, const int32_t input_ch,
            //output
            int8_t *output, const int32_t output_ch, const int32_t output_offset,
            //activation
            const int32_t act_min, const int32_t act_max)=NULL;
        if (act_min == -128 && act_max == 127) {
            if (input_ch % 8 == 0)
                mat_mult_kernel_1col_func = mat_mult_kernel_s8_s16_reordered_1col_8ch_ssat;
            else if (input_ch % 4 == 0)
                mat_mult_kernel_1col_func = mat_mult_kernel_s8_s16_reordered_1col_4ch_ssat;
        }
        else {
            if (input_ch % 8 == 0)
                mat_mult_kernel_1col_func = mat_mult_kernel_s8_s16_reordered_1col_8ch;
            else if (input_ch % 4 == 0)
                mat_mult_kernel_1col_func = mat_mult_kernel_s8_s16_reordered_1col_4ch;
        }
        mat_mult_kernel_1col_func(
            weight, bias, scale,
            input_buffer, input_ch,
            output, output_ch, output_offset,
            act_min, act_max);
    }
}
