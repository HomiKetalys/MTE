//
// Created by kyuliea on 2025/3/29.
//
#include "mte_core.h"

#define DW_CONV_FUNCTION(KERNEL_H,KERNEL_W,STRIDE_H,STRIDE_W,DILATE_H,DILATE_W) dw_conv2d_##KERNEL_H##x##KERNEL_W##_stride_##STRIDE_H##_##STRIDE_W##_dilate_##DILATE_H##_##DILATE_W##_s8
#define DW_CONV_FUNCTION_KERNEL(KERNEL_H,KERNEL_W,STRIDE_H,STRIDE_W,DILATE_H,DILATE_W,CLIP) dw_conv2d_##KERNEL_H##x##KERNEL_W##_stride_##STRIDE_H##_##STRIDE_W##_dilate_##DILATE_H##_##DILATE_W##_s8_kernel_1x4_##CLIP

void DW_CONV_FUNCTION_KERNEL(5,5,1,1,1,1,ssat)(
    const int8_t *input, int32_t input_h, int32_t input_w,int32_t padded_input_w,
    const int8_t *weight, int32_t scale, int32_t bias,
    int32_t act_min, int32_t act_max,
    int8_t *output, int32_t output_h, int32_t output_w, int32_t output_ch, int32_t output_offset);


void DW_CONV_FUNCTION_KERNEL(5,5,1,1,1,1,minmax)(
    const int8_t *input, int32_t input_h, int32_t input_w,int32_t padded_input_w,
    const int8_t *weight, int32_t scale, int32_t bias,
    int32_t act_min, int32_t act_max,
    int8_t *output, int32_t output_h, int32_t output_w, int32_t output_ch, int32_t output_offset);

void DW_CONV_FUNCTION(5,5,1,1,1,1)(
    //input
    const int8_t *input,
    const int32_t input_h,
    const int32_t input_w,
    const int32_t input_ch,
    const int32_t input_offset,
    const int32_t pad_h_low,
    const int32_t pad_h_high,
    const int32_t pad_w_low,
    const int32_t pad_w_high,
    //kernel
    const int8_t *weight,
    int8_t *weight_cache,
    const int32_t *bias,
    int32_t *bias_cache,
    const int32_t *scale,
    int32_t *scale_cache,
    int8_t *buffer,
    //activation
    const int32_t act_min,
    const int32_t act_max,
    //output
    int8_t *output,
    const int32_t output_h,
    const int32_t output_w,
    const int32_t output_ch,
    const int32_t output_offset)
{

    int32_t i, j, ch;

    int8_t *input_buffer = buffer;
    if (weight_cache != 0) {
        memcpy(weight_cache, weight, output_ch * 5*5);
        weight = weight_cache;
    }
    if (bias_cache != 0) {
        memcpy(bias_cache, bias, output_ch * 4);
        bias = bias_cache;
    }
    if (scale_cache != 0) {
        memcpy(scale_cache, scale, output_ch * 4);
        scale = scale_cache;
    }


    void (*dw_conv_kernel_func)(
        const int8_t *input, int32_t input_h, int32_t input_w,int32_t padded_input_w,
        const int8_t *weight, int32_t scale, int32_t bias,
        int32_t act_min, int32_t act_max,
        int8_t *output, int32_t output_h, int32_t output_w, int32_t output_ch,int32_t output_offset) = NULL;
    if (act_min == -128 && act_max == 127)
        dw_conv_kernel_func = DW_CONV_FUNCTION_KERNEL(5,5,1,1,1,1,ssat);
    else
        dw_conv_kernel_func = DW_CONV_FUNCTION_KERNEL(5,5,1,1,1,1,minmax);
    const int32_t padded_input_w=input_w+pad_w_low+pad_w_high;
    for (i = 0; i < pad_h_low ; i++) {
        for(j=0;j<padded_input_w;j++)
            *input_buffer++ = input_offset;
    }
    for (i = 0; i < input_h; i++) {
        for(j=0;j<pad_w_low;j++)
        {
            *input_buffer++ = input_offset;
        }
        input_buffer += input_w;
        for(j=0;j<pad_w_high;j++) {
            *input_buffer++ = input_offset;
        }
    }
    for (i = 0; i < pad_h_high ; i++) {
        for(j=0;j<padded_input_w;j++)
            *input_buffer++ = input_offset;
    }

    const int8_t *input_p;

    for (ch = 0; ch < input_ch; ch++) {
        input_buffer = (buffer + padded_input_w*pad_h_low);//skip pad_h_low rows
        input_p = input++;
        for (i = 0; i < input_h; i++) {
            input_buffer += pad_w_low;
            for (j = 0; j < input_w; j++) {
                *input_buffer++ = *input_p;
                input_p += input_ch;
            }
            input_buffer += pad_w_high;
        }
        dw_conv_kernel_func(
            buffer, input_h, input_w,padded_input_w,
            weight, *scale++, *bias++,
            act_min, act_max,
            output++, output_h, output_w, output_ch, output_offset);
        weight += 5*5;
    }
}

//#define CREATE_DW_CONV_KERNEL_1xN(KERNEL_H, KERNEL_W, STRIDE_H, STRIDE_W, DILATE_H, DILATE_W, N)                                 \
//    void dw_conv_##KERNEL_H##x##KERNEL_W##_stride_##STRIDE_H##_##STRIDE_W##_dilate_##DIALATE_H##_##DILATE_W##_s8_kernel_1x##N##( \
//        const int8_t *input, const int32_t input_h, const int32_t input_w,const int32_t padded_input_w,                                                       \
//        const int8_t *weight, const int32_t scale, const int32_t bias,                                                           \
//        const int32_t act_min, const int32_t act_max,                                                                            \
//        int8_t *output, const int32_t output_h, const int32_t output_w, const int32_t output_ch, const int32_t output_offset)    \
//    {                                                                                                                            \
//        int32_t i, j, k, v;                                                                                                      \
//        int32_t res_col=output_w%##N##;                                                                                            \
//        for(i=0;i<output_h;i++){                                                                                                 \
//            for(j=0;j<output_w;j+=##N##){                                                                                        \
//                const int8_t *input_window=input;                                                                                \
//                GEN_SUM(##N##);                                                                                                  \
//                \
//            }\
//        }\
//    }

void DW_CONV_FUNCTION_KERNEL(5,5,1,1,1,1,ssat)(
    const int8_t *input, const int32_t input_h, const int32_t input_w,const int32_t padded_input_w,
    const int8_t *weight, const int32_t scale, const int32_t bias,
    const int32_t act_min, const int32_t act_max,
    int8_t *output, const int32_t output_h, const int32_t output_w, const int32_t output_ch, const int32_t output_offset)
{
    int32_t i, j, k, v;
    for (i = 0; i < output_h; i++) {
        int32_t res_col = output_w % 4;
        for (j = 0; j < output_w; j += 4) {
            const int8_t *input_window = input;

            int32_t sum0 = bias;
            int32_t sum1 = bias;
            int32_t sum2 = bias;
            int32_t sum3 = bias;

            for (k = 0; k < 5; k++) {
                for (v = 0; v < 5; v++) {
                    sum0 += input_window[v] * weight[5 * k + v];
                    sum1 += input_window[v + 1] * weight[5 * k + v];
                    sum2 += input_window[v + 2 * 1] * weight[5 * k + v];
                    sum3 += input_window[v + 3 * 1] * weight[5 * k + v];
                }
                input_window += padded_input_w;
            }
            sum0 = __SMMLAR(sum0, scale, output_offset);
            sum1 = __SMMLAR(sum1, scale, output_offset);
            sum2 = __SMMLAR(sum2, scale, output_offset);
            sum3 = __SMMLAR(sum3, scale, output_offset);

            sum0 = __SSAT8(sum0);
            sum1 = __SSAT8(sum1);
            sum2 = __SSAT8(sum2);
            sum3 = __SSAT8(sum3);

            *output = (int8_t) sum0;
            output += output_ch;
            *output = (int8_t) sum1;
            output += output_ch;
            *output = (int8_t) sum2;
            output += output_ch;
            *output = (int8_t) sum3;
            output += output_ch;

            input += 1 * 4;
        }
        while (res_col--) {
            const int8_t *input_window = input;
            int32_t sum0 = bias;
            for (k = 0; k < 5; k++) {
                for (v = 0; v < 5; v++) {
                    sum0 += input_window[v] * weight[5 * k + v];
                }
                input_window += padded_input_w;
            }
            sum0 = __SMMLAR(sum0, scale, output_offset);
            sum0 = __SSAT8(sum0);
            *output = (int8_t) sum0;
            output += output_ch;
            input += 1;
        }
        input += 1 * padded_input_w-input_w;
    }
}

void DW_CONV_FUNCTION_KERNEL(5,5,1,1,1,1,minmax)(
    const int8_t *input, const int32_t input_h, const int32_t input_w,const int32_t padded_input_w,
    const int8_t *weight, const int32_t scale, const int32_t bias,
    const int32_t act_min, const int32_t act_max,
    int8_t *output, const int32_t output_h, const int32_t output_w, const int32_t output_ch, const int32_t output_offset)
{
    int32_t i, j, k, v;
    for (i = 0; i < output_h; i++) {
        int32_t res_col = output_w % 4;
        for (j = 0; j < output_w; j += 4) {
            const int8_t *input_window = input;

            int32_t sum0 = bias;
            int32_t sum1 = bias;
            int32_t sum2 = bias;
            int32_t sum3 = bias;

            for (k = 0; k < 5; k++) {
                for (v = 0; v < 5; v++) {
                    sum0 += input_window[v] * weight[5 * k + v];
                    sum1 += input_window[v + 1] * weight[5 * k + v];
                    sum2 += input_window[v + 2 * 1] * weight[5 * k + v];
                    sum3 += input_window[v + 3 * 1] * weight[5 * k + v];
                }
                input_window += padded_input_w;
            }
            sum0 = __SMMLAR(sum0, scale, output_offset);
            sum1 = __SMMLAR(sum1, scale, output_offset);
            sum2 = __SMMLAR(sum2, scale, output_offset);
            sum3 = __SMMLAR(sum3, scale, output_offset);

            sum0 = MAX(sum0,act_min);
            sum1 = MAX(sum1,act_min);
            sum2 = MAX(sum2,act_min);
            sum3 = MAX(sum3,act_min);

            sum0 = MIN(sum0,act_max);
            sum1 = MIN(sum1,act_max);
            sum2 = MIN(sum2,act_max);
            sum3 = MIN(sum3,act_max);

            *output = (int8_t) sum0;
            output += output_ch;
            *output = (int8_t) sum1;
            output += output_ch;
            *output = (int8_t) sum2;
            output += output_ch;
            *output = (int8_t) sum3;
            output += output_ch;

            input += 1 * 4;
        }
        while (res_col--) {
            const int8_t *input_window = input;
            int32_t sum0 = bias;
            for (k = 0; k < 5; k++) {
                for (v = 0; v < 5; v++) {
                    sum0 += input_window[v] * weight[5 * k + v];
                }
                input_window += padded_input_w;
            }
            sum0 = __SMMLAR(sum0, scale, output_offset);
            sum0 = MAX(sum0,act_min);
            sum0 = MIN(sum0,act_max);
            *output = (int8_t) sum0;
            output += output_ch;
            input += 1;
        }
        input += 1 * padded_input_w-input_w;
    }
}