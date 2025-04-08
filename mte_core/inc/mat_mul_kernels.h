//
// Created by kyuliea on 2024/10/29.
//

#ifndef MTE_MAT_MUL_KERNELS_H
#define MTE_MAT_MUL_KERNELS_H
#include "mte_core.h"

#define CREATE_MAT_MUL_KERNEL_DECLARE(COL_NUM, IN_CH_NUM, OUT_CH_NUM, MINMAX_TYPE, KERNEL_INFO)                      \
    int8_t *mat_mult_kernel_s8_s16r_##COL_NUM##col_##IN_CH_NUM##ich_##OUT_CH_NUM##och_##MINMAX_TYPE##_##KERNEL_INFO( \
        const int8_t *input_a, const int32_t *bias, const int32_t *scale,      /* input a    */                      \
        const int16_t *input_b, const uint32_t input_ch,                       /* input b    */                      \
        int8_t *output, const uint32_t output_ch, const int32_t output_offset, /* output     */                      \
        const int32_t act_min, const int32_t act_max);                         /* activation */


CREATE_MAT_MUL_KERNEL_DECLARE(2, 4, 2, ssat, spl)
CREATE_MAT_MUL_KERNEL_DECLARE(1, 4, 2, ssat, spl)
CREATE_MAT_MUL_KERNEL_DECLARE(2, 4, 2, minmax, spl)
CREATE_MAT_MUL_KERNEL_DECLARE(1, 4, 2, minmax, spl)

CREATE_MAT_MUL_KERNEL_DECLARE(2, 8, 2, ssat, spl)
CREATE_MAT_MUL_KERNEL_DECLARE(1, 8, 2, ssat, spl)
CREATE_MAT_MUL_KERNEL_DECLARE(2, 8, 2, minmax, spl)
CREATE_MAT_MUL_KERNEL_DECLARE(1, 8, 2, minmax, spl)


int8_t *mat_mult_kernel_s8_s16_reordered_1col_4ch_ssat(
    //input a
    const int8_t *input_a, const int32_t *bias, const int32_t *scale,
    //input b
    const int16_t *input_b, const int32_t input_ch,
    //output
    int8_t *output, const int32_t output_ch, const int32_t output_offset,
    //activation
    const int32_t act_min, const int32_t act_max);

int8_t *mat_mult_kernel_s8_s16_reordered_1col_4ch(
    //input a
    const int8_t *input_a, const int32_t *bias, const int32_t *scale,
    //input b
    const int16_t *input_b, const int32_t input_ch,
    //output
    int8_t *output, const int32_t output_ch, const int32_t output_offset,
    //activation
    const int32_t act_min, const int32_t act_max);

int8_t *mat_mult_kernel_s8_s16_reordered_2col_4ch_ssat(
    //input a
    const int8_t *input_a, const int32_t *bias, const int32_t *scale,
    //input b
    const int16_t *input_b, const int32_t input_ch,
    //output
    int8_t *output, const int32_t output_ch, const int32_t output_offset,
    //activation
    const int32_t act_min, const int32_t act_max);

int8_t *mat_mult_kernel_s8_s16_reordered_2col_4ch(
    //input a
    const int8_t *input_a, const int32_t *bias, const int32_t *scale,
    //input b
    const int16_t *input_b, const int32_t input_ch,
    //output
    int8_t *output, const int32_t output_ch, const int32_t output_offset,
    //activation
    const int32_t act_min, const int32_t act_max);

int8_t *mat_mult_kernel_s8_s16_reordered_1col_8ch_ssat(
    //input a
    const int8_t *input_a, const int32_t *bias, const int32_t *scale,
    //input b
    const int16_t *input_b, const int32_t input_ch,
    //output
    int8_t *output, const int32_t output_ch, const int32_t output_offset,
    //activation
    const int32_t act_min, const int32_t act_max);

int8_t *mat_mult_kernel_s8_s16_reordered_1col_8ch(
    //input a
    const int8_t *input_a, const int32_t *bias, const int32_t *scale,
    //input b
    const int16_t *input_b, const int32_t input_ch,
    //output
    int8_t *output, const int32_t output_ch, const int32_t output_offset,
    //activation
    const int32_t act_min, const int32_t act_max);

int8_t *mat_mult_kernel_s8_s16_reordered_2col_8ch_ssat(
    //input a
    const int8_t *input_a, const int32_t *bias, const int32_t *scale,
    //input b
    const int16_t *input_b, const int32_t input_ch,
    //output
    int8_t *output, const int32_t output_ch, const int32_t output_offset,
    //activation
    const int32_t act_min, const int32_t act_max);

int8_t *mat_mult_kernel_s8_s16_reordered_2col_8ch(
    //input a
    const int8_t *input_a, const int32_t *bias, const int32_t *scale,
    //input b
    const int16_t *input_b, const int32_t input_ch,
    //output
    int8_t *output, const int32_t output_ch, const int32_t output_offset,
    //activation
    const int32_t act_min, const int32_t act_max);

#endif//MTE_MAT_MUL_KERNELS_H
