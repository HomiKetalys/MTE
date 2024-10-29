//
// Created by kyuliea on 2024/10/29.
//

#include "../inc/mte_core.h"


#define GEN_SCALE_i(i) const int32_t scale_##i = *scale++
#define GEN_IPB_i(i) const int16_t *ip_b##i = input_b + i * input_ch;
#define GEN_IPA_i(i) const int8_t *ip_a##i = input_a + i * input_ch;
#define GEN_SCALES(N) REPEAT_ALIGN_ID(N, GEN_SCALE_i)
#define GEN_IPAS(N) REPEAT_ALIGN_ID(N, GEN_IPA_i)
#define GEN_IPBS(N) REPEAT_ALIGN_ID(N, GEN_IPB_i)


#define GEN_CHOUT_INIT_nm(n, m) int32_t ch_##n##_out_##m = *bias
#define GEN_CHOUT_INIT(N, M) REPEAT_ALIGN_2_ID_A(N, M, GEN_CHOUT_INIT_nm, bias++)

#define GEN_CHOUT_REQ_nm(n, m) ch_##n##_out_##m = __SMMLAR(ch_##n##_out_##m, scale_0, output_offset)
#define GEN_CHOUT_REQ(N, M) REPEAT_ALIGN_2_ID(N, M, GEN_CHOUT_REQ_nm)

#define GEN_CHOUT_MAX_nm(n, m) ch_##n##_out_##m = MAX(ch_##n##_out_##m, act_min)
#define GEN_CHOUT_MIN_nm(n, m) ch_##n##_out_##m = MIN(ch_##n##_out_##m, act_max)
#define GEN_CHOUT_SSAT_nm(n, m) ch_##n##_out_##m = __SSAT8(ch_##n##_out_##m)

#define GEN_CHOUT_ssat(N, M) REPEAT_ALIGN_2_ID(N, M, GEN_CHOUT_SSAT_nm)
#define GEN_CHOUT_minmax(N, M, ACT_MIN, ACT_MAX) \
    REPEAT_ALIGN_2_ID(N, M, GEN_CHOUT_MAX_nm);   \
    REPEAT_ALIGN_2_ID(N, M, GEN_CHOUT_MIN_nm)

#define GEN_OUT_i(i) int8_t *out_##i = output + i * output_ch
#define GEN_OUT(N) REPEAT_ALIGN_ID(N, GEN_OUT_i)

#define GEN_OUT_WRITE_nm(n, m) *out_##m = (int8_t) ch_##n##_out_##m
#define GEN_OUT_WRITE(N, M) REPEAT_ALIGN_2_ID(N, M, GEN_OUT_WRITE_nm)

#define MAT_MUL_KERNEL_1_4_2_norm                  \
    while (col_count) {                            \
        int32_t a01, a02, a11, a12, b0;            \
        b0 = mte_read_q15x2_ia(&ip_b0);            \
        a02 = mte_read_s8x4_ia(&ip_a0);            \
        a01 = __SXTB16(a02);                       \
        a02 = __SXTB16_ROR8(a02);                  \
        ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0); \
        a12 = mte_read_s8x4_ia(&ip_a1);            \
        a11 = __SXTB16(a12);                       \
        a12 = __SXTB16_ROR8(a12);                  \
        ch_1_out_0 = __SMLAD(a11, b0, ch_1_out_0); \
        b0 = mte_read_q15x2_ia(&ip_b0);            \
        ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0); \
        ch_1_out_0 = __SMLAD(a12, b0, ch_1_out_0); \
        col_count--;                               \
    }

#define MAT_MUL_KERNEL_1_4_1_norm                  \
while                                              \
    (col_count)                                    \
    {                                              \
        int32_t a01, a02, b0;                      \
        b0 = mte_read_q15x2_ia(&ip_b0);            \
        a02 = mte_read_s8x4_ia(&ip_a0);            \
        a01 = __SXTB16(a02);                       \
        a02 = __SXTB16_ROR8(a02);                  \
        ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0); \
        b0 = mte_read_q15x2_ia(&ip_b0);            \
        ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0); \
        col_count--;                               \
    }

#define CREATE_MAT_MUL_KERNEL(COL_NUM, IN_CH_NUM, OUT_CH_NUM, MINMAX_TYPE, KERNEL_INFO)                              \
    int8_t *mat_mult_kernel_s8_s16r_##COL_NUM##col_##IN_CH_NUM##ich_##OUT_CH_NUM##och_##MINMAX_TYPE##_##KERNEL_INFO( \
        const int8_t *input_a, const int32_t *bias, const int32_t *scale,      /* input a    */                      \
        const int16_t *input_b, const uint32_t input_ch,                       /* input b    */                      \
        int8_t *output, const uint32_t output_ch, const int32_t output_offset, /* output     */                      \
        const int32_t act_min, const int32_t act_max)                          /* activation */                      \
    {                                                                                                                \
        GEN_OUT(COL_NUM);                                                                                            \
        uint32_t row_count = output_ch / OUT_CH_NUM;                                                                 \
        while (row_count) {                                                                                          \
            GEN_SCALES(OUT_CH_NUM);                                                                                  \
            GEN_CHOUT_INIT(OUT_CH_NUM, COL_NUM);                                                                     \
            GEN_IPBS(OUT_CH_NUM);                                                                                    \
            GEN_IPAS(OUT_CH_NUM);                                                                                    \
            uint32_t col_count = input_ch / IN_CH_NUM;                                                               \
            MAT_MUL_KERNEL_##COL_NUM##_##IN_CH_NUM##_##OUT_CH_NUM##_##KERNEL_INFO;                                   \
            GEN_CHOUT_REQ(OUT_CH_NUM, COL_NUM);                                                                      \
            GEN_CHOUT_##MINMAX_TYPE(OUT_CH_NUM, COL_NUM);                                                            \
            GEN_OUT_WRITE(OUT_CH_NUM, COL_NUM);                                                                      \
            input_a += OUT_CH_NUM * input_ch;                                                                        \
            row_count--;                                                                                             \
        }                                                                                                            \
        row_count = output_ch % OUT_CH_NUM;                                                                          \
        while (row_count) {                                                                                          \
            GEN_SCALES(1);                                                                                           \
            GEN_CHOUT_INIT(1, COL_NUM);                                                                              \
            GEN_IPBS(1);                                                                                             \
            GEN_IPAS(1);                                                                                             \
            uint32_t col_count = input_ch / IN_CH_NUM;                                                               \
            MAT_MUL_KERNEL_##COL_NUM##_##IN_CH_NUM##_##1##_##KERNEL_INFO;                                            \
            GEN_CHOUT_REQ(1, COL_NUM);                                                                               \
            GEN_CHOUT_##MINMAX_TYPE(1, COL_NUM);                                                                     \
            GEN_OUT_WRITE(1, COL_NUM);                                                                               \
            input_a += input_ch;                                                                                     \
            row_count--;                                                                                             \
        }                                                                                                            \
        out_0 += COL_NUM * input_ch;                                                                                 \
        return out_0;                                                                                                \
    }

CREATE_MAT_MUL_KERNEL(1, 4, 2, ssat, norm)

int8_t *mat_mult_kernel_s8_s16_reordered_1col_4ch_ssat(
    //input a
    const int8_t *input_a, const int32_t *bias, const int32_t *scale,
    //input b
    const int16_t *input_b, const uint32_t input_ch,
    //output
    int8_t *output, const uint32_t output_ch, const int32_t output_offset,
    //activation
    const int32_t act_min, const int32_t act_max)
{
    /* Set the output pointer, because the input is one column, the output is also one column. */
    int8_t *out_0 = output;

    /* Set the pointer to matrix A */
    const int8_t *ip_a0 = input_a;

    /* Extract two rows from matrix A each time to calculate the results of two channels, so the number of iterations is half that of the output channel.*/
    uint32_t row_count = output_ch / 2;
    /* this loop over rows in A */
    while (row_count) {

        /* Matrix A shares the same scaling factor for each row and processes two rows at a time.*/
        const int32_t scale_0 = *scale++;
        const int32_t scale_1 = *scale++;

        /* Each channel uses the same bias and has one column, so each channel has one outputs. */
        /* channel 0*/
        int32_t ch_0_out_0 = *bias++;
        /* channel 1*/
        int32_t ch_1_out_0 = *bias++;

        /* Set pointers for the one columns of input */
        const int16_t *ip_b0 = input_b;

        /* Set the pointer to the second row of each two rows in matrix A */
        const int8_t *ip_a1 = ip_a0 + input_ch;

        /* Each time four elements are taken from a row of matrix A, a loop is repeated twice, so the number of loops is one eighth of the number of channels */
        uint32_t col_count = input_ch / 4;

        /* accumulate over the vector */
        while (col_count) {
            int32_t a01, a02, a11, a12, b0;

            b0 = mte_read_q15x2_ia(&ip_b0);
            a02 = mte_read_s8x4_ia(&ip_a0);
            a01 = __SXTB16(a02);
            a02 = __SXTB16_ROR8(a02);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            a12 = mte_read_s8x4_ia(&ip_a1);
            a11 = __SXTB16(a12);
            a12 = __SXTB16_ROR8(a12);
            ch_1_out_0 = __SMLAD(a11, b0, ch_1_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);

            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            ch_1_out_0 = __SMLAD(a12, b0, ch_1_out_0);
            col_count--;
        }
        ch_0_out_0 = __SMMLAR(ch_0_out_0, scale_0, output_offset);
        ch_1_out_0 = __SMMLAR(ch_1_out_0, scale_1, output_offset);
        ch_0_out_0 = MAX(ch_0_out_0, act_min);
        ch_1_out_0 = MAX(ch_1_out_0, act_min);
        ch_0_out_0 = MIN(ch_0_out_0, act_max);
        ch_1_out_0 = MIN(ch_1_out_0, act_max);
        *out_0++ = (int8_t) ch_0_out_0;
        *out_0++ = (int8_t) ch_1_out_0;

        /* The first row of each two rows in matrix A has already been moved by one row in matrix multiplication.
        Here, add another row, and this will be the first row of the next two rows */
        ip_a0 += input_ch;

        row_count--;
    }

    if (output_ch & 1) {

        /* Matrix A shares the same scaling factor for each row and processes two rows at a time.*/
        const int32_t scale_0 = *scale++;

        /* Each channel uses the same bias and has two columns, so each channel has two outputs. */
        /* channel 0*/
        int32_t ch_0_out_0 = *bias++;

        /* Set pointers for the two columns of input */
        const int16_t *ip_b0 = input_b;

        uint32_t col_count = input_ch / 4;
        while (col_count) {
            int32_t a01, a02, b0;
            b0 = mte_read_q15x2_ia(&ip_b0);

            a02 = mte_read_s8x4_ia(&ip_a0);
            a01 = __SXTB16(a02);
            a02 = __SXTB16_ROR8(a02);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);
            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);

            col_count--;
        } /* while over col_count */

        ch_0_out_0 = __SMMLAR(ch_0_out_0, scale_0, output_offset);
        ch_0_out_0 = MAX(ch_0_out_0, act_min);
        ch_0_out_0 = MIN(ch_0_out_0, act_max);
        *out_0++ = (int8_t) ch_0_out_0;
    }

    /* return the new output pointer with offset */
    return out_0;
}

int8_t *mat_mult_kernel_s8_s16_reordered_1col_4ch(
    //input a
    const int8_t *input_a, const int32_t *bias, const int32_t *scale,
    //input b
    const int16_t *input_b, const uint32_t input_ch,
    //output
    int8_t *output, const uint32_t output_ch, const int32_t output_offset,
    //activation
    const int32_t act_min, const int32_t act_max)
{
    /* Set the output pointer, because the input is one column, the output is also one column. */
    int8_t *out_0 = output;

    /* Set the pointer to matrix A */
    const int8_t *ip_a0 = input_a;

    /* Extract two rows from matrix A each time to calculate the results of two channels, so the number of iterations is half that of the output channel.*/
    uint32_t row_count = output_ch / 2;
    /* this loop over rows in A */
    while (row_count) {

        /* Matrix A shares the same scaling factor for each row and processes two rows at a time.*/
        const int32_t scale_0 = *scale++;
        const int32_t scale_1 = *scale++;

        /* Each channel uses the same bias and has one column, so each channel has one outputs. */
        /* channel 0*/
        int32_t ch_0_out_0 = *bias++;
        /* channel 1*/
        int32_t ch_1_out_0 = *bias++;

        /* Set pointers for the one columns of input */
        const int16_t *ip_b0 = input_b;

        /* Set the pointer to the second row of each two rows in matrix A */
        const int8_t *ip_a1 = ip_a0 + input_ch;

        /* Each time four elements are taken from a row of matrix A, a loop is repeated twice, so the number of loops is one eighth of the number of channels */
        uint32_t col_count = input_ch / 4;

        /* accumulate over the vector */
        while (col_count) {
            int32_t a01, a02, a11, a12, b0;

            b0 = mte_read_q15x2_ia(&ip_b0);
            a02 = mte_read_s8x4_ia(&ip_a0);
            a01 = __SXTB16(a02);
            a02 = __SXTB16_ROR8(a02);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            a12 = mte_read_s8x4_ia(&ip_a1);
            a11 = __SXTB16(a12);
            a12 = __SXTB16_ROR8(a12);
            ch_1_out_0 = __SMLAD(a11, b0, ch_1_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);

            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            ch_1_out_0 = __SMLAD(a12, b0, ch_1_out_0);
            col_count--;
        }
        ch_0_out_0 = __SMMLAR(ch_0_out_0, scale_0, output_offset);
        ch_1_out_0 = __SMMLAR(ch_1_out_0, scale_1, output_offset);
        ch_0_out_0 = MAX(ch_0_out_0, act_min);
        ch_1_out_0 = MAX(ch_1_out_0, act_min);
        ch_0_out_0 = MIN(ch_0_out_0, act_max);
        ch_1_out_0 = MIN(ch_1_out_0, act_max);
        *out_0++ = (int8_t) ch_0_out_0;
        *out_0++ = (int8_t) ch_1_out_0;

        /* The first row of each two rows in matrix A has already been moved by one row in matrix multiplication.
        Here, add another row, and this will be the first row of the next two rows */
        ip_a0 += input_ch;

        row_count--;
    }

    if (output_ch & 1) {

        /* Matrix A shares the same scaling factor for each row and processes two rows at a time.*/
        const int32_t scale_0 = *scale++;

        /* Each channel uses the same bias and has two columns, so each channel has two outputs. */
        /* channel 0*/
        int32_t ch_0_out_0 = *bias++;

        /* Set pointers for the two columns of input */
        const int16_t *ip_b0 = input_b;

        uint32_t col_count = input_ch / 4;
        while (col_count) {
            int32_t a01, a02, b0;
            b0 = mte_read_q15x2_ia(&ip_b0);

            a02 = mte_read_s8x4_ia(&ip_a0);
            a01 = __SXTB16(a02);
            a02 = __SXTB16_ROR8(a02);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);
            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);

            col_count--;
        } /* while over col_count */

        ch_0_out_0 = __SMMLAR(ch_0_out_0, scale_0, output_offset);
        ch_0_out_0 = MAX(ch_0_out_0, act_min);
        ch_0_out_0 = MIN(ch_0_out_0, act_max);
        *out_0++ = (int8_t) ch_0_out_0;
    }

    /* return the new output pointer with offset */
    return out_0;
}

int8_t *mat_mult_kernel_s8_s16_reordered_2col_4ch_ssat(
    //input a
    const int8_t *input_a, const int32_t *bias, const int32_t *scale,
    //input b
    const int16_t *input_b, const uint32_t input_ch,
    //output
    int8_t *output, const uint32_t output_ch, const int32_t output_offset,
    //activation
    const int32_t act_min, const int32_t act_max)
{
    /* Set the output pointer, because the input is two columns, the output is also two columns. */
    int8_t *out_0 = output;
    int8_t *out_1 = out_0 + output_ch;

    /* Set the pointer to matrix A */
    const int8_t *ip_a0 = input_a;

    /* Extract two rows from matrix A each time to calculate the results of two channels, so the number of iterations is half that of the output channel.*/
    uint32_t row_count = output_ch / 2;
    /* this loop over rows in A */
    while (row_count) {

        /* Matrix A shares the same scaling factor for each row and processes two rows at a time.*/
        const int32_t scale_0 = *scale++;
        const int32_t scale_1 = *scale++;

        /* Each channel uses the same bias and has two columns, so each channel has two outputs. */
        /* channel 0*/
        int32_t ch_0_out_0 = *bias;
        int32_t ch_0_out_1 = *bias++;
        /* channel 1*/
        int32_t ch_1_out_0 = *bias;
        int32_t ch_1_out_1 = *bias++;

        /* Set pointers for the two columns of input */
        const int16_t *ip_b0 = input_b;
        const int16_t *ip_b1 = ip_b0 + input_ch;

        /* Set the pointer to the second row of each two rows in matrix A */
        const int8_t *ip_a1 = ip_a0 + input_ch;

        /* Each time four elements are taken from a row of matrix A, a loop is repeated twice, so the number of loops is one eighth of the number of channels */
        uint32_t col_count = input_ch / 4;

        /* accumulate over the vector */
        while (col_count) {
            int32_t a01, a02, a11, a12, b0, b1;

            b0 = mte_read_q15x2_ia(&ip_b0);
            b1 = mte_read_q15x2_ia(&ip_b1);
            a02 = mte_read_s8x4_ia(&ip_a0);
            a01 = __SXTB16(a02);
            a02 = __SXTB16_ROR8(a02);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a01, b1, ch_0_out_1);
            a12 = mte_read_s8x4_ia(&ip_a1);
            a11 = __SXTB16(a12);
            a12 = __SXTB16_ROR8(a12);
            ch_1_out_0 = __SMLAD(a11, b0, ch_1_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);
            ch_1_out_1 = __SMLAD(a11, b1, ch_1_out_1);
            b1 = mte_read_q15x2_ia(&ip_b1);

            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a02, b1, ch_0_out_1);
            ch_1_out_0 = __SMLAD(a12, b0, ch_1_out_0);
            ch_1_out_1 = __SMLAD(a12, b1, ch_1_out_1);

            col_count--;
        }
        ch_0_out_0 = __SMMLAR(ch_0_out_0, scale_0, output_offset);
        ch_0_out_1 = __SMMLAR(ch_0_out_1, scale_0, output_offset);
        ch_1_out_0 = __SMMLAR(ch_1_out_0, scale_1, output_offset);
        ch_1_out_1 = __SMMLAR(ch_1_out_1, scale_1, output_offset);
        ch_0_out_0 = __SSAT8(ch_0_out_0);
        ch_0_out_1 = __SSAT8(ch_0_out_1);
        ch_1_out_0 = __SSAT8(ch_1_out_0);
        ch_1_out_1 = __SSAT8(ch_1_out_1);
        *out_0++ = (int8_t) ch_0_out_0;
        *out_1++ = (int8_t) ch_0_out_1;
        *out_0++ = (int8_t) ch_1_out_0;
        *out_1++ = (int8_t) ch_1_out_1;

        /* The first row of each two rows in matrix A has already been moved by one row in matrix multiplication.
        Here, add another row, and this will be the first row of the next two rows */
        ip_a0 += input_ch;

        row_count--;
    }

    if (output_ch & 1) {
        /* Matrix A shares the same scaling factor for each row and processes two rows at a time.*/
        const int32_t scale_0 = *scale++;

        /* Each channel uses the same bias and has two columns, so each channel has two outputs. */
        /* channel 0*/
        int32_t ch_0_out_0 = *bias;
        int32_t ch_0_out_1 = *bias++;

        /* Set pointers for the two columns of input */
        const int16_t *ip_b0 = input_b;
        const int16_t *ip_b1 = ip_b0 + input_ch;

        uint32_t col_count = input_ch / 4;
        while (col_count) {
            int32_t a01, a02, b0, b1;
            b0 = mte_read_q15x2_ia(&ip_b0);
            b1 = mte_read_q15x2_ia(&ip_b1);

            a02 = mte_read_s8x4_ia(&ip_a0);
            a01 = __SXTB16(a02);
            a02 = __SXTB16_ROR8(a02);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a01, b1, ch_0_out_1);
            b0 = mte_read_q15x2_ia(&ip_b0);
            b1 = mte_read_q15x2_ia(&ip_b1);

            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a02, b1, ch_0_out_1);
            col_count--;
        } /* while over col_count */

        ch_0_out_0 = __SMMLAR(ch_0_out_0, scale_0, output_offset);
        ch_0_out_1 = __SMMLAR(ch_0_out_1, scale_0, output_offset);
        ch_0_out_0 = __SSAT8(ch_0_out_0);
        ch_0_out_1 = __SSAT8(ch_0_out_1);
        *out_0++ = (int8_t) ch_0_out_0;
        *out_1++ = (int8_t) ch_0_out_1;
    }

    out_0 += output_ch;

    /* return the new output pointer with offset */
    return out_0;
}

int8_t *mat_mult_kernel_s8_s16_reordered_2col_4ch(
    //input a
    const int8_t *input_a, const int32_t *bias, const int32_t *scale,
    //input b
    const int16_t *input_b, const uint32_t input_ch,
    //output
    int8_t *output, const uint32_t output_ch, const int32_t output_offset,
    //activation
    const int32_t act_min, const int32_t act_max)
{
    /* Set the output pointer, because the input is two columns, the output is also two columns. */
    int8_t *out_0 = output;
    int8_t *out_1 = out_0 + output_ch;

    /* Set the pointer to matrix A */
    const int8_t *ip_a0 = input_a;

    /* Extract two rows from matrix A each time to calculate the results of two channels, so the number of iterations is half that of the output channel.*/
    uint32_t row_count = output_ch / 2;
    /* this loop over rows in A */
    while (row_count) {

        /* Matrix A shares the same scaling factor for each row and processes two rows at a time.*/
        const int32_t scale_0 = *scale++;
        const int32_t scale_1 = *scale++;

        /* Each channel uses the same bias and has two columns, so each channel has two outputs. */
        /* channel 0*/
        int32_t ch_0_out_0 = *bias;
        int32_t ch_0_out_1 = *bias++;
        /* channel 1*/
        int32_t ch_1_out_0 = *bias;
        int32_t ch_1_out_1 = *bias++;

        /* Set pointers for the two columns of input */
        const int16_t *ip_b0 = input_b;
        const int16_t *ip_b1 = ip_b0 + input_ch;

        /* Set the pointer to the second row of each two rows in matrix A */
        const int8_t *ip_a1 = ip_a0 + input_ch;

        /* Each time four elements are taken from a row of matrix A, a loop is repeated twice, so the number of loops is one eighth of the number of channels */
        uint32_t col_count = input_ch / 4;

        /* accumulate over the vector */
        while (col_count) {
            int32_t a01, a02, a11, a12, b0, b1;

            b0 = mte_read_q15x2_ia(&ip_b0);
            b1 = mte_read_q15x2_ia(&ip_b1);
            a02 = mte_read_s8x4_ia(&ip_a0);
            a01 = __SXTB16(a02);
            a02 = __SXTB16_ROR8(a02);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a01, b1, ch_0_out_1);
            a12 = mte_read_s8x4_ia(&ip_a1);
            a11 = __SXTB16(a12);
            a12 = __SXTB16_ROR8(a12);
            ch_1_out_0 = __SMLAD(a11, b0, ch_1_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);
            ch_1_out_1 = __SMLAD(a11, b1, ch_1_out_1);
            b1 = mte_read_q15x2_ia(&ip_b1);

            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a02, b1, ch_0_out_1);
            ch_1_out_0 = __SMLAD(a12, b0, ch_1_out_0);
            ch_1_out_1 = __SMLAD(a12, b1, ch_1_out_1);

            col_count--;
        }
        ch_0_out_0 = __SMMLAR(ch_0_out_0, scale_0, output_offset);
        ch_0_out_1 = __SMMLAR(ch_0_out_1, scale_0, output_offset);
        ch_1_out_0 = __SMMLAR(ch_1_out_0, scale_1, output_offset);
        ch_1_out_1 = __SMMLAR(ch_1_out_1, scale_1, output_offset);
        ch_0_out_0 = MAX(ch_0_out_0, act_min);
        ch_0_out_1 = MAX(ch_0_out_1, act_min);
        ch_1_out_0 = MAX(ch_1_out_0, act_min);
        ch_1_out_1 = MAX(ch_1_out_1, act_min);
        ch_0_out_0 = MIN(ch_0_out_0, act_max);
        ch_0_out_1 = MIN(ch_0_out_1, act_max);
        ch_1_out_0 = MIN(ch_1_out_0, act_max);
        ch_1_out_1 = MIN(ch_1_out_1, act_max);
        *out_0++ = (int8_t) ch_0_out_0;
        *out_1++ = (int8_t) ch_0_out_1;
        *out_0++ = (int8_t) ch_1_out_0;
        *out_1++ = (int8_t) ch_1_out_1;

        /* The first row of each two rows in matrix A has already been moved by one row in matrix multiplication.
        Here, add another row, and this will be the first row of the next two rows */
        ip_a0 += input_ch;

        row_count--;
    }

    if (output_ch & 1) {
        /* Matrix A shares the same scaling factor for each row and processes two rows at a time.*/
        const int32_t scale_0 = *scale++;

        /* Each channel uses the same bias and has two columns, so each channel has two outputs. */
        /* channel 0*/
        int32_t ch_0_out_0 = *bias;
        int32_t ch_0_out_1 = *bias++;

        /* Set pointers for the two columns of input */
        const int16_t *ip_b0 = input_b;
        const int16_t *ip_b1 = ip_b0 + input_ch;

        uint32_t col_count = input_ch / 4;
        while (col_count) {
            int32_t a01, a02, b0, b1;
            b0 = mte_read_q15x2_ia(&ip_b0);
            b1 = mte_read_q15x2_ia(&ip_b1);

            a02 = mte_read_s8x4_ia(&ip_a0);
            a01 = __SXTB16(a02);
            a02 = __SXTB16_ROR8(a02);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a01, b1, ch_0_out_1);
            b0 = mte_read_q15x2_ia(&ip_b0);
            b1 = mte_read_q15x2_ia(&ip_b1);

            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a02, b1, ch_0_out_1);
            col_count--;
        } /* while over col_count */

        ch_0_out_0 = __SMMLAR(ch_0_out_0, scale_0, output_offset);
        ch_0_out_1 = __SMMLAR(ch_0_out_1, scale_0, output_offset);
        ch_0_out_0 = MAX(ch_0_out_0, act_min);
        ch_0_out_0 = MIN(ch_0_out_0, act_max);
        *out_0++ = (int8_t) ch_0_out_0;
        *out_1++ = (int8_t) ch_0_out_1;
    }

    out_0 += output_ch;

    /* return the new output pointer with offset */
    return out_0;
}

int8_t *mat_mult_kernel_s8_s16_reordered_1col_8ch_ssat(
    //input a
    const int8_t *input_a, const int32_t *bias, const int32_t *scale,
    //input b
    const int16_t *input_b, const uint32_t input_ch,
    //output
    int8_t *output, const uint32_t output_ch, const int32_t output_offset,
    //activation
    const int32_t act_min, const int32_t act_max)
{
    /* Set the output pointer, because the input is one column, the output is also one column. */
    int8_t *out_0 = output;

    /* Set the pointer to matrix A */
    const int8_t *ip_a0 = input_a;

    /* Extract two rows from matrix A each time to calculate the results of two channels, so the number of iterations is half that of the output channel.*/
    uint32_t row_count = output_ch / 2;
    /* this loop over rows in A */
    while (row_count) {

        /* Matrix A shares the same scaling factor for each row and processes two rows at a time.*/
        const int32_t scale_0 = *scale++;
        const int32_t scale_1 = *scale++;

        /* Each channel uses the same bias and has one column, so each channel has one outputs. */
        /* channel 0*/
        int32_t ch_0_out_0 = *bias++;
        /* channel 1*/
        int32_t ch_1_out_0 = *bias++;

        /* Set pointers for the one columns of input */
        const int16_t *ip_b0 = input_b;

        /* Set the pointer to the second row of each two rows in matrix A */
        const int8_t *ip_a1 = ip_a0 + input_ch;

        /* Each time four elements are taken from a row of matrix A, a loop is repeated twice, so the number of loops is one eighth of the number of channels */
        uint32_t col_count = input_ch / 8;

        /* accumulate over the vector */
        while (col_count) {
            int32_t a01, a02, a11, a12, b0;

            b0 = mte_read_q15x2_ia(&ip_b0);
            a02 = mte_read_s8x4_ia(&ip_a0);
            a01 = __SXTB16(a02);
            a02 = __SXTB16_ROR8(a02);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            a12 = mte_read_s8x4_ia(&ip_a1);
            a11 = __SXTB16(a12);
            a12 = __SXTB16_ROR8(a12);
            ch_1_out_0 = __SMLAD(a11, b0, ch_1_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);

            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            ch_1_out_0 = __SMLAD(a12, b0, ch_1_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);


            a02 = mte_read_s8x4_ia(&ip_a0);
            a01 = __SXTB16(a02);
            a02 = __SXTB16_ROR8(a02);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            a12 = mte_read_s8x4_ia(&ip_a1);
            a11 = __SXTB16(a12);
            a12 = __SXTB16_ROR8(a12);
            ch_1_out_0 = __SMLAD(a11, b0, ch_1_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);

            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            ch_1_out_0 = __SMLAD(a12, b0, ch_1_out_0);
            col_count--;
        }
        ch_0_out_0 = __SMMLAR(ch_0_out_0, scale_0, output_offset);
        ch_1_out_0 = __SMMLAR(ch_1_out_0, scale_1, output_offset);
        ch_0_out_0 = __SSAT8(ch_0_out_0);
        ch_1_out_0 = __SSAT8(ch_1_out_0);
        *out_0++ = (int8_t) ch_0_out_0;
        *out_0++ = (int8_t) ch_1_out_0;

        /* The first row of each two rows in matrix A has already been moved by one row in matrix multiplication.
        Here, add another row, and this will be the first row of the next two rows */
        ip_a0 += input_ch;

        row_count--;
    }

    if (output_ch & 1) {

        /* Matrix A shares the same scaling factor for each row and processes two rows at a time.*/
        const int32_t scale_0 = *scale++;

        /* Each channel uses the same bias and has two columns, so each channel has two outputs. */
        /* channel 0*/
        int32_t ch_0_out_0 = *bias++;

        /* Set pointers for the two columns of input */
        const int16_t *ip_b0 = input_b;

        uint32_t col_count = input_ch / 8;
        while (col_count) {
            int32_t a01, a02, b0;
            b0 = mte_read_q15x2_ia(&ip_b0);

            a02 = mte_read_s8x4_ia(&ip_a0);
            a01 = __SXTB16(a02);
            a02 = __SXTB16_ROR8(a02);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);
            a01 = mte_read_s8x4_ia(&ip_a0);
            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);

            b0 = mte_read_q15x2_ia(&ip_b0);
            a02 = __SXTB16(a01);
            a01 = __SXTB16_ROR8(a01);

            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);
            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);

            col_count--;
        } /* while over col_count */

        ch_0_out_0 = __SMMLAR(ch_0_out_0, scale_0, output_offset);
        ch_0_out_0 = __SSAT8(ch_0_out_0);
        *out_0++ = (int8_t) ch_0_out_0;
    }

    /* return the new output pointer with offset */
    return out_0;
}

int8_t *mat_mult_kernel_s8_s16_reordered_1col_8ch(
    //input a
    const int8_t *input_a, const int32_t *bias, const int32_t *scale,
    //input b
    const int16_t *input_b, const uint32_t input_ch,
    //output
    int8_t *output, const uint32_t output_ch, const int32_t output_offset,
    //activation
    const int32_t act_min, const int32_t act_max)
{
    /* Set the output pointer, because the input is one column, the output is also one column. */
    int8_t *out_0 = output;

    /* Set the pointer to matrix A */
    const int8_t *ip_a0 = input_a;

    /* Extract two rows from matrix A each time to calculate the results of two channels, so the number of iterations is half that of the output channel.*/
    uint32_t row_count = output_ch / 2;
    /* this loop over rows in A */
    while (row_count) {

        /* Matrix A shares the same scaling factor for each row and processes two rows at a time.*/
        const int32_t scale_0 = *scale++;
        const int32_t scale_1 = *scale++;

        /* Each channel uses the same bias and has one column, so each channel has one outputs. */
        /* channel 0*/
        int32_t ch_0_out_0 = *bias++;
        /* channel 1*/
        int32_t ch_1_out_0 = *bias++;

        /* Set pointers for the one columns of input */
        const int16_t *ip_b0 = input_b;

        /* Set the pointer to the second row of each two rows in matrix A */
        const int8_t *ip_a1 = ip_a0 + input_ch;

        /* Each time four elements are taken from a row of matrix A, a loop is repeated twice, so the number of loops is one eighth of the number of channels */
        uint32_t col_count = input_ch / 8;

        /* accumulate over the vector */
        while (col_count) {
            int32_t a01, a02, a11, a12, b0;

            b0 = mte_read_q15x2_ia(&ip_b0);
            a02 = mte_read_s8x4_ia(&ip_a0);
            a01 = __SXTB16(a02);
            a02 = __SXTB16_ROR8(a02);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            a12 = mte_read_s8x4_ia(&ip_a1);
            a11 = __SXTB16(a12);
            a12 = __SXTB16_ROR8(a12);
            ch_1_out_0 = __SMLAD(a11, b0, ch_1_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);

            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            ch_1_out_0 = __SMLAD(a12, b0, ch_1_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);


            a02 = mte_read_s8x4_ia(&ip_a0);
            a01 = __SXTB16(a02);
            a02 = __SXTB16_ROR8(a02);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            a12 = mte_read_s8x4_ia(&ip_a1);
            a11 = __SXTB16(a12);
            a12 = __SXTB16_ROR8(a12);
            ch_1_out_0 = __SMLAD(a11, b0, ch_1_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);

            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            ch_1_out_0 = __SMLAD(a12, b0, ch_1_out_0);
            col_count--;
        }
        ch_0_out_0 = __SMMLAR(ch_0_out_0, scale_0, output_offset);
        ch_1_out_0 = __SMMLAR(ch_1_out_0, scale_1, output_offset);
        ch_0_out_0 = MAX(ch_0_out_0, act_min);
        ch_1_out_0 = MAX(ch_1_out_0, act_min);
        ch_0_out_0 = MIN(ch_0_out_0, act_max);
        ch_1_out_0 = MIN(ch_1_out_0, act_max);
        *out_0++ = (int8_t) ch_0_out_0;
        *out_0++ = (int8_t) ch_1_out_0;

        /* The first row of each two rows in matrix A has already been moved by one row in matrix multiplication.
        Here, add another row, and this will be the first row of the next two rows */
        ip_a0 += input_ch;

        row_count--;
    }

    if (output_ch & 1) {

        /* Matrix A shares the same scaling factor for each row and processes two rows at a time.*/
        const int32_t scale_0 = *scale++;

        /* Each channel uses the same bias and has two columns, so each channel has two outputs. */
        /* channel 0*/
        int32_t ch_0_out_0 = *bias++;

        /* Set pointers for the two columns of input */
        const int16_t *ip_b0 = input_b;

        uint32_t col_count = input_ch / 8;
        while (col_count) {
            int32_t a01, a02, b0;
            b0 = mte_read_q15x2_ia(&ip_b0);

            a02 = mte_read_s8x4_ia(&ip_a0);
            a01 = __SXTB16(a02);
            a02 = __SXTB16_ROR8(a02);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);
            a01 = mte_read_s8x4_ia(&ip_a0);
            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);

            b0 = mte_read_q15x2_ia(&ip_b0);
            a02 = __SXTB16(a01);
            a01 = __SXTB16_ROR8(a01);

            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);
            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);

            col_count--;
        } /* while over col_count */

        ch_0_out_0 = __SMMLAR(ch_0_out_0, scale_0, output_offset);
        ch_0_out_0 = MAX(ch_0_out_0, act_min);
        ch_0_out_0 = MIN(ch_0_out_0, act_max);
        *out_0++ = (int8_t) ch_0_out_0;
    }

    /* return the new output pointer with offset */
    return out_0;
}

int8_t *mat_mult_kernel_s8_s16_reordered_2col_8ch_ssat(
    //input a
    const int8_t *input_a, const int32_t *bias, const int32_t *scale,
    //input b
    const int16_t *input_b, const uint32_t input_ch,
    //output
    int8_t *output, const uint32_t output_ch, const int32_t output_offset,
    //activation
    const int32_t act_min, const int32_t act_max)
{
    /* Set the output pointer, because the input is two columns, the output is also two columns. */
    int8_t *out_0 = output;
    int8_t *out_1 = out_0 + output_ch;

    /* Set the pointer to matrix A */
    const int8_t *ip_a0 = input_a;

    /* Extract two rows from matrix A each time to calculate the results of two channels, so the number of iterations is half that of the output channel.*/
    uint32_t row_count = output_ch / 2;
    /* this loop over rows in A */
    while (row_count) {

        /* Matrix A shares the same scaling factor for each row and processes two rows at a time.*/
        const int32_t scale_0 = *scale++;
        const int32_t scale_1 = *scale++;

        /* Each channel uses the same bias and has two columns, so each channel has two outputs. */
        /* channel 0*/
        int32_t ch_0_out_0 = *bias;
        int32_t ch_0_out_1 = *bias++;
        /* channel 1*/
        int32_t ch_1_out_0 = *bias;
        int32_t ch_1_out_1 = *bias++;

        /* Set pointers for the two columns of input */
        const int16_t *ip_b0 = input_b;
        const int16_t *ip_b1 = ip_b0 + input_ch;

        /* Set the pointer to the second row of each two rows in matrix A */
        const int8_t *ip_a1 = ip_a0 + input_ch;

        /* Each time four elements are taken from a row of matrix A, a loop is repeated twice, so the number of loops is one eighth of the number of channels */
        uint32_t col_count = input_ch / 8;

        /* accumulate over the vector */
        while (col_count) {
            int32_t a01, a02, a11, a12, b0, b1;

            b0 = mte_read_q15x2_ia(&ip_b0);
            b1 = mte_read_q15x2_ia(&ip_b1);
            a02 = mte_read_s8x4_ia(&ip_a0);
            a01 = __SXTB16(a02);
            a02 = __SXTB16_ROR8(a02);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a01, b1, ch_0_out_1);
            a12 = mte_read_s8x4_ia(&ip_a1);
            a11 = __SXTB16(a12);
            a12 = __SXTB16_ROR8(a12);
            ch_1_out_0 = __SMLAD(a11, b0, ch_1_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);
            ch_1_out_1 = __SMLAD(a11, b1, ch_1_out_1);
            b1 = mte_read_q15x2_ia(&ip_b1);

            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a02, b1, ch_0_out_1);
            ch_1_out_0 = __SMLAD(a12, b0, ch_1_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);
            ch_1_out_1 = __SMLAD(a12, b1, ch_1_out_1);
            b1 = mte_read_q15x2_ia(&ip_b1);


            a02 = mte_read_s8x4_ia(&ip_a0);
            a01 = __SXTB16(a02);
            a02 = __SXTB16_ROR8(a02);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a01, b1, ch_0_out_1);
            a12 = mte_read_s8x4_ia(&ip_a1);
            a11 = __SXTB16(a12);
            a12 = __SXTB16_ROR8(a12);
            ch_1_out_0 = __SMLAD(a11, b0, ch_1_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);
            ch_1_out_1 = __SMLAD(a11, b1, ch_1_out_1);
            b1 = mte_read_q15x2_ia(&ip_b1);

            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a02, b1, ch_0_out_1);
            ch_1_out_0 = __SMLAD(a12, b0, ch_1_out_0);
            ch_1_out_1 = __SMLAD(a12, b1, ch_1_out_1);
            col_count--;
        }
        ch_0_out_0 = __SMMLAR(ch_0_out_0, scale_0, output_offset);
        ch_0_out_1 = __SMMLAR(ch_0_out_1, scale_0, output_offset);
        ch_1_out_0 = __SMMLAR(ch_1_out_0, scale_1, output_offset);
        ch_1_out_1 = __SMMLAR(ch_1_out_1, scale_1, output_offset);
        ch_0_out_0 = __SSAT8(ch_0_out_0);
        ch_0_out_1 = __SSAT8(ch_0_out_1);
        ch_1_out_0 = __SSAT8(ch_1_out_0);
        ch_1_out_1 = __SSAT8(ch_1_out_1);
        *out_0++ = (int8_t) ch_0_out_0;
        *out_1++ = (int8_t) ch_0_out_1;
        *out_0++ = (int8_t) ch_1_out_0;
        *out_1++ = (int8_t) ch_1_out_1;

        /* The first row of each two rows in matrix A has already been moved by one row in matrix multiplication.
        Here, add another row, and this will be the first row of the next two rows */
        ip_a0 += input_ch;

        row_count--;
    }

    if (output_ch & 1) {
        /* Matrix A shares the same scaling factor for each row and processes two rows at a time.*/
        const int32_t scale_0 = *scale++;

        /* Each channel uses the same bias and has two columns, so each channel has two outputs. */
        /* channel 0*/
        int32_t ch_0_out_0 = *bias;
        int32_t ch_0_out_1 = *bias++;

        /* Set pointers for the two columns of input */
        const int16_t *ip_b0 = input_b;
        const int16_t *ip_b1 = ip_b0 + input_ch;

        uint32_t col_count = input_ch / 8;
        while (col_count) {
            int32_t a01, a02, b0, b1;
            b0 = mte_read_q15x2_ia(&ip_b0);
            b1 = mte_read_q15x2_ia(&ip_b1);

            a02 = mte_read_s8x4_ia(&ip_a0);
            a01 = __SXTB16(a02);
            a02 = __SXTB16_ROR8(a02);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a01, b1, ch_0_out_1);
            a01 = mte_read_s8x4_ia(&ip_a0);
            b0 = mte_read_q15x2_ia(&ip_b0);
            b1 = mte_read_q15x2_ia(&ip_b1);

            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);
            ch_0_out_1 = __SMLAD(a02, b1, ch_0_out_1);
            b1 = mte_read_q15x2_ia(&ip_b1);


            a02 = __SXTB16(a01);
            a01 = __SXTB16_ROR8(a01);

            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);
            ch_0_out_1 = __SMLAD(a02, b1, ch_0_out_1);
            b1 = mte_read_q15x2_ia(&ip_b1);
            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a01, b1, ch_0_out_1);

            col_count--;
        } /* while over col_count */

        ch_0_out_0 = __SMMLAR(ch_0_out_0, scale_0, output_offset);
        ch_0_out_1 = __SMMLAR(ch_0_out_1, scale_0, output_offset);
        ch_0_out_0 = __SSAT8(ch_0_out_0);
        ch_0_out_1 = __SSAT8(ch_0_out_1);
        *out_0++ = (int8_t) ch_0_out_0;
        *out_1++ = (int8_t) ch_0_out_1;
    }

    out_0 += output_ch;

    /* return the new output pointer with offset */
    return out_0;
}

int8_t *mat_mult_kernel_s8_s16_reordered_2col_8ch(
    //input a
    const int8_t *input_a, const int32_t *bias, const int32_t *scale,
    //input b
    const int16_t *input_b, const uint32_t input_ch,
    //output
    int8_t *output, const uint32_t output_ch, const int32_t output_offset,
    //activation
    const int32_t act_min, const int32_t act_max)
{
    /* Set the output pointer, because the input is two columns, the output is also two columns. */
    int8_t *out_0 = output;
    int8_t *out_1 = out_0 + output_ch;

    /* Set the pointer to matrix A */
    const int8_t *ip_a0 = input_a;

    /* Extract two rows from matrix A each time to calculate the results of two channels, so the number of iterations is half that of the output channel.*/
    uint32_t row_count = output_ch / 2;
    /* this loop over rows in A */
    while (row_count) {

        /* Matrix A shares the same scaling factor for each row and processes two rows at a time.*/
        const int32_t scale_0 = *scale++;
        const int32_t scale_1 = *scale++;

        /* Each channel uses the same bias and has two columns, so each channel has two outputs. */
        /* channel 0*/
        int32_t ch_0_out_0 = *bias;
        int32_t ch_0_out_1 = *bias++;
        /* channel 1*/
        int32_t ch_1_out_0 = *bias;
        int32_t ch_1_out_1 = *bias++;

        /* Set pointers for the two columns of input */
        const int16_t *ip_b0 = input_b;
        const int16_t *ip_b1 = ip_b0 + input_ch;

        /* Set the pointer to the second row of each two rows in matrix A */
        const int8_t *ip_a1 = ip_a0 + input_ch;

        /* Each time four elements are taken from a row of matrix A, a loop is repeated twice, so the number of loops is one eighth of the number of channels */
        uint32_t col_count = input_ch / 8;

        /* accumulate over the vector */
        while (col_count) {
            int32_t a01, a02, a11, a12, b0, b1;

            b0 = mte_read_q15x2_ia(&ip_b0);
            b1 = mte_read_q15x2_ia(&ip_b1);
            a02 = mte_read_s8x4_ia(&ip_a0);
            a01 = __SXTB16(a02);
            a02 = __SXTB16_ROR8(a02);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a01, b1, ch_0_out_1);
            a12 = mte_read_s8x4_ia(&ip_a1);
            a11 = __SXTB16(a12);
            a12 = __SXTB16_ROR8(a12);
            ch_1_out_0 = __SMLAD(a11, b0, ch_1_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);
            ch_1_out_1 = __SMLAD(a11, b1, ch_1_out_1);
            b1 = mte_read_q15x2_ia(&ip_b1);

            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a02, b1, ch_0_out_1);
            ch_1_out_0 = __SMLAD(a12, b0, ch_1_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);
            ch_1_out_1 = __SMLAD(a12, b1, ch_1_out_1);
            b1 = mte_read_q15x2_ia(&ip_b1);


            a02 = mte_read_s8x4_ia(&ip_a0);
            a01 = __SXTB16(a02);
            a02 = __SXTB16_ROR8(a02);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a01, b1, ch_0_out_1);
            a12 = mte_read_s8x4_ia(&ip_a1);
            a11 = __SXTB16(a12);
            a12 = __SXTB16_ROR8(a12);
            ch_1_out_0 = __SMLAD(a11, b0, ch_1_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);
            ch_1_out_1 = __SMLAD(a11, b1, ch_1_out_1);
            b1 = mte_read_q15x2_ia(&ip_b1);

            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a02, b1, ch_0_out_1);
            ch_1_out_0 = __SMLAD(a12, b0, ch_1_out_0);
            ch_1_out_1 = __SMLAD(a12, b1, ch_1_out_1);
            col_count--;
        }
        ch_0_out_0 = __SMMLAR(ch_0_out_0, scale_0, output_offset);
        ch_0_out_1 = __SMMLAR(ch_0_out_1, scale_0, output_offset);
        ch_1_out_0 = __SMMLAR(ch_1_out_0, scale_1, output_offset);
        ch_1_out_1 = __SMMLAR(ch_1_out_1, scale_1, output_offset);
        ch_0_out_0 = MAX(ch_0_out_0, act_min);
        ch_0_out_1 = MAX(ch_0_out_1, act_min);
        ch_1_out_0 = MAX(ch_1_out_0, act_min);
        ch_1_out_1 = MAX(ch_1_out_1, act_min);
        ch_0_out_0 = MIN(ch_0_out_0, act_max);
        ch_0_out_1 = MIN(ch_0_out_1, act_max);
        ch_1_out_0 = MIN(ch_1_out_0, act_max);
        ch_1_out_1 = MIN(ch_1_out_1, act_max);
        *out_0++ = (int8_t) ch_0_out_0;
        *out_1++ = (int8_t) ch_0_out_1;
        *out_0++ = (int8_t) ch_1_out_0;
        *out_1++ = (int8_t) ch_1_out_1;

        /* The first row of each two rows in matrix A has already been moved by one row in matrix multiplication.
        Here, add another row, and this will be the first row of the next two rows */
        ip_a0 += input_ch;

        row_count--;
    }

    if (output_ch & 1) {

        /* Matrix A shares the same scaling factor for each row and processes two rows at a time.*/
        const int32_t scale_0 = *scale++;

        /* Each channel uses the same bias and has two columns, so each channel has two outputs. */
        /* channel 0*/
        int32_t ch_0_out_0 = *bias;
        int32_t ch_0_out_1 = *bias++;

        /* Set pointers for the two columns of input */
        const int16_t *ip_b0 = input_b;
        const int16_t *ip_b1 = ip_b0 + input_ch;

        uint32_t col_count = input_ch / 8;
        while (col_count) {
            int32_t a01, a02, b0, b1;
            b0 = mte_read_q15x2_ia(&ip_b0);
            b1 = mte_read_q15x2_ia(&ip_b1);

            a02 = mte_read_s8x4_ia(&ip_a0);
            a01 = __SXTB16(a02);
            a02 = __SXTB16_ROR8(a02);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a01, b1, ch_0_out_1);
            a01 = mte_read_s8x4_ia(&ip_a0);
            b0 = mte_read_q15x2_ia(&ip_b0);
            b1 = mte_read_q15x2_ia(&ip_b1);

            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);
            ch_0_out_1 = __SMLAD(a02, b1, ch_0_out_1);
            b1 = mte_read_q15x2_ia(&ip_b1);


            a02 = __SXTB16(a01);
            a01 = __SXTB16_ROR8(a01);

            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            b0 = mte_read_q15x2_ia(&ip_b0);
            ch_0_out_1 = __SMLAD(a02, b1, ch_0_out_1);
            b1 = mte_read_q15x2_ia(&ip_b1);
            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a01, b1, ch_0_out_1);

            col_count--;
        } /* while over col_count */

        ch_0_out_0 = __SMMLAR(ch_0_out_0, scale_0, output_offset);
        ch_0_out_1 = __SMMLAR(ch_0_out_1, scale_0, output_offset);
        ch_0_out_0 = MAX(ch_0_out_0, act_min);
        ch_0_out_1 = MAX(ch_0_out_1, act_min);
        ch_0_out_0 = MIN(ch_0_out_0, act_max);
        ch_0_out_1 = MIN(ch_0_out_1, act_max);
        *out_0++ = (int8_t) ch_0_out_0;
        *out_1++ = (int8_t) ch_0_out_1;
    }

    out_0 += output_ch;

    /* return the new output pointer with offset */
    return out_0;
}


void mte_convolve_1x1_s8(
    //input
    const int8_t *input,
    const uint16_t input_h,
    const uint32_t input_w,
    const uint32_t input_ch,
    const int32_t input_offset,
    //kernel
    const int8_t *weight,
    const int32_t *bias,
    const int32_t *scale,
    const uint32_t stride_h,
    const uint32_t stride_w,
    //output
    int8_t *output,
    const uint16_t output_h,
    const uint32_t output_w,
    const uint32_t output_ch,
    const int32_t output_offset,
    //activation
    const int32_t act_min,
    const int32_t act_max,
    //buffer
    int16_t *input_buf,
    int8_t *kernel_buf)
{
    uint32_t i;
    /* Partial(two columns) im2col buffer */
    const uint32_t num_elements = output_h * output_w;
    const uint32_t channel_div4 = (input_ch >> 2);

    const int32_t offset_i16x2 = __PKHBT(input_offset, input_offset, 16);
    const uint32_t weight_byte_size = input_ch * output_ch;
    const uint32_t bias_byte_size = 4 * output_ch;
    const uint32_t scale_byte_size = bias_byte_size;
    int8_t *weight_buf = kernel_buf;
    int8_t *bias_buf = kernel_buf + weight_byte_size;
    int8_t *scale_buf = kernel_buf + weight_byte_size + bias_byte_size;
    memcpy(weight_buf, weight, weight_byte_size);
    memcpy(bias_buf, bias, bias_byte_size);
#if defined(INT_SCALE)
    memcpy(scale_buf, scale, scale_byte_size);
#else
    int32_t *int_scale_buf = (int32_t *) scale_buf;
    float *float_scale = (float *) scale;
    for (i = 0; i < output_ch; i++) {
        *int_scale_buf++ = (int32_t) ((*float_scale++) * 4294967296.f);
    }
#endif
    int8_t *(*mat_mult_kernel_2col_func)() = (int8_t *(*) ())((int8_t *(*) ())(
                                                                  act_min == -128 && act_max == 127)
                                                                  ? mat_mult_kernel_s8_s16_reordered_2col_8ch_ssat
                                                                  : mat_mult_kernel_s8_s16_reordered_2col_8ch);
    const uint32_t var0 = input_w * input_ch * stride_h;
    const uint32_t var1 = input_ch * stride_w;
    for (i = 0; i < num_elements / 2; i++) {
        /* Fill buffer for partial im2col - two columns at a time */
        const uint32_t _2i = 2 * i;
        const int8_t *src = &input[(_2i / output_w) * var0 + (_2i % output_w) * var1];
        int16_t *dst = input_buf;
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
            weight_buf, bias_buf, scale_buf,
            input_buf, input_ch,
            output, output_ch, output_offset,
            act_min, act_max);
    }

    /* check if there is an odd column left-over for computation */
    if (num_elements & 0x1) {
        const uint32_t _2i = num_elements & 0x1;
        const int8_t *src = &input[(_2i / output_w) * var0 + (_2i % output_w) * var1];
        int16_t *dst = input_buf;
        uint32_t cnt = channel_div4;//two * numof2col columns
        while (cnt > 0) {
            i8x4_to_2xi16x2_offset_reordered_ele_ia((int32_t **) &src,(int32_t **) &dst, offset_i16x2);
                cnt--;
        }
        int8_t *(*mat_mult_kernel_1col_func)() = (int8_t *(*) ())((act_min == -128 && act_max == 127)
                                                                      ? mat_mult_kernel_s8_s16_reordered_1col_8ch_ssat
                                                                      : mat_mult_kernel_s8_s16_reordered_1col_8ch);
        output = mat_mult_kernel_1col_func(
            weight_buf, bias_buf, scale_buf,
            input_buf, input_ch,
            output, output_ch, output_offset,
            act_min, act_max);
    }
}

void convolve_1x1_s8_fpreq(
    const int8_t *input,
    const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
    const int8_t *kernel, const int32_t *bias, const float *scales,
    const int32_t out_offset, const int32_t input_offset,
    const int32_t out_activation_min, const int32_t out_activation_max,
    int8_t *output, const uint16_t output_x, const uint16_t output_y,
    const uint16_t output_ch, int16_t *runtime_buf, int8_t *kbuf)
{
    mte_convolve_1x1_s8(
        (const int8_t *) input,
        (const uint32_t) input_x,
        (const uint32_t) input_y,
        (const uint32_t) input_ch,
        (const int32_t) input_offset,
        (const int8_t *) kernel,
        (const int32_t *) bias,
        (const int32_t *) scales,
        (const uint32_t) 1,
        (const uint32_t) 1,
        (int8_t *) output,
        (const uint32_t) output_x,
        (const uint32_t) output_y,
        (const uint32_t) output_ch,
        (const int32_t) out_offset,
        (int32_t) out_activation_min,
        (int32_t) out_activation_max,
        (int16_t *) runtime_buf,
        (int8_t *) kbuf);
}