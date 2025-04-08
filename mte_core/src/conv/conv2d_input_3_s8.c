//
// Created by kyuliea on 2025/4/1.
//
#include "mte_core.h"

void mat_mult_kernel_input3_3x3_s16_s16_2col_ssat(
    const int16_t *input_a,
    const int16_t *input_b,
    const int32_t *bias,
    const int32_t *scale,
    int32_t act_min,int32_t act_max,
    int8_t *output,int32_t output_ch,int32_t output_offset);

void mat_mult_kernel_input3_3x3_s16_s16_2col_minmax(
    const int16_t *input_a,
    const int16_t *input_b,
    const int32_t *bias,
    const int32_t *scale,
    int32_t act_min,int32_t act_max,
    int8_t *output,int32_t output_ch,int32_t output_offset);

void mat_mult_kernel_input3_3x3_s16_s16_1col_ssat(
    const int16_t *input_a,
    const int16_t *input_b,
    const int32_t *bias,
    const int32_t *scale,
    int32_t act_min,int32_t act_max,
    int8_t *output,int32_t output_ch,int32_t output_offset);

void mat_mult_kernel_input3_3x3_s16_s16_1col_minmax(
    const int16_t *input_a,
    const int16_t *input_b,
    const int32_t *bias,
    const int32_t *scale,
    int32_t act_min,int32_t act_max,
    int8_t *output,int32_t output_ch,int32_t output_offset);


void conv2d_input_3_3x3_stride_2_2_dilate_1_1_s8(
    const int8_t *input, const int32_t input_h, const int32_t input_w,const int32_t input_ch,const int32_t input_offset,
    const int16_t *weight,int16_t *weight_cache,
    const int32_t *bias,int32_t *bias_cache,
    const int32_t *scale,int32_t *scale_cache,
    int8_t *input_buffer,
    const int32_t act_min,const int32_t act_max,
    int8_t *output,
    const int32_t output_h,const int32_t output_w,const int32_t output_ch,const int32_t output_offset
    ){

    void (*mat_mult_kernel_func)(
        const int16_t *input_a,
        const int16_t *input_b,
        const int32_t *bias,
        const int32_t *scale,
        int32_t act_min,int32_t act_max,
        int8_t *output,int32_t output_ch,int32_t output_offset)=NULL;
    if(act_min==-128&&act_max==127)
        mat_mult_kernel_func=mat_mult_kernel_input3_3x3_s16_s16_2col_ssat;
    else
        mat_mult_kernel_func=mat_mult_kernel_input3_3x3_s16_s16_2col_minmax;

    if(weight_cache!=0){
        memcpy(weight_cache,weight,output_ch*27*2);
        weight=weight_cache;
    }
    if(bias_cache!=0){
        memcpy(bias_cache,bias,output_ch*4);
        bias=bias_cache;
    }
    if(scale_cache!=0){
        memcpy(scale_cache,scale,output_ch*4);
        scale=scale_cache;
    }

    int32_t h,w;
    int16_t *two_col_buffer=(int16_t *)input_buffer;
    for (h = 0; h < output_h; h++) {
        for (w = 0; w < output_w; w++) {
            const int32_t in_h=2*h;
            const int32_t in_w=2*w;
            if(in_h==0||in_h+2==input_h+1||in_w==0||in_w==input_w+1)
            {
                memset(two_col_buffer,0,2*27);
            }
            const int32_t h_begin=MAX(in_h-1,0);
            const int32_t h_end=MIN(in_h+3-1,input_h);
            const int32_t w_begin=MAX(in_w-1,0);
            const int32_t w_end=MIN(in_w+3-1,input_w);
            int32_t win_h;
            int32_t win_w;
            for(win_h=h_begin;win_h<h_end;win_h++){
                for(win_w=w_begin;win_w<w_end;win_w++){
                    int32_t idx0=(win_h-in_h+1)*9+(win_w-in_w+1)*3;
                    int32_t idx1=win_h*input_h*3+win_w*3;
                    two_col_buffer[idx0]=((int16_t)input[idx1])-(int16_t)input_offset;
                    two_col_buffer[idx0+1]=((int16_t)input[idx1+1])-(int16_t)input_offset;
                    two_col_buffer[idx0+2]=((int16_t)input[idx1+2])-(int16_t)input_offset;
                }
            }
            two_col_buffer+=27;
            if (two_col_buffer == (int16_t *)input_buffer + 2 * 27) {
                mat_mult_kernel_func(
                    weight,
                    (int16_t *)input_buffer,
                    bias,
                    scale,
                    act_min,act_max,
                    output,output_ch,output_offset
                    );
                output+=2*output_ch;
                two_col_buffer=input_buffer;
            }
        }
    }
    if(act_min==-128&&act_max==127)
        mat_mult_kernel_func=mat_mult_kernel_input3_3x3_s16_s16_1col_ssat;
    else
        mat_mult_kernel_func=mat_mult_kernel_input3_3x3_s16_s16_1col_minmax;
    if (two_col_buffer != input_buffer) {
        mat_mult_kernel_func(
            weight,
            (int16_t *)two_col_buffer,
            bias,
            scale,
            act_min,act_max,
            output,output_ch,output_offset
        );
    }
}

void mat_mult_kernel_input3_3x3_s16_s16_2col_ssat(
    const int16_t *input_a,
    const int16_t *input_b,
    const int32_t *bias,
    const int32_t *scale,
    const int32_t act_min,const int32_t act_max,
    int8_t *output,const int32_t output_ch,const int32_t output_offset)
{
    /* set up the second output pointers */
    int8_t *output0=output;
    int8_t *output1=output+output_ch;
    int32_t row_count = output_ch / 3;
    while (row_count) {
        /* setup pointers for B */
        const int16_t *ip_b0 = input_b;

        /* Init accumulator with bias for channel N and N + 1 */
        int32_t ch_0_out_0 = *bias;
        int32_t ch_0_out_1 = *bias++;
        int32_t ch_1_out_0 = *bias;
        int32_t ch_1_out_1 = *bias++;
        int32_t ch_2_out_0 = *bias;
        int32_t ch_2_out_1 = *bias++;

        //------------------4
        int32_t a0,a1,a2,b0;

        b0 = mte_read_q15x2(&ip_b0[0]);
        a0=mte_read_q15x2(input_a);
        a1=mte_read_q15x2(input_a+27);
        a2=mte_read_q15x2(input_a+54);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+0]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);


        b0 = mte_read_q15x2(&ip_b0[2]);
        a0=mte_read_q15x2(input_a+2);
        a1=mte_read_q15x2(input_a+27+2);
        a2=mte_read_q15x2(input_a+54+2);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+2]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);

        //------------------8
        b0 = mte_read_q15x2(&ip_b0[4]);

        a0=mte_read_q15x2(input_a+4);
        a1=mte_read_q15x2(input_a+27+4);
        a2=mte_read_q15x2(input_a+54+4);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+4]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);

        b0 = mte_read_q15x2(&ip_b0[6]);

        a0=mte_read_q15x2(input_a+6);
        a1=mte_read_q15x2(input_a+27+6);
        a2=mte_read_q15x2(input_a+54+6);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+6]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);

        //------------------12
        b0 = mte_read_q15x2(&ip_b0[8]);

        a0=mte_read_q15x2(input_a+8);
        a1=mte_read_q15x2(input_a+27+8);
        a2=mte_read_q15x2(input_a+54+8);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+8]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);

        b0 = mte_read_q15x2(&ip_b0[10]);

        a0=mte_read_q15x2(input_a+10);
        a1=mte_read_q15x2(input_a+27+10);
        a2=mte_read_q15x2(input_a+54+10);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+10]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);

        //------------------16
        b0 = mte_read_q15x2(&ip_b0[12]);

        a0=mte_read_q15x2(input_a+12);
        a1=mte_read_q15x2(input_a+27+12);
        a2=mte_read_q15x2(input_a+54+12);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+12]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);

        b0 = mte_read_q15x2(&ip_b0[14]);

        a0=mte_read_q15x2(input_a+14);
        a1=mte_read_q15x2(input_a+27+14);
        a2=mte_read_q15x2(input_a+54+14);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+14]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);

        //------------------20
        b0 = mte_read_q15x2(&ip_b0[16]);

        a0=mte_read_q15x2(input_a+16);
        a1=mte_read_q15x2(input_a+27+16);
        a2=mte_read_q15x2(input_a+54+16);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+16]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);

        b0 = mte_read_q15x2(&ip_b0[18]);

        a0=mte_read_q15x2(input_a+18);
        a1=mte_read_q15x2(input_a+27+18);
        a2=mte_read_q15x2(input_a+54+18);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+18]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);

        //------------------24
        b0 = mte_read_q15x2(&ip_b0[20]);

        a0=mte_read_q15x2(input_a+20);
        a1=mte_read_q15x2(input_a+27+20);
        a2=mte_read_q15x2(input_a+54+20);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+20]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);

        b0 = mte_read_q15x2(&ip_b0[22]);

        a0=mte_read_q15x2(input_a+22);
        a1=mte_read_q15x2(input_a+27+22);
        a2=mte_read_q15x2(input_a+54+22);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+22]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);

        //------------------25,26,27
        b0 = mte_read_q15x2(&ip_b0[24]);

        a0=mte_read_q15x2(input_a+24);
        a1=mte_read_q15x2(input_a+27+24);
        a2=mte_read_q15x2(input_a+54+24);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+24]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);


        b0 = mte_read_q15x2(&ip_b0[26]);
        a0=mte_read_q15x2(input_a+26);
        a1=mte_read_q15x2(input_a+27+26);
        a2=mte_read_q15x2(input_a+54+26);
        ch_0_out_0 =__SMLABB(a0,b0,ch_0_out_0);
        ch_1_out_0 =__SMLABB(a1,b0,ch_1_out_0);
        ch_2_out_0 = __SMLABB(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+26]);
        ch_0_out_1 =__SMLABB(a0,b0,ch_0_out_1);
        ch_1_out_1 =__SMLABB(a1,b0,ch_1_out_1);
        ch_2_out_1 = __SMLABB(a2, b0, ch_2_out_1);


        ch_0_out_0=__SMMLAR(ch_0_out_0,scale[0],output_offset);
        ch_0_out_1=__SMMLAR(ch_0_out_1,scale[0],output_offset);
        ch_1_out_0=__SMMLAR(ch_1_out_0,scale[1],output_offset);
        ch_1_out_1=__SMMLAR(ch_1_out_1,scale[1],output_offset);
        ch_2_out_0=__SMMLAR(ch_2_out_0,scale[2],output_offset);
        ch_2_out_1=__SMMLAR(ch_2_out_1,scale[2],output_offset);
        ch_0_out_0=__SSAT8(ch_0_out_0);
        ch_0_out_1=__SSAT8(ch_0_out_1);
        ch_1_out_0=__SSAT8(ch_1_out_0);
        ch_1_out_1=__SSAT8(ch_1_out_1);
        ch_2_out_0=__SSAT8(ch_2_out_0);
        ch_2_out_1=__SSAT8(ch_2_out_1);
        *output0++ = (int8_t) ch_0_out_0;
        *output1++ = (int8_t) ch_0_out_1;
        *output0++ = (int8_t) ch_1_out_0;
        *output1++ = (int8_t) ch_1_out_1;
        *output0++ = (int8_t) ch_2_out_0;
        *output1++ = (int8_t) ch_2_out_1;

        /* skip row */
        scale += 3;
        input_a += 81;
        row_count--;
    }
}

void mat_mult_kernel_input3_3x3_s16_s16_2col_minmax(
    const int16_t *input_a,
    const int16_t *input_b,
    const int32_t *bias,
    const int32_t *scale,
    const int32_t act_min,const int32_t act_max,
    int8_t *output,const int32_t output_ch,const int32_t output_offset)
{
    /* set up the second output pointers */
    int8_t *output0=output;
    int8_t *output1=output+output_ch;
    int32_t row_count = output_ch / 3;
    while (row_count) {
        /* setup pointers for B */
        const int16_t *ip_b0 = input_b;

        /* Init accumulator with bias for channel N and N + 1 */
        int32_t ch_0_out_0 = *bias;
        int32_t ch_0_out_1 = *bias++;
        int32_t ch_1_out_0 = *bias;
        int32_t ch_1_out_1 = *bias++;
        int32_t ch_2_out_0 = *bias;
        int32_t ch_2_out_1 = *bias++;

        //------------------4
        int32_t a0,a1,a2,b0;

        b0 = mte_read_q15x2(&ip_b0[0]);
        a0=mte_read_q15x2(input_a);
        a1=mte_read_q15x2(input_a+27);
        a2=mte_read_q15x2(input_a+54);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+0]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);


        b0 = mte_read_q15x2(&ip_b0[2]);
        a0=mte_read_q15x2(input_a+2);
        a1=mte_read_q15x2(input_a+27+2);
        a2=mte_read_q15x2(input_a+54+2);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+2]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);

        //------------------8
        b0 = mte_read_q15x2(&ip_b0[4]);

        a0=mte_read_q15x2(input_a+4);
        a1=mte_read_q15x2(input_a+27+4);
        a2=mte_read_q15x2(input_a+54+4);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+4]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);

        b0 = mte_read_q15x2(&ip_b0[6]);

        a0=mte_read_q15x2(input_a+6);
        a1=mte_read_q15x2(input_a+27+6);
        a2=mte_read_q15x2(input_a+54+6);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+6]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);

        //------------------12
        b0 = mte_read_q15x2(&ip_b0[8]);

        a0=mte_read_q15x2(input_a+8);
        a1=mte_read_q15x2(input_a+27+8);
        a2=mte_read_q15x2(input_a+54+8);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+8]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);

        b0 = mte_read_q15x2(&ip_b0[10]);

        a0=mte_read_q15x2(input_a+10);
        a1=mte_read_q15x2(input_a+27+10);
        a2=mte_read_q15x2(input_a+54+10);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+10]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);

        //------------------16
        b0 = mte_read_q15x2(&ip_b0[12]);

        a0=mte_read_q15x2(input_a+12);
        a1=mte_read_q15x2(input_a+27+12);
        a2=mte_read_q15x2(input_a+54+12);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+12]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);

        b0 = mte_read_q15x2(&ip_b0[14]);

        a0=mte_read_q15x2(input_a+14);
        a1=mte_read_q15x2(input_a+27+14);
        a2=mte_read_q15x2(input_a+54+14);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+14]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);

        //------------------20
        b0 = mte_read_q15x2(&ip_b0[16]);

        a0=mte_read_q15x2(input_a+16);
        a1=mte_read_q15x2(input_a+27+16);
        a2=mte_read_q15x2(input_a+54+16);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+16]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);

        b0 = mte_read_q15x2(&ip_b0[18]);

        a0=mte_read_q15x2(input_a+18);
        a1=mte_read_q15x2(input_a+27+18);
        a2=mte_read_q15x2(input_a+54+18);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+18]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);

        //------------------24
        b0 = mte_read_q15x2(&ip_b0[20]);

        a0=mte_read_q15x2(input_a+20);
        a1=mte_read_q15x2(input_a+27+20);
        a2=mte_read_q15x2(input_a+54+20);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+20]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);

        b0 = mte_read_q15x2(&ip_b0[22]);

        a0=mte_read_q15x2(input_a+22);
        a1=mte_read_q15x2(input_a+27+22);
        a2=mte_read_q15x2(input_a+54+22);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+22]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);

        //------------------25,26,27
        b0 = mte_read_q15x2(&ip_b0[24]);

        a0=mte_read_q15x2(input_a+24);
        a1=mte_read_q15x2(input_a+27+24);
        a2=mte_read_q15x2(input_a+54+24);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+24]);
        ch_0_out_1 = __SMLAD(a0, b0, ch_0_out_1);
        ch_1_out_1 = __SMLAD(a1, b0, ch_1_out_1);
        ch_2_out_1 = __SMLAD(a2, b0, ch_2_out_1);


        b0 = mte_read_q15x2(&ip_b0[26]);
        a0=mte_read_q15x2(input_a+26);
        a1=mte_read_q15x2(input_a+27+26);
        a2=mte_read_q15x2(input_a+54+26);
        ch_0_out_0 =__SMLABB(a0,b0,ch_0_out_0);
        ch_1_out_0 =__SMLABB(a1,b0,ch_1_out_0);
        ch_2_out_0 = __SMLABB(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[27+26]);
        ch_0_out_1 =__SMLABB(a0,b0,ch_0_out_1);
        ch_1_out_1 =__SMLABB(a1,b0,ch_1_out_1);
        ch_2_out_1 = __SMLABB(a2, b0, ch_2_out_1);


        ch_0_out_0=__SMMLAR(ch_0_out_0,scale[0],output_offset);
        ch_0_out_1=__SMMLAR(ch_0_out_1,scale[0],output_offset);
        ch_1_out_0=__SMMLAR(ch_1_out_0,scale[1],output_offset);
        ch_1_out_1=__SMMLAR(ch_1_out_1,scale[1],output_offset);
        ch_2_out_0=__SMMLAR(ch_2_out_0,scale[2],output_offset);
        ch_2_out_1=__SMMLAR(ch_2_out_1,scale[2],output_offset);
        ch_0_out_0=MAX(ch_0_out_0,act_min);
        ch_0_out_1=MAX(ch_0_out_1,act_min);
        ch_1_out_0=MAX(ch_1_out_0,act_min);
        ch_1_out_1=MAX(ch_1_out_1,act_min);
        ch_2_out_0=MAX(ch_2_out_0,act_min);
        ch_2_out_1=MAX(ch_2_out_1,act_min);
        ch_0_out_0=MIN(ch_0_out_0,act_max);
        ch_0_out_1=MIN(ch_0_out_1,act_max);
        ch_1_out_0=MIN(ch_1_out_0,act_max);
        ch_1_out_1=MIN(ch_1_out_1,act_max);
        ch_2_out_0=MIN(ch_2_out_0,act_max);
        ch_2_out_1=MIN(ch_2_out_1,act_max);
        *output0++ = (int8_t) ch_0_out_0;
        *output1++ = (int8_t) ch_0_out_1;
        *output0++ = (int8_t) ch_1_out_0;
        *output1++ = (int8_t) ch_1_out_1;
        *output0++ = (int8_t) ch_2_out_0;
        *output1++ = (int8_t) ch_2_out_1;

        /* skip row */
        scale += 3;
        input_a += 81;
        row_count--;
    }
}

void mat_mult_kernel_input3_3x3_s16_s16_1col_ssat(
    const int16_t *input_a,
    const int16_t *input_b,
    const int32_t *bias,
    const int32_t *scale,
    const int32_t act_min,const int32_t act_max,
    int8_t *output,const int32_t output_ch,const int32_t output_offset)
{
    /* set up the second output pointers */
    int8_t *output0=output;
    int32_t row_count = output_ch / 3;
    while (row_count) {
        /* setup pointers for B */
        const int16_t *ip_b0 = input_b;

        /* Init accumulator with bias for channel N and N + 1 */
        int32_t ch_0_out_0 = *bias;
        int32_t ch_1_out_0 = *bias;
        int32_t ch_2_out_0 = *bias;

        //------------------4
        int32_t a0,a1,a2,b0;

        b0 = mte_read_q15x2(&ip_b0[0]);
        a0=mte_read_q15x2(input_a);
        a1=mte_read_q15x2(input_a+27);
        a2=mte_read_q15x2(input_a+54);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[2]);
        a0=mte_read_q15x2(input_a+2);
        a1=mte_read_q15x2(input_a+27+2);
        a2=mte_read_q15x2(input_a+54+2);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        //------------------8
        b0 = mte_read_q15x2(&ip_b0[4]);

        a0=mte_read_q15x2(input_a+4);
        a1=mte_read_q15x2(input_a+27+4);
        a2=mte_read_q15x2(input_a+54+4);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[6]);

        a0=mte_read_q15x2(input_a+6);
        a1=mte_read_q15x2(input_a+27+6);
        a2=mte_read_q15x2(input_a+54+6);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        //------------------12
        b0 = mte_read_q15x2(&ip_b0[8]);

        a0=mte_read_q15x2(input_a+8);
        a1=mte_read_q15x2(input_a+27+8);
        a2=mte_read_q15x2(input_a+54+8);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[10]);

        a0=mte_read_q15x2(input_a+10);
        a1=mte_read_q15x2(input_a+27+10);
        a2=mte_read_q15x2(input_a+54+10);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        //------------------16
        b0 = mte_read_q15x2(&ip_b0[12]);

        a0=mte_read_q15x2(input_a+12);
        a1=mte_read_q15x2(input_a+27+12);
        a2=mte_read_q15x2(input_a+54+12);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[14]);

        a0=mte_read_q15x2(input_a+14);
        a1=mte_read_q15x2(input_a+27+14);
        a2=mte_read_q15x2(input_a+54+14);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        //------------------20
        b0 = mte_read_q15x2(&ip_b0[16]);

        a0=mte_read_q15x2(input_a+16);
        a1=mte_read_q15x2(input_a+27+16);
        a2=mte_read_q15x2(input_a+54+16);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[18]);

        a0=mte_read_q15x2(input_a+18);
        a1=mte_read_q15x2(input_a+27+18);
        a2=mte_read_q15x2(input_a+54+18);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        //------------------24
        b0 = mte_read_q15x2(&ip_b0[20]);

        a0=mte_read_q15x2(input_a+20);
        a1=mte_read_q15x2(input_a+27+20);
        a2=mte_read_q15x2(input_a+54+20);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[22]);

        a0=mte_read_q15x2(input_a+22);
        a1=mte_read_q15x2(input_a+27+22);
        a2=mte_read_q15x2(input_a+54+22);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        //------------------25,26,27
        b0 = mte_read_q15x2(&ip_b0[24]);

        a0=mte_read_q15x2(input_a+24);
        a1=mte_read_q15x2(input_a+27+24);
        a2=mte_read_q15x2(input_a+54+24);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);


        b0 = mte_read_q15x2(&ip_b0[26]);
        a0=mte_read_q15x2(input_a+26);
        a1=mte_read_q15x2(input_a+27+26);
        a2=mte_read_q15x2(input_a+54+26);
        ch_0_out_0 =__SMLABB(a0,b0,ch_0_out_0);
        ch_1_out_0 =__SMLABB(a1,b0,ch_1_out_0);
        ch_2_out_0 = __SMLABB(a2, b0, ch_2_out_0);


        ch_0_out_0=__SMMLAR(ch_0_out_0,scale[0],output_offset);
        ch_1_out_0=__SMMLAR(ch_1_out_0,scale[1],output_offset);
        ch_2_out_0=__SMMLAR(ch_2_out_0,scale[2],output_offset);
        ch_0_out_0=__SSAT8(ch_0_out_0);
        ch_1_out_0=__SSAT8(ch_1_out_0);
        ch_2_out_0=__SSAT8(ch_2_out_0);
        *output0++ = (int8_t) ch_0_out_0;
        *output0++ = (int8_t) ch_1_out_0;
        *output0++ = (int8_t) ch_2_out_0;

        /* skip row */
        scale += 3;
        input_a += 81;
        row_count--;
    }
}

void mat_mult_kernel_input3_3x3_s16_s16_1col_minmax(
    const int16_t *input_a,
    const int16_t *input_b,
    const int32_t *bias,
    const int32_t *scale,
    const int32_t act_min,const int32_t act_max,
    int8_t *output,const int32_t output_ch,const int32_t output_offset)
{
    /* set up the second output pointers */
    int8_t *output0=output;
    int32_t row_count = output_ch / 3;
    while (row_count) {
        /* setup pointers for B */
        const int16_t *ip_b0 = input_b;

        /* Init accumulator with bias for channel N and N + 1 */
        int32_t ch_0_out_0 = *bias;
        int32_t ch_1_out_0 = *bias;
        int32_t ch_2_out_0 = *bias;

        //------------------4
        int32_t a0,a1,a2,b0;

        b0 = mte_read_q15x2(&ip_b0[0]);
        a0=mte_read_q15x2(input_a);
        a1=mte_read_q15x2(input_a+27);
        a2=mte_read_q15x2(input_a+54);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[2]);
        a0=mte_read_q15x2(input_a+2);
        a1=mte_read_q15x2(input_a+27+2);
        a2=mte_read_q15x2(input_a+54+2);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        //------------------8
        b0 = mte_read_q15x2(&ip_b0[4]);

        a0=mte_read_q15x2(input_a+4);
        a1=mte_read_q15x2(input_a+27+4);
        a2=mte_read_q15x2(input_a+54+4);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[6]);

        a0=mte_read_q15x2(input_a+6);
        a1=mte_read_q15x2(input_a+27+6);
        a2=mte_read_q15x2(input_a+54+6);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        //------------------12
        b0 = mte_read_q15x2(&ip_b0[8]);

        a0=mte_read_q15x2(input_a+8);
        a1=mte_read_q15x2(input_a+27+8);
        a2=mte_read_q15x2(input_a+54+8);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[10]);

        a0=mte_read_q15x2(input_a+10);
        a1=mte_read_q15x2(input_a+27+10);
        a2=mte_read_q15x2(input_a+54+10);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        //------------------16
        b0 = mte_read_q15x2(&ip_b0[12]);

        a0=mte_read_q15x2(input_a+12);
        a1=mte_read_q15x2(input_a+27+12);
        a2=mte_read_q15x2(input_a+54+12);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[14]);

        a0=mte_read_q15x2(input_a+14);
        a1=mte_read_q15x2(input_a+27+14);
        a2=mte_read_q15x2(input_a+54+14);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        //------------------20
        b0 = mte_read_q15x2(&ip_b0[16]);

        a0=mte_read_q15x2(input_a+16);
        a1=mte_read_q15x2(input_a+27+16);
        a2=mte_read_q15x2(input_a+54+16);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[18]);

        a0=mte_read_q15x2(input_a+18);
        a1=mte_read_q15x2(input_a+27+18);
        a2=mte_read_q15x2(input_a+54+18);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        //------------------24
        b0 = mte_read_q15x2(&ip_b0[20]);

        a0=mte_read_q15x2(input_a+20);
        a1=mte_read_q15x2(input_a+27+20);
        a2=mte_read_q15x2(input_a+54+20);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        b0 = mte_read_q15x2(&ip_b0[22]);

        a0=mte_read_q15x2(input_a+22);
        a1=mte_read_q15x2(input_a+27+22);
        a2=mte_read_q15x2(input_a+54+22);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);

        //------------------25,26,27
        b0 = mte_read_q15x2(&ip_b0[24]);

        a0=mte_read_q15x2(input_a+24);
        a1=mte_read_q15x2(input_a+27+24);
        a2=mte_read_q15x2(input_a+54+24);
        ch_0_out_0 = __SMLAD(a0, b0, ch_0_out_0);
        ch_1_out_0 = __SMLAD(a1, b0, ch_1_out_0);
        ch_2_out_0 = __SMLAD(a2, b0, ch_2_out_0);


        b0 = mte_read_q15x2(&ip_b0[26]);
        a0=mte_read_q15x2(input_a+26);
        a1=mte_read_q15x2(input_a+27+26);
        a2=mte_read_q15x2(input_a+54+26);
        ch_0_out_0 =__SMLABB(a0,b0,ch_0_out_0);
        ch_1_out_0 =__SMLABB(a1,b0,ch_1_out_0);
        ch_2_out_0 = __SMLABB(a2, b0, ch_2_out_0);


        ch_0_out_0=__SMMLAR(ch_0_out_0,scale[0],output_offset);
        ch_1_out_0=__SMMLAR(ch_1_out_0,scale[1],output_offset);
        ch_2_out_0=__SMMLAR(ch_2_out_0,scale[2],output_offset);
        ch_0_out_0=MAX(ch_0_out_0,act_min);
        ch_1_out_0=MAX(ch_1_out_0,act_min);
        ch_2_out_0=MAX(ch_2_out_0,act_min);
        ch_0_out_0=MIN(ch_0_out_0,act_max);
        ch_1_out_0=MIN(ch_1_out_0,act_max);
        ch_2_out_0=MIN(ch_2_out_0,act_max);
        *output0++ = (int8_t) ch_0_out_0;
        *output0++ = (int8_t) ch_1_out_0;
        *output0++ = (int8_t) ch_2_out_0;

        /* skip row */
        scale += 3;
        input_a += 81;
        row_count--;
    }
}