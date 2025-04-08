//
// Created by kyuliea on 2025/4/1.
//
#include "mte_core.h"

void add(const int8_t* input0, const int32_t scale0,
         const int8_t* input1, const int32_t scale1,
         const int32_t ele_nums,
         int8_t *output,const int32_t offset
         ){

    int8_t *final_output=output+4*(ele_nums/4);
    const int32_t scale= __PKHBT(scale0,scale1,16);
    while(output<final_output) {
        int32_t in0,in1,in2,in3;
        in0= __PKHBT(input0[0],input1[0],16);
        in1= __PKHBT(input0[1],input1[1],16);
        in2= __PKHBT(input0[2],input1[2],16);
        in3= __PKHBT(input0[3],input1[3],16);
        int32_t out0=offset,out1=offset,out2=offset,out3=offset;
        out0=__SMLAD(in0,scale,out0);
        out1=__SMLAD(in1,scale,out1);
        out2=__SMLAD(in2,scale,out2);
        out3=__SMLAD(in3,scale,out3);
        out0=out0>>14;
        out1=out1>>14;
        out2=out2>>14;
        out3=out3>>14;
        out0= __SSAT8(out0);
        out1= __SSAT8(out1);
        out2= __SSAT8(out2);
        out3= __SSAT8(out3);
        output[0] = (int8_t)out0;
        output[1] = (int8_t)out1;
        output[2] = (int8_t)out2;
        output[3] = (int8_t)out3;
        input0+=4;
        input1+=4;
        output+=4;
    }
    final_output=final_output+ele_nums%4;
    while(output<final_output)
    {
        int32_t in0;
        in0= __PKHBT(input0[0],input1[0],16);
        int32_t out0=offset;
        out0=__SMLAD(in0,scale,out0);
        out0=out0>>14;
        out0= __SSAT8(out0);
        output[0] = (int8_t)out0;
        input0++;
        input1++;
        output++;
    }
}