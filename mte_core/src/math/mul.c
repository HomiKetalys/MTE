//
// Created by kyuliea on 2025/4/18.
//
#include "mte_core.h"

void mul(
    const int8_t *input0,const int32_t input0_offset,const int8_t *input1,const int32_t input1_offset,
    const int32_t ele_nums,
    int8_t *output,const int32_t scale,const int32_t output_offset
    ){

    int32_t i;
    for(i=0;i<ele_nums;i+=4)
    {
        int32_t in00xin10=(input0[i]-input0_offset)*(input1[i]-input1_offset);
        int32_t in01xin11=(input0[i+1]-input0_offset)*(input1[i+1]-input1_offset);
        int32_t in02xin12=(input0[i+2]-input0_offset)*(input1[i+2]-input1_offset);
        int32_t in03xin13=(input0[i+3]-input0_offset)*(input1[i+3]-input1_offset);
        in00xin10=__SMMLAR(in00xin10,scale,output_offset);
        in01xin11=__SMMLAR(in01xin11,scale,output_offset);
        in02xin12=__SMMLAR(in02xin12,scale,output_offset);
        in03xin13=__SMMLAR(in03xin13,scale,output_offset);
        in00xin10= __SSAT8(in00xin10);
        in01xin11= __SSAT8(in01xin11);
        in02xin12= __SSAT8(in02xin12);
        in03xin13= __SSAT8(in03xin13);
        output[i]=(int8_t)in00xin10;
        output[i+1]=(int8_t)in01xin11;
        output[i+2]=(int8_t)in02xin12;
        output[i+3]=(int8_t)in03xin13;
    }
}