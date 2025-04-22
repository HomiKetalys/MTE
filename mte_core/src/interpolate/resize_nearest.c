//
// Created by kyuliea on 2025/4/18.
//
#include "mte_core.h"

void resize_nearest(
    const int8_t *input,const int32_t input_ch,const int32_t input_h,const int32_t input_w,
    int8_t *output,const int32_t output_h,const int32_t output_w
    ){
    int32_t h,w;
    const int32_t l15_output_h=output_h<<15;
    const int32_t l15_output_w=output_w<<15;
    const int32_t l16_input_h=input_h<<16;
    const int32_t l16_input_w=input_w<<16;
    const int32_t l16_output_h=output_h<<16;
    const int32_t l16_output_w=output_w<<16;

    const int32_t var0=input_w*input_ch;
    const int32_t var1=output_w*input_ch;
    for(h=0;h<output_h;h++){
        for(w=0;w<output_w;w++){
            const int32_t in_h = (h * l16_input_h + l15_output_h) / l16_output_h;
            const int32_t in_w = (w * l16_input_w + l15_output_w) / l16_output_w;
            const int8_t *input_p=input+in_h*var1+in_w*input_ch;
            int8_t *output_p=output+h*var0+w*input_ch;
            memcpy(output_p,input_p,input_ch);
        }
    }
}