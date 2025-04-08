//
// Created by kyuliea on 2025/4/1.
//

#include "mte_core.h"

void max_pool2d(
    const int8_t* input, const int32_t input_h, const int32_t input_w,const uint16_t input_ch,const int32_t input_offset,
    const int32_t pad_h_low,const int32_t pad_h_high,const int32_t pad_w_low,const int32_t pad_w_high,
    const int32_t kernel_h, const int32_t kernel_w,const int32_t stride_h,const int32_t stride_w,
    const int32_t act_min,const int32_t act_max,
    int8_t *output,const int32_t output_h, const int32_t output_w)
{

    int32_t h, w, ch,win_h,win_w;
    for(ch = 0; ch < input_ch; ch++){
        for(h = 0; h < output_h; h++){
            for(w = 0; w < output_w; w++){
                int32_t in_w=w*stride_h,in_h=h*stride_w;
                int32_t max=(in_h<pad_h_low||input_h+pad_h_low<in_h+kernel_h||in_w<pad_w_low||input_w+pad_w_low<in_w+kernel_w)?input_offset:act_min;
                int32_t h_begin=MAX(in_h-pad_h_low,0);
                int32_t h_end=MIN(in_h+kernel_h-pad_h_low,input_h);
                for(win_h=h_begin;win_h<h_end;win_h++){
                    int32_t w_begin=MAX(in_w-pad_w_low,0);
                    int32_t w_end=MIN(in_w+kernel_h-pad_w_low,input_w);
                    for(win_w=w_begin;win_w<w_end;win_w++){
                        max=MAX(max,input[(win_h*input_w+win_w)*input_ch+ch]);
                    }
                }
                max = MAX(max, act_min);
                max = MIN(max, act_max);
                output[(h * output_w+w) * input_ch + ch] = (int8_t)max;
            }
        }
    }
}