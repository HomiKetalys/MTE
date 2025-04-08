//
// Created by kyuliea on 2025/4/1.
//
#include "mte_core.h"

void pack(const int8_t  *input_addrs,const int32_t channels,const int32_t pack_nums,const int32_t repeat_nums,int8_t *output) {
    int32_t i,j;
    for (i = 0; i < pack_nums; i++) {
        for(j=0;j<repeat_nums;j++)
        {
            memcpy(output,input_addrs,channels);
            output+=channels;
        }
        input_addrs+=channels;
    }
}