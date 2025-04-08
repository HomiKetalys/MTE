//
// Created by kyuliea on 2025/4/1.
//
#include "mte_core.h"
extern int8_t *mte_mem;
void concat(const int32_t  *input_addrs,const int32_t *channels,const int32_t addr_nums,const int32_t concat_nums,int8_t *output) {
    int32_t i,j;
    for (i = 0; i < concat_nums; i++) {
        for(j=0;j<addr_nums;j++)
        {
            memcpy(output,mte_mem+input_addrs[j]+i*channels[j],channels[j]);
            output+=channels[j];
        }
    }
}