//
// Created by kyuliea on 2025/4/1.
//
#include "mte_core.h"

void gather(const int8_t *input,const int32_t gather_nums,const int32_t gather_block_size,const int32_t block_size,
            const int32_t *gather_idx,const int32_t idx_nums,int8_t *output) {
    int32_t i,j;
    for (i = 0; i < gather_nums; i++) {
        const int8_t *mini_gather_addr=input+i*gather_block_size;
        for(j=0;j<idx_nums;j++)
        {
            memcpy(output,mini_gather_addr+gather_idx[j]*block_size,block_size);
            output+=block_size;
        }
    }
}