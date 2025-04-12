//
// Created by kyuliea on 2025/4/10.
//
#include "mte_core.h"

void transpose(
    const int8_t *input, const int32_t ele_nums,const int32_t *perm, const int32_t *input_shape,const int32_t *output_shape, const int32_t dim_nums,
    int32_t *idx_buffer,
    int8_t *output
    ){
    int32_t i,j,k,output_idx;
    for(i=0;i<ele_nums;i++){
        k=i;
        output_idx=0;
        for(j=0;j<dim_nums;j++) {
            idx_buffer[dim_nums-1-j]=k%input_shape[dim_nums-1-j];
            k/=input_shape[dim_nums-1-j];
        }
        for(j=0;j<dim_nums;j++) {
            output_idx*=output_shape[j];
            output_idx+=idx_buffer[perm[j]];
        }
        output[output_idx]=input[i];
    }
}